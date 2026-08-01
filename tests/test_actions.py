"""Actions: the link tree, deletion safety and idempotency.

Driven through the `applying` service, which is what both front ends call, so
these tests exercise the same wiring a real run uses.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from app.actions import NameAllocator, PlannedAction
from app.actions.registry import default_registry
from app.domain.decision import Decision
from app.domain.rules import Always, HasClass, InDoubt, Rule, RuleSet
from app.domain.rules.ruleset import DOUBT_RULE_NAME, doubt_rule
from app.services import applying, library, maintenance


def _id_of(ctx, filename):
    row = ctx.storage.engine.conn.execute(
        "SELECT id FROM images WHERE filename = ?", (filename,)
    ).fetchone()
    return row["id"]


def with_doubt(ruleset, action):
    """The same rules, with the doubt rule set to a given action."""
    ordinary = [r for r in ruleset.rules if r.name != DOUBT_RULE_NAME]
    return replace(ruleset, rules=(*ordinary, doubt_rule(action)))


def categorise(storage, mapping: dict[str, tuple[str, str]], review: set[str] = frozenset()):
    """Pretend the detector ran: assign category/action per filename."""
    for row in storage.engine.conn.execute("SELECT id, path FROM images").fetchall():
        name = os.path.basename(row["path"])
        category, action = mapping.get(name, ("none", "ignore"))
        storage.results.finish_detect(
            row["id"], [], Decision(category, action, name in review), "test.pt"
        )


@pytest.fixture
def populated(ctx):
    library.scan_library(ctx)
    categorise(ctx.storage, {
        "a.jpg": ("cat", "link"),
        "b.jpg": ("dog", "link"),
        "c.png": ("cat-dog", "link"),
        "d.jpeg": ("none", "ignore"),
        "e.webp": ("none", "ignore"),
    }, review={"e.webp"})
    return ctx


# -------------------------------------------------------------------- linking


def test_builds_the_expected_folders(populated, ruleset):
    output = populated.settings.output.folder
    stats, _ = applying.apply(populated, ruleset)
    # 3 categorised. The fourth, e.webp, is in doubt and the doubt rule moves by
    # default — which needs a confirmation this call did not give.
    assert stats.created == 3
    assert stats.skipped == 1
    assert sorted(p.name for p in output.iterdir()) == ["Cat", "Cat-Dog", "Dog"]
    assert len(list((output / "Cat").iterdir())) == 1


def test_copies_are_real_independent_files(populated, ruleset, library_root):
    applying.apply(populated, ruleset)
    copied = next((populated.settings.output.folder / "Cat").iterdir())
    original = library_root / "a.jpg"
    assert copied.is_file() and not copied.is_symlink()
    assert copied.read_bytes() == original.read_bytes()
    # copy2, so a folder sorted by date is still sorted by the photo's date.
    assert copied.stat().st_mtime == original.stat().st_mtime


def test_originals_are_never_touched(populated, ruleset, library_root):
    before = sorted(p.name for p in library_root.rglob("*") if p.is_file())
    applying.apply(populated, ruleset)
    assert sorted(p.name for p in library_root.rglob("*") if p.is_file()) == before


def test_applying_twice_changes_nothing(populated, ruleset):
    applying.apply(populated, ruleset)
    stats, _ = applying.apply(populated, ruleset)
    assert (stats.created, stats.existing, stats.pruned) == (0, 3, 0)


def test_recategorising_prunes_the_old_link(populated, ruleset):
    output = populated.settings.output.folder
    applying.apply(populated, ruleset)
    target = populated.storage.engine.conn.execute(
        "SELECT id FROM images WHERE path LIKE '%/a.jpg'"
    ).fetchone()["id"]
    populated.storage.images.set_decision(target, "dog", "link", False)

    stats, _ = applying.apply(populated, ruleset)
    assert stats.pruned == 1
    assert not (output / "Cat").exists()      # emptied, then removed
    assert len(list((output / "Dog").iterdir())) == 2


# ----------------------------------------------------------------- the doubt rule


def test_doubt_applies_even_to_an_image_the_rules_ignore(populated, ruleset):
    """e.webp matched "none"/ignore but the detector was unsure — the whole
    point of the doubt rule is that a human still sees it."""
    applying.apply(populated, with_doubt(ruleset, "copy"))
    surfaced = [p.name for p in (populated.settings.output.folder / "_Review").iterdir()]
    assert surfaced == ["e.webp"]


def test_doubt_set_to_copy_leaves_the_original_and_the_normal_rule_alone(populated, ruleset):
    """A copy can coexist with whatever the matched rule wanted, so both run."""
    populated.storage.images.set_decision(_id_of(populated, "a.jpg"), "cat", "copy", True)

    applying.apply(populated, with_doubt(ruleset, "copy"), confirmed=True)
    output = populated.settings.output.folder
    assert (output / "_Review" / "a.jpg").is_file()
    assert (output / "Cat" / "a.jpg").is_file()      # still sorted normally


def test_doubt_set_to_move_takes_the_photo_instead_of_the_normal_rule(populated, ruleset,
                                                                     library_root):
    """A photo nobody could settle has not earned a place in a category folder,
    and cannot be in two at once."""
    populated.storage.images.set_decision(_id_of(populated, "a.jpg"), "cat", "copy", True)

    applying.apply(populated, with_doubt(ruleset, "move"), confirmed=True)
    output = populated.settings.output.folder
    assert (output / "_Review" / "a.jpg").is_file()
    assert not (output / "Cat").exists()             # not filed as a cat
    assert not (library_root / "a.jpg").exists()     # and gone from the library


def test_doubt_set_to_move_runs_before_a_rule_that_would_consume_the_photo(populated,
                                                                          library_root):
    """Ordering matters: a `copy` doubt rule has to run before a `move` rule, or
    there is nothing left to copy by the time it does."""
    populated.storage.images.set_decision(_id_of(populated, "a.jpg"), "cat", "move", True)
    rules = RuleSet((Rule("cat", HasClass("cat"), action="move"), doubt_rule("copy")))

    applying.apply(populated, rules, confirmed=True)
    output = populated.settings.output.folder
    assert (output / "_Review" / "a.jpg").is_file()
    assert (output / "Cat" / "a.jpg").is_file()
    assert not (library_root / "a.jpg").exists()


def test_doubt_set_to_ignore_surfaces_nothing(populated, ruleset):
    applying.apply(populated, with_doubt(ruleset, "ignore"))
    assert not (populated.settings.output.folder / "_Review").exists()


def test_doubt_moves_to_review_even_when_the_matched_rule_deletes(ctx, delete_rules,
                                                                   library_root):
    """The doubt rule's default action (move) always wins over a consuming
    matched rule, so an uncertain photo is never silently dropped. Regressed
    once, back when moving and deleting needed separate settings-level
    permissions: granting only one left the other's plan quietly filtered out,
    so the photo ended up neither deleted nor reviewed."""
    library.scan_library(ctx)
    categorise(ctx.storage, {"a.jpg": ("junk", "delete")}, review={"a.jpg"})
    applying.apply(ctx, delete_rules, confirmed=True)
    assert (ctx.settings.output.folder / "_Review" / "a.jpg").is_file()
    assert not (library_root / "a.jpg").exists()


def test_name_collisions_are_disambiguated_stably():
    allocator = NameAllocator()
    first = allocator.allocate(Path("/out"), "x.jpg", 1)
    second = allocator.allocate(Path("/out"), "x.jpg", 2)
    assert first == "/out/x.jpg"
    assert second == "/out/x_2.jpg"


# ------------------------------------------------------- output writability


def test_probe_passes_for_a_writable_folder(tmp_path):
    from app import filesystem

    assert filesystem.unwritable(tmp_path / "out") is None


def test_probe_leaves_nothing_behind(tmp_path):
    from app import filesystem

    out = tmp_path / "out"
    filesystem.unwritable(out)
    assert list(out.iterdir()) == []


def test_probe_reports_a_folder_it_cannot_write_to(tmp_path):
    from app import filesystem

    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o500)
    try:
        assert filesystem.unwritable(out) is not None
    finally:
        out.chmod(0o700)


def test_apply_fails_once_instead_of_per_photo(populated, ruleset):
    """The complaint this answers: one warning per photo, thousands of them,
    all saying the same thing about the same folder."""
    from app.errors import OutputNotWritable

    output = populated.settings.output.folder
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o500)
    try:
        with pytest.raises(OutputNotWritable) as caught:
            applying.apply(populated, ruleset)
        assert str(output) in str(caught.value)
    finally:
        output.chmod(0o700)


# --------------------------------------------------------------------- moving


@pytest.fixture
def move_rules():
    return RuleSet((
        Rule("cat", HasClass("cat"), action="move"),
        Rule("none", Always(), action="ignore"),
        doubt_rule("move"),
    ))


@pytest.fixture
def marked_for_moving(ctx):
    library.scan_library(ctx)
    categorise(ctx.storage, {"a.jpg": ("cat", "move"), "e.webp": ("cat", "move")},
               review={"e.webp"})
    return ctx


def test_move_relocates_the_original(marked_for_moving, move_rules, library_root):
    stats, _ = applying.apply(marked_for_moving, move_rules, confirmed=True)
    assert stats.moved == 2
    assert not (library_root / "a.jpg").exists()
    assert (marked_for_moving.settings.output.folder / "Cat" / "a.jpg").is_file()


def test_move_is_blocked_unless_explicitly_allowed(marked_for_moving, move_rules, library_root):
    stats, _ = applying.apply(marked_for_moving, move_rules, confirmed=False)
    assert stats.moved == 0
    assert stats.skipped == 2
    assert (library_root / "a.jpg").exists()


def test_move_updates_the_index_so_the_photo_is_still_found(marked_for_moving, move_rules):
    applying.apply(marked_for_moving, move_rules, confirmed=True)
    moved = marked_for_moving.settings.output.folder / "Cat" / "a.jpg"
    row = marked_for_moving.storage.engine.conn.execute(
        "SELECT path, filename, root, deleted FROM images WHERE filename = 'a.jpg'"
    ).fetchone()
    assert row["path"] == str(moved)
    assert row["root"] == str(moved.parent)
    assert not row["deleted"]  # filed, not gone


def test_a_second_run_leaves_moved_photos_alone(marked_for_moving, move_rules):
    """No link record to recognise them by, so the check is the path itself."""
    applying.apply(marked_for_moving, move_rules, confirmed=True)
    moved = marked_for_moving.settings.output.folder / "Cat" / "a.jpg"
    before = moved.stat().st_mtime_ns

    stats, planned = applying.apply(marked_for_moving, move_rules, confirmed=True)
    assert stats.moved == 0
    assert not [p for p in planned if p.action == "move"]
    assert moved.stat().st_mtime_ns == before
    assert len(list(moved.parent.iterdir())) == 1  # no duplicate alongside it


def test_a_doubtful_photo_goes_to_the_doubt_folder_not_its_category(marked_for_moving,
                                                                   move_rules):
    """e.webp needs review, so the doubt rule claims it and Cat/ does not."""
    applying.apply(marked_for_moving, move_rules, confirmed=True)
    output = marked_for_moving.settings.output.folder
    assert (output / "_Review" / "e.webp").is_file()
    assert not (output / "Cat" / "e.webp").exists()


def test_move_is_recorded_in_the_activity_log(marked_for_moving, move_rules):
    applying.apply(marked_for_moving, move_rules, confirmed=True)
    rows = marked_for_moving.storage.engine.conn.execute(
        "SELECT detail FROM action_log WHERE action = 'move'"
    ).fetchall()
    # The destination is logged, so a move can be traced (and undone) by hand.
    assert sorted(Path(r["detail"]).name for r in rows) == ["a.jpg", "e.webp"]


# ------------------------------------------------------------------- deleting


@pytest.fixture
def delete_rules():
    return RuleSet((
        Rule("junk", HasClass("cat"), action="delete"),
        Rule("none", Always(), action="ignore"),
    ))


@pytest.fixture
def marked_for_deletion(ctx):
    library.scan_library(ctx)
    categorise(ctx.storage, {"a.jpg": ("junk", "delete"), "b.jpg": ("junk", "delete")})
    return ctx


def test_deletion_is_blocked_unless_explicitly_allowed(marked_for_deletion, delete_rules,
                                                       library_root):
    stats, _ = applying.apply(marked_for_deletion, delete_rules, confirmed=False)
    assert stats.deleted == 0
    assert stats.skipped == 2
    assert (library_root / "a.jpg").exists()


def test_delete_moves_to_trash_by_default(marked_for_deletion, delete_rules, library_root):
    stats, _ = applying.apply(marked_for_deletion, delete_rules, confirmed=True)
    assert stats.deleted == 2
    assert not (library_root / "a.jpg").exists()
    assert (marked_for_deletion.settings.output.trash_folder / "a.jpg").exists()  # recoverable


def test_trash_preserves_the_folder_structure(ctx, delete_rules):
    library.scan_library(ctx)
    categorise(ctx.storage, {"c.png": ("junk", "delete")})
    applying.apply(ctx, delete_rules, confirmed=True)
    assert (ctx.settings.output.trash_folder / "sub" / "c.png").exists()


def test_deleted_images_are_recorded_and_not_reprocessed(marked_for_deletion, delete_rules):
    storage = marked_for_deletion.storage
    applying.apply(marked_for_deletion, delete_rules, confirmed=True)
    assert storage.images.count("deleted = 1") == 2
    assert storage.activity.recent()[0]["action"] == "delete"

    # A second pass must not try again on files that are already gone.
    stats, _ = applying.apply(marked_for_deletion, delete_rules, confirmed=True)
    assert stats.deleted == 0


def test_dry_run_writes_nothing(marked_for_deletion, delete_rules, library_root):
    stats, planned = applying.apply(
        marked_for_deletion, delete_rules, dry_run=True, confirmed=True
    )
    assert stats.by_action["delete"] == 2
    assert len(planned) == 2
    assert (library_root / "a.jpg").exists()
    assert not marked_for_deletion.settings.output.trash_folder.exists()


def test_dry_run_reports_what_would_be_blocked(marked_for_deletion, delete_rules):
    stats, planned = applying.apply(
        marked_for_deletion, delete_rules, dry_run=True, confirmed=False
    )
    assert stats.skipped == 2
    assert planned == []


# ------------------------------------------------------------------ registry


def test_registry_lists_actions_and_flags_the_consuming_ones():
    registry = default_registry()
    assert set(registry.names()) == {"copy", "move", "delete", "ignore"}
    assert registry.consuming_names() == {"delete", "move"}
    assert registry.get("copy").consumes_original is False


def test_unknown_action_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown action"):
        default_registry().get("teleport")


def test_a_new_action_needs_no_changes_elsewhere(populated):
    """The extension point: register a class, use it from a rule."""
    from app.actions import Action

    seen: list[str] = []

    class TagAction(Action):
        name = "tag"
        description = "test action"

        def plan(self, action_ctx, target, rule, namer):
            return [PlannedAction(target.image_id, self.name, target.path, detail=rule.name)]

        def execute(self, action_ctx, planned):
            seen.append(planned.source)
            return True

    registry = default_registry()
    registry.register(TagAction())
    ctx = replace(populated, actions=registry)

    for row in ctx.storage.engine.conn.execute("SELECT id FROM images").fetchall():
        ctx.storage.images.set_decision(row["id"], "tagged", "tag", False)
    ruleset = RuleSet((Rule("tagged", Always(), action="tag"),))

    stats, _ = applying.apply(ctx, ruleset)
    assert stats.by_action["tag"] == 5
    assert len(seen) == 5


# -------------------------------------------------------------------- verify


def test_verify_is_clean_after_apply(populated, ruleset):
    applying.apply(populated, ruleset)
    report = maintenance.verify(populated)
    assert report.as_dict() == {
        "copies_ok": 3, "copies_broken": 0, "copies_missing": 0, "sources_missing": 0
    }


def test_verify_notices_a_deleted_copy(populated, ruleset):
    applying.apply(populated, ruleset)
    next((populated.settings.output.folder / "Cat").iterdir()).unlink()
    assert maintenance.verify(populated).copies_missing == 1


def test_verify_notices_a_missing_original(populated, ruleset, library_root):
    applying.apply(populated, ruleset)
    (library_root / "a.jpg").unlink()
    assert maintenance.verify(populated).sources_missing == 1
