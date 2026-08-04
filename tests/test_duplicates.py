"""Near-duplicate detection: grouping, similarity scoring, marking, dismissing.

Driven through the `duplicates` service, the same way the web UI's Duplicates
page calls it: its own folders scanned into its own index (`ctx.dupes`), so the
scanner's hashing, the schema, and the clustering all run together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import filesystem
from app.errors import ConfigError
from app.services import duplicates


def _gradient(seed: int, size: tuple[int, int] = (64, 64)) -> Image.Image:
    """A smooth gradient whose direction depends on `seed`.

    Different seeds look different, and hash differently; the same seed
    re-saved is a near-duplicate. Unlike the solid colours
    `conftest.make_image` uses (fine for scanner/pipeline tests), a solid
    colour has no gradient at all, so every one of them dHashes to the same
    all-zero value — useless for telling "distinct photo" apart from
    "near-duplicate" here.
    """
    x, y = np.meshgrid(np.linspace(0, 255, size[0]), np.linspace(0, 255, size[1]))
    angle = np.radians((seed * 37) % 360)
    pattern = (x * np.cos(angle) + y * np.sin(angle)) % 256
    return Image.fromarray(np.stack([pattern] * 3, axis=-1).astype("uint8"), "RGB")


@pytest.fixture
def library_root(tmp_path: Path) -> Path:
    """Overrides `conftest`'s fixture of the same name: two near-duplicates
    (one re-encoded at a lower quality) and one genuinely different photo."""
    root = tmp_path / "photos"
    root.mkdir()
    _gradient(1).save(root / "original.png")
    _gradient(1).save(root / "resaved.jpg", quality=70)
    _gradient(2).save(root / "different.png")
    return root


def _id_of(ctx, filename: str) -> int:
    """An id from the *duplicate* index — its own database, with its own ids."""
    row = ctx.dupes.engine.conn.execute(
        "SELECT id FROM images WHERE filename = ?", (filename,)
    ).fetchone()
    return row["id"]


@pytest.fixture
def ctx(ctx):  # noqa: F811 - shadowing conftest's `ctx` on purpose, to scan once per test
    duplicates.scan_folders(ctx)
    return ctx


def test_groups_finds_the_near_duplicate_pair(ctx):
    result = duplicates.groups(ctx, max_distance=10)
    assert len(result) == 1
    ids = {img["id"] for img in result[0]["images"]}
    assert ids == {_id_of(ctx, "original.png"), _id_of(ctx, "resaved.jpg")}


def test_distinct_photo_is_never_grouped(ctx):
    result = duplicates.groups(ctx, max_distance=10)
    grouped_ids = {img["id"] for group in result for img in group["images"]}
    assert _id_of(ctx, "different.png") not in grouped_ids


def test_keeper_score_prefers_higher_resolution_then_larger_file():
    small = {"width": 64, "height": 64, "size": 5000}
    large = {"width": 128, "height": 128, "size": 3000}
    same_res_bigger_file = {"width": 64, "height": 64, "size": 9000}
    assert duplicates._keeper_score(large) > duplicates._keeper_score(small)
    assert duplicates._keeper_score(same_res_bigger_file) > duplicates._keeper_score(small)


def test_every_photo_carries_a_similarity_percentage(ctx):
    """Quoted against the group's suggested keeper, which is therefore 100%.

    Both photos here score 100: a perceptual hash is meant to survive a
    re-encode, and `resaved.jpg` is `original.png` at quality 70 — different
    bytes, different format, same picture. That is the whole point of grouping
    on the hash rather than on `filesystem.sha256_file`.
    """
    group = duplicates.groups(ctx, max_distance=10)[0]

    assert all(0.0 <= i["similarity"] <= 100.0 for i in group["images"])
    keeper = next(i for i in group["images"] if i["id"] == group["keep"])
    assert keeper["similarity"] == 100.0
    # The group's own score is its loosest pairing.
    assert group["similarity"] == min(i["similarity"] for i in group["images"])


def test_images_are_ordered_most_alike_first(ctx):
    group = duplicates.groups(ctx, max_distance=10)[0]
    scores = [i["similarity"] for i in group["images"]]
    assert scores == sorted(scores, reverse=True)


def test_photos_start_unmarked(ctx):
    group = duplicates.groups(ctx, max_distance=10)[0]
    assert all(image["mark"] is None for image in group["images"])


def test_a_mark_is_remembered_and_changes_nothing_on_disk(ctx):
    image_id = _id_of(ctx, "resaved.jpg")
    path = Path(ctx.dupes.images.path_of(image_id))

    duplicates.mark(ctx, [image_id], "discard")

    # Recorded...
    group = duplicates.groups(ctx, max_distance=10)[0]
    marks = {i["id"]: i["mark"] for i in group["images"]}
    assert marks[image_id] == "discard"
    # ...and that is all it did. The photo is exactly where it was.
    assert path.exists()
    row = ctx.dupes.engine.conn.execute(
        "SELECT deleted FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    assert row["deleted"] == 0
    assert not list(ctx.settings.output.trash_folder.rglob("*.jpg"))


def test_a_marked_photo_still_appears_in_its_group(ctx):
    """A mark is a note, not a resolution — the group does not disappear."""
    duplicates.mark(ctx, [_id_of(ctx, "resaved.jpg")], "discard")
    assert len(duplicates.groups(ctx, max_distance=10)) == 1


def test_a_mark_can_be_changed_and_cleared(ctx):
    image_id = _id_of(ctx, "resaved.jpg")

    duplicates.mark(ctx, [image_id], "discard")
    duplicates.mark(ctx, [image_id], "keep")
    assert duplicates.marked(ctx, "keep") == [image_id]
    assert duplicates.marked(ctx, "discard") == []

    duplicates.mark(ctx, [image_id], None)
    assert duplicates.marked(ctx, "keep") == []


def test_an_unknown_mark_is_refused(ctx):
    with pytest.raises(ValueError, match="unknown mark"):
        duplicates.mark(ctx, [_id_of(ctx, "resaved.jpg")], "maybe")


def test_marked_lists_ids_for_a_later_action(ctx):
    """Nothing acts on marks yet; this is the handle for whatever eventually does."""
    ids = [_id_of(ctx, "original.png"), _id_of(ctx, "resaved.jpg")]
    duplicates.mark(ctx, ids, "discard")
    assert duplicates.marked(ctx, "discard") == sorted(ids)



def test_dismiss_excludes_the_pair_from_future_groups(ctx):
    assert duplicates.groups(ctx, max_distance=10)  # sanity: grouped before dismissal

    duplicates.dismiss(ctx, [_id_of(ctx, "original.png"), _id_of(ctx, "resaved.jpg")])

    assert duplicates.groups(ctx, max_distance=10) == []


# ------------------------------------------------- separate from the sorting index


def test_a_duplicate_scan_leaves_the_sorting_index_empty(ctx):
    """The whole point of the split: pointing the duplicate finder at a folder
    must not enrol that folder in the library the pipeline detects and sorts."""
    assert ctx.dupes.images.count("") == 3
    assert ctx.storage.images.count("") == 0


def test_the_two_indexes_are_different_databases(ctx):
    assert ctx.dupes.engine is not ctx.storage.engine
    assert ctx.dupes.marks.all() == {}
    ids = [_id_of(ctx, "original.png")]
    duplicates.mark(ctx, ids, "keep")
    # Recorded in the duplicate index, and nowhere near the sorting one.
    assert ctx.dupes.marks.all() == {ids[0]: "keep"}
    assert ctx.storage.marks.all() == {}


def test_no_folder_configured_says_so_rather_than_finding_nothing(ctx, monkeypatch):
    """"No duplicates" and "you never said where to look" are different answers."""
    from app.config import load_settings
    monkeypatch.delenv("MEDIASORT_DUPES_FOLDERS", raising=False)
    reloaded = ctx.reloaded(load_settings())
    with pytest.raises(ConfigError, match="no folder to check for duplicates"):
        duplicates.scan_folders(reloaded)


# -------------------------------------------------- the threshold means what it says


def test_a_group_never_holds_a_pair_looser_than_the_threshold():
    """The bug this pins: `cluster` is transitive, so a chain of near-misses used
    to produce one group holding photos far further apart than `max_distance` —
    a slider reading 10 above a pair labelled 56% alike."""
    from app.services.duplicates import _split_around_keepers

    # A chain: each hop is 12 bits, the ends are 24 apart.
    hashes = {1: 0, 2: (1 << 12) - 1, 3: ((1 << 24) - 1) ^ ((1 << 12) - 1)}
    rows = {i: {"id": i, "path": f"/{i}.jpg", "filename": f"{i}.jpg", "size": 100,
                "width": 10, "height": 10, "taken_at": None} for i in hashes}

    built = _split_around_keepers([1, 2, 3], rows, hashes, {}, max_distance=12)

    # `similarity` rounds to one decimal, so compare against the same rounding.
    floor = round((64 - 12) * 100.0 / 64, 1)
    for group in built:
        assert group["similarity"] >= floor
        for image in group["images"]:
            assert image["similarity"] >= floor


def test_the_group_percentage_is_never_below_the_threshold(ctx):
    floor = round((64 - 10) * 100.0 / 64, 1)
    for group in duplicates.groups(ctx, max_distance=10):
        assert group["similarity"] >= floor


# --------------------------------------------------------------- discarding


def test_discarding_moves_marked_photos_and_deletes_nothing(ctx, library_root):
    discard_me = _id_of(ctx, "resaved.jpg")
    keep_me = _id_of(ctx, "original.png")
    duplicates.mark(ctx, [discard_me], "discard")

    result = duplicates.discard_marked(ctx)

    assert result["count"] == 1
    assert result["errors"] == 0
    assert result["deleted"] is False
    # Gone from where it was, still on disk in the trash folder.
    assert not (library_root / "resaved.jpg").exists()
    assert (library_root / "_Duplicates" / "resaved.jpg").is_file()
    # And the one being kept was not touched.
    assert (library_root / "original.png").is_file()
    assert ctx.dupes.images.path_of(keep_me) == str(library_root / "original.png")


def test_a_discarded_photo_stops_being_offered(ctx):
    duplicates.mark(ctx, [_id_of(ctx, "resaved.jpg")], "discard")
    duplicates.discard_marked(ctx)
    assert duplicates.groups(ctx, max_distance=10) == []


def test_a_rescan_does_not_index_the_trash_folder_back(ctx, library_root):
    duplicates.mark(ctx, [_id_of(ctx, "resaved.jpg")], "discard")
    duplicates.discard_marked(ctx)

    duplicates.scan_folders(ctx)

    live = {Path(row["path"]).name for row in ctx.dupes.images.iter_live_paths()}
    assert "resaved.jpg" not in live
    assert duplicates.groups(ctx, max_distance=10) == []


def test_discarding_nothing_is_not_an_error(ctx):
    assert duplicates.discard_marked(ctx)["count"] == 0


def test_direct_delete_trashes_the_file_instead_of_moving_it(ctx, library_root, monkeypatch):
    # The desktop trash is the one destination a test must not really use, so
    # the primitive is stubbed: what matters here is that `delete` takes the
    # trash route rather than the `_Duplicates` one.
    trashed: list[str] = []
    monkeypatch.setattr(filesystem, "trash", lambda path: trashed.append(str(path)) or True)
    discard_me = _id_of(ctx, "resaved.jpg")
    duplicates.mark(ctx, [discard_me], "discard")

    result = duplicates.discard_marked(ctx, delete=True)

    assert result == {"count": 1, "errors": 0, "deleted": True,
                      "destination": "the desktop trash"}
    assert trashed == [str(library_root / "resaved.jpg")]
    assert not (library_root / "_Duplicates").exists()
    assert duplicates.groups(ctx, max_distance=10) == []


def test_discarding_clears_the_marks_it_acted_on(ctx):
    duplicates.mark(ctx, [_id_of(ctx, "resaved.jpg")], "discard")
    duplicates.discard_marked(ctx)
    assert duplicates.marked(ctx, "discard") == []
