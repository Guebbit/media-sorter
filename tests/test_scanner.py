"""The scanner: incremental indexing is what makes big libraries survivable."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace

import pytest

from app.domain.decision import Decision
from app.domain.detection import Detection
from app.errors import ConfigError
from app.services import library
from tests.conftest import make_image

CAT = Detection("cat", 0.9, 0, 0, 1, 1)


def test_finds_images_and_skips_everything_else(ctx, storage):
    stats = library.scan_library(ctx)
    assert stats.seen == 5
    assert stats.added == 5
    assert storage.images.count("") == 5

    indexed = {
        os.path.basename(r["path"])
        for r in storage.engine.conn.execute("SELECT path FROM images")
    }
    assert "notes.txt" not in indexed          # not an image extension
    assert "skip.jpg" not in indexed           # inside a dot-directory


def test_video_files_are_indexed_but_never_probed_as_images(ctx, storage, library_root):
    (library_root / "clip.mp4").write_bytes(b"not a real video, just bytes")
    stats = library.scan_library(ctx)
    assert stats.seen == 6
    row = storage.engine.conn.execute(
        "SELECT width, format FROM images WHERE path LIKE '%clip.mp4'"
    ).fetchone()
    assert row is not None
    assert (row["width"], row["format"]) == (None, None)  # indexed, just not decoded


def test_records_hash_and_dimensions(ctx, storage):
    library.scan_library(ctx)
    row = storage.engine.conn.execute(
        "SELECT hash, width, height, format FROM images LIMIT 1"
    ).fetchone()
    assert len(row["hash"]) == 64
    assert (row["width"], row["height"]) == (64, 48)
    assert row["format"] in {"JPEG", "PNG", "WEBP"}


def test_rescan_touches_nothing(ctx):
    library.scan_library(ctx)
    again = library.scan_library(ctx)
    assert (again.added, again.changed, again.unchanged) == (0, 0, 5)


def test_changed_file_is_reprocessed(ctx, storage, library_root):
    library.scan_library(ctx)
    row = storage.images.claim_detect(1)[0]
    storage.results.finish_detect(row.id, [CAT], Decision("cats", "copy", False), "test.pt")

    time.sleep(0.01)
    make_image(library_root / "a.jpg", size=(100, 100), color=(200, 10, 10))
    stats = library.scan_library(ctx)
    assert stats.changed == 1

    changed = storage.engine.conn.execute(
        "SELECT detect_state, category, action, width FROM images WHERE path LIKE '%/a.jpg'"
    ).fetchone()
    assert changed["detect_state"] == 0        # back at the start of the pipeline
    assert changed["category"] is None
    assert changed["action"] is None
    assert changed["width"] == 100           # metadata refreshed


def test_changed_file_discards_its_stale_detections(ctx, storage, library_root):
    library.scan_library(ctx)
    target = storage.engine.conn.execute(
        "SELECT id FROM images WHERE path LIKE '%/a.jpg'"
    ).fetchone()["id"]
    storage.results.finish_detect(target, [CAT], Decision("cats", "copy", False), "test.pt")
    assert storage.detections.class_counts() == {"cat": 1}

    time.sleep(0.01)
    make_image(library_root / "a.jpg", size=(100, 100), color=(1, 2, 3))
    library.scan_library(ctx)
    assert storage.detections.class_counts() == {}


def test_deleted_file_is_flagged_not_dropped(ctx, storage, library_root):
    library.scan_library(ctx)
    (library_root / "a.jpg").unlink()
    library.scan_library(ctx)

    assert storage.images.count("missing = 1") == 1
    # The row survives, so an unplugged drive loses nothing.
    assert storage.images.count("") == 5


def test_restored_file_comes_back(ctx, storage, library_root, tmp_path):
    library.scan_library(ctx)
    backup = tmp_path / "backup.jpg"
    shutil.move(str(library_root / "a.jpg"), backup)
    library.scan_library(ctx)
    assert storage.images.count("missing = 1") == 1

    shutil.move(str(backup), library_root / "a.jpg")
    library.scan_library(ctx)
    assert storage.images.count("missing = 1") == 0


def test_duplicate_content_is_detected(ctx, storage, library_root):
    shutil.copy2(library_root / "a.jpg", library_root / "sub" / "copy.jpg")
    library.scan_library(ctx)
    duplicates = storage.reporting.duplicates()
    assert len(duplicates) == 1
    assert duplicates[0]["n"] == 2


def test_unreadable_file_does_not_abort_the_scan(ctx, storage, library_root):
    (library_root / "broken.jpg").write_bytes(b"this is not a JPEG")
    stats = library.scan_library(ctx)
    assert stats.seen == 6
    # It is still indexed (hash + size are valid); only the probe failed.
    row = storage.engine.conn.execute(
        "SELECT width FROM images WHERE path LIKE '%broken%'"
    ).fetchone()
    assert row["width"] is None


def test_missing_input_folder_is_skipped(ctx, tmp_path):
    absent = replace(ctx.settings.library, input_folders=(tmp_path / "does-not-exist",))
    ctx = replace(ctx, settings=replace(ctx.settings, library=absent))
    assert library.scan_library(ctx).seen == 0


def _pointed_elsewhere(ctx, tmp_path):
    """`ctx`, reconfigured to a second, freshly-populated folder — as if the
    user had changed which folder they point app at."""
    other = tmp_path / "photos-2"
    make_image(other / "z.jpg")
    elsewhere = replace(ctx.settings.library, input_folders=(other,))
    return replace(ctx, settings=replace(ctx.settings, library=elsewhere))


def test_a_folder_dropped_from_configuration_is_purged_on_rescan(ctx, storage, tmp_path):
    """Changing INPUT_FOLDERS must not leave the old folder's rows counted
    forever — the "Indexed" total (and everything derived from it) would just
    keep growing across every folder ever pointed at, never shrinking."""
    library.scan_library(ctx)
    assert storage.images.count("") == 5

    elsewhere = _pointed_elsewhere(ctx, tmp_path)
    library.scan_library(elsewhere)

    assert storage.images.count("") == 1     # only the new folder's photo; the old rows are gone
    assert storage.images.count("missing = 1") == 0


def test_dropping_a_folder_prunes_the_output_it_produced(ctx, storage, library_root):
    """A dropped folder's copies in the output tree must not become orphans
    nobody ever cleans up — the same pruning `apply` does for any other stale
    link, just triggered by the folder itself going away instead of a rule."""
    from app.domain.decision import Decision

    library.scan_library(ctx)
    row = storage.images.claim_detect(1)[0]
    storage.results.finish_detect(row.id, [], Decision("none", "copy", False), "test.pt")
    output_copy = ctx.settings.output.folder / "leftover.jpg"
    output_copy.parent.mkdir(parents=True, exist_ok=True)
    output_copy.write_bytes(b"stand-in for a real copy")
    storage.links.record([(row.id, str(output_copy), "copy")])
    assert output_copy.exists()

    elsewhere = _pointed_elsewhere(ctx, tmp_path=library_root.parent)
    library.scan_library(elsewhere)

    assert not output_copy.exists()
    assert storage.links.known() == {}


def test_a_one_off_input_override_does_not_touch_the_rest_of_the_library(ctx, storage, tmp_path):
    """`--input` for a single run must not read as "the user removed every
    other folder" — only a scan against the saved configuration may prune."""
    library.scan_library(ctx)
    assert storage.images.count("") == 5

    elsewhere = _pointed_elsewhere(ctx, tmp_path)
    library.scan_library(elsewhere, authoritative=False)

    assert storage.images.count("") == 6     # the original five, untouched, plus the new one


def test_a_dropped_folder_starts_fresh_if_reconfigured_and_rescanned(ctx, storage, tmp_path):
    """Coming back is a real rescan, not a restore — the point is that nothing
    from a dropped folder lingers, not that it is recoverable."""
    library.scan_library(ctx)
    elsewhere = _pointed_elsewhere(ctx, tmp_path)
    library.scan_library(elsewhere)
    assert storage.images.count("") == 1

    library.scan_library(ctx)  # back to the original folder
    assert storage.images.count("") == 5   # indexed fresh
    # `elsewhere` is itself unconfigured now, so it is what gets purged this time.
    other_root = str((tmp_path / "photos-2").resolve())
    assert storage.images.count("root = ?", (other_root,)) == 0


def test_our_own_output_inside_the_library_is_not_indexed(ctx, library_root):
    """The default output folder lives in the library, so this is the normal case:
    `copy` mode would otherwise index its own copies, and a `delete` run would
    hand the trash folder back as new photos on the next scan."""
    output = replace(ctx.settings.output, folder=library_root / "Sorted")
    output = replace(output, trash_folder=output.folder / "_Trash")
    ctx = replace(ctx, settings=replace(ctx.settings, output=output))
    make_image(output.folder / "Cat" / "copied.jpg")        # as `copy` mode leaves it
    make_image(output.trash_folder / "deleted.jpg")         # as a `delete` rule leaves it

    stats = library.scan_library(ctx)

    assert stats.seen == 5  # the library's own five, neither of the two above


def test_an_output_folder_elsewhere_in_the_library_is_also_skipped(ctx, library_root):
    """Pruning is by location, not by the name `Sorted`."""
    output = replace(ctx.settings.output, folder=library_root / "sub" / "DONE")
    ctx = replace(ctx, settings=replace(ctx.settings, output=output))
    make_image(output.folder / "Cat" / "copied.jpg")

    assert library.scan_library(ctx).seen == 5


def test_scanning_with_no_folders_configured_says_so(ctx):
    """A folder that is gone is a warning; no folders at all is a question for the user."""
    unset = replace(ctx.settings.library, input_folders=())
    ctx = replace(ctx, settings=replace(ctx.settings, library=unset))
    with pytest.raises(ConfigError, match="no photo folders configured"):
        library.scan_library(ctx)
