"""The index: claiming, resume semantics and schema compatibility."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from app.domain.decision import Decision
from app.domain.detection import Detection

from app.services import library
from app.storage import (DONE, ERROR, PENDING, RUNNING, SKIPPED, SCHEMA_VERSION, Stage,
                               Storage)

CAT = Detection("cat", 0.9, 1, 2, 3, 4)
DOG = Detection("dog", 0.7, 5, 6, 7, 8)


@pytest.fixture
def indexed(ctx, storage):
    library.scan_library(ctx)
    return storage


def test_claiming_is_exclusive(indexed):
    first = indexed.images.claim_detect(2)
    second = indexed.images.claim_detect(2)
    assert len(first) == 2
    assert len(second) == 2
    assert {r.id for r in first}.isdisjoint({r.id for r in second})


def test_claimed_rows_are_marked_running(indexed):
    indexed.images.claim_detect(1)
    assert indexed.images.count("detect_state = ?", (RUNNING,)) == 1


def test_reset_running_recovers_from_a_crash(indexed):
    indexed.images.claim_detect(3)
    assert indexed.images.reset_running(Stage.DETECT) == 3
    assert indexed.images.count("detect_state = ?", (PENDING,)) == 5


def test_finish_detect_stores_detections_and_decision(indexed):
    row = indexed.images.claim_detect(1)[0]
    indexed.results.finish_detect(
        row.id, [CAT, DOG], Decision("cat-dog", "copy", True), "yolo11m.pt"
    )

    stored = indexed.engine.conn.execute(
        "SELECT category, action, needs_review, detect_state FROM images WHERE id = ?", (row.id,)
    ).fetchone()
    assert (stored["category"], stored["action"], stored["detect_state"]) == \
        ("cat-dog", "copy", DONE)
    assert stored["needs_review"] == 1
    assert len(indexed.detections.for_image(row.id)) == 2


def test_finish_detect_replaces_previous_detections(indexed):
    row = indexed.images.claim_detect(1)[0]
    decision = Decision("cats", "copy", False)
    indexed.results.finish_detect(row.id, [CAT], decision, "m")
    indexed.results.finish_detect(row.id, [CAT], decision, "m")
    assert len(indexed.detections.for_image(row.id)) == 1


def test_analysis_only_claims_images_the_rules_kept(indexed):
    for index, row in enumerate(indexed.images.claim_detect(5)):
        keep = index < 2
        indexed.results.finish_detect(
            row.id, [], Decision("cats" if keep else "none", "copy" if keep else "ignore", False), "m"
        )
    indexed.images.skip_analysis_for_ignored()

    claimed = indexed.images.claim_analyze(10)
    assert len(claimed) == 2
    assert indexed.images.count("analyze_state = ?", (SKIPPED,)) == 3


def test_errors_are_recorded_and_retryable(indexed):
    row = indexed.images.claim_detect(1)[0]
    indexed.images.fail(row.id, Stage.DETECT, "boom")
    assert indexed.images.count("detect_state = ?", (ERROR,)) == 1

    assert indexed.images.retry_errors(Stage.DETECT) == 1
    assert indexed.images.count("detect_state = ?", (PENDING,)) == 5


def test_set_decision_updates_without_touching_detections(indexed):
    row = indexed.images.claim_detect(1)[0]
    indexed.results.finish_detect(row.id, [CAT], Decision("cats", "copy", False), "m")
    indexed.images.set_decision(row.id, "dogs", "delete", True)

    stored = indexed.engine.conn.execute(
        "SELECT category, action, needs_review FROM images WHERE id = ?", (row.id,)
    ).fetchone()
    assert (stored["category"], stored["action"], stored["needs_review"]) == ("dogs", "delete", 1)
    # Detections survive a re-decision — that is what makes `recheck` cheap.
    assert len(indexed.detections.for_image(row.id)) == 1


def test_counts(indexed):
    counts = indexed.reporting.pending_counts()
    assert counts["total"] == 5
    assert counts["detect_pending"] == 5
    assert counts["deleted"] == 0


def test_analyze_total_counts_every_candidate_whatever_its_state(indexed):
    """The denominator of a progress bar must not move while the stage runs.

    Two images are decided `link` and one of them is then claimed by the analyze
    stage. Summing done + pending would report 1; the answer is 2 either way.
    """
    rows = indexed.images.claim_detect(3)
    for row, action in zip(rows, ["copy", "copy", "ignore"]):
        indexed.results.finish_detect(row.id, [CAT], Decision("cat", action, False), "m")
    indexed.images.skip_analysis_for_ignored()

    assert indexed.reporting.pending_counts()["analyze_total"] == 2
    assert len(indexed.images.claim_analyze(1)) == 1
    counts = indexed.reporting.pending_counts()
    assert counts["analyze_total"] == 2
    assert counts["analyze_done"] + counts["analyze_pending"] == 1


def test_purge_missing_drops_only_vanished_rows(ctx, indexed, library_root):
    (library_root / "a.jpg").unlink()
    library.scan_library(ctx)
    assert indexed.images.purge_missing() == 1
    assert indexed.images.count("") == 4


def test_nested_writes_commit_once(indexed):
    """Composing repository writes into one transaction is just nesting them."""
    with indexed.engine.write():
        indexed.images.reset_stage(Stage.DETECT)
        indexed.detections.clear()
    assert indexed.images.count("detect_state = ?", (PENDING,)) == 5


def test_a_failed_nested_write_rolls_the_whole_thing_back(indexed):
    before = indexed.images.count("category IS NULL")
    with pytest.raises(RuntimeError):
        with indexed.engine.write():
            indexed.images.set_decision(1, "cats", "copy", False)
            raise RuntimeError("boom")
    assert indexed.images.count("category IS NULL") == before


def test_an_index_from_an_older_schema_is_rebuilt_not_refused(tmp_path):
    """The index is a rebuildable cache, so an old one is thrown away and
    recreated rather than half-migrated — or, worse, turned into an error the
    user has to read and act on before they can sort any photos."""
    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
        CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_info VALUES ('version', '2');
        INSERT INTO images (path) VALUES ('/gone.jpg');
        """
    )
    legacy.commit()
    legacy.close()

    storage = Storage(path)
    storage.init()

    # Usable straight away, on the current schema, with the stale rows gone.
    assert storage.images.count("") == 0
    assert storage.engine.conn.execute(
        "SELECT value FROM schema_info WHERE key='version'"
    ).fetchone()[0] == SCHEMA_VERSION
    # A v2 `images` table had no `phash`; the rebuilt one does.
    assert "phash" in {r["name"] for r in storage.engine.conn.execute("PRAGMA table_info(images)")}


def test_a_fresh_file_is_not_mistaken_for_an_old_one(tmp_path):
    storage = Storage(tmp_path / "new.db")
    storage.init()
    assert storage.images.count("") == 0


def test_init_is_idempotent(settings):
    storage = Storage(settings.paths.database)
    storage.init()
    storage.init()
    stored = storage.engine.conn.execute(
        "SELECT value FROM schema_info WHERE key='version'"
    ).fetchone()[0]
    assert stored == SCHEMA_VERSION


def test_a_read_on_another_thread_survives_a_write_in_flight():
    """The default in-memory index is shared-cache, which locks per *table* and
    refuses a conflicting read outright instead of waiting out `busy_timeout`.
    The web UI polls stats from its own thread while the stages write, so that
    read has to come back rather than raising "database table is locked"."""
    storage = Storage(None)
    storage.init()
    assert not storage.engine.persistent
    result: list[object] = []

    def read() -> None:
        try:
            result.append(storage.images.count("deleted = 1"))
        except BaseException as exc:  # noqa: BLE001 - the regression itself
            result.append(exc)

    with storage.engine.write() as conn:
        # Any statement that takes the write lock on `images` will do; the
        # reader below contends with the lock, not with the rows.
        conn.execute("UPDATE images SET deleted = 1")
        reader = threading.Thread(target=read)
        reader.start()
        reader.join(timeout=10)

    assert not reader.is_alive(), "the read hung"
    assert result and not isinstance(result[0], BaseException), result
