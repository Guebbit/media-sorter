"""The images table: the row every stage claims, updates and reports on.

This is where resumability lives. Nothing about a run is held in memory: stages
claim rows atomically, so a process can be killed at any point and the next run
picks up exactly where it left off — and several processes can work the same
index (on a local filesystem, not NFS).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .engine import Engine
from .state import DONE, ERROR, PENDING, RUNNING, SKIPPED, Stage


@dataclass(frozen=True, slots=True)
class ImageRow:
    """The minimum a stage needs to process one image."""

    id: int
    path: str


#: "Ready for this stage" predicates, each missing only its own state column's
#: comparison (`= ?` for a claim/count, `IN (?, ?)` for an in-flight check).
#: Defined once so `claim_*`, `count_pending` and `*_in_flight` can never drift
#: apart on what "ready" means for a given stage.
_DETECT_READY = "missing = 0"
_ADJUDICATE_READY = "detect_state = ? AND needs_review = 1 AND missing = 0 AND deleted = 0"
_ANALYZE_READY = (
    "detect_state = ? AND adjudicate_state NOT IN (?, ?) AND missing = 0 AND deleted = 0 "
    "AND action IS NOT NULL AND action != 'ignore'"
)


class ImageRepository:
    """The `images` table: one row per indexed file, claimed and updated by
    each stage in turn, and the source of every pending/count query."""

    def __init__(self, engine: Engine):
        self._engine = engine

    @property
    def _conn(self) -> sqlite3.Connection:
        """Shorthand for the current thread's connection, for read-only queries."""
        return self._engine.conn

    # ---------------------------------------------------------------- scanning

    def stat_of(self, path: str) -> sqlite3.Row | None:
        """The stored size/mtime/hash for `path`, or None if it isn't indexed
        yet — what the scanner compares against to decide if a file changed."""
        return self._conn.execute(
            "SELECT id, size, mtime, hash FROM images WHERE path = ?", (path,)
        ).fetchone()

    def path_of(self, image_id: int) -> str | None:
        """The current path for `image_id`, or None if the row doesn't exist
        — for a caller that only has an id (not a full row) and needs to
        touch the file itself."""
        row = self._conn.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
        return row["path"] if row else None

    def flag_root_missing(self, root: str) -> None:
        """Mark everything under a root as missing; the scan clears what it finds."""
        with self._engine.write() as conn:
            conn.execute("UPDATE images SET missing = 1 WHERE root = ? AND missing = 0", (root,))

    def distinct_roots(self) -> list[str]:
        """Every root at least one indexed image is attributed to — including
        ones already flagged missing, since a root whose drive happens to be
        unplugged is still a root that needs to be recognised as dropped once
        it is no longer configured, not silently skipped."""
        rows = self._conn.execute("SELECT DISTINCT root FROM images").fetchall()
        return [row["root"] for row in rows]

    def purge_root(self, root: str) -> int:
        """Delete every row attributed to `root` — cascading its detections,
        adjudications, metadata, links and activity log with it (`ON DELETE
        CASCADE`). Used when a folder drops out of the configured library, so
        nothing it produced lingers in a count once it is no longer part of the
        library — unlike a single vanished file, a whole dropped root is not
        coming back on its own; only reconfiguring and rescanning does that,
        which indexes it fresh."""
        with self._engine.write() as conn:
            cur = conn.execute("DELETE FROM images WHERE root = ?", (root,))
            return cur.rowcount

    def touch_present(self, image_ids: Sequence[int]) -> None:
        """Clears `deleted` too: if a file was restored from the trash it is a
        real image again and should flow through the pipeline normally."""
        if not image_ids:
            return
        with self._engine.write() as conn:
            conn.executemany(
                "UPDATE images SET missing = 0, deleted = 0 WHERE id = ?",
                [(i,) for i in image_ids],
            )

    def upsert(self, rows: Iterable[dict]) -> None:
        """Insert new images, or reset a changed one back to the start of the pipeline."""
        now = time.time()
        payload = [
            (
                r["path"], r["filename"], r["root"], r["hash"], r["phash"], r["size"], r["mtime"],
                r["width"], r["height"], r["format"], r["taken_at"], now, now,
            )
            for r in rows
        ]
        if not payload:
            return
        with self._engine.write() as conn:
            conn.executemany(
                """
                INSERT INTO images (path, filename, root, hash, phash, size, mtime, width, height,
                                    format, taken_at, first_seen, updated_at, missing)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                ON CONFLICT(path) DO UPDATE SET
                    hash=excluded.hash, phash=excluded.phash, size=excluded.size,
                    mtime=excluded.mtime, width=excluded.width, height=excluded.height,
                    format=excluded.format, taken_at=excluded.taken_at,
                    updated_at=excluded.updated_at,
                    missing=0, deleted=0, category=NULL, action=NULL,
                    needs_review=0, error=NULL, detect_state=0, analyze_state=0
                """,
                payload,
            )
            # A changed file invalidates everything derived from its old content.
            placeholders = ",".join("?" * len(payload))
            paths = [r[0] for r in payload]
            conn.execute(
                f"""
                DELETE FROM detections WHERE image_id IN
                    (SELECT id FROM images WHERE path IN ({placeholders}) AND detect_state = 0)
                """,
                paths,
            )
            conn.execute(
                f"""
                DELETE FROM metadata WHERE image_id IN
                    (SELECT id FROM images WHERE path IN ({placeholders}) AND analyze_state = 0)
                """,
                paths,
            )

    # ---------------------------------------------------------------- claiming

    def reset_running(self, stage: Stage) -> int:
        """Recover rows a crashed run left in-flight."""
        column = Stage(stage).column
        with self._engine.write() as conn:
            cur = conn.execute(
                f"UPDATE images SET {column} = ? WHERE {column} = ?", (PENDING, RUNNING)
            )
            return cur.rowcount

    def claim_detect(self, limit: int) -> list[ImageRow]:
        """Up to `limit` images never through the detector, marked RUNNING."""
        return self._claim(
            f"SELECT id, path FROM images WHERE detect_state = ? AND {_DETECT_READY} "
            "ORDER BY id LIMIT ?",
            (PENDING, limit),
            Stage.DETECT,
        )

    def claim_adjudicate(self, limit: int) -> list[ImageRow]:
        """Only the images the detector could not settle.

        `needs_review` is the detector's own admission of doubt, so it is the
        entire queue for the second opinion: everything else has an answer
        already, and asking a vision model to confirm what YOLO was sure about
        would cost hours to change nothing.
        """
        return self._claim(
            f"SELECT id, path FROM images WHERE adjudicate_state = ? AND {_ADJUDICATE_READY} "
            "ORDER BY id LIMIT ?",
            (PENDING, DONE, limit),
            Stage.ADJUDICATE,
        )

    def claim_analyze(self, limit: int) -> list[ImageRow]:
        """Only images the rules actually care about, and only once settled.

        A rule whose action is `ignore` is the user saying "not interesting",
        so the expensive semantic pass skips it — that is the whole point of
        the staged design. Waiting for adjudication matters for the same reason
        in reverse: a photo the second opinion is about to rescue from `ignore`
        must not be written off before the verdict lands. A stage that errored
        does not block, or one unreachable Ollama would strand the description
        pass for good.
        """
        return self._claim(
            f"SELECT id, path FROM images WHERE analyze_state = ? AND {_ANALYZE_READY} "
            "ORDER BY id LIMIT ?",
            (PENDING, DONE, PENDING, RUNNING, limit),
            Stage.ANALYZE,
        )

    def _claim(self, sql: str, params: Sequence, stage: Stage) -> list[ImageRow]:
        """Select and mark RUNNING in one transaction, so two workers never get
        the same row."""
        with self._engine.write() as conn:
            rows = conn.execute(sql, params).fetchall()
            if rows:
                conn.executemany(
                    f"UPDATE images SET {stage.column} = ? WHERE id = ?",
                    [(RUNNING, r["id"]) for r in rows],
                )
        return [ImageRow(r["id"], r["path"]) for r in rows]

    def sync_adjudication_queue(self) -> int:
        """Make the queue agree with the current decisions, in both directions.

        `finish_detect` gets this right for the image in front of it, but a
        decision can be rewritten later — a rule edit, a lowered threshold —
        and then an image that was certain is not, or the reverse. Only rows
        the stage has never touched are moved: a `DONE` verdict is a fact about
        an image, and re-asking a model that already answered every time the
        rules change would cost hours to hear the same thing.
        """
        with self._engine.write() as conn:
            queued = conn.execute(
                "UPDATE images SET adjudicate_state = ? "
                "WHERE detect_state = ? AND adjudicate_state = ? AND needs_review = 1 "
                "AND missing = 0 AND deleted = 0",
                (PENDING, DONE, SKIPPED),
            ).rowcount
            skipped = conn.execute(
                "UPDATE images SET adjudicate_state = ? "
                "WHERE detect_state = ? AND adjudicate_state = ? AND needs_review = 0",
                (SKIPPED, DONE, PENDING),
            ).rowcount
            return queued + skipped

    def skip_analysis_for_ignored(self) -> int:
        """Images the rules chose to ignore never need the semantic pass.

        Only once adjudication has had its say: writing off a photo whose
        verdict is still outstanding would skip the one the second opinion was
        about to rescue.
        """
        with self._engine.write() as conn:
            cur = conn.execute(
                "UPDATE images SET analyze_state = ? WHERE detect_state = ? AND analyze_state = ?"
                " AND adjudicate_state NOT IN (?, ?)"
                " AND (action IS NULL OR action = 'ignore')",
                (SKIPPED, DONE, PENDING, PENDING, RUNNING),
            )
            return cur.rowcount

    # ----------------------------------------------------------------- results

    def set_decision(self, image_id: int, category: str, action: str, needs_review: bool) -> None:
        """Re-apply the decision engine without re-running the detector — what
        makes "try different rules" cheap."""
        with self._engine.write() as conn:
            conn.execute(
                "UPDATE images SET category=?, action=?, needs_review=?, updated_at=? WHERE id=?",
                (category, action, int(needs_review), time.time(), image_id),
            )

    def fail(self, image_id: int, stage: Stage, message: str) -> None:
        """Mark one image's `stage` ERROR with `message`, so it stops being
        claimed until `retry_errors` puts it back in the queue."""
        column = Stage(stage).column
        with self._engine.write() as conn:
            conn.execute(
                f"UPDATE images SET {column}=?, error=?, updated_at=? WHERE id=?",
                (ERROR, message[:500], time.time(), image_id),
            )

    def retry_errors(self, stage: Stage | None = None) -> int:
        """Put every ERROR row for `stage` (or every stage) back to PENDING,
        clearing the recorded error. Returns how many rows changed."""
        stages = [Stage(stage)] if stage else list(Stage)
        total = 0
        with self._engine.write() as conn:
            for member in stages:
                cur = conn.execute(
                    f"UPDATE images SET {member.column}=?, error=NULL WHERE {member.column}=?",
                    (PENDING, ERROR),
                )
                total += cur.rowcount
        return total

    def reset_stage(self, stage: Stage) -> int:
        """Queue a stage again for every image. The derived rows are dropped by
        whoever calls this — see `services.maintenance.reset`."""
        with self._engine.write() as conn:
            if stage is Stage.DETECT:
                # Everything downstream was derived from detections that are
                # about to be dropped, so the whole chain goes back to pending.
                cur = conn.execute(
                    "UPDATE images SET detect_state = ?, adjudicate_state = ?, "
                    "category = NULL, action = NULL, needs_review = 0",
                    (PENDING, PENDING),
                )
            elif stage is Stage.ADJUDICATE:
                cur = conn.execute("UPDATE images SET adjudicate_state = ?", (PENDING,))
            else:
                cur = conn.execute("UPDATE images SET analyze_state = ?", (PENDING,))
            return cur.rowcount

    def relocate(self, image_id: int, path: str) -> None:
        """The original is still ours, at a new address.

        `root` moves with it. Rescans clear and re-set `missing` one root at a
        time, and a filed photo now lives under the output folder — which the
        walker prunes — so leaving it attributed to the folder it came from
        would have the next scan report it as gone.
        """
        target = Path(path)
        with self._engine.write() as conn:
            conn.execute(
                "UPDATE images SET path = ?, filename = ?, root = ?, updated_at = ? "
                "WHERE id = ?",
                (str(target), target.name, str(target.parent), time.time(), image_id),
            )

    def mark_deleted(self, image_id: int) -> None:
        """The original is gone; keep the row so the file is not re-indexed."""
        with self._engine.write() as conn:
            conn.execute(
                "UPDATE images SET deleted = 1, updated_at = ? WHERE id = ?",
                (time.time(), image_id),
            )

    def purge_missing(self) -> int:
        """Delete every row flagged `missing` — a file gone from disk long
        enough that it is no longer worth tracking. Returns how many rows went."""
        with self._engine.write() as conn:
            cur = conn.execute("DELETE FROM images WHERE missing = 1")
            return cur.rowcount

    # ----------------------------------------------------------------- queries

    def count(self, where: str = "", params: Sequence = ()) -> int:
        """`SELECT COUNT(*) FROM images`, optionally filtered by a raw `where`
        clause — the one general-purpose counting primitive every more
        specific count (`count_pending`, `Reporting.pending_counts`, ...)
        builds on."""
        sql = "SELECT COUNT(*) AS n FROM images"
        if where:
            sql += " WHERE " + where
        return self._conn.execute(sql, params).fetchone()["n"]

    def count_pending(self, stage: Stage) -> int:
        """Rows a stage would claim right now — the same predicate as its claim."""
        if stage is Stage.DETECT:
            return self.count(f"detect_state = ? AND {_DETECT_READY}", (PENDING,))
        if stage is Stage.ADJUDICATE:
            return self.count(f"adjudicate_state = ? AND {_ADJUDICATE_READY}", (PENDING, DONE))
        return self.count(
            f"analyze_state = ? AND {_ANALYZE_READY}", (PENDING, DONE, PENDING, RUNNING)
        )

    def detect_in_flight(self) -> bool:
        """Is detection still going to produce work for the stages after it?"""
        return self.count(f"detect_state IN (?, ?) AND {_DETECT_READY}", (PENDING, RUNNING)) > 0

    def adjudicate_in_flight(self) -> bool:
        """Is a verdict still outstanding that could change what gets described?"""
        return self.count(
            f"adjudicate_state IN (?, ?) AND {_ADJUDICATE_READY}", (PENDING, RUNNING, DONE)
        ) > 0

    def iter_decided_ids(self) -> list[int]:
        """Images with usable detections — the input to re-running the rules."""
        rows = self._conn.execute(
            "SELECT id FROM images WHERE detect_state = ? AND missing = 0 AND deleted = 0",
            (DONE,),
        ).fetchall()
        return [row["id"] for row in rows]

    def iter_actionable(self) -> Iterator[sqlite3.Row]:
        """Every categorised image, in a stable order.

        Rules decide what happens — including "nothing" — so the filtering
        belongs to the action registry rather than to SQL. Rows already
        deleted or gone from disk are excluded.
        """
        yield from self._conn.execute(
            """SELECT id, path, filename, category, action, needs_review FROM images
               WHERE missing = 0 AND deleted = 0 AND detect_state = ? AND category IS NOT NULL
               ORDER BY id""",
            (DONE,),
        )

    def iter_live_paths(self) -> Iterator[sqlite3.Row]:
        """Originals we believe are still on disk — what `verify` re-checks."""
        yield from self._conn.execute(
            "SELECT id, path FROM images WHERE missing = 0 AND deleted = 0"
        )

    def iter_export(self) -> Iterator[sqlite3.Row]:
        """Every non-missing image joined with its stored metadata JSON, in
        id order — the row shape `services.exporting` writes out as-is."""
        yield from self._conn.execute(
            """
            SELECT i.id, i.path, i.filename, i.hash, i.width, i.height, i.taken_at,
                   i.category, i.needs_review, i.detect_state, i.analyze_state, m.json AS metadata
            FROM images i LEFT JOIN metadata m ON m.image_id = i.id
            WHERE i.missing = 0
            ORDER BY i.id
            """
        )
