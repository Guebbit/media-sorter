"""Read-only questions about the index: progress, tallies, lookups.

Separate from the repositories because these queries belong to no single table
and mutate nothing. Both front ends are built almost entirely out of this
module, which is why none of them needs to write SQL of its own.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .engine import Engine
from .images import ImageRepository
from .state import DONE, ERROR, Stage


class Reporting:
    """Cross-table read queries: progress counts, tallies, duplicate/search
    lookups. Neither front end writes SQL of its own — everything they show
    that isn't a single repository's job comes from here."""

    def __init__(self, engine: Engine, images: ImageRepository):
        self._engine = engine
        self._images = images

    def pending_counts(self) -> dict[str, int]:
        """Every number `mediasort stats`/the web UI's overview shows: totals,
        per-stage done/pending/error, and the two "total eligible" denominators
        that stay stable while a stage is mid-run (see the comments below)."""
        count = self._images.count
        return {
            "total": count("missing = 0"),
            "missing": count("missing = 1"),
            "deleted": count("deleted = 1"),
            "detect_pending": self._images.count_pending(Stage.DETECT),
            "detect_done": count("detect_state = ?", (DONE,)),
            "detect_error": count("detect_state = ?", (ERROR,)),
            "adjudicate_pending": self._images.count_pending(Stage.ADJUDICATE),
            "adjudicate_done": count("adjudicate_state = ?", (DONE,)),
            "adjudicate_error": count("adjudicate_state = ?", (ERROR,)),
            # Every image the detector could not settle, whatever has since
            # happened to it — the denominator that does not move mid-stage.
            # A settled verdict clears `needs_review`, so counting the flag
            # instead would shrink as the stage worked.
            "adjudicate_total": count(
                "detect_state = ? AND missing = 0 AND deleted = 0 "
                "AND (needs_review = 1 OR adjudicate_state IN (?, ?))",
                (DONE, DONE, ERROR),
            ),
            "analyze_pending": self._images.count_pending(Stage.ANALYZE),
            "analyze_done": count("analyze_state = ?", (DONE,)),
            "analyze_error": count("analyze_state = ?", (ERROR,)),
            # Everything the semantic pass is ever going to look at, whatever
            # state those rows are in. Summing done + pending instead would count
            # the rows currently in flight as neither, and a denominator that
            # shrinks while a stage runs makes a progress bar go backwards.
            "analyze_total": count(
                "detect_state = ? AND missing = 0 AND deleted = 0 "
                "AND action IS NOT NULL AND action != 'ignore'",
                (DONE,),
            ),
            "review": count("needs_review = 1 AND missing = 0"),
            # Whether the output tree has been built yet, which is the difference
            # between "apply has nothing to do" and "apply has not been run".
            "links_total": self._engine.conn.execute(
                "SELECT COUNT(*) AS n FROM links"
            ).fetchone()["n"],
        }

    def category_counts(self) -> dict[str, int]:
        """How many images fall into each rule's category, most populous
        first; uncategorised images are grouped under `"unprocessed"`."""
        rows = self._engine.conn.execute(
            "SELECT COALESCE(category, 'unprocessed') AS c, COUNT(*) AS n "
            "FROM images WHERE missing = 0 GROUP BY c ORDER BY n DESC"
        ).fetchall()
        return {r["c"]: r["n"] for r in rows}

    def action_counts(self) -> dict[str, int]:
        """How many images are headed toward each action, most common first."""
        rows = self._engine.conn.execute(
            "SELECT action, COUNT(*) AS n FROM images "
            "WHERE missing = 0 AND action IS NOT NULL GROUP BY action ORDER BY n DESC"
        ).fetchall()
        return {r["action"]: r["n"] for r in rows}

    def duplicates(self, limit: int = 100) -> list[sqlite3.Row]:
        """Groups of indexed files sharing a hash — exact byte-for-byte
        copies, up to `limit` groups, largest group first."""
        return self._engine.conn.execute(
            """SELECT hash, COUNT(*) AS n, GROUP_CONCAT(path, char(10)) AS paths
               FROM images WHERE hash IS NOT NULL AND missing = 0
               GROUP BY hash HAVING n > 1 ORDER BY n DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    def phashes(self) -> list[sqlite3.Row]:
        """Every present, non-deleted image with a perceptual hash — id,
        path, filename, size, dimensions, capture date, and the hash itself.
        What the near-duplicate grouper (`domain.similarity.cluster`) reads
        before clustering."""
        return self._engine.conn.execute(
            "SELECT id, path, filename, size, width, height, taken_at, phash FROM images "
            "WHERE missing = 0 AND deleted = 0 AND phash IS NOT NULL"
        ).fetchall()

    def samples(self, category: str | None = None, limit: int = 60) -> list[dict[str, Any]]:
        """A page of categorised images, for a UI that wants to show its work."""
        sql = ("SELECT id, path, filename, category, action, needs_review FROM images "
               "WHERE missing = 0 AND deleted = 0 AND category IS NOT NULL")
        params: list[Any] = []
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY id LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._engine.conn.execute(sql, params)]

    def search_metadata(self, term: str, limit: int = 50) -> list[sqlite3.Row]:
        """Substring search over the stored analysis JSON.

        Crude by design: the payload's shape follows the user's rules, so there
        is no fixed field to index. It is fast enough for a personal library.
        """
        return self._engine.conn.execute(
            """SELECT i.path, i.category, m.json FROM metadata m JOIN images i ON i.id = m.image_id
               WHERE i.missing = 0 AND lower(m.json) LIKE ? ORDER BY i.id LIMIT ?""",
            (f"%{term.lower()}%", limit),
        ).fetchall()
