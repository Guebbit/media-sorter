"""The adjudications table: what the second opinion said, kept so rules can be re-run.

The counterpart of `detections`, and separate from it for the same reason
`metadata` is separate: each table holds one kind of evidence from one kind of
engine, so `recheck` can rebuild a decision from all of it without either
having overwritten the other.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Sequence

from ..domain.adjudication import Adjudication
from .engine import Engine


class AdjudicationRepository:
    """The `adjudications` table: the second opinion's verdicts, replaced
    wholesale each time an image is re-adjudicated."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def for_image(self, image_id: int) -> list[sqlite3.Row]:
        """Raw verdict rows for one image, alphabetical by class."""
        return self._engine.conn.execute(
            "SELECT class, verdict, confidence FROM adjudications WHERE image_id = ? "
            "ORDER BY class",
            (image_id,),
        ).fetchall()

    def objects_for_image(self, image_id: int) -> list[Adjudication]:
        """The same rows as domain values — what re-deciding wants."""
        return [Adjudication.from_row(row) for row in self.for_image(image_id)]

    def replace_for_image(self, image_id: int, adjudications: Sequence[Adjudication],
                          model: str) -> None:
        """Drop whatever this image had and store `adjudications` in its
        place — the write side of one `OllamaClient.adjudicate` reply."""
        now = time.time()
        with self._engine.write() as conn:
            conn.execute("DELETE FROM adjudications WHERE image_id = ?", (image_id,))
            if adjudications:
                conn.executemany(
                    """INSERT INTO adjudications
                           (image_id, class, verdict, confidence, model, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    [(image_id, a.cls, a.verdict, a.confidence, model, now)
                     for a in adjudications],
                )

    def clear(self) -> None:
        """Drop every stored verdict — used by a full `reset`."""
        self._engine.truncate("adjudications")

    def verdict_counts(self) -> dict[str, int]:
        """How the second opinion has been ruling — the honest measure of
        whether escalating is buying anything."""
        rows = self._engine.conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM adjudications GROUP BY verdict ORDER BY n DESC"
        ).fetchall()
        return {r["verdict"]: r["n"] for r in rows}
