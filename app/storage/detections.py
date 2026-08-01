"""The detections table: what the detector saw, kept so rules can be re-run."""

from __future__ import annotations

import sqlite3
import time
from typing import Sequence

from ..domain.detection import Detection
from .engine import Engine


class DetectionRepository:
    """The `detections` table: every box YOLO found for an image, replaced
    wholesale each time that image is (re)detected."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def for_image(self, image_id: int) -> list[sqlite3.Row]:
        """Raw rows for one image's detections, most confident first."""
        return self._engine.conn.execute(
            "SELECT class, confidence, x1, y1, x2, y2 FROM detections WHERE image_id = ? "
            "ORDER BY confidence DESC",
            (image_id,),
        ).fetchall()

    def objects_for_image(self, image_id: int) -> list[Detection]:
        """The same rows as domain values — what the decision engine wants."""
        return [Detection.from_row(row) for row in self.for_image(image_id)]

    def replace_for_image(self, image_id: int, detections: Sequence[Detection], model: str) -> None:
        """Drop whatever this image had and store `detections` in its place —
        the write side of a fresh `detect_batch` result, one image at a time."""
        now = time.time()
        with self._engine.write() as conn:
            conn.execute("DELETE FROM detections WHERE image_id = ?", (image_id,))
            if detections:
                conn.executemany(
                    """INSERT INTO detections
                           (image_id, class, confidence, x1, y1, x2, y2, model, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    [
                        (image_id, d.cls, d.confidence, d.x1, d.y1, d.x2, d.y2, model, now)
                        for d in detections
                    ],
                )

    def clear(self) -> None:
        """Drop every stored detection — used by a full `reset`."""
        self._engine.truncate("detections")

    def class_counts(self) -> dict[str, int]:
        """How many detections exist per class, most common first — the
        `mediasort stats` breakdown of what the detector has actually seen."""
        rows = self._engine.conn.execute(
            "SELECT class, COUNT(*) AS n FROM detections GROUP BY class ORDER BY n DESC"
        ).fetchall()
        return {r["class"]: r["n"] for r in rows}
