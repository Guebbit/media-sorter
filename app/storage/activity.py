"""The action log: an append-only record of everything irreversible.

Links are re-derivable and therefore uninteresting to log. A deletion is not,
so it is written down here before the run forgets about it.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Sequence

from .engine import Engine


class ActivityLog:
    """The `action_log` table: an append-only record of every `move`/`delete`
    `apply` has ever carried out. Never updated or deleted from — a history,
    not a cache."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def record(self, entries: Sequence[tuple[int, str, str]]) -> None:
        """Append `(image_id, action, detail)` rows in bulk — called once per
        batch by `_execute`, not once per file."""
        if not entries:
            return
        now = time.time()
        with self._engine.write() as conn:
            conn.executemany(
                "INSERT INTO action_log (image_id, action, detail, created_at) VALUES (?,?,?,?)",
                [(image_id, action, detail, now) for image_id, action, detail in entries],
            )

    def recent(self, limit: int = 100) -> list[sqlite3.Row]:
        """The most recent `limit` log entries, newest first, joined with
        each image's current path — the `mediasort history` view."""
        return self._engine.conn.execute(
            """SELECT a.action, a.detail, a.created_at, i.path FROM action_log a
               JOIN images i ON i.id = a.image_id ORDER BY a.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
