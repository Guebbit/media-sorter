"""What a person decided while reviewing near-duplicates.

Two tables, two different questions, one per class:

  `dupe_dismissals`  a verdict on a *pair* — "these two are not duplicates"
  `dupe_marks`       a verdict on one *photo* — "keep this" / "discard this"

Both are records, not instructions: nothing here moves, deletes or otherwise
touches a file. Deciding what a mark should eventually cause is a separate
question, deliberately not answered yet.

Kept apart from the grouping itself (`domain.similarity`) — these tables only
remember verdicts; working out what a verdict means for a group of three or
more is the grouper's job, not theirs.
"""

from __future__ import annotations

import time
from typing import Sequence

from .engine import Engine

#: The two things a photo can be marked. Anything else is rejected on the way
#: in, so a typo cannot reach storage and quietly mean nothing later.
KEEP = "keep"
DISCARD = "discard"
MARKS = (KEEP, DISCARD)


class DupeDismissals:
    """The `dupe_dismissals` table: every pair a user has dismissed as "not
    duplicates", each one stored once with its smaller id first so the same
    pair can never be recorded both ways round."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def dismiss(self, image_ids: Sequence[int]) -> None:
        """Record every pair within `image_ids` as dismissed — called with a
        whole group, so the grouper stops proposing any pairing inside it."""
        pairs = {
            (a, b) if a < b else (b, a)
            for i, a in enumerate(image_ids)
            for b in image_ids[i + 1:]
            if a != b
        }
        if not pairs:
            return
        now = time.time()
        with self._engine.write() as conn:
            conn.executemany(
                "INSERT INTO dupe_dismissals (image_id_a, image_id_b, created_at) "
                "VALUES (?,?,?) ON CONFLICT(image_id_a, image_id_b) DO NOTHING",
                [(a, b, now) for a, b in pairs],
            )

    def dismissed_pairs(self) -> frozenset[tuple[int, int]]:
        """Every dismissed pair, `(smaller_id, larger_id)` — what the
        grouper excludes before clustering."""
        rows = self._engine.conn.execute(
            "SELECT image_id_a, image_id_b FROM dupe_dismissals"
        ).fetchall()
        return frozenset((r["image_id_a"], r["image_id_b"]) for r in rows)


class DupeMarks:
    """The `dupe_marks` table: one photo, one recorded decision.

    A mark is remembered so a large library can be reviewed across several
    sittings without losing your place. It causes nothing on its own.
    """

    def __init__(self, engine: Engine):
        self._engine = engine

    def set(self, image_ids: Sequence[int], mark: str) -> int:
        """Mark every id `keep` or `discard`. Returns how many were marked.

        Raises `ValueError` on any other word: an unrecognised mark stored
        now is a silently ignored decision later, which is worse than a
        refusal here.
        """
        if mark not in MARKS:
            raise ValueError(f"unknown mark {mark!r}; expected one of {', '.join(MARKS)}")
        if not image_ids:
            return 0
        now = time.time()
        with self._engine.write() as conn:
            conn.executemany(
                "INSERT INTO dupe_marks (image_id, mark, created_at) VALUES (?,?,?) "
                "ON CONFLICT(image_id) DO UPDATE SET mark=excluded.mark, "
                "created_at=excluded.created_at",
                [(image_id, mark, now) for image_id in image_ids],
            )
        return len(image_ids)

    def clear(self, image_ids: Sequence[int]) -> int:
        """Forget the decision for every id — back to unmarked."""
        if not image_ids:
            return 0
        with self._engine.write() as conn:
            conn.executemany(
                "DELETE FROM dupe_marks WHERE image_id = ?", [(i,) for i in image_ids]
            )
        return len(image_ids)

    def all(self) -> dict[int, str]:
        """Every marked image id mapped to its mark — read once per review
        page rather than once per photo."""
        rows = self._engine.conn.execute("SELECT image_id, mark FROM dupe_marks").fetchall()
        return {row["image_id"]: row["mark"] for row in rows}
