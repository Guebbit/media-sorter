"""The links table: every entry we created in the output tree.

Recording them is what makes the output tree disposable and idempotent — a
second run recognises its own work, and a re-categorised image has its old link
pruned instead of accumulating.
"""

from __future__ import annotations

import time
from typing import Iterator, Sequence

from .engine import Engine


class LinkRepository:
    """The `links` table: every path `apply` has written into the output
    tree, one row per output file, so a later run can recognise its own work."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def record(self, entries: Sequence[tuple[int, str]]) -> None:
        """Record (or update) `(image_id, link_path)` rows in bulk — called
        once per batch by `_execute`, not once per file."""
        if not entries:
            return
        now = time.time()
        with self._engine.write() as conn:
            conn.executemany(
                "INSERT INTO links (image_id, link_path, created_at) VALUES (?,?,?) "
                "ON CONFLICT(link_path) DO UPDATE SET image_id=excluded.image_id",
                [(image_id, path, now) for image_id, path in entries],
            )

    def known(self) -> dict[str, int]:
        """Every recorded output path mapped to the image it belongs to —
        what pruning compares the current plan against."""
        rows = self._engine.conn.execute("SELECT link_path, image_id FROM links").fetchall()
        return {r["link_path"]: r["image_id"] for r in rows}

    def paths_for_root(self, root: str) -> list[str]:
        """Every output path recorded for an image under `root` — read before
        `images.purge_root(root)` runs, since deleting the image rows cascades
        their link records away before an `apply` ever gets a chance to notice
        they are now stale and remove the files they point at."""
        rows = self._engine.conn.execute(
            "SELECT l.link_path FROM links l JOIN images i ON i.id = l.image_id WHERE i.root = ?",
            (root,),
        ).fetchall()
        return [r["link_path"] for r in rows]

    def forget(self, paths: Sequence[str]) -> None:
        """Drop the records for `paths` — called after they are pruned from disk."""
        if not paths:
            return
        with self._engine.write() as conn:
            conn.executemany("DELETE FROM links WHERE link_path = ?", [(p,) for p in paths])

    def clear(self) -> None:
        """Forget every recorded output path — used by `clean --links`."""
        self._engine.truncate("links")

    def recorded_paths(self) -> Iterator[str]:
        """Every output path we have written down — what `verify` walks,
        checking each is still on disk."""
        for row in self._engine.conn.execute("SELECT link_path FROM links"):
            yield row["link_path"]
