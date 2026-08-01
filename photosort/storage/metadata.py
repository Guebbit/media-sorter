"""The metadata table: whatever the semantic pass returned, stored as JSON.

Kept opaque on purpose. The schema the vision engine is asked for follows the
user's rules, so pinning it into columns would mean a migration every time
someone adds a class to a rule.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .engine import Engine


class MetadataRepository:
    """The `metadata` table: one opaque JSON blob per image, the parsed reply
    from `OllamaClient.analyze`."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def upsert(self, image_id: int, payload: dict[str, Any], model: str) -> None:
        """Store (or replace) `payload` — a semantic-pass result, already
        coerced by `analyzing.contract.coerce` — as this image's metadata JSON."""
        with self._engine.write() as conn:
            conn.execute(
                """INSERT INTO metadata (image_id, json, model, created_at) VALUES (?,?,?,?)
                   ON CONFLICT(image_id) DO UPDATE SET
                       json=excluded.json, model=excluded.model, created_at=excluded.created_at""",
                (image_id, json.dumps(payload, ensure_ascii=False), model, time.time()),
            )

    def clear(self) -> None:
        """Drop every stored metadata blob — used by a full `reset`."""
        self._engine.truncate("metadata")
