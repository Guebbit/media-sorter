"""Connections, transactions and schema — the mechanics of talking to SQLite.

Separated from the repositories so that "how a write is serialised" is decided
once. `write()` nests: a repository method that opens a transaction can be
called on its own, or wrapped by a caller that needs several of them to commit
together, without either side knowing about the other.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..errors import IncompatibleIndex
from .schema import SCHEMA, SCHEMA_VERSION

log = logging.getLogger(__name__)


class Engine:
    """Thread-safe SQLite access: one connection per thread, one writer at a time."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's connection, opened and tuned on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-64000")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close this thread's connection, if it opened one."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A serialized write transaction. Re-entrant: only the outermost
        `with` block begins and commits, so composing repository calls into one
        atomic unit is just nesting them."""
        with self._write_lock:
            depth = getattr(self._local, "depth", 0)
            conn = self.conn
            if depth == 0:
                conn.execute("BEGIN IMMEDIATE")
            self._local.depth = depth + 1
            try:
                yield conn
            except BaseException:
                if depth == 0:
                    conn.execute("ROLLBACK")
                raise
            else:
                if depth == 0:
                    conn.execute("COMMIT")
            finally:
                self._local.depth = depth

    # ------------------------------------------------------------------ schema

    def init_schema(self) -> None:
        """Create any missing tables and stamp the current schema version —
        safe to call on every startup, on a fresh database or an existing one."""
        with self._write_lock:
            self._check_version()
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT INTO schema_info(key, value) VALUES('version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (SCHEMA_VERSION,),
            )

    def _check_version(self) -> None:
        """Refuse to open an index from an older layout.

        The index is a cache, not a system of record: everything in it is
        re-derivable by re-scanning. Carrying migration code for a rebuildable
        cache costs more than it saves, so an old file gets a clear error
        instead of a silent half-upgrade.
        """
        tables = {
            row["name"] for row in
            self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "images" not in tables:
            return  # fresh database
        found = "0"
        if "schema_info" in tables:
            row = self.conn.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()
            found = row["value"] if row else "0"
        if found != SCHEMA_VERSION:
            raise IncompatibleIndex(
                f"{self.path} was written by schema v{found}, this build needs "
                f"v{SCHEMA_VERSION}. Delete it and re-run `photosort scan` "
                f"(detections are re-derived; nothing in your library is affected)."
            )

    def vacuum(self) -> None:
        """Reclaim disk space SQLite is holding onto after deletes."""
        self.conn.execute("VACUUM")

    def truncate(self, table: str) -> None:
        """Delete every row of `table` in its own write transaction.

        `table` is always one of our own hardcoded schema names, never user
        input, so building the statement by formatting is safe here — SQLite
        has no way to parameterize a table name.
        """
        with self.write() as conn:
            conn.execute(f"DELETE FROM {table}")
