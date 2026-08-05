"""Connections, transactions and schema — the mechanics of talking to SQLite.

Separated from the repositories so that "how a write is serialised" is decided
once. `write()` nests: a repository method that opens a transaction can be
called on its own, or wrapped by a caller that needs several of them to commit
together, without either side knowing about the other.
"""

from __future__ import annotations

import itertools
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schema import SCHEMA, SCHEMA_VERSION

log = logging.getLogger(__name__)

#: Names for in-memory databases. Process-wide and never reused, so two engines
#: alive at once — the sorting index and the duplicate index — cannot collide,
#: and neither can an engine built after an earlier one was collected.
_MEMORY_NAMES = itertools.count()


class Engine:
    """Thread-safe SQLite access: one connection per thread, one writer at a time.

    `path` may be `None`, meaning keep the index in memory and leave nothing
    behind. That is the ordinary case for a tool pointed at a folder once: the
    index only pays for itself when the same folder is worked on twice, so
    writing one by default would litter every library it ever touched.
    """

    def __init__(self, path: Path | None):
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.path = Path(path) if path is not None else None
        self._keepalive: sqlite3.Connection | None = None

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._dsn, self._uri = str(self.path), False
        else:
            # A plain ":memory:" database belongs to the one connection that
            # opened it, and this engine opens one per thread — each would get
            # its own empty database and the stages would not see each other's
            # work. A shared-cache URI makes every connection the same database;
            # the name is per-engine so two Storages cannot collide.
            #
            # Counted, not `id(self)`: CPython reuses the address of a collected
            # object, so a short-lived engine could hand its name — and with it
            # any rows still held alive by a worker thread's connection — to the
            # next engine built after it. A counter is never reused.
            self._dsn = f"file:mediasort-{next(_MEMORY_NAMES)}?mode=memory&cache=shared"
            self._uri = True
            # An in-memory database lives only as long as a connection to it
            # does. Stage threads come and go, so the engine holds one itself.
            self._keepalive = self._open()

    @property
    def persistent(self) -> bool:
        """Whether this index outlives the run — i.e. whether it has a file."""
        return self.path is not None

    def _open(self) -> sqlite3.Connection:
        """A new connection to this engine's database, tuned the same way
        whether it is a file or the shared in-memory one."""
        conn = sqlite3.connect(self._dsn, timeout=60.0, isolation_level=None, uri=self._uri)
        conn.row_factory = sqlite3.Row
        # journal_mode is a no-op on an in-memory database (it reports "memory"
        # rather than failing), so the same list serves both.
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "foreign_keys=ON",
                       "busy_timeout=60000", "temp_store=MEMORY", "cache_size=-64000"):
            conn.execute(f"PRAGMA {pragma}")
        if self._uri:
            # Shared cache — the in-memory mode — locks per *table*, not per
            # database, and a reader that meets a held table lock gets
            # SQLITE_LOCKED back immediately. Not SQLITE_BUSY: the busy handler
            # is never consulted, so the `busy_timeout` above does not apply and
            # WAL is a no-op here anyway. The web UI polls `/api/stats` on its
            # own thread while the stages write, so without this every poll that
            # lands mid-transaction fails outright ("database table is locked:
            # images"). Dropping the read lock is the documented way out; the
            # cost is that such a read can see a transaction that later rolls
            # back, which for progress counters beats not answering at all.
            # File-backed databases skip this and keep WAL's real isolation.
            conn.execute("PRAGMA read_uncommitted=1")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's connection, opened and tuned on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open()
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
            self._rebuild_if_stale()
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT INTO schema_info(key, value) VALUES('version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (SCHEMA_VERSION,),
            )

    def _rebuild_if_stale(self) -> None:
        """Throw away an index from an older layout and start over.

        Everything in here is re-derivable by re-scanning, which is what makes
        this safe: the index is a cache, and a cache that argues with you is a
        cache doing its job badly. This used to refuse to open and tell the
        user to delete the file by hand — a chore with exactly one possible
        answer, and one that arrives at the worst moment, when they wanted to
        sort some photos. Nothing in the library is touched either way.
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
        if found == SCHEMA_VERSION:
            return

        log.warning(
            "index at %s was written by schema v%s and this build needs v%s — rebuilding it; "
            "the next scan re-derives everything, your library is untouched",
            self.path or "memory", found, SCHEMA_VERSION,
        )
        # Dropped rather than unlinked, so this works identically for a file and
        # for the in-memory database, and needs no reconnect either way.
        conn = self.conn
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            for name in sorted(tables):
                conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

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
