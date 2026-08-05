"""Incremental indexing.

Cheap check first: a file whose size and mtime match the index is skipped
without ever being opened. Only new or changed files get hashed and probed,
which is what keeps a re-scan of 300k photos to a couple of minutes.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import filesystem, imaging
from ..config import LibrarySettings
from ..storage import ImageRepository
from .walker import walk

log = logging.getLogger(__name__)

UPSERT_BATCH = 500


@dataclass(slots=True)
class ScanStats:
    seen: int = 0
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    errors: int = 0
    missing: int = 0
    roots: list[str] = field(default_factory=list)


def _build_row(path: str, stat: os.stat_result, root: str) -> dict:
    """Everything the index stores about a file, for a file worth reading."""
    meta, phash = imaging.inspect(path)
    return {
        "path": path,
        "filename": os.path.basename(path),
        "root": root,
        "hash": filesystem.sha256_file(path),
        "phash": phash,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "width": meta.width,
        "height": meta.height,
        "format": meta.format,
        "taken_at": meta.taken_at,
    }


def scan(library: LibrarySettings, images: ImageRepository, workers: int = 4,
         on_progress: Callable[[ScanStats], None] | None = None,
         exclude_paths: frozenset[Path] = frozenset()) -> ScanStats:
    """Walk the configured folders and bring the index up to date.

    `exclude_paths` are the folders this tool writes into — see `walker.walk`.
    Whether a folder no longer configured should be dropped from the index is
    decided a layer up, in `services.library`, before this is called — pure
    indexing here, no opinion about what "no longer wanted" means.
    """
    # The one place every front end passes through on its way to reading files,
    # so the "you have not said where your photos are" error lives here once.
    library.require_folders()
    stats = ScanStats()

    for configured in library.input_folders:
        root = Path(configured).resolve()
        if not root.is_dir():
            log.warning("input folder does not exist, skipping: %s", root)
            continue
        stats.roots.append(str(root))
        log.info("scanning %s", root)

        # Assume everything under this root is gone; anything we find clears it.
        images.flag_root_missing(str(root))
        _scan_root(root, library, images, workers, stats, on_progress, exclude_paths)

    stats.missing = images.count("missing = 1")
    if on_progress:
        on_progress(stats)
    return stats


def _scan_root(root: Path, library: LibrarySettings, images: ImageRepository, workers: int,
               stats: ScanStats, on_progress: Callable[[ScanStats], None] | None,
               exclude_paths: frozenset[Path] = frozenset()) -> None:
    """Walk one input root, dispatching changed/new files to a thread pool for
    metadata extraction while batching writes so a large scan does not hit
    the database once per file."""
    pending: list[tuple[str, os.stat_result]] = []
    present_ids: list[int] = []
    upserts: list[dict] = []

    def flush_upserts(force: bool = False) -> None:
        """Write buffered upserts once `UPSERT_BATCH` is reached, or always
        if `force` (used at the end of the walk)."""
        if upserts and (force or len(upserts) >= UPSERT_BATCH):
            images.upsert(upserts)
            upserts.clear()

    def flush_present(force: bool = False) -> None:
        """Write buffered "still here" ids once `UPSERT_BATCH` is reached, or
        always if `force`."""
        if present_ids and (force or len(present_ids) >= UPSERT_BATCH):
            images.touch_present(present_ids)
            present_ids.clear()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        def drain() -> None:
            """Send every pending (path, stat) pair to the pool for metadata
            extraction, collect the results, and flush them to storage."""
            nonlocal pending
            if not pending:
                return
            futures = {
                pool.submit(_build_row, path, stat, str(root)): path for path, stat in pending
            }
            pending = []
            for future, path in futures.items():
                try:
                    upserts.append(future.result())
                except OSError as exc:
                    stats.errors += 1
                    log.warning("cannot read %s: %s", path, exc)
            flush_upserts()

        for path, stat in walk(root, library, exclude_paths):
            stats.seen += 1
            existing = images.stat_of(path)
            if _unchanged(existing, stat):
                stats.unchanged += 1
                present_ids.append(existing["id"])
                flush_present()
            else:
                if existing is None:
                    stats.added += 1
                else:
                    stats.changed += 1
                pending.append((path, stat))
                if len(pending) >= workers * 8:
                    drain()

            if on_progress and stats.seen % 200 == 0:
                on_progress(stats)

        drain()

    flush_upserts(force=True)
    flush_present(force=True)


def _unchanged(existing, stat: os.stat_result) -> bool:
    """Size and mtime agreeing is good enough to skip opening the file."""
    return (
        existing is not None
        and existing["size"] == stat.st_size
        and abs((existing["mtime"] or 0) - stat.st_mtime) < 1e-6
    )
