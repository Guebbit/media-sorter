"""Deciding which files on disk are part of the library.

Only this module knows the filtering rules, so "what counts as a photo" is one
question with one answer, and the scanner proper is free to be about
incremental bookkeeping.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator

from ..config import LibrarySettings

log = logging.getLogger(__name__)


def _resolve(path: Path | str) -> Path:
    """Comparable form of a path that may not exist yet, and never raises."""
    try:
        return Path(path).resolve()
    except OSError:
        return Path(os.path.abspath(path))


def walk(root: Path, library: LibrarySettings,
         exclude_paths: frozenset[Path] = frozenset()) -> Iterator[tuple[str, os.stat_result]]:
    """Every library file under `root`, with the stat we already paid for.

    Prunes excluded and hidden directories, and does not follow symlinks unless
    asked to — otherwise a link back into the tree would loop forever.

    `exclude_paths` prunes by location rather than by name: the folders this tool
    writes into. Nothing stops an output or trash folder from sitting inside the
    library — it is the default — and without this a `copy` run would index its
    own copies, and a `delete` run would hand the trash back as new photos.
    """
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (PermissionError, FileNotFoundError, NotADirectoryError) as exc:
            log.warning("cannot read %s: %s", current, exc)
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=library.follow_symlinks):
                    if entry.name in library.exclude_dirs or entry.name.startswith("."):
                        continue
                    if exclude_paths and _resolve(entry.path) in exclude_paths:
                        log.debug("not indexing our own output: %s", entry.path)
                        continue
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=library.follow_symlinks):
                    if os.path.splitext(entry.name)[1].lower() in library.extensions:
                        yield entry.path, entry.stat()
            except OSError as exc:
                log.warning("cannot stat %s: %s", entry.path, exc)
