"""Mutating the filesystem, expressed as intents.

Everything that creates, moves or removes a file goes through here, so the
layers above talk about "copy it there" or "move to trash" rather than naming
`shutil` functions. Keeping the primitives in one module is also what makes the
destructive surface auditable: this file is the complete list of places that can
touch a byte on disk.

There is deliberately no symlink or hardlink here. Every rule action names the
thing it does — copy, move, delete — so there is no second setting deciding what
"put it in the folder" turns into, and nothing that a filesystem might refuse.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from app.errors import MediaSortError

log = logging.getLogger(__name__)

HASH_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """The file's SHA-256 hex digest, read in chunks so a large photo never
    needs to fit in memory whole — what the scanner keys duplicate detection on."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def exists(path: str | Path | None) -> bool:
    """Whether `path` is set and points at something real."""
    if not path:
        return False
    return Path(path).exists()


def copy(target: str | Path, source: str | Path) -> None:
    """Duplicate `source` at `target`, replacing whatever was there.

    `copy2` rather than `copy`, so the duplicate keeps the original's timestamps
    — a sorted folder ordered by date is most of the point of sorting it.
    """
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remove(destination)
    shutil.copy2(source, destination)


def unwritable(directory: str | Path) -> str | None:
    """Why the output folder cannot be written to, or None if it can.

    Asks by doing it. Whatever stops the first file stops all of them, so this
    runs once per apply and turns what would be one warning per photo into one
    answer before anything starts.
    """
    target = Path(directory)
    try:
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target, prefix=".mediasort-probe-"):
            pass
    except OSError as exc:
        return exc.strerror or str(exc)
    return None


def move(source: str | Path, destination: str | Path, disambiguate_with: object = "") -> Path:
    """Move a file, never overwriting: a taken name gets a suffix instead."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if exists(target):
        target = target.with_name(f"{target.stem}_{disambiguate_with}{target.suffix}")
    shutil.move(str(source), str(target))
    return target


def remove(path: str | Path) -> bool:
    """Unlink a file or symlink. False when there was nothing there."""
    target = Path(path)
    if not exists(target):
        return False
    target.unlink()
    return True


def trash(path: str | Path) -> bool:
    """Send a file to the desktop trash. False when there was nothing there.

    Distinct from `remove`, which unlinks: this one is recoverable from the
    file manager, which is the only reason a delete button is offered at all.
    `send2trash` rather than an XDG implementation here — the `.trashinfo`
    sidecar, per-filesystem trash directories and name collisions are exactly
    what that library exists to get right, and a half-done trash is a delete.
    """
    target = Path(path)
    if not exists(target):
        return False
    try:
        from send2trash import send2trash
    except ImportError as exc:  # optional: only the delete path needs it
        raise MediaSortError(
            "deleting to the desktop trash needs send2trash "
            "(pip install send2trash) — until then, discard moves files into "
            "the duplicates folder instead"
        ) from exc
    send2trash(os.fspath(target))
    return True


def remove_tree_contents(root: str | Path) -> tuple[int, list[str]]:
    """Empty a generated directory out, deepest entries first.

    Returns the number of entries removed and one message per entry that could
    not be. Only ever pointed at output we created ourselves.
    """
    target = Path(root)
    removed = 0
    problems: list[str] = []
    if not target.is_dir():
        return removed, problems
    for path in sorted(target.rglob("*"), reverse=True):
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                path.rmdir()
        except OSError as exc:
            problems.append(f"{path}: {exc}")
    return removed, problems


def prune_empty_dirs(root: str | Path) -> None:
    """Remove directories a pruned link left behind. The root itself stays."""
    start = Path(root)
    if not start.is_dir():
        return
    for directory, dirnames, filenames in os.walk(start, topdown=False):
        path = Path(directory)
        if path == start:
            continue
        if not dirnames and not filenames:
            try:
                path.rmdir()
            except OSError:
                pass


def relative_to_any(path: str | Path, roots: Iterable[str | Path]) -> Path:
    """The path as seen from whichever root contains it; bare filename if none."""
    candidate = Path(path)
    for root in roots:
        try:
            return candidate.relative_to(root)
        except ValueError:
            continue
    return Path(candidate.name)
