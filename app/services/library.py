"""Use case: bring the index up to date with what is on disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .. import filesystem
from ..scanning import ScanStats, scan
from .context import AppContext

log = logging.getLogger(__name__)


def _our_own_output(ctx: AppContext) -> frozenset[Path]:
    """The folders we write into, so a scan never indexes its own results.

    The output tree defaults to a subfolder of the library, and the trash folder
    defaults to a subfolder of that, so overlap is the normal case rather than a
    misconfiguration. Resolved here because the scanner compares resolved paths.
    """
    output = ctx.settings.output
    return frozenset({
        Path(output.folder).expanduser().resolve(),
        Path(output.trash_folder).expanduser().resolve(),
    })


def _drop_unconfigured_roots(ctx: AppContext) -> None:
    """Purge every indexed root that is no longer a configured input folder.

    Not flagged, gone: changing INPUT_FOLDERS is the user saying that folder is
    no longer part of this library, so its rows should stop counting the moment
    that is true, not linger as "missing" forever. The tracked output it
    produced is pruned first — the same cleanup an `apply` run already does for
    any other stale link — and only then the index rows, so nothing is left
    behind in either place. Nothing on the original drive is touched; a folder
    added back and rescanned is indexed fresh, from scratch.
    """
    configured = {
        str(Path(folder).expanduser().resolve()) for folder in ctx.settings.library.input_folders
    }
    images = ctx.storage.images
    links = ctx.storage.links
    pruned_any = False

    for root in images.distinct_roots():
        if root in configured:
            continue
        for link_path in links.paths_for_root(root):
            filesystem.remove(link_path)
            pruned_any = True
        removed = images.purge_root(root)
        log.info("%s is no longer a configured folder — dropped %d indexed file(s)", root, removed)

    if pruned_any:
        filesystem.prune_empty_dirs(ctx.settings.output.folder)


def scan_library(ctx: AppContext,
                 on_progress: Callable[[ScanStats], None] | None = None,
                 authoritative: bool = True) -> ScanStats:
    """Walk the configured input folders and bring the index up to date,
    excluding the tool's own output/trash folders.

    `authoritative=False` is for a one-off `--input` override: `ctx` then
    reflects only where this run looked, not the whole library, so a folder
    absent from it must not be treated as one the user dropped — see
    `_drop_unconfigured_roots`.
    """
    if authoritative:
        _drop_unconfigured_roots(ctx)
    return scan(
        ctx.settings.library,
        ctx.storage.images,
        workers=ctx.settings.workers.scan,
        on_progress=on_progress,
        exclude_paths=_our_own_output(ctx),
    )
