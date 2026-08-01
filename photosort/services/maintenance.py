"""Use cases that repair, re-queue or throw away derived state.

None of them can touch an original: only `move` and `delete` do that, they live
in `actions`, and each sits behind two locks of its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .. import filesystem
from ..actions.executor import verify_links
from ..storage import Stage
from .context import AppContext


@dataclass(slots=True)
class VerifyReport:
    """The result of `verify`: whether originals and their output-tree copies
    still agree with what the index believes."""

    copies_ok: int = 0
    copies_broken: int = 0
    copies_missing: int = 0
    sources_missing: int = 0

    def as_dict(self) -> dict[str, int]:
        """Plain-dict form, for rendering or serializing."""
        return {
            "copies_ok": self.copies_ok, "copies_broken": self.copies_broken,
            "copies_missing": self.copies_missing, "sources_missing": self.sources_missing,
        }


@dataclass(slots=True)
class CleanReport:
    """The result of `clean_links`/`purge_missing`/`vacuum`, whichever ran."""

    removed: int = 0
    purged: int = 0
    vacuumed: bool = False
    problems: list[str] = field(default_factory=list)


def verify(ctx: AppContext) -> VerifyReport:
    """Check that indexed originals and generated links still exist."""
    report = VerifyReport(**verify_links(ctx.action_context, ctx.storage.links))
    for row in ctx.storage.images.iter_live_paths():
        if not os.path.exists(row["path"]):
            report.sources_missing += 1
    return report


def retry_errors(ctx: AppContext, stage: Stage | None = None) -> int:
    """Put failed images back in the queue."""
    return ctx.storage.images.retry_errors(stage)


#: Which derived-data repositories (attribute names on `Storage`) a stage's
#: reset must also clear, beyond re-queuing the stage itself. Resetting
#: detection cascades into adjudication too, since everything downstream was
#: derived from detections that are about to be dropped. A dict instead of an
#: if/elif chain, so a new `Stage` only needs an entry here, not a new branch.
_RESET_CLEARS: dict[Stage, tuple[str, ...]] = {
    Stage.DETECT: ("detections", "adjudications"),
    Stage.ADJUDICATE: ("adjudications",),
    Stage.ANALYZE: ("metadata",),
}


def reset(ctx: AppContext, stage: Stage | None = None) -> None:
    """Queue a stage again for every image, dropping what it had derived.

    One transaction: an interrupted reset would otherwise leave detections
    behind for images that are about to be re-detected.

    Resetting adjudication has to undo the decision as well as the verdicts. A
    settled `absent` cleared the review flag, and the flag is the queue — leave
    it cleared and the images just released would never be picked up again. So
    the caller re-decides from the raw detections afterwards, which is exactly
    the state the stage found them in the first time.
    """
    stages = [stage] if stage else list(Stage)
    with ctx.storage.engine.write():
        for member in stages:
            ctx.storage.images.reset_stage(member)
            for repo_name in _RESET_CLEARS[member]:
                getattr(ctx.storage, repo_name).clear()


def clean_links(ctx: AppContext) -> CleanReport:
    """Delete the whole generated output tree and forget it ever existed."""
    removed, problems = filesystem.remove_tree_contents(ctx.settings.output.folder)
    ctx.storage.links.clear()
    return CleanReport(removed=removed, problems=problems)


def purge_missing(ctx: AppContext) -> int:
    """Drop rows whose file is gone from disk."""
    return ctx.storage.images.purge_missing()


def vacuum(ctx: AppContext) -> None:
    """Reclaim space SQLite is holding onto after deletes — `VACUUM`."""
    ctx.storage.engine.vacuum()
