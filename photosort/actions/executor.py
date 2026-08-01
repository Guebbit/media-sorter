"""Carrying out a plan, idempotently.

Everything here exists to make a second run cheap and a re-categorisation
clean: recognise the links we already made, prune the ones that no longer
belong, and never treat a deletion as something that can be "un-applied".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import filesystem
from ..domain.rules import RuleSet
from ..errors import OutputNotWritable
from ..storage import ActivityLog, ImageRepository, LinkRepository
from .base import ActionContext, PlannedAction
from .planner import plan_all
from .registry import ActionRegistry

log = logging.getLogger(__name__)

BATCH = 500
PROGRESS_EVERY = 200


@dataclass(slots=True)
class ApplyStats:
    """What one `apply_actions` call did (or, for `dry_run=True`, would do) —
    the tally both front ends report from."""

    created: int = 0
    existing: int = 0
    pruned: int = 0
    skipped: int = 0
    errors: int = 0
    by_action: dict[str, int] = field(default_factory=dict)

    def bump(self, action: str) -> None:
        """Count one more planned or executed step under `action`'s name."""
        self.by_action[action] = self.by_action.get(action, 0) + 1

    # Every executed action is counted in `by_action` already, so the two that
    # front ends report separately are read back from it rather than tallied
    # twice — a second counter is a second thing that can disagree.
    @property
    def deleted(self) -> int:
        """How many `delete` actions ran."""
        return self.by_action.get("delete", 0)

    @property
    def moved(self) -> int:
        """How many `move` actions ran."""
        return self.by_action.get("move", 0)


def apply_actions(ctx: ActionContext, images: ImageRepository, links: LinkRepository,
                  activity: ActivityLog, ruleset: RuleSet, registry: ActionRegistry, *,
                  prune: bool = True, dry_run: bool = False,
                  permitted: frozenset[str] = frozenset(),
                  on_progress: Callable[[int], None] | None = None
                  ) -> tuple[ApplyStats, list[PlannedAction]]:
    """Plan everything, prune what no longer belongs, then execute.

    `permitted` names the consuming actions this run is allowed to carry out;
    anything that consumes an original and is absent from it is planned,
    counted, and dropped.
    """
    stats = ApplyStats()
    planned = plan_all(ctx, images, ruleset, registry)
    planned = _filter_unconfirmed(planned, registry, permitted, stats)
    _require_writable_output(ctx, planned)

    if dry_run:
        for item in planned:
            stats.bump(item.action)
        return stats, planned

    known_links = links.known()
    if prune:
        _prune(planned, known_links, links, registry, stats)
    _execute(ctx, planned, known_links, images, links, activity, registry, stats, on_progress)
    if prune:
        filesystem.prune_empty_dirs(ctx.output.folder)
    return stats, planned


def _require_writable_output(ctx: ActionContext, planned: list[PlannedAction]) -> None:
    """Fail before the first file rather than once per file.

    Whatever stops the output folder being written stops every photo in the
    library identically, so reporting it per photo buries the one fact that
    matters under thousands of copies of itself. Skipped when nothing is going
    to write there — a ruleset that only deletes needs no output folder at all.
    """
    if not any(p.target for p in planned):
        return
    problem = filesystem.unwritable(ctx.output.folder)
    if problem:
        raise OutputNotWritable(f"cannot write to {ctx.output.folder}: {problem}")


def _filter_unconfirmed(planned: list[PlannedAction], registry: ActionRegistry,
                        permitted: frozenset[str], stats: ApplyStats) -> list[PlannedAction]:
    """Drop any planned step whose action consumes the original but is not in
    `permitted`, counting each as skipped rather than silently vanishing."""
    withheld = registry.consuming_names() - permitted
    blocked = [p for p in planned if p.action in withheld]
    if not blocked:
        return planned
    for name in sorted({p.action for p in blocked}):
        log.warning(
            "%d %r action(s) skipped; they need --yes (or the UI checkbox) to run",
            sum(p.action == name for p in blocked), name,
        )
    stats.skipped += len(blocked)
    return [p for p in planned if p not in blocked]


def _prune(planned: list[PlannedAction], known_links: dict[str, int], links: LinkRepository,
           registry: ActionRegistry, stats: ApplyStats) -> None:
    """Tracked output is re-derivable, so it is the only thing that gets pruned."""
    wanted = {
        p.target: p.image_id for p in planned if p.target and registry.get(p.action).tracked
    }
    stale = [path for path, image_id in known_links.items() if wanted.get(path) != image_id]
    for path in stale:
        try:
            filesystem.remove(path)
            stats.pruned += 1
        except OSError as exc:
            stats.errors += 1
            log.warning("cannot remove stale link %s: %s", path, exc)
    links.forget(stale)


def _execute(ctx: ActionContext, planned: list[PlannedAction], known_links: dict[str, int],
             images: ImageRepository, links: LinkRepository, activity: ActivityLog,
             registry: ActionRegistry, stats: ApplyStats,
             on_progress: Callable[[int], None] | None) -> None:
    """Run every planned action in order, batching the resulting link/activity
    rows so a large apply does not write to the database once per image."""
    link_batch: list[tuple[int, str, str]] = []
    log_batch: list[tuple[int, str, str]] = []

    for done, item in enumerate(planned, 1):
        action = registry.get(item.action)
        already_there = (
            action.tracked
            and known_links.get(item.target or "") == item.image_id
            and filesystem.exists(item.target)
        )
        if already_there:
            stats.existing += 1
        else:
            try:
                if action.execute(ctx, item):
                    stats.bump(item.action)
                    if action.tracked:
                        link_batch.append((item.image_id, item.target or "", item.detail))
                        stats.created += 1
                    if action.consumes_original:
                        # After execute, `target` is where the file actually
                        # ended up, which is not always where it was planned.
                        log_batch.append((item.image_id, item.action, item.target or item.detail))
                        action.settle(item, images)
                else:
                    stats.skipped += 1
            except OSError as exc:
                stats.errors += 1
                log.warning("%s failed: %s", item.describe(), exc)

        if len(link_batch) >= BATCH:
            links.record(link_batch)
            link_batch.clear()
        if len(log_batch) >= BATCH:
            activity.record(log_batch)
            log_batch.clear()
        if on_progress and done % PROGRESS_EVERY == 0:
            on_progress(done)

    links.record(link_batch)
    activity.record(log_batch)


def verify_links(ctx: ActionContext, links: LinkRepository) -> dict[str, int]:
    """Check the output tree still holds everything we recorded.

    A copy either exists or it does not — there is no third state now that
    nothing points at anything, so "broken" only survives to name the leftovers
    of an older run: symlinks whose originals have since gone.
    """
    report = {"copies_ok": 0, "copies_broken": 0, "copies_missing": 0}
    for row in links.iter_with_sources():
        copied = Path(row["link_path"])
        if not filesystem.exists(copied):
            report["copies_missing"] += 1
        elif copied.is_symlink() and not copied.exists():
            report["copies_broken"] += 1
        else:
            report["copies_ok"] += 1
    return report
