"""Use case: carry out (or preview) whatever the rules decided."""

from __future__ import annotations

from typing import Any, Callable

from ..actions import ApplyStats, PlannedAction, apply_actions
from ..domain.rules import RuleSet
from .context import AppContext


def apply(ctx: AppContext, ruleset: RuleSet, prune: bool = True, dry_run: bool = False,
          on_progress: Callable[[int], None] | None = None
          ) -> tuple[ApplyStats, list[PlannedAction]]:
    """Carry out every action the rules decided on, `move` and `delete` included."""
    return apply_actions(
        ctx.action_context,
        ctx.storage.images,
        ctx.storage.links,
        ctx.storage.activity,
        ruleset,
        prune=prune,
        dry_run=dry_run,
        registry=ctx.actions,
        on_progress=on_progress,
    )


def describe_stats(stats: ApplyStats) -> str:
    """A plain-text tally of what `apply` did or would do — created, existing,
    pruned, plus whichever of moved/deleted/skipped/errors are nonzero —
    `skipped` meaning the original was already gone when its turn came. Shared
    by both front ends so the CLI and the web UI can never report different
    numbers for the same run; each is free to style or frame the string
    however suits it."""
    parts = [f"{stats.created} created", f"{stats.existing} existing", f"{stats.pruned} pruned"]
    if stats.moved:
        parts.append(f"{stats.moved} moved")
    if stats.deleted:
        parts.append(f"{stats.deleted} deleted")
    if stats.skipped:
        parts.append(f"{stats.skipped} skipped")
    if stats.errors:
        parts.append(f"{stats.errors} errors")
    return ", ".join(parts)


def preview(ctx: AppContext, ruleset: RuleSet, limit: int = 200) -> dict[str, Any]:
    """What would happen right now, without touching anything."""
    stats, planned = apply(ctx, ruleset, dry_run=True)
    return {
        "counts": stats.by_action,
        "sample": [
            {"action": p.action, "source": p.source, "target": p.target} for p in planned[:limit]
        ],
        "total": len(planned),
    }
