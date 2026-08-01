"""Use case: carry out (or preview) whatever the rules decided."""

from __future__ import annotations

from typing import Any, Callable

from ..actions import ApplyStats, PlannedAction, apply_actions
from ..domain.rules import RuleSet
from .context import AppContext


def permitted_consuming(ctx: AppContext, requested: bool) -> frozenset[str]:
    """Which consuming actions may run: all of them, but only if this request
    explicitly confirmed it — the one gate left on moving or deleting originals.
    """
    return frozenset(ctx.actions.consuming_names()) if requested else frozenset()


def apply(ctx: AppContext, ruleset: RuleSet, prune: bool = True, dry_run: bool = False,
          confirmed: bool = False,
          on_progress: Callable[[int], None] | None = None
          ) -> tuple[ApplyStats, list[PlannedAction]]:
    """Moving and deleting need an explicit request — `confirmed=True`."""
    return apply_actions(
        ctx.action_context,
        ctx.storage.images,
        ctx.storage.links,
        ctx.storage.activity,
        ruleset,
        prune=prune,
        dry_run=dry_run,
        permitted=permitted_consuming(ctx, confirmed),
        registry=ctx.actions,
        on_progress=on_progress,
    )


def describe_stats(stats: ApplyStats) -> str:
    """A plain-text tally of what `apply` did or would do — created, existing,
    pruned, plus whichever of moved/deleted/skipped/errors are nonzero. Shared
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
    stats, planned = apply(ctx, ruleset, dry_run=True, confirmed=True)
    return {
        "counts": stats.by_action,
        "skipped_unconfirmed": stats.skipped,
        "sample": [
            {"action": p.action, "source": p.source, "target": p.target} for p in planned[:limit]
        ],
        "total": len(planned),
    }
