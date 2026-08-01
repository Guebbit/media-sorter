"""Read-only use cases: what is in the index, and what has happened to it."""

from __future__ import annotations

import json
from typing import Any

from ..domain.rules import RuleSet
from .context import AppContext


def overview(ctx: AppContext) -> dict[str, Any]:
    """Pipeline progress and tallies — the payload behind `stats` and the UI."""
    reporting = ctx.storage.reporting
    return {
        "pipeline": reporting.pending_counts(),
        "categories": reporting.category_counts(),
        "actions": reporting.action_counts(),
        "classes": ctx.storage.detections.class_counts(),
        "verdicts": ctx.storage.adjudications.verdict_counts(),
    }


def duplicates(ctx: AppContext, limit: int = 20) -> list[dict[str, Any]]:
    """Groups of byte-identical indexed files, each with its list of paths."""
    return [
        {"hash": row["hash"], "count": row["n"], "paths": row["paths"].split("\n")}
        for row in ctx.storage.reporting.duplicates(limit)
    ]


def history(ctx: AppContext, limit: int = 30) -> list[dict[str, Any]]:
    """Recent irreversible actions."""
    return [
        {"action": row["action"], "path": row["path"], "detail": row["detail"] or "-",
         "created_at": row["created_at"]}
        for row in ctx.storage.activity.recent(limit)
    ]


def samples(ctx: AppContext, category: str | None = None, limit: int = 60) -> list[dict[str, Any]]:
    """Categorised images with their detections, for a UI that shows its work."""
    rows = ctx.storage.reporting.samples(category, limit)
    for row in rows:
        row["detections"] = [dict(d) for d in ctx.storage.detections.for_image(row["id"])]
    return rows


def search(ctx: AppContext, term: str, ruleset: RuleSet, limit: int = 50) -> list[dict[str, Any]]:
    """Free-text search over stored analysis metadata.

    The interesting fields are named after the classes the rules mention, so the
    ruleset is what turns an opaque JSON blob into a readable summary.
    """
    classes = ruleset.required_classes()
    results: list[dict[str, Any]] = []
    for row in ctx.storage.reporting.search_metadata(term, limit):
        data = json.loads(row["json"])
        highlights = []
        for cls in sorted(classes):
            kinds = data.get(f"{cls}_kinds") or []
            if kinds:
                highlights.append(f"{cls}: {', '.join(kinds)}")
        if data.get("activities"):
            highlights.append(", ".join(data["activities"]))
        results.append(
            {"path": row["path"], "category": row["category"], "highlights": highlights}
        )
    return results
