"""Deciding what should happen, for every image, without doing any of it.

Kept apart from the executor because planning is pure: given the index and the
rules it produces a list, which is equally the input to a real run and the
entire content of `--dry-run`.
"""

from __future__ import annotations

from ..domain.rules import RuleSet
from ..storage import ImageRepository
from .base import ActionContext, ImageTarget, NameAllocator, PlannedAction
from .registry import ActionRegistry


def plan_all(ctx: ActionContext, images: ImageRepository, ruleset: RuleSet,
             registry: ActionRegistry) -> list[PlannedAction]:
    """Turn every categorised image into concrete, not-yet-executed actions."""
    namer = NameAllocator()
    doubt = ruleset.for_doubt()
    doubt_action = registry.get(doubt.action)
    planned: list[PlannedAction] = []

    for row in images.iter_actionable():
        rule = ruleset.by_name(row["category"] or "")
        if rule is None:
            continue
        target = ImageTarget(
            image_id=row["id"], path=row["path"], filename=row["filename"],
            category=row["category"], needs_review=bool(row["needs_review"]),
        )
        in_doubt = target.needs_review and doubt.action != "ignore"

        # What to do about doubt is the detector's question, not the rule's, so
        # it applies whatever the matched rule decided — including "ignore".
        if in_doubt and doubt_action.consumes_original:
            # The doubt pile takes the photo itself, so the matched rule cannot
            # also have it. Deliberate: a photo nobody could settle has not
            # earned a place in a category folder.
            planned.extend(doubt_action.plan(ctx, target, doubt, namer))
            continue
        if in_doubt:
            # Copied *before* the matched rule runs, which is what lets the rule
            # be a `move`: after it, there would be nothing left to copy.
            planned.extend(doubt_action.plan(ctx, target, doubt, namer))
        planned.extend(registry.get(rule.action).plan(ctx, target, rule, namer))
    return planned
