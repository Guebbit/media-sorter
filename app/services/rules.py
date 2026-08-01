"""Use cases around the ruleset."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.detection import VIDEO_CLASS
from ..domain.rules import RuleSet, reorder
from ..errors import RuleError
from .context import AppContext

log = logging.getLogger(__name__)


def active_ruleset(ctx: AppContext, validate: bool = True) -> RuleSet:
    """The rules in force. Nothing seeds this file for you — run
    `mediasort rules init --classes ...` once, and from then on it is the
    single source of truth."""
    if not ctx.rules.exists():
        raise RuleError(f"{ctx.rules.path} does not exist yet — "
                        "run `mediasort rules init --classes cat,dog,...` first")
    ruleset = ctx.rules.load()
    if validate:
        ruleset.validate(ctx.actions.names())
    return ruleset


def save_ruleset(ctx: AppContext, payload: Any) -> int:
    """Validate first, then write: a rejected edit leaves the file untouched."""
    ruleset = RuleSet.from_json(payload)
    ruleset.validate(ctx.actions.names())
    ctx.rules.save(ruleset)
    return len(ruleset.rules)


def move_rule(ctx: AppContext, name: str, offset: int) -> RuleSet:
    """Shift rule `name` by `offset` places in priority order, and save it."""
    ruleset = reorder(active_ruleset(ctx, validate=False), name, offset)
    ctx.rules.save(ruleset)
    return ruleset


def init_ruleset(ctx: AppContext, classes: str | None = None, force: bool = False) -> RuleSet:
    """Write a starter ruleset from a list of classes."""
    if ctx.rules.exists() and not force:
        raise RuleError(f"{ctx.rules.path} already exists")
    if not classes:
        raise RuleError("no classes given — pass --classes (e.g. cat,dog,bird)")
    names = [c.strip().lower() for c in classes.split(",") if c.strip()]
    ruleset = RuleSet.starter(names)
    ctx.rules.save(ruleset)
    return ruleset


def validate_ruleset(ctx: AppContext, check_classes: bool = False) -> RuleSet:
    """Check the file parses and refers to real actions — and optionally that
    the configured detector can actually find every class it mentions."""
    if not ctx.rules.exists():
        raise RuleError(f"{ctx.rules.path} does not exist yet — "
                        "run `mediasort rules init --classes cat,dog,...` first")
    ruleset = ctx.rules.load()
    known_classes = detector_classes(ctx) if check_classes else None
    ruleset.validate(ctx.actions.names(), known_classes)
    return ruleset


def detector_classes(ctx: AppContext) -> list[str]:
    """Every class a rule can match: what the configured weights can find,
    plus `video` — the one class no model looks for, since a rule matches a
    video by its extension instead (`domain.detection.is_video`). Needs the
    detector installed."""
    # Deferred: `detecting` imports ultralytics/torch, which most calls into
    # this service (validating a rule, moving one) never need to pay to load.
    from ..detecting import available_classes, ensure_available

    ensure_available()
    return sorted({*available_classes(ctx.settings.detect), VIDEO_CLASS})


def offered_classes(ctx: AppContext) -> list[str]:
    """The class list for a rules editor, degrading to whatever the saved
    rules already reference when the detector cannot be reached.

    A UI must still work on a machine without the model installed; being wrong
    about the menu is better than showing no editor at all.
    """
    try:
        return detector_classes(ctx)
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot list model classes: %s", exc)
        try:
            return sorted(ctx.rules.load().required_classes() | {VIDEO_CLASS})
        except Exception:  # noqa: BLE001
            return [VIDEO_CLASS]
