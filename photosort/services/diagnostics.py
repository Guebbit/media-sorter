"""Use case: is this installation actually able to do its job?

Returns a list of findings rather than printing them, so the same checks can be
rendered as a table, served as JSON, or asserted on in a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .. import filesystem
from ..analyzing import OllamaClient
from ..config import dotenv, overrides
from ..errors import PhotosortError
from . import rules as rules_service
from .context import AppContext

Status = Literal["ok", "warn", "fail", "info"]


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    detail: str
    status: Status = "info"


def doctor(ctx: AppContext) -> list[Check]:
    """Every diagnostic check, in the order a user would want to read them:
    config layering, the index, the library, the rules, the detector, Ollama."""
    checks: list[Check] = _config_checks(ctx)
    checks.append(Check("index", str(ctx.settings.paths.database)))
    checks += _library_checks(ctx)
    checks += _rules_checks(ctx)
    checks += _detector_checks(ctx)
    checks += _analysis_checks(ctx)
    return checks


def _config_checks(ctx: AppContext) -> list[Check]:
    """Which layers are in play. The first question when a setting looks ignored."""
    env_file = dotenv.find_env_file()
    overlay = ctx.settings.paths.settings
    stored = overrides.load(overlay)
    return [
        Check(".env", str(env_file) if env_file else "none found — using defaults"),
        Check(
            "saved settings",
            f"{overlay} ({', '.join(sorted(stored))})" if stored else f"{overlay} (empty)",
        ),
    ]


def _library_checks(ctx: AppContext) -> list[Check]:
    """Whether the configured input folders exist and the output folder is
    writable — the two things a fresh install most commonly gets wrong."""
    checks = [
        Check("input", str(folder), "ok" if Path(folder).is_dir() else "fail")
        for folder in ctx.settings.library.input_folders
    ]
    if not checks:
        # The one setting with no default, so an empty list is the likeliest
        # reason a fresh installation does nothing. Say it here, not on the scan.
        checks.append(Check(
            "input", "not set — `photosort config set --input /path/to/photos`", "fail"
        ))
    output = ctx.settings.output
    checks.append(Check("output", str(output.folder)))
    problem = filesystem.unwritable(output.folder)
    checks.append(Check("output writable", problem, "fail") if problem
                  else Check("output writable", "yes", "ok"))
    return checks


def _rules_checks(ctx: AppContext) -> list[Check]:
    """Whether the rules file parses, and which consuming actions it uses."""
    # A ruleset can be unusable in more than one way — unparseable, unknown
    # action, or asking the detector for nothing — and a diagnostic has to
    # report each of them rather than raise.
    try:
        ruleset = rules_service.active_ruleset(ctx)
        checks = [
            Check("rules", f"{ctx.rules.path} ({len(ruleset.rules)} rules)", "ok"),
            Check("looking for", ", ".join(ruleset.detector_classes())),
        ]
    except PhotosortError as exc:
        return [Check("rules", str(exc), "fail")]

    # `move` and `delete` consume the original, so it is worth naming which
    # rules use them — they only run once a run explicitly confirms it
    # (`apply --yes`, or the confirm checkbox in the web UI).
    for action in sorted(ctx.actions.consuming_names()):
        rules = [r.name for r in ruleset.rules if r.action == action]
        if rules:
            checks.append(Check(
                f"{action} rules",
                f"{', '.join(rules)} — needs confirmation to run",
            ))
    return checks


def _detector_checks(ctx: AppContext) -> list[Check]:
    """Whether the detection dependencies are installed, which device YOLO
    will run on, and whether the configured weights are actually present."""
    # Deferred: `detecting` imports ultralytics/torch, which every other caller
    # of this module (front ends just want the `Check` type, or another check
    # function) would otherwise pay to load.
    from ..detecting import describe_device, is_available, resolve_model

    if not is_available():
        return [Check("detector", "dependencies are not installed", "fail")]
    try:
        detect = ctx.settings.detect
        path = resolve_model(detect.model, detect.model_search_paths)
        found = Path(path).exists()
        return [
            Check("torch device", describe_device(detect)),
            Check("detector weights", path, "ok" if found else "fail"),
        ]
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never itself crash
        return [Check("detector", str(exc), "fail")]


def _analysis_checks(ctx: AppContext) -> list[Check]:
    """Ollama, and the two independent jobs it is asked to do.

    Reported separately because they fail differently: without the second
    opinion, uncertain photos are sorted more crudely; without the semantic
    pass, they are sorted identically but cannot be searched by description.
    Either can be off while the other runs, so neither may speak for Ollama.
    """
    settings = ctx.settings.analyze
    jobs = [
        Check("second opinion", "on" if settings.adjudicate else
              "disabled — uncertain photos go straight to review",
              "ok" if settings.adjudicate else "info"),
        Check("semantic pass", "on" if settings.enabled else "disabled",
              "ok" if settings.enabled else "info"),
    ]
    if not (settings.enabled or settings.adjudicate):
        return [*jobs, Check("ollama", "not needed — nothing asks for it")]
    try:
        client = OllamaClient(settings)
        models = client.health()
        checks = [*jobs, Check("ollama", f"up at {settings.url} ({len(models)} models)", "ok")]
        client.ensure_model()
        checks.append(Check("vision model", client.model, "ok"))
        return checks
    except Exception as exc:  # noqa: BLE001
        return [*jobs, Check("ollama", str(exc), "fail")]


def output_problem(ctx: AppContext) -> str | None:
    """Why the output folder cannot be written to, for a front end that wants to
    say so before anything is started."""
    return filesystem.unwritable(ctx.settings.output.folder)


def ollama_problem(ctx: AppContext) -> str | None:
    """Why Ollama cannot be reached, for a front end that wants to say so before
    a run starts rather than after every uncertain photo lands in `_Review`
    unresolved and every semantic tag goes missing.

    None when nothing needs Ollama: `health()` costs a request, so it is only
    worth making when the second opinion or the semantic pass is switched on.
    """
    settings = ctx.settings.analyze
    if not (settings.enabled or settings.adjudicate):
        return None
    try:
        OllamaClient(settings).health()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never itself crash
        return str(exc)
    return None
