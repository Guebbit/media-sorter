"""Use cases around the settings a front end may change while running.

Reading is a projection of the live `Settings` plus the layer each value came
from; writing goes through `config.overrides`, which validates before it saves.
Neither knows how it will be rendered, and neither reloads anything — rebuilding
the application around new settings belongs to whoever owns the context, which
is the front end.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..analyzing import OllamaClient
from ..config import env as env_config
from ..config import overrides
from ..errors import ConfigError
from .context import AppContext


def _effective(ctx: AppContext) -> dict[str, Any]:
    """The live value of every editable key, in the shape a form wants."""
    settings = ctx.settings
    return {
        "INPUT_FOLDERS": [str(folder) for folder in settings.library.input_folders],
        "OUTPUT_FOLDER": str(settings.output.folder),
        "DETECT_CONFIDENCE": settings.detect.confidence,
        "REVIEW_CONFIDENCE": settings.detect.review_confidence,
        "OLLAMA_URL": settings.analyze.url,
        "OLLAMA_MODEL": settings.analyze.model,
        "ADJUDICATE_ENABLED": settings.analyze.adjudicate,
        "ANALYZE_ENABLED": settings.analyze.enabled,
    }


def _source(field: overrides.Editable, value: Any, stored: Mapping[str, str]) -> str:
    """Which layer decided this field — or "unset" when nothing did.

    `source_of` answers from the layers alone, which would call an empty list of
    photo folders a "default". It is not one: that setting has no default, and a
    form badge saying so is the difference between a field a user has to fill in
    and a field they can leave alone.
    """
    if value == [] or value == "":
        return "unset"
    return env_config.source_of(field.key, stored)


def describe(ctx: AppContext) -> dict[str, Any]:
    """Everything a settings form needs: the fields, the values, the provenance."""
    stored = overrides.load(ctx.settings.paths.settings)
    values = _effective(ctx)
    return {
        "fields": [
            {
                "key": field.key,
                "label": field.label,
                "kind": field.kind,
                "help": field.help,
                "value": values[field.key],
                "source": _source(field, values[field.key], stored),
            }
            for field in overrides.EDITABLE
        ],
        "file": str(ctx.settings.paths.settings),
        "env_file": str(env_config.dotenv.find_env_file() or ""),
    }


def update(ctx: AppContext, values: Mapping[str, Any]) -> dict[str, str]:
    """Validate and persist an edit. Returns the stored overlay.

    Raises `ConfigError` on anything unusable, before writing, so a rejected edit
    leaves the previous settings in force.
    """
    if not values:
        raise ConfigError("no settings were submitted")
    return overrides.save(ctx.settings.paths.settings, values)


def reset(ctx: AppContext, keys: Mapping[str, Any] | list[str]) -> dict[str, str]:
    """Forget the stored value for some keys, handing them back to `.env`."""
    return overrides.clear(ctx.settings.paths.settings, list(keys))


def vision_models(ctx: AppContext, url: str | None = None) -> dict[str, Any]:
    """Ask an Ollama what it has installed.

    `url` lets a UI test a server the user has typed but not yet saved, which is
    the difference between a field that validates and one that reassures.
    """
    settings = ctx.settings.analyze
    if url:
        settings = replace(settings, url=overrides.normalize("OLLAMA_URL", url))

    client = OllamaClient(settings)
    try:
        models = client.health()
    except Exception as exc:  # noqa: BLE001 - reported, never raised at the user
        return {"url": settings.url, "reachable": False, "error": str(exc), "models": []}
    return {"url": settings.url, "reachable": True, "error": "", "models": sorted(models)}
