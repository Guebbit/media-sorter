"""Settings: what the user chose, grouped by the concern it belongs to.

`sections` describes the shape and validates it; `env` assembles one from the
layers a real installation has — defaults, a `.env` (`dotenv`), the environment,
and the runtime overlay the web UI writes (`overrides`). Splitting shape from
source means a caller — a test, a front end — can build a `Settings` directly
without going through any of it, and nothing outside this package ever reads
`os.environ`.
"""

from __future__ import annotations

from . import dotenv, overrides
from .env import load_settings, settings_file, source_of
from .sections import (AnalyzeSettings, DetectSettings, DupesSettings, LibrarySettings,
                       OutputSettings, Paths, Settings, WebSettings, WorkerSettings)

__all__ = [
    "AnalyzeSettings", "DetectSettings", "DupesSettings", "LibrarySettings", "OutputSettings",
    "Paths", "Settings", "WebSettings", "WorkerSettings",
    "dotenv", "load_settings", "overrides", "settings_file", "source_of",
]
