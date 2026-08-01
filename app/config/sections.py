"""The shape of the configuration, one dataclass per concern.

Grouped rather than flat so a module can be handed exactly the settings it has
any business reading: the scanner takes `LibrarySettings`, the actions take
`OutputSettings`, and neither can accidentally start depending on the Ollama
timeout. Frozen, because a setting that changes under a running stage is a bug —
front ends override with `dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ConfigError


@dataclass(frozen=True, slots=True)
class Paths:
    """Where the tool keeps its own state. All of it is re-derivable."""

    database: Path
    rules: Path
    models: Path
    #: The runtime overlay written by the web UI — see `config.overrides`. Held
    #: here so a front end can offer to edit it without re-deriving the location.
    settings: Path


@dataclass(frozen=True, slots=True)
class LibrarySettings:
    """Which files count as part of the library."""

    input_folders: tuple[Path, ...]
    extensions: frozenset[str]
    exclude_dirs: frozenset[str]
    follow_symlinks: bool = False

    def validate(self) -> None:
        """Raise if the extension list is empty — everything else here is
        optional or has a safe fallback."""
        if not self.extensions:
            raise ConfigError("at least one file extension is required")

    def require_folders(self) -> None:
        """Raise unless there is something to walk.

        Deliberately not part of `validate`: no folders is an *unconfigured*
        installation, not a broken one, and settings that refuse to load would
        take `mediasort config set` and the web UI down with them — the two
        places the folders are chosen.
        """
        if not self.input_folders:
            raise ConfigError(
                "no photo folders configured. Name them for one run with "
                "`mediasort scan --input /path/to/photos`, save them with "
                "`mediasort config set --input /path/to/photos`, or use the "
                "Settings tab in `mediasort web`."
            )


@dataclass(frozen=True, slots=True)
class DetectSettings:
    """The object detector: weights, batching, device.

    The confidence thresholds used to live here as two global settings; they
    are now per rule condition (`domain.rules.conditions.HasClass`), resolved
    through `RuleSet.class_bands()` — see `domain.rules.conditions` for the
    hardcoded fallback a condition gets when it does not set its own.
    """

    model: str
    #: Directories searched for weights before falling back to a download.
    #: A deployment that ships weights inside its image adds them here.
    model_search_paths: tuple[Path, ...]
    batch: int
    imgsz: int
    device_name: str

    @property
    def device(self) -> str | None:
        """None lets the backend pick (GPU when available)."""
        return None if self.device_name in {"auto", ""} else self.device_name


@dataclass(frozen=True, slots=True)
class AnalyzeSettings:
    """The vision-language engine: where it is, and how to talk to it.

    Used for two independent jobs — the semantic description pass, which
    always runs, and the adjudication ("second opinion") pass, which runs
    whenever `RuleSet.needs_adjudication()` says some rule asked for it. Both
    used to have their own on/off setting here; both are gone in favour of
    that per-rule decision.
    """

    url: str
    model: str
    timeout: int
    num_ctx: int
    max_edge: int
    keep_alive: str


@dataclass(frozen=True, slots=True)
class OutputSettings:
    """What the tool is allowed to produce, and to destroy."""

    folder: Path
    trash_folder: Path


@dataclass(frozen=True, slots=True)
class WebSettings:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    scan: int
    detect: int
    analyze: int


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything, assembled. Built once by the composition root."""

    paths: Paths
    library: LibrarySettings
    detect: DetectSettings
    analyze: AnalyzeSettings
    output: OutputSettings
    web: WebSettings
    workers: WorkerSettings
    log_level: str = "INFO"

    def validate(self) -> None:
        """Validate every section that has anything to check."""
        self.library.validate()
