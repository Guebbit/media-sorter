"""The shape of the configuration, one dataclass per concern.

Grouped rather than flat so a module can be handed exactly the settings it has
any business reading: the scanner takes `LibrarySettings`, the actions take
`OutputSettings`, and neither can accidentally start depending on the Ollama
timeout. Frozen, because a setting that changes under a running stage is a bug —
front ends override with `dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigError


@dataclass(frozen=True, slots=True)
class Paths:
    """Where the tool keeps its own state. All of it is re-derivable."""

    #: `None` means the index is kept in memory for this run and nothing is
    #: written — the default, since a tool pointed at a folder once has nothing
    #: to gain from leaving a file behind. See `config.env.load_settings`.
    database: Path | None
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

    Deliberately no confidence thresholds: they belong to the rule condition
    that uses them (`domain.rules.conditions.HasClass`), resolved through
    `RuleSet.class_bands()`, so "how sure about a cat" is answered per class
    rather than once for the whole run. See `domain.rules.conditions` for the
    hardcoded band a condition gets when it sets none of its own.
    """

    model: str
    #: Directories searched for weights before falling back to a download.
    #: A deployment that ships weights inside its image adds them here.
    model_search_paths: tuple[Path, ...]
    batch: int
    imgsz: int
    device_name: str
    #: How many frames to sample from each video and run through the model,
    #: spread across its length. Zero turns it off, and a video is then sorted
    #: by its extension alone — the `video` pseudo-class and nothing else.
    #: A video costs this many images of detection time, so it trades accuracy
    #: (a subject that appears in only part of the clip) against a run that
    #: takes eight times longer per video than per photo.
    video_frames: int

    @property
    def device(self) -> str | None:
        """None lets the backend pick (GPU when available)."""
        return None if self.device_name in {"auto", ""} else self.device_name

    @property
    def samples_video(self) -> bool:
        """Whether anything will look inside a video this run."""
        return self.video_frames > 0


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
class DupesSettings:
    """The duplicate finder, which is its own tool sharing this process.

    It has its own folders on purpose: "which photos am I sorting" and "which
    folder do I want to de-duplicate" are unrelated questions, and answering
    the second should never index a folder into the sorting pipeline or drag
    the sorting library into a duplicate review.
    """

    folders: tuple[Path, ...]
    extensions: frozenset[str]
    exclude_dirs: frozenset[str]
    #: `None` keeps this index in memory and leaves nothing behind — the same
    #: default, and the same reasoning, as `Paths.database`.
    database: Path | None = None

    @property
    def trash_folder(self) -> Path:
        """Where a discarded duplicate goes. Inside the folder being checked,
        not the sorting output tree: this tool is self-contained, and a discard
        here has nothing to do with a `delete` rule over there.

        Discarding *moves* — the file stays readable and putting it back is a
        drag in a file manager. There is no code path from this page that
        unlinks anything.
        """
        return self.folders[0] / "_Duplicates"

    def as_library(self) -> "LibrarySettings":
        """This folder set as something `scanning.scan` can walk. The scanner
        takes a `LibrarySettings`, and a duplicate scan is an ordinary scan —
        only the index it fills and the folders it walks are different."""
        return LibrarySettings(
            input_folders=self.folders,
            extensions=self.extensions,
            exclude_dirs=self.exclude_dirs,
        )

    def require_folders(self) -> None:
        """Raise unless a folder to check has been chosen."""
        if not self.folders:
            raise ConfigError(
                "no folder to check for duplicates. Pick one on the Duplicates "
                "page of `mediasort web`, or set MEDIASORT_DUPES_FOLDERS."
            )


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
    dupes: DupesSettings
    web: WebSettings
    workers: WorkerSettings
    log_level: str = "INFO"

    def validate(self) -> None:
        """Validate every section that has anything to check."""
        self.library.validate()
