"""Filling in the settings, from the layers a real installation has.

The only module in the codebase that touches `os.environ`. Three layers, lowest
first:

1. the defaults written below — a complete, working local run on their own;
2. the environment, including whatever a project-local `.env` supplied;
3. the runtime overlay the web UI and `mediasort config set` write, for the
   handful of keys in `config.overrides`.

Every key reads all three layers the same way; `INPUT_FOLDERS` is simply the one
with no default at all.

Nothing downstream can tell the difference: by the time a `Settings` exists it is
a frozen value with no memory of where any of it came from. `source_of` reports
that separately, for a front end that wants to say so.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ..video import VIDEO_EXTENSIONS
from . import dotenv, overrides
from .sections import (AnalyzeSettings, DetectSettings, DupesSettings, LibrarySettings,
                       OutputSettings, Paths, Settings, WebSettings, WorkerSettings)

PREFIX = "MEDIASORT_"

#: Everything Pillow can open here, HEIC/HEIF included — that one needs
#: `pillow-heif`, which is a hard dependency, so it is always available — plus
#: every video extension. A video is sampled for frames rather than decoded
#: whole (`DETECT_VIDEO_FRAMES`), and always also carries the "video"
#: pseudo-class (`video.is_video`).
DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.heic,.heif," + ",".join(
    sorted(VIDEO_EXTENSIONS)
)
#: Hidden directories are pruned by the walker regardless, so this list is only
#: for the visible ones. `output` and `Sorted` are ours: the output tree is
#: made of real files — copies, or the originals themselves after a `move` —
#: and a library that happens to contain it would otherwise index its own
#: results.
DEFAULT_EXCLUDE_DIRS = ".git,node_modules,@eaDir,output,Output,Sorted"
DEFAULT_SETTINGS_FILE = "data/settings.json"

#: Where a saved index lives inside the library it describes. Hidden, and in the
#: walker's blind spot already — the scanner skips dotted directories.
INDEX_DIRNAME = ".mediasort"

#: Point at a specific `.env`; a missing file then loads nothing, which is how
#: tests and hand-configured deployments opt out of the upward search entirely.
ENV_FILE_VAR = PREFIX + "ENV_FILE"
SETTINGS_FILE_VAR = PREFIX + "SETTINGS"


class _Layers:
    """Reads one key across the layers, and parses it into the type asked for."""

    def __init__(self, overlay: Mapping[str, str] | None = None):
        self.overlay = dict(overlay or {})

    def raw(self, key: str) -> str | None:
        """The winning value for `key`, or None when no layer sets it —
        the overlay first, then the environment."""
        candidates = [self.overlay.get(key), os.environ.get(PREFIX + key)]
        for candidate in candidates:
            if candidate is not None and candidate.strip() != "":
                return candidate.strip()
        return None

    def str_(self, key: str, default: str) -> str:
        """`key`'s winning value, or `default` if no layer sets it."""
        value = self.raw(key)
        return default if value is None else value

    def int_(self, key: str, default: int) -> int:
        """`key` parsed as an int, or `default` if unset or unparseable."""
        try:
            return int(self.str_(key, str(default)))
        except ValueError:
            return default

    def bool_(self, key: str, default: bool) -> bool:
        """`key` as a bool, treating any of `overrides.TRUTHY_STRINGS` as true."""
        return self.str_(key, "1" if default else "0").lower() in overrides.TRUTHY_STRINGS

    def list_(self, key: str, default: str) -> list[str]:
        """`key` split on commas into a list of trimmed, non-empty strings."""
        return [item.strip() for item in self.str_(key, default).split(",") if item.strip()]

    def paths_(self, key: str, default: str) -> tuple[Path, ...]:
        """`key` as a comma-separated list of `Path`s, `~` expanded."""
        return tuple(Path(item).expanduser() for item in self.list_(key, default))

    def extensions_(self, key: str, default: str) -> frozenset[str]:
        """`key` as a set of lowercase file extensions, each guaranteed a
        leading dot regardless of how the user wrote it."""
        return frozenset(
            item if item.startswith(".") else "." + item
            for item in (raw.lower() for raw in self.list_(key, default))
        )


def settings_file() -> Path:
    """Where the runtime overlay lives.

    Resolved from the environment alone, because it has to be readable *before*
    there are any settings to read it from.
    """
    configured = os.environ.get(SETTINGS_FILE_VAR, "").strip()
    return Path(configured or DEFAULT_SETTINGS_FILE).expanduser()


def source_of(key: str, overlay: Mapping[str, str] | None = None) -> str:
    """Which layer decides `key`: "settings", "environment" or "default"."""
    values = overrides.load(settings_file()) if overlay is None else overlay
    if values.get(key):
        return "settings"
    if os.environ.get(PREFIX + key, "").strip():
        return "environment"
    return "default"


def _index_path(layers: _Layers, input_folders: tuple[Path, ...]) -> Path | None:
    """Where this run's index lives, or `None` to keep it in memory.

    Three answers, in order:

    1. `DB` names a file — an explicit location always wins and always saves.
    2. `SAVE_INDEX` is on — the index goes *beside the library*, in
       `<first photo folder>/{INDEX_DIRNAME}/index.db`, not in the app folder.
       One index per library rather than one shared by every folder ever
       pointed at, so libraries cannot contaminate each other and the saved
       work travels with the drive it describes.
    3. Neither — nothing is written.

    Off by default because the index only pays for itself when the same folder
    is worked on twice. It is worth turning on for a library big enough that
    losing a half-finished detection run to a Ctrl-C would hurt.
    """
    explicit = layers.raw("DB")
    if explicit:
        return Path(explicit).expanduser()
    if not layers.bool_("SAVE_INDEX", False):
        return None
    if not input_folders:
        # Nothing to sit beside yet; `require_folders` will ask for one.
        return None
    return input_folders[0] / INDEX_DIRNAME / "index.db"


def _build_paths(layers: _Layers, overlay_path: Path, models_dir: Path,
                 input_folders: tuple[Path, ...]) -> Paths:
    """Where the tool keeps its own state, from the `DB`/`RULES` layers plus
    the already-resolved overlay/models locations."""
    return Paths(
        database=_index_path(layers, input_folders),
        rules=Path(layers.str_("RULES", "data/rules.json")).expanduser(),
        models=models_dir,
        settings=overlay_path,
    )


def _dupes_index_path(layers: _Layers, folders: tuple[Path, ...]) -> Path | None:
    """Where the duplicate finder's index lives, or `None` for memory only.

    Same two answers, and the same default, as `_index_path`: nothing is
    written unless asked for, and when it is asked for it goes beside the
    folder it describes rather than in the app directory. A separate file from
    the sorting index because it describes a separate folder set.
    """
    if not layers.bool_("DUPES_SAVE_INDEX", False):
        return None
    if not folders:
        return None
    return folders[0] / INDEX_DIRNAME / "dupes.db"


def _build_dupes(layers: _Layers) -> DupesSettings:
    """The duplicate finder's own folders and index, sharing only the file-type
    rules with the sorting library — what counts as an image does not change
    because the question did."""
    folders = layers.paths_("DUPES_FOLDERS", "")
    return DupesSettings(
        folders=folders,
        extensions=layers.extensions_("EXTENSIONS", DEFAULT_EXTENSIONS),
        exclude_dirs=frozenset(layers.list_("EXCLUDE_DIRS", DEFAULT_EXCLUDE_DIRS)),
        database=_dupes_index_path(layers, folders),
    )


def _build_library(layers: _Layers, input_folders: tuple[Path, ...]) -> LibrarySettings:
    """Which files count as the library, from the `EXTENSIONS`/`EXCLUDE_DIRS`
    layers plus the already-resolved `input_folders`."""
    return LibrarySettings(
        input_folders=input_folders,
        extensions=layers.extensions_("EXTENSIONS", DEFAULT_EXTENSIONS),
        exclude_dirs=frozenset(layers.list_("EXCLUDE_DIRS", DEFAULT_EXCLUDE_DIRS)),
        follow_symlinks=layers.bool_("FOLLOW_SYMLINKS", False),
    )


def _build_detect(layers: _Layers, models_dir: Path) -> DetectSettings:
    """The YOLO detector's settings: weights, thresholds, batching, device."""
    return DetectSettings(
        model=layers.str_("DETECT_MODEL", "yolo11m.pt"),
        model_search_paths=(models_dir, *layers.paths_("MODEL_SEARCH_PATHS", "")),
        batch=max(1, layers.int_("DETECT_BATCH", 16)),
        imgsz=layers.int_("DETECT_IMGSZ", 640),
        device_name=layers.str_("DETECT_DEVICE", "auto"),
        # Eight stills is enough to catch a subject that is only in part of a
        # clip without making one video cost as much as a folder of photos.
        # `0` opts out and goes back to sorting videos by extension alone.
        video_frames=max(0, layers.int_("DETECT_VIDEO_FRAMES", 8)),
    )


def _build_analyze(layers: _Layers) -> AnalyzeSettings:
    """The Ollama vision engine's settings: where it is, which model, timeouts."""
    return AnalyzeSettings(
        url=layers.str_("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
        model=layers.str_("OLLAMA_MODEL", "llava-llama3"),
        timeout=layers.int_("OLLAMA_TIMEOUT", 180),
        num_ctx=layers.int_("OLLAMA_NUM_CTX", 4096),
        max_edge=layers.int_("OLLAMA_MAX_EDGE", 1024),
        keep_alive=layers.str_("OLLAMA_KEEP_ALIVE", "10m"),
    )


def _build_output(layers: _Layers, output_folder: Path) -> OutputSettings:
    """What the tool may write and destroy, from the already-resolved `output_folder`."""
    return OutputSettings(
        folder=output_folder,
        trash_folder=Path(
            layers.str_("TRASH_FOLDER", str(output_folder / "_Trash"))
        ).expanduser(),
    )


def _build_web(layers: _Layers) -> WebSettings:
    """Where `mediasort web` binds."""
    return WebSettings(
        host=layers.str_("WEB_HOST", "127.0.0.1"),
        port=layers.int_("WEB_PORT", 8765),
    )


def _build_workers(layers: _Layers) -> WorkerSettings:
    """Thread-pool sizes for scanning, detecting and analyzing."""
    return WorkerSettings(
        scan=max(1, layers.int_("WORKERS_SCAN", 4)),
        detect=max(1, layers.int_("WORKERS_DETECT", 4)),
        analyze=max(1, layers.int_("WORKERS_ANALYZE", 2)),
    )


def load_settings(env_file: Path | str | None = None) -> Settings:
    """Read every layer, validate the combination, and freeze it.

    `env_file` says where the `.env` is; `MEDIASORT_ENV_FILE` does the same from
    the environment. Neither is needed for the normal case, where the file is
    found next to the project.
    """
    requested = env_file if env_file is not None else os.environ.get(ENV_FILE_VAR) or None
    dotenv.load(requested)

    overlay_path = settings_file()
    layers = _Layers(overrides.load(overlay_path))

    # No default: an unconfigured library is empty, not "./photos". The error
    # belongs to whoever needs the folders — see `LibrarySettings.require_folders`
    # — so `config set` can still run.
    input_folders = layers.paths_("INPUT_FOLDERS", "")

    # The sorted tree belongs with the photos it sorts, not in whichever folder
    # the command happened to be run from. In a subfolder of its own, though, and
    # one whose name is in DEFAULT_EXCLUDE_DIRS: writing it directly into the
    # library would mean the next scan indexing our own output, and `_Trash`
    # handing deleted originals back as new photos.
    default_output = str(input_folders[0] / "Sorted") if input_folders else "output"
    output_folder = Path(layers.str_("OUTPUT_FOLDER", default_output)).expanduser()
    models_dir = Path(layers.str_("MODELS_DIR", "data/models")).expanduser()

    settings = Settings(
        paths=_build_paths(layers, overlay_path, models_dir, input_folders),
        library=_build_library(layers, input_folders),
        detect=_build_detect(layers, models_dir),
        analyze=_build_analyze(layers),
        output=_build_output(layers, output_folder),
        dupes=_build_dupes(layers),
        web=_build_web(layers),
        workers=_build_workers(layers),
        log_level=layers.str_("LOG_LEVEL", "INFO").upper(),
    )
    settings.validate()
    return settings
