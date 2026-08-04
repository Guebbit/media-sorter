"""The settings a front end is allowed to change at runtime.

Environment variables are the right way to configure a machine and the wrong way
to configure a session: picking a folder or pointing at a different Ollama should
not mean editing a file and restarting. So a small, explicitly enumerated set of
keys can also live in a JSON overlay, written by the web UI's Settings tab and by
`mediasort config set` — the same file, the same validation, either way.

Three decisions worth stating:

* The overlay **wins** over the environment for the keys it contains. A setting
  changed in the UI that quietly did nothing because `.env` disagreed would be a
  worse surprise than the reverse, and `sources()` reports which layer each
  effective value came from so the UI can show it.
* Values are stored as the same strings the environment would hold, so there is
  exactly one parser (`config.env`) and one validator (`config.sections`). The
  overlay adds a layer, not a second dialect.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import ConfigError
from .sections import Settings

log = logging.getLogger(__name__)

#: Strings a user or a checkbox might send for "true", shared with `config.env`
#: so the environment and the runtime overlay agree on what counts as truthy.
TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def _atomic_write(file: Path, data: Mapping[str, str]) -> None:
    """Write `data` as indented JSON, visible atomically: a reader never sees a
    half-written file, since `replace` is a single filesystem rename."""
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(file)


@dataclass(frozen=True, slots=True)
class Editable:
    """One setting the UI may change, and enough about it to render a field."""

    key: str
    label: str
    #: folders | text | url | bool — what the UI should draw, not a type system.
    kind: str
    help: str = ""
    #: How to describe the value the settings derive when nothing is stored, for
    #: a form to show as a placeholder. Its presence also means "leave the field
    #: empty until someone fills it in": a derived path sitting in the box looks
    #: like a choice that was made, and clearing the box is how it is undone.
    derived: str = ""


EDITABLE: tuple[Editable, ...] = (
    Editable(
        "INPUT_FOLDERS", "Input folders", "folders",
        "The folders to index. Anything already indexed from a folder you remove "
        "stays in the database.",
    ),
    Editable(
        "OUTPUT_FOLDER", "Output folder", "text",
        "Where the sorted tree and the trash folder are written. Left unset, it is "
        "a 'Sorted' subfolder of your first photo folder, so results land next to "
        "the photos they came from. Only a 'move' rule puts originals in here; "
        "'copy' leaves them where they are.",
        derived="a 'Sorted' folder inside your first input folder",
    ),
    Editable(
        "DUPES_FOLDERS", "Folders to check for duplicates", "folders",
        "Where the duplicate finder looks. Nothing to do with the sorting "
        "folders above: checking a folder for duplicates does not add it to the "
        "library, and does not run the detector over it.",
    ),
    Editable(
        "DUPES_SAVE_INDEX", "Remember the duplicate scan", "bool",
        "Off by default, so a duplicate check leaves nothing behind. Turn it on "
        "for a folder big enough that re-hashing it every time is a chore — the "
        "hashes are then kept in a hidden '.mediasort' folder inside it, "
        "separately from the sorting index.",
    ),
    Editable(
        "OLLAMA_URL", "Ollama URL", "url",
        "The Ollama you already run. Nothing is started for you.",
    ),
    Editable(
        "OLLAMA_MODEL", "Vision model", "text",
        "Any vision-capable model installed in that Ollama.",
    ),
    Editable(
        "SAVE_INDEX", "Remember this library", "bool",
        "Off by default: the tool works out what is in your photos, sorts them, and "
        "leaves nothing behind. Turn it on for a folder large enough that you would "
        "not want to redo the detection — what it has learned is then saved in a "
        "hidden '.mediasort' folder inside that library, so a second run picks up "
        "where the first stopped. One per library; other folders are unaffected.",
    ),
)

BY_KEY = {field.key: field for field in EDITABLE}


def _require_known(key: str) -> Editable:
    """`key`'s `Editable` spec, or `ConfigError` if it isn't a runtime-editable setting."""
    field = BY_KEY.get(key)
    if field is None:
        raise ConfigError(
            f"{key!r} is not a runtime-editable setting; "
            f"expected one of {', '.join(sorted(BY_KEY))}"
        )
    return field


def normalize(key: str, value: Any) -> str:
    """A submitted value as the string the environment would have held."""
    field = _require_known(key)

    if field.kind == "folders":
        # Both shapes reach here and both are the user's: a JSON array from the
        # settings form, and a repeated or comma-joined `--input` from the CLI.
        given = value if isinstance(value, (list, tuple)) else [value]
        items = [part for entry in given for part in str(entry).split(",")]
        folders = [item.strip() for item in items if item.strip()]
        if not folders:
            raise ConfigError("at least one photo folder is required")
        for folder in folders:
            path = Path(folder).expanduser()
            if not path.is_dir():
                raise ConfigError(f"not a folder: {folder}")
        return ",".join(str(Path(f).expanduser()) for f in folders)

    if field.kind == "bool":
        # A checkbox sends a real boolean; `config set` and `.env` send a string.
        if isinstance(value, str):
            return "1" if value.strip().lower() in TRUTHY_STRINGS else "0"
        return "1" if value else "0"

    text = str(value).strip()
    if not text:
        raise ConfigError(f"{field.label} cannot be empty")

    if field.kind == "url":
        if not text.startswith(("http://", "https://")):
            raise ConfigError(f"{field.label} must start with http:// or https://")
        return text.rstrip("/")

    return text


def load(path: Path | str) -> dict[str, str]:
    """The overlay, or an empty one. Unknown keys are dropped, not fatal.

    Tolerant on read and strict on write: a file left behind by a newer version
    must not stop the tool from starting, whereas a bad key coming from the UI is
    a mistake worth reporting.
    """
    file = Path(path)
    if not file.is_file():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable settings overlay %s: %s", file, exc)
        return {}
    if not isinstance(data, Mapping):
        log.warning("ignoring settings overlay %s: expected a JSON object", file)
        return {}

    values: dict[str, str] = {}
    for key, value in data.items():
        if key in BY_KEY and value is not None:
            values[str(key)] = str(value)
        else:
            log.debug("ignoring unknown key %r in %s", key, file)
    return values


def save(path: Path | str, values: Mapping[str, Any]) -> dict[str, str]:
    """Validate, merge into what is already stored, and write atomically.

    Merged rather than replaced so a front end can submit one field without
    having to send back everything it did not touch.
    """
    file = Path(path)
    merged = load(file)
    for key, value in values.items():
        merged[key] = normalize(key, value)

    _atomic_write(file, merged)
    return merged


def with_folders(settings: Settings, input_folders: Any = None,
                 output_folder: Any = None) -> Settings:
    """`settings` with the folders replaced, for this run only.

    The counterpart to `save` for a CLI flag: same keys, same validation, nothing
    written. Both are None for the ordinary case, which returns `settings`
    unchanged rather than a copy.
    """
    if not input_folders and output_folder is None:
        return settings

    library, output = settings.library, settings.output
    if input_folders:
        folders = tuple(
            Path(item) for item in normalize("INPUT_FOLDERS", input_folders).split(",")
        )
        library = replace(library, input_folders=folders)
    if output_folder is not None:
        folder = Path(normalize("OUTPUT_FOLDER", output_folder)).expanduser()
        # A trash folder that was only ever derived from the output folder follows
        # it; one somebody set on purpose stays where they put it.
        derived = output.trash_folder == output.folder / "_Trash"
        output = replace(
            output,
            folder=folder,
            trash_folder=folder / "_Trash" if derived else output.trash_folder,
        )

    settings = replace(settings, library=library, output=output)
    settings.validate()
    return settings


def clear(path: Path | str, keys: Iterable[str]) -> dict[str, str]:
    """Drop keys from the overlay, handing those settings back to the environment."""
    file = Path(path)
    remaining = load(file)
    for key in keys:
        _require_known(key)
        remaining.pop(key, None)

    _atomic_write(file, remaining)
    return remaining
