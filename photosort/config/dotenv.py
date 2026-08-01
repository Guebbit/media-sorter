"""Reading a `.env` file into the environment.

The application is configured through environment variables, which is fine for a
shell and awkward for a person: a project-local `.env` is how you keep your
choices somewhere you can edit them. Nothing else in the codebase knows this
file exists — by the time settings are read it is indistinguishable from a
variable exported by hand.

A real environment variable always wins. That ordering is what makes a one-off
override possible (`PHOTOSORT_OLLAMA_MODEL=llava:13b photosort analyze`) without
editing the file first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

FILENAME = ".env"
#: How far up from the working directory to look. Deep enough to run a command
#: from a subfolder of the project, shallow enough not to wander into $HOME.
SEARCH_DEPTH = 4


def find_env_file(start: Path | None = None) -> Path | None:
    """The nearest `.env` at or above `start`, or None."""
    current = (start or Path.cwd()).resolve()
    for parent in (current, *current.parents)[:SEARCH_DEPTH]:
        candidate = parent / FILENAME
        if candidate.is_file():
            return candidate
    return None


def parse(text: str) -> dict[str, str]:
    """`KEY=value` lines to a dict, ignoring blanks and comments.

    Deliberately not a shell: no interpolation, no multi-line values. Quotes are
    stripped when they wrap the whole value, which is the only reason to write
    them here, and a trailing `# comment` is dropped only outside quotes.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        values[key] = _clean(value.strip())
    return values


def _clean(value: str) -> str:
    """One `.env` value stripped of surrounding quotes, or truncated at an
    inline `#` comment when unquoted."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # Unquoted: an inline comment needs whitespace in front of the #, so a value
    # that legitimately contains one (a colour, a URL fragment) survives.
    for index, char in enumerate(value):
        if char == "#" and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value


def load(path: Path | None = None) -> dict[str, str]:
    """Fill in `os.environ` from a `.env` file, without overriding it.

    `path` given explicitly means exactly that file and no search — a missing
    one loads nothing, which is how a test (or a deployment that configures
    everything by hand) opts out. Returns the keys actually applied.
    """
    env_file = Path(path) if path is not None else find_env_file()
    if env_file is None or not env_file.is_file():
        return {}

    try:
        text = env_file.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("cannot read %s: %s", env_file, exc)
        return {}

    applied = {}
    for key, value in parse(text).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    log.debug("loaded %d values from %s", len(applied), env_file)
    return applied
