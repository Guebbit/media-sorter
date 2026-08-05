"""Where the index lives, and whether it exists at all.

The tool is pointed at one folder after another, so the default is to leave
nothing behind: what the detector worked out is kept in memory for the run and
dropped. `SAVE_INDEX` turns that around for a library big enough to be worth
resuming — and then the index belongs *to that library*, not to the app, so two
folders can never contaminate each other.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.config import load_settings, overrides
from app.config.env import INDEX_DIRNAME
from app.services import library
from app.storage import Storage


@pytest.fixture
def unpinned(env, monkeypatch):
    """`conftest` pins MEDIASORT_DB so most tests get a real file. These tests
    are about what happens when nobody pins it, which is the normal case."""
    monkeypatch.delenv("MEDIASORT_DB", raising=False)
    return env


# ------------------------------------------------------------------ location


def test_by_default_no_index_file_is_written(unpinned, library_root):
    settings = load_settings()
    assert settings.paths.database is None

    storage = Storage(settings.paths.database)
    storage.init()
    assert not storage.persistent
    assert not (library_root / INDEX_DIRNAME).exists()


def test_saving_puts_the_index_inside_the_library_it_describes(unpinned, library_root,
                                                               monkeypatch):
    monkeypatch.setenv("MEDIASORT_SAVE_INDEX", "1")
    database = load_settings().paths.database

    assert database == library_root / INDEX_DIRNAME / "index.db"


def test_two_libraries_get_two_indexes(unpinned, tmp_path, monkeypatch):
    """The whole point of putting it in the library: pointing the tool at a
    second folder must not disturb the first one's work."""
    monkeypatch.setenv("MEDIASORT_SAVE_INDEX", "1")
    other = tmp_path / "other-library"
    other.mkdir()

    first = load_settings().paths.database
    # INPUT_FOLDERS is deliberately not an environment variable, so point at the
    # second library the way `config set` and the Settings tab do.
    overrides.save(Path(unpinned["MEDIASORT_SETTINGS"]), {"INPUT_FOLDERS": [str(other)]})
    second = load_settings().paths.database

    assert first != second
    assert second == other / INDEX_DIRNAME / "index.db"


def test_an_explicit_db_path_still_wins(unpinned, tmp_path, monkeypatch):
    """Naming a file is itself a request to save one, whatever the toggle says."""
    monkeypatch.setenv("MEDIASORT_DB", str(tmp_path / "chosen.db"))
    assert load_settings().paths.database == tmp_path / "chosen.db"


# -------------------------------------------------------------- in memory


def test_an_in_memory_index_is_shared_by_every_thread(unpinned):
    """The stages run in their own threads and the engine opens a connection
    per thread. A private `:memory:` database per connection would leave each
    stage looking at an empty index.
    """
    storage = Storage(None)
    storage.init()
    storage.images.upsert([{
        "path": "/x.jpg", "filename": "x.jpg", "root": "/", "hash": "h", "phash": None,
        "size": 1, "mtime": 0.0, "width": None, "height": None, "format": None,
        "taken_at": None,
    }])

    seen: list[int] = []
    thread = threading.Thread(target=lambda: seen.append(storage.images.count("")))
    thread.start()
    thread.join()

    assert seen == [1]


def test_a_full_scan_works_with_nothing_on_disk(unpinned, ctx, library_root):
    """The default path end to end: index the library, keep it all in memory."""
    assert ctx.storage.path is None
    stats = library.scan_library(ctx)

    assert stats.added == 5
    assert ctx.storage.images.count("") == 5
    assert not (library_root / INDEX_DIRNAME).exists()
