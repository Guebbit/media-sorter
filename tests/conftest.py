"""Shared fixtures.

Everything here builds a throwaway library + index under tmp_path, so the tests
never touch a real photo folder and can run in any order. `ctx` is the same
composition root the front ends use, which is what lets a test drive a use case
exactly the way the CLI does.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings, load_settings, overrides
from app.domain.detection import Detection
from app.domain.rules import RuleSet
from app.services import AppContext
from app.storage import RulesStore, Storage


def make_image(path: Path, size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def detection(cls: str, confidence: float) -> Detection:
    return Detection(cls, confidence, 0.0, 0.0, 10.0, 10.0)


@pytest.fixture
def library_root(tmp_path: Path) -> Path:
    """A small photo library: 5 images, one nested, plus files that must be ignored.

    Named for the folder rather than "library" so it cannot be confused with the
    `services.library` module in tests that use both.
    """
    root = tmp_path / "photos"
    for index, name in enumerate(["a.jpg", "b.jpg", "sub/c.png", "sub/d.jpeg", "e.webp"]):
        make_image(root / name, color=(index * 20, 50, 90))
    (root / "notes.txt").write_text("not an image")
    make_image(root / ".hidden" / "skip.jpg")
    return root


@pytest.fixture
def env(tmp_path: Path, library_root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Configuration pointing at the temporary library.

    The folders go through the overlay, the same way `app config set` and the
    Settings tab write them.
    """
    values = {
        # Both layers that would otherwise reach outside tmp_path: the project's
        # own .env (tests run from the repo root) and the UI's settings overlay.
        # An explicit path that does not exist means "no such layer".
        "MEDIASORT_ENV_FILE": str(tmp_path / "absent.env"),
        "MEDIASORT_SETTINGS": str(tmp_path / "data" / "settings.json"),
        "MEDIASORT_DB": str(tmp_path / "data" / "app.db"),
        "MEDIASORT_MODELS_DIR": str(tmp_path / "data" / "models"),
        "MEDIASORT_RULES": str(tmp_path / "data" / "rules.json"),
        "MEDIASORT_LOG_LEVEL": "CRITICAL",
        # The duplicate finder has its own folders; pointing it at the same
        # library is the case most likely to expose the two indexes bleeding
        # into each other, so it is the default here.
        "MEDIASORT_DUPES_FOLDERS": str(library_root),
        "MEDIASORT_WORKERS_SCAN": "2",
        "MEDIASORT_WORKERS_ANALYZE": "2",
    }
    # Clear anything inherited from the developer's shell so tests are hermetic.
    for key in list(os.environ):
        if key.startswith("MEDIASORT_"):
            monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    overrides.save(Path(values["MEDIASORT_SETTINGS"]), {
        "INPUT_FOLDERS": [str(library_root)],
        "OUTPUT_FOLDER": str(tmp_path / "output"),
    })
    # Nothing seeds the rules file at runtime any more (that's `rules init`'s
    # job) — write the default cat/dog set directly so every other test can
    # assume one exists, exactly as the CLI would leave it.
    RulesStore(Path(values["MEDIASORT_RULES"])).save(RuleSet.starter(["cat", "dog"]))
    return values


@pytest.fixture
def settings(env: dict[str, str]) -> Settings:
    return load_settings()


@pytest.fixture
def ctx(settings: Settings) -> AppContext:
    """The wired application, exactly as a front end would build it."""
    return AppContext.create(settings)


@pytest.fixture
def storage(ctx: AppContext) -> Storage:
    return ctx.storage


@pytest.fixture
def ruleset() -> RuleSet:
    """The default cat/dog ruleset — the behaviour from the original spec."""
    return RuleSet.starter(["cat", "dog"])
