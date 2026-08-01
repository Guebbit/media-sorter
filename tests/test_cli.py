"""CLI smoke tests: every command runs and exits cleanly.

Detection commands are covered elsewhere; here we only assert that they fail
with a helpful message rather than a traceback when torch is absent.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from photosort.domain.decision import Decision
from photosort.domain.detection import Detection
from photosort.interfaces.cli import app
from photosort.services import library

runner = CliRunner()


def invoke(*args):
    result = runner.invoke(app, list(args))
    if result.exit_code not in (0, 1, 2):
        raise AssertionError(f"photosort {' '.join(args)} crashed:\n{result.output}\n{result.exception}")
    return result


@pytest.fixture
def indexed(ctx, storage):
    """An index that looks like detection already ran."""
    library.scan_library(ctx)
    for index, row in enumerate(storage.images.claim_detect(10)):
        hit = index < 2
        storage.results.finish_detect(
            row.id,
            [Detection("cat", 0.9, 0, 0, 1, 1)] if hit else [],
            Decision("cat", "link", False) if hit else Decision("none", "ignore", False),
            "m",
        )
    return storage


def test_help_lists_the_commands(env):
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("scan", "run", "apply", "rules", "web", "stats", "recheck", "detect"):
        assert command in result.output


def test_scan(env):
    result = invoke("scan")
    assert result.exit_code == 0
    assert "files seen" in result.output


def test_scan_takes_the_folders_from_a_flag(env, tmp_path, storage):
    """The one-run override: this folder, not the saved one, and nothing persisted."""
    from photosort.config import load_settings, overrides, settings_file
    from tests.conftest import make_image

    elsewhere = tmp_path / "elsewhere"
    make_image(elsewhere / "x.jpg")

    result = invoke("scan", "--input", str(elsewhere))

    assert result.exit_code == 0
    assert storage.images.count("") == 1  # just the one image, not the library's five
    assert str(elsewhere) in result.output
    assert overrides.load(settings_file())["INPUT_FOLDERS"] != str(elsewhere)
    assert load_settings().library.input_folders != (elsewhere,)


def test_scan_without_folders_explains_itself(env, monkeypatch):
    from photosort.config import overrides, settings_file

    overrides.clear(settings_file(), ["INPUT_FOLDERS"])
    monkeypatch.setenv("PHOTOSORT_INPUT_FOLDERS", "/stale/export")  # must not rescue it

    result = invoke("scan")

    assert result.exit_code != 0
    assert "no photo folders configured" in result.output


def test_config_show_reports_the_folders_and_their_layer(env, library_root, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")  # a tmp_path is long; do not let rich elide it
    result = invoke("config", "show")
    assert result.exit_code == 0
    assert str(library_root) in result.output
    assert "INPUT_FOLDERS" in result.output
    assert "saved" in result.output   # never "from .env" for a folder


def test_config_set_persists_for_later_commands(env, tmp_path):
    from photosort.config import load_settings
    from tests.conftest import make_image

    elsewhere = tmp_path / "elsewhere"
    make_image(elsewhere / "x.jpg")
    out = tmp_path / "sorted"

    assert invoke("config", "set", "--input", str(elsewhere), "--output", str(out)).exit_code == 0

    settings = load_settings()
    assert settings.library.input_folders == (elsewhere,)
    assert settings.output.folder == out


def test_config_set_refuses_a_folder_that_is_not_there(env, tmp_path):
    result = invoke("config", "set", "--input", str(tmp_path / "absent"))
    assert result.exit_code == 2
    assert "not a folder" in result.output


def test_config_set_needs_something_to_set(env):
    result = invoke("config", "set")
    assert result.exit_code == 2
    assert "nothing to set" in result.output


def test_config_unset_hands_a_folder_back(env):
    from photosort.config import load_settings

    assert invoke("config", "unset", "INPUT_FOLDERS").exit_code == 0
    assert load_settings().library.input_folders == ()


def test_stats(env, indexed):
    result = invoke("stats")
    assert result.exit_code == 0
    assert "pipeline" in result.output


def test_rules_show_seeds_the_file(env, settings):
    result = invoke("rules", "show")
    assert result.exit_code == 0
    assert "cat-dog" in result.output
    assert settings.paths.rules.exists()


def test_rules_init_with_custom_classes(env, settings):
    result = invoke("rules", "init", "--classes", "bird,horse", "--force")
    assert result.exit_code == 0
    names = [r["name"] for r in json.loads(settings.paths.rules.read_text())["rules"]]
    assert names == ["bird-horse", "bird", "horse", "none", "in-doubt"]


def test_rules_init_refuses_to_clobber(env, settings):
    invoke("rules", "init", "--force")
    result = invoke("rules", "init")
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_rules_validate_accepts_the_default(env):
    assert invoke("rules", "validate").exit_code == 0


def test_rules_validate_rejects_a_broken_file(env, settings):
    settings.paths.rules.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.rules.write_text(json.dumps({"rules": [{"name": "x", "when": {"nope": 1}}]}))
    result = invoke("rules", "validate")
    assert result.exit_code == 1
    assert "invalid" in result.output


def test_rules_actions_lists_the_registry(env):
    result = invoke("rules", "actions")
    assert result.exit_code == 0
    assert "delete" in result.output


def test_apply_dry_run_writes_nothing(env, settings, indexed):
    result = invoke("apply", "--dry-run")
    assert result.exit_code == 0
    assert "dry run" in result.output
    assert not settings.output.folder.exists() or not any(settings.output.folder.iterdir())


def test_apply_builds_the_tree(env, settings, indexed):
    assert invoke("apply").exit_code == 0
    assert (settings.output.folder / "Cat").is_dir()


def test_recheck_reapplies_rules_without_the_detector(env, settings, indexed):
    settings.paths.rules.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.rules.write_text(json.dumps({"rules": [
        {"name": "felines", "when": {"class": "cat"}, "action": "copy"},
        {"name": "rest", "when": {"always": True}, "action": "ignore"},
    ]}))
    result = invoke("recheck")
    assert result.exit_code == 0
    assert indexed.images.count("category = 'felines'") == 2


def test_verify(env, indexed):
    result = invoke("verify")
    assert result.exit_code == 0
    assert "copies_ok" in result.output


def test_duplicates(env, indexed):
    assert invoke("duplicates").exit_code == 0


def test_history_is_empty_without_deletions(env, indexed):
    result = invoke("history")
    assert "no originals have been moved or removed" in result.output


def test_export_json(env, tmp_path, indexed):
    out = tmp_path / "export.json"
    assert invoke("export", "--out", str(out)).exit_code == 0
    assert len(json.loads(out.read_text())) == 5


def test_export_csv(env, tmp_path, indexed):
    out = tmp_path / "export.csv"
    assert invoke("export", "--format", "csv", "--out", str(out)).exit_code == 0
    assert out.read_text().startswith("id,path,category")


def test_export_rejects_a_bad_format(env, tmp_path):
    assert invoke("export", "--format", "xml", "--out", str(tmp_path / "x")).exit_code == 2


def test_search_reports_no_matches(env, indexed):
    result = invoke("search", "maine coon")
    assert "no matches" in result.output


def test_retry(env, indexed):
    assert invoke("retry").exit_code == 0


def test_reset(env, indexed):
    assert invoke("reset", "detect", "--yes").exit_code == 0
    assert indexed.images.count("detect_state = 0") == 5


def test_clean_requires_a_target(env):
    assert invoke("clean").exit_code == 2


def test_clean_links(env, settings, indexed):
    invoke("apply")
    assert invoke("clean", "--links", "--yes").exit_code == 0


def test_doctor_runs_without_the_model(env):
    result = invoke("doctor")
    assert result.exit_code == 0
    assert "rules" in result.output


def test_detector_commands_fail_helpfully_without_torch(env):
    pytest.importorskip("typer")
    try:
        import ultralytics  # noqa: F401  # probe only
        pytest.skip("ultralytics is installed; the guard cannot trigger")
    except ImportError:
        pass
    result = invoke("detect")
    assert result.exit_code == 2
    assert "dependencies are not installed" in result.output
