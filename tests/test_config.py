"""The configuration layers: a `.env` file, the environment, the UI's overlay.

Layering is the whole point of these three modules, so most of what follows is
about precedence rather than parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from photosort.config import dotenv, load_settings, overrides, settings_file, source_of
from photosort.errors import ConfigError


# ----------------------------------------------------------------- .env parsing


def test_parses_the_ordinary_shapes():
    values = dotenv.parse(
        "\n".join([
            "# a comment",
            "",
            "PHOTOSORT_A=plain",
            "  PHOTOSORT_B = spaced ",
            'PHOTOSORT_C="quoted value"',
            "PHOTOSORT_D='single'",
            "export PHOTOSORT_E=exported",
            "PHOTOSORT_F=trailing   # inline comment",
            "not a setting",
        ])
    )
    assert values == {
        "PHOTOSORT_A": "plain",
        "PHOTOSORT_B": "spaced",
        "PHOTOSORT_C": "quoted value",
        "PHOTOSORT_D": "single",
        "PHOTOSORT_E": "exported",
        "PHOTOSORT_F": "trailing",
    }


def test_a_hash_inside_a_value_survives():
    """A URL fragment or a colour is not a comment."""
    assert dotenv.parse("PHOTOSORT_X=http://host/#frag")["PHOTOSORT_X"] == "http://host/#frag"
    assert dotenv.parse('PHOTOSORT_Y="a # b"')["PHOTOSORT_Y"] == "a # b"


def test_env_file_fills_gaps_but_never_overrides(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("PHOTOSORT_OLLAMA_MODEL=from-file\nPHOTOSORT_DETECT_MODEL=from-file\n")
    monkeypatch.setenv("PHOTOSORT_OLLAMA_MODEL", "from-environment")

    applied = dotenv.load(path)

    assert applied == {"PHOTOSORT_DETECT_MODEL": "from-file"}
    import os
    assert os.environ["PHOTOSORT_OLLAMA_MODEL"] == "from-environment"


def test_an_explicit_missing_env_file_loads_nothing(tmp_path):
    assert dotenv.load(tmp_path / "absent.env") == {}


def test_the_search_walks_up_to_the_project(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PHOTOSORT_DETECT_MODEL=found\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert dotenv.find_env_file() == tmp_path / ".env"


# -------------------------------------------------------------------- defaults


def test_heif_is_indexed_without_being_configured(env, monkeypatch):
    """`.env` used to add these, so dropping the line must not drop the support."""
    monkeypatch.delenv("PHOTOSORT_EXTENSIONS", raising=False)
    extensions = load_settings().library.extensions
    assert {".heic", ".heif"} <= extensions
    assert ".jpg" in extensions


def test_our_own_output_tree_is_skipped_without_being_configured(env, monkeypatch):
    """Only matters for copy/hardlink modes, where the output is real files —
    but that is exactly when a library containing it would index its results."""
    monkeypatch.delenv("PHOTOSORT_EXCLUDE_DIRS", raising=False)
    excluded = load_settings().library.exclude_dirs
    assert {"output", "Output", "Sorted"} <= excluded
    assert "node_modules" in excluded


# -------------------------------------------------------------------- overlay


def test_only_enumerated_keys_are_editable(tmp_path):
    with pytest.raises(ConfigError, match="not a runtime-editable setting"):
        overrides.save(tmp_path / "settings.json", {"PHOTOSORT_TELEPORT_ENABLED": "1"})
    with pytest.raises(ConfigError, match="not a runtime-editable setting"):
        overrides.save(tmp_path / "settings.json", {"DB": "/somewhere/else.db"})


def test_folders_are_checked_before_they_are_stored(tmp_path):
    file = tmp_path / "settings.json"
    with pytest.raises(ConfigError, match="not a folder"):
        overrides.save(file, {"INPUT_FOLDERS": [str(tmp_path / "nope")]})
    assert not file.exists()  # a rejected edit writes nothing


def test_a_url_must_look_like_one(tmp_path):
    with pytest.raises(ConfigError, match="must start with http"):
        overrides.save(tmp_path / "settings.json", {"OLLAMA_URL": "localhost:11434"})


def test_values_are_stored_as_the_environment_would_hold_them(tmp_path, library_root):
    stored = overrides.save(tmp_path / "settings.json", {
        "INPUT_FOLDERS": [str(library_root), str(tmp_path)],
        "OLLAMA_URL": "http://box:11434/",
    })
    assert stored["INPUT_FOLDERS"] == f"{library_root},{tmp_path}"
    assert stored["OLLAMA_URL"] == "http://box:11434"


def test_saving_one_key_keeps_the_others(tmp_path, library_root):
    file = tmp_path / "settings.json"
    overrides.save(file, {"INPUT_FOLDERS": [str(library_root)]})
    stored = overrides.save(file, {"OLLAMA_MODEL": "llava:13b"})
    assert stored["INPUT_FOLDERS"] == str(library_root)
    assert stored["OLLAMA_MODEL"] == "llava:13b"


def test_clearing_hands_a_key_back(tmp_path, library_root):
    file = tmp_path / "settings.json"
    overrides.save(file, {"OLLAMA_MODEL": "llava:13b", "INPUT_FOLDERS": [str(library_root)]})
    remaining = overrides.clear(file, ["OLLAMA_MODEL"])
    assert "OLLAMA_MODEL" not in remaining
    assert "INPUT_FOLDERS" in remaining


def test_an_unreadable_overlay_does_not_stop_the_tool(tmp_path):
    """Tolerant on read: a file from a newer version must not be fatal."""
    file = tmp_path / "settings.json"
    file.write_text("{ not json")
    assert overrides.load(file) == {}
    file.write_text(json.dumps({"OLLAMA_MODEL": "llava:13b", "FUTURE_KEY": "x"}))
    assert overrides.load(file) == {"OLLAMA_MODEL": "llava:13b"}


# ------------------------------------------------------------------ precedence


def test_the_overlay_wins_over_the_environment(env, tmp_path):
    """The setting a user changed in the UI has to be the one that takes effect."""
    assert load_settings().analyze.model == "llava-llama3"  # the default
    overrides.save(settings_file(), {"OLLAMA_MODEL": "from-overlay"})
    assert load_settings().analyze.model == "from-overlay"


def test_the_environment_wins_when_the_overlay_is_silent(env, monkeypatch):
    monkeypatch.setenv("PHOTOSORT_OLLAMA_MODEL", "from-environment")
    overrides.save(settings_file(), {"OLLAMA_MODEL": "from-overlay"})
    overrides.clear(settings_file(), ["OLLAMA_MODEL"])
    assert load_settings().analyze.model == "from-environment"


def test_folders_come_back_as_paths(env, library_root, tmp_path):
    overrides.save(settings_file(), {"INPUT_FOLDERS": [str(library_root), str(tmp_path)]})
    assert load_settings().library.input_folders == (library_root, tmp_path)


def test_source_is_reported_per_key(env, monkeypatch):
    monkeypatch.setenv("PHOTOSORT_OLLAMA_URL", "http://box:11434")
    overrides.save(settings_file(), {"OLLAMA_MODEL": "llava:13b"})
    assert source_of("OLLAMA_MODEL") == "settings"
    assert source_of("OLLAMA_URL") == "environment"
    assert source_of("OUTPUT_FOLDER") == "settings"  # the fixture saves it


def test_a_folder_is_never_reported_as_coming_from_the_environment(env, monkeypatch):
    """Even with the variable exported, `.env` is not a layer for these two."""
    monkeypatch.setenv("PHOTOSORT_INPUT_FOLDERS", "/stale")
    overrides.clear(settings_file(), ["INPUT_FOLDERS"])
    assert source_of("INPUT_FOLDERS") == "default"


# ------------------------------------------------------- the folders, specifically


def test_the_environment_cannot_set_the_folders(env, monkeypatch, tmp_path, library_root):
    """The variables used to work, so a leftover export must not silently win."""
    monkeypatch.setenv("PHOTOSORT_INPUT_FOLDERS", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("PHOTOSORT_OUTPUT_FOLDER", str(tmp_path / "elsewhere-out"))

    settings = load_settings()

    assert settings.library.input_folders == (library_root,)
    assert settings.output.folder == tmp_path / "output"


def test_an_unconfigured_library_still_loads(env, monkeypatch):
    """`config set` and the web UI have to run before anything is configured."""
    monkeypatch.delenv("PHOTOSORT_INPUT_FOLDERS", raising=False)
    overrides.clear(settings_file(), ["INPUT_FOLDERS"])

    settings = load_settings()          # must not raise

    assert settings.library.input_folders == ()
    with pytest.raises(ConfigError, match="no photo folders configured"):
        settings.library.require_folders()


def test_the_output_folder_defaults_to_a_subfolder_of_the_library(env, library_root):
    """Unlike the input folders, this one has an answer nobody has to give: the
    results belong next to the photos, not wherever the command was run from."""
    overrides.clear(settings_file(), ["OUTPUT_FOLDER"])

    output = load_settings().output

    assert output.folder == library_root / "Sorted"
    assert output.trash_folder == library_root / "Sorted" / "_Trash"


def test_the_default_output_is_a_folder_the_scanner_already_ignores(env):
    """Otherwise the default would make every re-scan index its own results."""
    overrides.clear(settings_file(), ["OUTPUT_FOLDER"])
    settings = load_settings()
    assert settings.output.folder.name in settings.library.exclude_dirs


def test_without_input_folders_there_is_nothing_to_derive_it_from(env):
    overrides.clear(settings_file(), ["INPUT_FOLDERS", "OUTPUT_FOLDER"])
    assert load_settings().output.folder == Path("output")


# ---------------------------------------------------------- one-run overrides


def test_a_flag_overrides_the_saved_folders(settings, tmp_path, library_root):
    other = tmp_path / "other-photos"
    other.mkdir()

    overridden = overrides.with_folders(
        settings, input_folders=[str(other)], output_folder=str(tmp_path / "elsewhere")
    )

    assert overridden.library.input_folders == (other,)
    assert overridden.output.folder == tmp_path / "elsewhere"
    # ...and the saved settings are untouched: this is a flag, not an edit.
    assert load_settings().library.input_folders == (library_root,)


def test_no_flags_is_the_same_settings_object(settings):
    assert overrides.with_folders(settings) is settings
    assert overrides.with_folders(settings, input_folders=[]) is settings


def test_a_flag_is_validated_like_a_saved_value(settings, tmp_path):
    with pytest.raises(ConfigError, match="not a folder"):
        overrides.with_folders(settings, input_folders=[str(tmp_path / "nope")])


def test_repeated_and_comma_joined_flags_mean_the_same_thing(settings, tmp_path, library_root):
    other = tmp_path / "other-photos"
    other.mkdir()
    expected = (library_root, other)

    repeated = overrides.with_folders(settings, input_folders=[str(library_root), str(other)])
    joined = overrides.with_folders(settings, input_folders=[f"{library_root},{other}"])

    assert repeated.library.input_folders == expected
    assert joined.library.input_folders == expected


def test_a_derived_trash_folder_follows_the_output_folder(settings, tmp_path):
    assert settings.output.trash_folder == settings.output.folder / "_Trash"
    moved = overrides.with_folders(settings, output_folder=str(tmp_path / "elsewhere"))
    assert moved.output.trash_folder == tmp_path / "elsewhere" / "_Trash"


def test_an_explicit_trash_folder_stays_put(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSORT_TRASH_FOLDER", str(tmp_path / "bin"))
    fixed = load_settings()

    moved = overrides.with_folders(fixed, output_folder=str(tmp_path / "elsewhere"))

    assert moved.output.trash_folder == tmp_path / "bin"


def test_a_broken_overlay_cannot_make_settings_unloadable(env):
    settings_file().parent.mkdir(parents=True, exist_ok=True)
    settings_file().write_text("nonsense")
    assert load_settings().analyze.model == "llava-llama3"
