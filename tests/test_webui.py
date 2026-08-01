"""The web UI's operations, exercised without a socket.

Everything below calls `WebApi` directly, which is the point of splitting it
from the HTTP layer: the interface is testable as plain Python.
"""

from __future__ import annotations

import pytest

from photosort.domain.decision import Decision
from photosort.domain.detection import Detection
from photosort.errors import ConfigError
from photosort.interfaces.web import JobRunner, WebApi
from photosort.services import library
from photosort.storage import RulesStore


CAT = Detection("cat", 0.9, 0, 0, 1, 1)


@pytest.fixture
def app(ctx):
    return WebApi(ctx)


# --------------------------------------------------------------------- rules


def test_get_rules_returns_the_active_ruleset(app, settings):
    payload = app.get_rules()
    assert [r["name"] for r in payload["rules"]] == ["cat-dog", "cat", "dog", "none", "in-doubt"]
    assert settings.paths.rules.exists()


def test_summary_describes_each_rule_for_the_ui(app):
    summary = {row["name"]: row for row in app.get_rules()["summary"]}
    assert summary["cat-dog"]["describes"] == "cat and dog"
    assert summary["cat-dog"]["folder"] == "Cat-Dog"
    assert summary["none"]["action"] == "ignore"


def test_saving_rules_persists_them(app, settings):
    app.put_rules({"rules": [
        {"name": "only-cats", "when": {"class": "cat"}, "action": "copy", "folder": "Felines"},
        {"name": "rest", "when": {"always": True}, "action": "ignore"},
    ]})
    reloaded = RulesStore(settings.paths.rules).load()
    assert reloaded.names() == ["only-cats", "rest", "in-doubt"]
    assert reloaded.rules[0].target_folder() == "Felines"


def test_saving_an_unknown_action_is_rejected(app, settings):
    from photosort.errors import RuleError

    path = settings.paths.rules
    before = path.read_text() if path.exists() else None
    with pytest.raises(RuleError, match="unknown action"):
        app.put_rules({"rules": [{"name": "x", "when": {"class": "cat"}, "action": "teleport"}]})
    if before is not None:
        assert path.read_text() == before  # nothing was written


def test_moving_a_rule_changes_priority(app, settings):
    app.get_rules()
    app.move_rule("dog", -1)
    assert RulesStore(settings.paths.rules).load().names() == ["cat-dog", "dog", "cat", "none", "in-doubt"]


# ---------------------------------------------------------------------- meta


def test_meta_exposes_actions_and_config(app, monkeypatch):
    # available_classes needs the model; fall back to the configured seed.
    meta = app.get_meta()
    assert {"copy", "delete", "ignore"} <= {a["name"] for a in meta["actions"]}
    assert any(a["consumes_original"] for a in meta["actions"])
    # Each says what it does to the original; "destructive" is wrong for move.
    assert {a["name"]: a["consequence"] for a in meta["actions"]}["move"]
    assert "cat" in meta["classes"]
    # Offered even though no model looks for it — a rule matches it by extension.
    assert "video" in meta["classes"]
    # The starter ruleset leaves every class band unset, which resolves to
    # `ollama_review=True` by default — so the fixture ruleset does ask for a
    # second opinion, unlike the old global `ADJUDICATE_ENABLED=0` test default.
    assert meta["config"]["adjudication_needed"] is True


def test_meta_reports_an_unreachable_ollama_when_a_job_needs_it(app, monkeypatch):
    monkeypatch.setenv("PHOTOSORT_OLLAMA_URL", "http://127.0.0.1:1")
    app._reload()
    assert app.get_meta()["config"]["ollama_problem"]


def test_meta_is_quiet_about_ollama_when_it_is_reachable(app, monkeypatch):
    from .test_analyze import FakeOllama

    server = FakeOllama()
    try:
        monkeypatch.setenv("PHOTOSORT_OLLAMA_URL", server.url)
        app._reload()
        assert app.get_meta()["config"]["ollama_problem"] is None
    finally:
        server.stop()


def test_stats_shape(app, ctx):
    library.scan_library(ctx)
    stats = app.get_stats()
    assert stats["pipeline"]["total"] == 5
    assert "job" in stats


# ------------------------------------------------------------------ previews


def test_preview_plans_without_writing(app, ctx, storage):
    library.scan_library(ctx)
    for row in storage.images.claim_detect(10):
        storage.results.finish_detect(row.id, [], Decision("cat", "copy", False), "m")

    preview = app.get_preview()
    assert preview["total"] == 5
    assert preview["counts"]["copy"] == 5
    assert not (ctx.settings.output.folder / "Cats").exists()


def test_recheck_reports_what_new_rules_would_do(app, ctx, storage):
    library.scan_library(ctx)
    for row in storage.images.claim_detect(10):
        storage.results.finish_detect(row.id, [CAT], Decision("cat", "copy", False), "m")

    # Swap in rules that treat cats as junk; detections are reused, not re-run.
    app.put_rules({"rules": [
        {"name": "junk", "when": {"class": "cat"}, "action": "delete"},
        {"name": "rest", "when": {"always": True}, "action": "ignore"},
    ]})
    outcome = app.get_recheck()
    assert outcome["categories"]["junk"] == 5
    assert outcome["actions"]["delete"] == 5
    # Still only a projection: nothing in the database changed.
    assert storage.images.count("category = 'cat'") == 5


def test_samples_can_be_filtered(app, ctx, storage):
    library.scan_library(ctx)
    for index, row in enumerate(storage.images.claim_detect(10)):
        storage.results.finish_detect(
            row.id, [], Decision("cat" if index < 2 else "dog", "copy", False), "m"
        )

    assert len(app.get_samples("cat")["images"]) == 2
    assert len(app.get_samples(None)["images"]) == 5


def test_meta_reports_what_the_detector_looks_for(app):
    """The UI explains an empty result with it, so it has to be there."""
    assert app.get_meta()["looking_for"] == ["cat", "dog"]


def test_the_confirmation_covers_moves_and_not_only_deletions(app, tmp_path):
    """The bug this pins: the UI's one checkbox was labelled "confirm deletions"
    and wired to deletions only, so a library configured to move sorted nothing
    and reported every photo as skipped."""
    app.put_rules({"rules": [
        {"name": "cat", "when": {"class": "cat"}, "action": "move"},
        {"name": "none", "when": {"always": True}, "action": "ignore"},
    ]})
    library.scan_library(app.ctx)
    for row in app.ctx.storage.images.claim_detect(10):
        app.ctx.storage.results.finish_detect(row.id, [], Decision("cat", "move", False), "m")

    assert app.start_apply(confirmed=True) is True
    app.jobs.join()

    snapshot = app.jobs.snapshot()
    assert snapshot["error"] == ""
    assert "moved" in snapshot["message"]
    assert "skipped" not in snapshot["message"]


def test_apply_explains_itself_when_there_is_nothing_to_do(app):
    """The complaint this answers is "Apply actions does nothing"."""
    assert app.start_apply(confirmed=False) is True
    app.jobs.join()

    snapshot = app.jobs.snapshot()
    assert snapshot["error"] == ""
    assert "nothing to do" in snapshot["message"]


def test_scan_says_so_when_nothing_changed(app, ctx):
    """Re-scanning a caught-up library is the common case, and it looks like a
    dead button unless the result says what it found."""
    library.scan_library(ctx)
    assert app.start_scan() is True
    app.jobs.join()

    snapshot = app.jobs.snapshot()
    assert snapshot["error"] == ""
    assert "nothing new or changed" in snapshot["message"]


def test_a_finished_job_still_says_which_step_it_was(app, ctx):
    """The result is rendered next to the button that started it, so the name has
    to outlive the job — `state` is back to "idle" by then."""
    library.scan_library(ctx)
    app.start_scan()
    app.jobs.join()

    snapshot = app.jobs.snapshot()
    assert snapshot["busy"] is False
    assert snapshot["name"] == "scan"
    assert snapshot["elapsed"] >= 0


def test_stats_say_whether_the_output_tree_exists_yet(app, ctx, storage):
    """"ready to apply" and "already applied" are different sentences."""
    from photosort.services import applying

    library.scan_library(ctx)
    for row in storage.images.claim_detect(10):
        storage.results.finish_detect(row.id, [CAT], Decision("cat", "copy", False), "m")

    assert app.get_stats()["pipeline"]["links_total"] == 0
    applying.apply(ctx, app.load_rules())
    assert app.get_stats()["pipeline"]["links_total"] == 5


# --------------------------------------------------------------- settings


def test_settings_report_their_value_and_layer(app, library_root):
    fields = {f["key"]: f for f in app.get_settings()["fields"]}
    assert fields["INPUT_FOLDERS"]["value"] == [str(library_root)]
    # Saved, never "from .env" — the environment is not a layer for a folder.
    assert fields["INPUT_FOLDERS"]["source"] == "settings"
    assert fields["OLLAMA_MODEL"]["kind"] == "text"


def test_saving_a_setting_takes_effect_without_a_restart(app, tmp_path):
    other = tmp_path / "more-photos"
    other.mkdir()

    payload = app.put_settings({"values": {"OLLAMA_MODEL": "llava:13b",
                                           "INPUT_FOLDERS": [str(other)]}})

    assert payload["saved"] is True
    # The live context, not just the file: this is what the next job will use.
    assert app.ctx.settings.analyze.model == "llava:13b"
    assert app.ctx.settings.library.input_folders == (other,)
    fields = {f["key"]: f for f in payload["fields"]}
    assert fields["OLLAMA_MODEL"]["source"] == "settings"


def test_a_rejected_setting_leaves_the_previous_one_in_force(app, tmp_path):
    from photosort.config import overrides

    overlay = app.ctx.settings.paths.settings
    before = app.ctx.settings.library.input_folders
    with pytest.raises(ConfigError, match="not a folder"):
        app.put_settings({"values": {"INPUT_FOLDERS": [str(tmp_path / "absent")]}})
    assert app.ctx.settings.library.input_folders == before
    assert overrides.load(overlay)["INPUT_FOLDERS"] == str(before[0])  # nothing written


def test_the_folders_can_be_set_from_nothing(app, tmp_path):
    """The bootstrap the UI exists for: no folders saved, no `.env` to fall back on."""
    payload = app.reset_settings({"keys": ["INPUT_FOLDERS"]})

    assert app.ctx.settings.library.input_folders == ()
    fields = {f["key"]: f for f in payload["fields"]}
    assert fields["INPUT_FOLDERS"]["source"] == "unset"

    picked = tmp_path / "picked"
    picked.mkdir()
    app.put_settings({"values": {"INPUT_FOLDERS": [str(picked)]}})

    assert app.ctx.settings.library.input_folders == (picked,)


def test_a_scan_with_no_folders_reports_the_reason(app):
    """A job that cannot start has to say why in the UI, not just die in a thread."""
    app.reset_settings({"keys": ["INPUT_FOLDERS"]})

    assert app.start_scan() is True
    app.jobs.join()

    assert "no photo folders configured" in app.jobs.snapshot()["error"]


def test_settings_cannot_change_under_a_running_job(app):
    gate = threading_event()
    app.jobs.start("slow", lambda: (gate.wait(2), "done")[1])
    try:
        with pytest.raises(ConfigError, match="wait for it to finish"):
            app.put_settings({"values": {"OLLAMA_MODEL": "llava:13b"}})
    finally:
        gate.set()
        app.jobs.join()


def test_a_setting_can_be_handed_back_to_the_env_file(app, monkeypatch):
    monkeypatch.setenv("PHOTOSORT_OLLAMA_MODEL", "from-env-file")
    app.put_settings({"values": {"OLLAMA_MODEL": "llava:13b"}})
    assert app.ctx.settings.analyze.model == "llava:13b"   # the overlay wins

    payload = app.reset_settings({"keys": ["OLLAMA_MODEL"]})

    fields = {f["key"]: f for f in payload["fields"]}
    assert fields["OLLAMA_MODEL"]["source"] == "environment"
    assert app.ctx.settings.analyze.model == "from-env-file"


def test_the_index_is_not_reopened_when_it_has_not_moved(app):
    """No editable setting can move the database, so the connection survives."""
    before = app.ctx.storage
    app.put_settings({"values": {"OLLAMA_MODEL": "llava:13b"}})
    assert app.ctx.storage is before


def test_unknown_keys_are_refused(app):
    with pytest.raises(ConfigError, match="not a runtime-editable setting"):
        app.put_settings({"values": {"TELEPORT_ENABLED": "1"}})


# ------------------------------------------------------- ollama model probe


def test_an_unreachable_ollama_is_reported_not_raised(app):
    result = app.get_vision_models("http://127.0.0.1:1")
    assert result["reachable"] is False
    assert result["error"]
    assert result["models"] == []


def test_a_url_the_user_has_not_saved_yet_can_be_tested(app):
    from .test_analyze import FakeOllama

    server = FakeOllama()
    try:
        result = app.get_vision_models(server.url)
    finally:
        server.stop()

    assert result["reachable"] is True
    assert result["models"] == ["llava-llama3:8b", "llava:13b"]
    # Testing a URL must not save it.
    assert app.ctx.settings.analyze.url != server.url


def test_a_bad_url_is_refused_before_any_request(app):
    with pytest.raises(ConfigError, match="must start with http"):
        app.get_vision_models("localhost:11434")


# ------------------------------------------------------------------- jobs


def test_job_runner_reports_success():
    runner = JobRunner()
    done = threading_event()
    assert runner.start("demo", lambda: (done.set(), "finished")[1])
    done.wait(2)
    runner.join()
    assert runner.snapshot()["message"] == "finished"
    assert runner.snapshot()["error"] == ""


def test_job_runner_captures_failures():
    runner = JobRunner()

    def boom():
        raise RuntimeError("kaboom")

    runner.start("demo", boom)
    runner.join()
    assert runner.snapshot()["error"] == "kaboom"
    assert runner.snapshot()["busy"] is False


def test_only_one_job_runs_at_a_time():
    runner = JobRunner()
    gate = threading_event()
    runner.start("slow", lambda: (gate.wait(2), "done")[1])
    assert runner.start("second", lambda: "nope") is False
    gate.set()
    runner.join()


def threading_event():
    import threading

    return threading.Event()


def test_meta_names_the_doubt_rule_and_its_allowed_actions(app):
    """The editor renders that one rule differently, so it has to be told which
    it is rather than hardcoding the name on both sides."""
    meta = app.get_meta()
    assert meta["doubt_rule"] == "in-doubt"
    assert set(meta["doubt_actions"]) == {"move", "copy", "ignore"}
    assert "delete" not in meta["doubt_actions"]


def test_saving_rules_cannot_drop_the_doubt_rule(app, settings):
    """Whatever the editor posts, the answer to "and the unsure ones?" survives."""
    app.put_rules({"rules": [
        {"name": "only-cats", "when": {"class": "cat"}, "action": "copy"},
    ]})
    reloaded = RulesStore(settings.paths.rules).load()
    assert reloaded.names() == ["only-cats", "in-doubt"]
    assert reloaded.for_doubt().action == "move"


def test_the_doubt_rule_action_can_be_changed_and_persists(app, settings):
    app.put_rules({"rules": [
        {"name": "only-cats", "when": {"class": "cat"}, "action": "copy"},
        {"name": "in-doubt", "when": {"in_doubt": True}, "action": "copy"},
    ]})
    assert RulesStore(settings.paths.rules).load().for_doubt().action == "copy"


def test_saving_a_doubt_rule_that_deletes_is_rejected(app):
    from photosort.errors import RuleError

    with pytest.raises(RuleError, match="must be one of"):
        app.put_rules({"rules": [
            {"name": "in-doubt", "when": {"in_doubt": True}, "action": "delete"},
        ]})
