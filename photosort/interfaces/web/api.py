"""The web UI's operations, as plain Python returning plain data.

Deliberately socket-free: every endpoint below is a method that can be called
and asserted on directly, and each one is a thin translation of a service call
into JSON. Where the CLI renders a table, this renders a dict — the decision in
between is neither one's business.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ...config import load_settings
from ...domain.rules import DOUBT_ACTIONS, DOUBT_RULE_NAME, RuleSet
from ...errors import ConfigError, PhotosortError
from ...pipeline import Stopper
from ...services import AppContext, applying, configuring, diagnostics, insights, library
from ...services import processing
from ...services import rules as rules_service
from ...storage import RulesStore, Storage
from .jobs import JobRunner


class WebApi:
    """Every operation the web UI can trigger: read endpoints that project
    live state into JSON, and job endpoints that hand a long-running service
    call to `self.jobs` and return immediately. `server.py` is the only thing
    that calls these methods over HTTP; nothing stops a test from calling
    them directly."""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.jobs = JobRunner()

    # ------------------------------------------------------------------ rules

    def load_rules(self) -> RuleSet:
        """The active ruleset, unvalidated — read endpoints show whatever is
        on disk even if it wouldn't currently pass `rules validate`."""
        return rules_service.active_ruleset(self.ctx, validate=False)

    def get_rules(self) -> dict[str, Any]:
        """The active ruleset, as raw JSON plus a rendered summary per rule."""
        ruleset = self.load_rules()
        return {
            "rules": ruleset.to_json()["rules"],
            "summary": [
                {"name": r.name, "action": r.action, "folder": r.target_folder(),
                 "enabled": r.enabled, "describes": r.condition.describe()}
                for r in ruleset.rules
            ],
        }

    def put_rules(self, payload: Any) -> dict[str, Any]:
        """Validate and save a whole new ruleset from the editor."""
        return {"saved": True, "count": rules_service.save_ruleset(self.ctx, payload)}

    def move_rule(self, name: str, offset: int) -> dict[str, Any]:
        """Shift one rule's priority by `offset` places."""
        rules_service.move_rule(self.ctx, name, offset)
        return {"saved": True}

    # ------------------------------------------------------------------ info

    def looking_for(self) -> list[str]:
        """What the rules make the detector search for.

        Empty when the ruleset cannot be read, because this feeds an explanation
        of why nothing was found — and refusing to explain is the worst answer
        available at that point.
        """
        try:
            return sorted(self.load_rules().detector_classes())
        except PhotosortError:
            return []

    def needs_adjudication(self) -> bool:
        """Whether any rule's class band asks for a second opinion at all —
        the replacement for the old global `ADJUDICATE_ENABLED`, read from the
        ruleset instead of settings. False when the ruleset cannot be read,
        same reasoning as `looking_for`."""
        try:
            return self.load_rules().needs_adjudication()
        except PhotosortError:
            return False

    def get_meta(self) -> dict[str, Any]:
        """Everything the rules editor's page needs on load: offered classes,
        the action catalog, the doubt-rule slot, and a snapshot of settings
        relevant to what will happen when rules run."""
        settings = self.ctx.settings
        return {
            "classes": rules_service.offered_classes(self.ctx),
            "looking_for": self.looking_for(),
            "actions": self.ctx.actions.catalog(),
            # The doubt rule is a fixed slot in the rules list, so the editor has
            # to know which one it is and what it may be set to.
            "doubt_rule": DOUBT_RULE_NAME,
            "doubt_actions": list(DOUBT_ACTIONS),
            "config": {
                "input_folders": [str(folder) for folder in settings.library.input_folders],
                "output_folder": str(settings.output.folder),
                "trash_folder": str(settings.output.trash_folder),
                # Non-null means every file this run writes is going to fail.
                # Sent with the page so the UI can say so before a button is
                # pressed, rather than after a few thousand identical warnings.
                "output_problem": diagnostics.output_problem(self.ctx),
                # Same idea, for Ollama: said once here rather than once per
                # photo in a log the UI is not watching.
                "ollama_problem": diagnostics.ollama_problem(self.ctx),
                "model": settings.detect.model,
                "ollama_url": settings.analyze.url,
                "ollama_model": settings.analyze.model,
                # Per-rule now (each class condition sets its own band in the
                # editor), so the one thing left worth saying up front is
                # whether *anything* currently asks for a second opinion.
                "adjudication_needed": self.needs_adjudication(),
            },
        }

    # ------------------------------------------------------------------ settings

    def get_settings(self) -> dict[str, Any]:
        """The Settings tab's data: every editable field, its value, its source."""
        return configuring.describe(self.ctx)

    def put_settings(self, payload: Any) -> dict[str, Any]:
        """Persist an edit and rebuild the application around it.

        Refused mid-job: a stage holding a `Settings` that has just been replaced
        would be working from two different configurations at once.
        """
        values = (payload or {}).get("values", payload)
        if not isinstance(values, dict):
            raise ConfigError("expected an object of settings to save")
        self._require_idle()
        configuring.update(self.ctx, values)
        self._reload()
        return {"saved": True, **self.get_settings()}

    def reset_settings(self, payload: Any) -> dict[str, Any]:
        """Drop stored values, handing those settings back to `.env`."""
        keys = (payload or {}).get("keys") or []
        if not keys:
            raise ConfigError("no settings were named")
        self._require_idle()
        configuring.reset(self.ctx, keys)
        self._reload()
        return {"saved": True, **self.get_settings()}

    def get_vision_models(self, url: str | None = None) -> dict[str, Any]:
        """What an Ollama (the saved one, or `url` if the user is testing a
        different one) reports having installed."""
        return configuring.vision_models(self.ctx, url)

    def _require_idle(self) -> None:
        """Refuse a settings change while a job holds the current context."""
        if self.jobs.busy:
            raise ConfigError("something is running — wait for it to finish first")

    def _reload(self) -> None:
        """Re-read every layer and swap the context in place.

        The index is kept rather than reopened, since no editable setting can
        move it; a deployment that moves the database in `.env` while the server
        runs gets it on the next start, which is the same answer as before.
        """
        settings = load_settings()
        storage = self.ctx.storage
        if Path(settings.paths.database) != Path(storage.path):
            storage = Storage(settings.paths.database)
            storage.init()
            self.ctx.storage.close()
        self.ctx = replace(
            self.ctx,
            settings=settings,
            storage=storage,
            rules=RulesStore(settings.paths.rules),
        )

    def get_stats(self) -> dict[str, Any]:
        """The overview tab's data, plus whatever job is currently running."""
        return {**insights.overview(self.ctx), "job": self.jobs.snapshot()}

    def get_preview(self, limit: int = 200) -> dict[str, Any]:
        """What `apply` would do right now, without writing anything."""
        return applying.preview(self.ctx, self.load_rules(), limit)

    def get_recheck(self) -> dict[str, Any]:
        """What would the current rules do? A projection, nothing is written."""
        return processing.recheck(self.ctx, self.load_rules(), persist=False)

    def get_samples(self, category: str | None, limit: int = 60) -> dict[str, Any]:
        """A page of categorised images, for the "Where you are" gallery."""
        return {"images": insights.samples(self.ctx, category, limit)}

    # ------------------------------------------------------------------ jobs

    def start_scan(self) -> bool:
        """Kick off a library scan in the background."""
        def job() -> str:
            """Run the scan; the string returned becomes the job's final message."""
            stats = library.scan_library(self.ctx)
            if not (stats.added or stats.changed):
                # Re-scanning a caught-up library is the common case; silence here
                # reads as a broken button.
                return (f"scanned {stats.seen} files — nothing new or changed since "
                        f"the last scan, so the index is already up to date")
            return f"scanned {stats.seen} files: {stats.added} new, {stats.changed} changed"

        return self.jobs.start("scan", job)

    def start_pipeline(self, with_analyze: bool = True) -> bool:
        """Kick off detect (+ adjudicate/analyze, if enabled) in the background."""
        def job() -> str:
            """Run the pipeline; the string returned becomes the job's final message."""
            ruleset = rules_service.active_ruleset(self.ctx)
            results = processing.run_all(
                self.ctx, ruleset, Stopper(), with_analyze=with_analyze
            )
            done = "; ".join(
                f"{name}: {s.processed} examined, {s.errors} failed"
                for name, s in results.items()
            )
            if not any(s.processed or s.errors for s in results.values()):
                # A run with nothing left to claim is the ordinary case once the
                # library is caught up, and it has to read as such.
                return ("nothing left to examine — every indexed photo has already "
                        "been through the detector. Scan again if you added photos.")
            return done or "nothing to do"

        return self.jobs.start("pipeline", job)

    def start_apply(self, confirmed: bool) -> bool:
        """Kick off applying the rules' actions in the background."""
        def job() -> str:
            """Run apply; the string returned becomes the job's final message."""
            stats, _ = applying.apply(self.ctx, self.load_rules(), confirmed=confirmed)
            if not any((stats.created, stats.existing, stats.pruned, stats.moved,
                        stats.deleted, stats.skipped)):
                # Silence here reads as a broken button; it is almost always one
                # of two things, and both are visible on this tab.
                return ("nothing to do — no photo has an action that produces "
                        "anything yet. See “Where you are” above.")
            summary = applying.describe_stats(stats)
            if stats.skipped:
                summary += " (skipped: not confirmed — tick the box beside Sort now)"
            return summary

        return self.jobs.start("apply", job)

    def start_recheck(self) -> bool:
        """Re-run the decision engine over stored detections and persist it."""
        def job() -> str:
            """Run recheck; the string returned becomes the job's final message."""
            ruleset = rules_service.active_ruleset(self.ctx)
            outcome = processing.recheck(self.ctx, ruleset)
            return f"re-evaluated {sum(outcome['categories'].values())} images"

        return self.jobs.start("recheck", job)
