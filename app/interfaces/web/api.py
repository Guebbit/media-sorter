"""The web UI's operations, as plain Python returning plain data.

Deliberately socket-free: every endpoint below is a method that can be called
and asserted on directly, and each one is a thin translation of a service call
into JSON. Where the CLI renders a table, this renders a dict — the decision in
between is neither one's business.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from ... import imaging
from ...config import load_settings
from ...domain.rules import DOUBT_ACTIONS, DOUBT_RULE_NAME, RuleSet
from ...errors import ConfigError, MediaSortError
from ...pipeline import Stopper
from ...services import (AppContext, applying, configuring, diagnostics, duplicates,
                         insights, library, maintenance)
from ...services import processing
from ...services import rules as rules_service
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
        except MediaSortError:
            return []

    def needs_adjudication(self) -> bool:
        """Whether any rule's class band asks for a second opinion at all,
        read from the ruleset rather than from settings. False when the
        ruleset cannot be read, same reasoning as `looking_for`."""
        try:
            return self.load_rules().needs_adjudication()
        except MediaSortError:
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
                # Each class condition sets its own band in the editor, so the
                # one thing worth saying up front is whether *anything*
                # currently asks for a second opinion.
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
        """Re-read every layer and swap the context in place. *When* to reload
        is this front end's call; what a reload rebuilds is `AppContext`'s."""
        self.ctx = self.ctx.reloaded(load_settings())

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

    # --------------------------------------------------------------- duplicates
    #
    # The duplicate finder is its own page over its own folders and its own
    # index, so its ids are `ctx.dupes` ids and mean nothing to the sorting
    # index. That is why it has its own image routes below rather than sharing
    # `/api/image`: the same number is a different photo in each database.

    def get_dupes_config(self) -> dict[str, Any]:
        """What the Duplicates page needs on load: its folders, whether its
        scan is remembered, and how many images it currently knows about."""
        dupes = self.ctx.settings.dupes
        return {
            "folders": [str(folder) for folder in dupes.folders],
            "saved": dupes.database is not None,
            # Live rows only: a discarded photo is flagged deleted, and counting
            # it would make the folder look unchanged after a discard.
            "indexed": self.ctx.dupes.images.count("missing = 0 AND deleted = 0"),
            "default_distance": duplicates.DEFAULT_MAX_DISTANCE,
            # Named on the page before the discard button is pressed, so where
            # the files go is never a surprise.
            "trash": str(dupes.trash_folder) if dupes.folders else "",
        }

    def get_duplicate_groups(self, distance: int = duplicates.DEFAULT_MAX_DISTANCE,
                             limit: int = 200) -> dict[str, Any]:
        """Groups of near-duplicate photos, for the Duplicates page."""
        return {"groups": duplicates.groups(self.ctx, distance, limit)}

    def start_dupes_scan(self) -> bool:
        """Hash the duplicate folders in the background.

        A job like any other, so the page can poll `/api/stats` for progress
        and the browser can be closed while it runs.
        """
        def job() -> str:
            """Scan; the string returned becomes the job's final message."""
            # Hashing a large folder is minutes of work with nothing on screen
            # unless the job says so as it goes. `scan` calls this every batch,
            # and the page polls the same message.
            def progress(stats: Any) -> None:
                """Report running totals into the job's status line."""
                self.jobs.message = f"hashed {stats.seen} files…"

            stats = duplicates.scan_folders(self.ctx, on_progress=progress)
            return (f"checked {stats.seen} files "
                    f"({stats.added} new, {stats.changed} changed)")

        return self.jobs.start("dupes-scan", job)

    def discard_marked_duplicates(self, delete: bool = False) -> dict[str, Any]:
        """Move every `discard`-marked photo into the duplicate trash folder,
        or with `delete`, send it to the desktop trash instead.

        The only endpoint on this page that touches a file. Neither destination
        unlinks anything, and it acts on exactly what is marked at the moment it
        runs — nothing implicit, nothing on a schedule. `delete` is per request:
        there is no stored setting that quietly turns discarding into deleting.
        """
        return duplicates.discard_marked(self.ctx, delete=delete)

    def get_job(self) -> dict[str, Any]:
        """Just the running job. The Duplicates page polls this rather than
        `/api/stats`, which would compute the whole sorting overview — a
        different index, and none of it on screen here."""
        return {"job": self.jobs.snapshot()}

    def mark_duplicates(self, image_ids: list[int], mark: str | None) -> dict[str, Any]:
        """Record `keep` / `discard` / undecided for these photos.

        Nothing on disk changes. This endpoint cannot delete a photo even if
        asked to — there is no file operation behind it.
        """
        ids = [int(i) for i in image_ids]
        return {"marked": duplicates.mark(self.ctx, ids, mark), "mark": mark}

    def dismiss_duplicates(self, image_ids: list[int]) -> dict[str, Any]:
        """Record a whole group as "not duplicates" — it will not be
        proposed again."""
        duplicates.dismiss(self.ctx, [int(i) for i in image_ids])
        return {"dismissed": True}

    def image_bytes(self, image_id: int) -> tuple[bytes, str]:
        """An original's raw bytes and content-type — the web UI's full-size view."""
        return self._read_image(self._require_path(image_id))

    def thumbnail_bytes(self, image_id: int, size: int = 240) -> tuple[bytes, str]:
        """A downscaled JPEG for an image in the sorting index."""
        return imaging.to_jpeg_bytes(self._require_path(image_id), size), "image/jpeg"

    def dupe_image_bytes(self, image_id: int) -> tuple[bytes, str]:
        """The same, for an id belonging to the duplicate index."""
        return self._read_image(self._require_path(image_id, dupes=True))

    def dupe_thumbnail_bytes(self, image_id: int, size: int = 240) -> tuple[bytes, str]:
        """A downscaled JPEG for `image_id` — what a duplicate group's card requests."""
        return imaging.to_jpeg_bytes(self._require_path(image_id, dupes=True), size), "image/jpeg"

    @staticmethod
    def _read_image(path: str) -> tuple[bytes, str]:
        """A file's bytes and its guessed content type."""
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return Path(path).read_bytes(), content_type

    def _require_path(self, image_id: int, dupes: bool = False) -> str:
        """Where `image_id` lives, in whichever of the two indexes owns it."""
        storage = self.ctx.dupes if dupes else self.ctx.storage
        path = storage.images.path_of(image_id)
        if path is None:
            raise MediaSortError(f"no such image: {image_id}")
        return path

    # ------------------------------------------------------------------ jobs

    # Each step is a plain method returning the line the UI shows when it
    # finishes, so `start_everything` can run the three of them back to back
    # and get exactly what pressing the three buttons in order would have said.

    def _scan(self) -> str:
        """Run the scan; the string returned becomes the job's final message."""
        stats = library.scan_library(self.ctx)
        if not (stats.added or stats.changed):
            # Re-scanning a caught-up library is the common case; silence here
            # reads as a broken button.
            return (f"scanned {stats.seen} files — nothing new or changed since "
                    f"the last scan, so the index is already up to date")
        return f"scanned {stats.seen} files: {stats.added} new, {stats.changed} changed"

    def _pipeline(self, with_analyze: bool) -> str:
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

    def _apply(self) -> str:
        """Run apply; the string returned becomes the job's final message."""
        stats, _ = applying.apply(self.ctx, self.load_rules())
        if not any((stats.created, stats.existing, stats.pruned, stats.moved,
                    stats.deleted, stats.skipped)):
            # Silence here reads as a broken button; it is almost always one
            # of two things, and both are visible on this tab.
            return ("nothing to do — no photo has an action that produces "
                    "anything yet. See “Where you are” above.")
        return applying.describe_stats(stats)

    def start_scan(self) -> bool:
        """Kick off a library scan in the background."""
        return self.jobs.start("scan", self._scan)

    def start_pipeline(self, with_analyze: bool = True) -> bool:
        """Kick off detect (+ adjudicate/analyze, if enabled) in the background."""
        return self.jobs.start("pipeline", lambda: self._pipeline(with_analyze))

    def start_apply(self) -> bool:
        """Kick off applying the rules' actions in the background."""
        return self.jobs.start("apply", self._apply)

    def start_everything(self, with_analyze: bool = True) -> bool:
        """Scan, examine and sort, back to back, as one job — the web UI's
        equivalent of `mediasort run`.

        One job rather than the page firing three: the browser can then be
        closed halfway through without the chain stopping where it stood. It
        renames itself at each step (see `JobRunner.phase`), so the Run tab
        lights up the step actually working, exactly as if it had been pressed.
        """
        def job() -> str:
            """Run all three steps; each one's message is kept, so a step that
            finished with nothing to do still says so at the end."""
            done = [f"scan — {self._scan()}"]
            self.jobs.phase("pipeline")
            done.append(f"examine — {self._pipeline(with_analyze)}")
            self.jobs.phase("apply")
            done.append(f"sort — {self._apply()}")
            return " · ".join(done)

        # Starts life as the scan, which is the step it begins with: a name no
        # step recognises would render the first stretch of the run nowhere.
        return self.jobs.start("scan", job)

    def start_recheck(self) -> bool:
        """Re-run the decision engine over stored detections and persist it."""
        def job() -> str:
            """Run recheck; the string returned becomes the job's final message."""
            ruleset = rules_service.active_ruleset(self.ctx)
            outcome = processing.recheck(self.ctx, ruleset)
            return f"re-evaluated {sum(outcome['categories'].values())} images"

        return self.jobs.start("recheck", job)

