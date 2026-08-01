"""The three stage loops: claim a batch, process it, record it, repeat.

Each takes a *factory* rather than a ready engine, so the expensive and
failure-prone construction (loading weights, waiting for an HTTP server) happens
inside the thread that owns the stage.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from ..config import AnalyzeSettings, DetectSettings, WorkerSettings
from ..domain.adjudication import Adjudication, adjudicated, uncertain_classes
from ..domain.decision import decide
from ..domain.detection import VIDEO_CLASS, VIDEO_MODEL, Detection, is_video
from ..domain.rules import (ClassBand, DEFAULT_CONFIDENCE, DEFAULT_OLLAMA_REVIEW,
                            DEFAULT_REVIEW_CONFIDENCE, RuleSet)
from ..errors import EngineError
from ..storage import ImageRow, Stage, Storage
from .ports import AdjudicatorEngine, DetectorEngine, VisionEngine
from .stopper import Stopper

log = logging.getLogger(__name__)

#: What a class with no band at all gets — one no rule mentions, but whose
#: detections still landed in storage.
_DEFAULT_BAND = ClassBand(DEFAULT_CONFIDENCE, DEFAULT_REVIEW_CONFIDENCE, DEFAULT_OLLAMA_REVIEW)

#: stage name, how many more are done, how many are left
ProgressFn = Callable[[str, int, int], None]


@dataclass(slots=True)
class StageStats:
    """What one call to a `run_*_stage` function did, for the caller to print
    or fold into a bigger summary."""

    processed: int = 0
    errors: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    #: How a second opinion ruled, when there was one. Empty for every other stage.
    verdicts: dict[str, int] = field(default_factory=dict)

    def bump(self, category: str) -> None:
        """Count one more image decided into `category`."""
        self.categories[category] = self.categories.get(category, 0) + 1


def _batch_size(base: int, remaining_budget: int | None) -> int | None:
    """How many rows to claim next, capped by `remaining_budget` (a `--limit`
    on the CLI). `None` means the budget for this run is spent, so the caller
    should stop instead of claiming more."""
    if remaining_budget is not None:
        if remaining_budget <= 0:
            return None
        return min(base, remaining_budget)
    return base


def _spend_budget(remaining_budget: int | None, spent: int) -> int | None:
    """`remaining_budget` after claiming `spent` more rows; unbounded stays unbounded."""
    return remaining_budget if remaining_budget is None else remaining_budget - spent


def _report(on_progress: ProgressFn | None, stage: Stage, done: int, images) -> None:
    """Tell the caller's progress bar what just happened, if it's listening."""
    if on_progress:
        on_progress(stage.value, done, images.count_pending(stage))


def run_detect_stage(storage: Storage, settings: DetectSettings, ruleset: RuleSet,
                     engine_factory: Callable[[], DetectorEngine], stopper: Stopper,
                     on_progress: ProgressFn | None = None,
                     limit: int | None = None) -> StageStats:
    """Claim pending images in batches, run them through `engine_factory()`'s
    detector, decide each one against `ruleset`, and persist the result —
    until the queue is empty, `limit` is spent, or `stopper` says to stop.

    A video never reaches the detector — there is nothing here that decodes
    one — so it is picked out of the batch by extension and given a synthetic
    `video` detection instead, and `engine_factory()` is only ever called for a
    batch that actually has a real image in it, lazily, so a library (or a
    single batch) with no photos at all never pays to load YOLO.
    """
    stats = StageStats()
    images = storage.images
    images.reset_running(Stage.DETECT)
    engine: DetectorEngine | None = None
    remaining_budget = limit

    def _finish(row: ImageRow, detections: list[Detection], model: str,
                skip_analyze: bool = False) -> None:
        """Score `row` against `ruleset`, persist the decision, and fold it
        into `stats` — the one place both the video shortcut and the real
        detector path land."""
        decision = decide(detections, ruleset)
        storage.results.finish_detect(row.id, detections, decision, model, skip_analyze)
        stats.processed += 1
        stats.bump(decision.category)

    try:
        while not stopper.stopped:
            batch_size = _batch_size(settings.batch, remaining_budget)
            if batch_size is None:
                break

            rows = images.claim_detect(batch_size)
            if not rows:
                break
            remaining_budget = _spend_budget(remaining_budget, len(rows))

            video_rows, image_rows = [], []
            for row in rows:
                (video_rows if is_video(row.path) else image_rows).append(row)

            for row in video_rows:
                # No vision model in this pipeline can look at a video either,
                # so the semantic pass has nothing to do here — skip it now
                # rather than leave it pending on the strength of an action
                # that only turns out to be `ignore` for some rules.
                _finish(row, [Detection(VIDEO_CLASS, 1.0, 0, 0, 0, 0)], VIDEO_MODEL,
                        skip_analyze=True)

            if image_rows:
                if engine is None:
                    engine = engine_factory()
                by_path = {row.path: row for row in image_rows}
                try:
                    # The one call into YOLO for this batch — `engine` is a
                    # `DetectorEngine` (in practice a `detecting.Detector`), so
                    # nothing here knows it is ultralytics on the other end.
                    results = engine.detect_batch(list(by_path))
                except Exception as exc:  # noqa: BLE001 - a bad batch must not kill the run
                    log.error("detection batch failed: %s", exc)
                    for row in image_rows:
                        images.fail(row.id, Stage.DETECT, str(exc))
                    stats.errors += len(image_rows)
                    results = {}

                for path, detections in results.items():
                    row = by_path[path]
                    if detections is None:
                        images.fail(row.id, Stage.DETECT, "unreadable image")
                        stats.errors += 1
                        continue
                    _finish(row, detections, engine.model_name)

            images.sync_adjudication_queue()
            images.skip_analysis_for_ignored()
            _report(on_progress, Stage.DETECT, len(rows), images)
    finally:
        if engine is not None:
            engine.close()
    return stats


def run_adjudicate_stage(storage: Storage, ruleset: RuleSet,
                         workers: WorkerSettings,
                         engine_factory: Callable[[], AdjudicatorEngine], stopper: Stopper,
                         on_progress: ProgressFn | None = None, limit: int | None = None,
                         wait_for_detect: bool = False) -> StageStats:
    """The second opinion, on the images the detector could not settle.

    The verdict is not stored as an opinion and left there: it rewrites the
    decision, because that is the only thing that makes the stage worth its
    seconds. A confirmed class stops falling through to the catch-all rule; a
    rejected one stops being flagged for a human; an unsettled one is left
    exactly as uncertain as it was, which is what the review folder is for.
    """
    stats = StageStats()
    images = storage.images
    images.reset_running(Stage.ADJUDICATE)
    engine = engine_factory()
    remaining_budget = limit

    def process(row: ImageRow) -> tuple[int, list[Adjudication] | None, str | None]:
        """One image's worth of work: find what's still uncertain, ask the
        engine, and return `(image_id, verdicts, error)` for the caller to
        record — run inside the thread pool, so it must not touch shared state."""
        detections = storage.detections.objects_for_image(row.id)
        classes = uncertain_classes(detections, ruleset)
        if not classes:
            # The flag outlived what raised it — a rule edit, a threshold change.
            # Nothing to ask, and nothing to record.
            return row.id, [], None
        try:
            # The one call into Ollama for this image — `engine` is an
            # `AdjudicatorEngine` (in practice an `analyzing.OllamaClient`
            # asked the verdict question), one HTTP request per uncertain image.
            return row.id, engine.adjudicate(row.path, classes), None
        except (EngineError, OSError) as exc:
            return row.id, None, str(exc)

    with ThreadPoolExecutor(max_workers=workers.analyze, thread_name_prefix="adjudicate") as pool:
        while not stopper.stopped:
            batch_size = _batch_size(workers.analyze * 2, remaining_budget)
            if batch_size is None:
                break

            images.sync_adjudication_queue()
            rows = images.claim_adjudicate(batch_size)
            if not rows:
                if wait_for_detect and not stopper.stopped and images.detect_in_flight():
                    stopper.wait(2.0)
                    continue
                break
            remaining_budget = _spend_budget(remaining_budget, len(rows))

            by_id = {row.id: row for row in rows}
            futures = [pool.submit(process, row) for row in rows]
            for future in as_completed(futures):
                image_id, verdicts, error = future.result()
                if error is not None:
                    # An image nobody could settle stays flagged, which is the
                    # same answer the tool gave before this stage existed.
                    images.fail(image_id, Stage.ADJUDICATE, error)
                    stats.errors += 1
                    log.warning("adjudication failed on image %s: %s", image_id, error)
                    continue
                decision = decide(
                    adjudicated(
                        storage.detections.objects_for_image(image_id),
                        verdicts or [],
                        ruleset,
                    ),
                    ruleset,
                )
                storage.results.finish_adjudicate(
                    image_id, verdicts or [], decision, engine.model
                )
                stats.processed += 1
                stats.bump(decision.category)
                for verdict in verdicts or []:
                    stats.verdicts[verdict.verdict] = stats.verdicts.get(verdict.verdict, 0) + 1
                log.debug("adjudicated %s -> %s", by_id[image_id].path, decision.category)

            # A verdict can move a photo out of `ignore`, so the skip has to be
            # re-evaluated after it rather than only when detection commits.
            images.skip_analysis_for_ignored()
            _report(on_progress, Stage.ADJUDICATE, len(rows), images)
    return stats


def run_analyze_stage(storage: Storage, settings: AnalyzeSettings, ruleset: RuleSet,
                      workers: WorkerSettings, engine_factory: Callable[[], VisionEngine],
                      stopper: Stopper, on_progress: ProgressFn | None = None,
                      limit: int | None = None, wait_for_detect: bool = False) -> StageStats:
    """`wait_for_detect` keeps the stage alive while anything upstream still feeds it.

    Upstream is both earlier stages: adjudication can move a photo out of
    `ignore` and into this queue, so finishing as soon as detection is done
    would leave exactly the rescued images undescribed.
    """
    stats = StageStats()
    images = storage.images
    images.reset_running(Stage.ANALYZE)
    engine = engine_factory()
    remaining_budget = limit

    def process(row: ImageRow) -> tuple[int, str | None]:
        """One image's worth of work: describe it and persist the result,
        returning `(image_id, error)` for the caller — run inside the thread
        pool, so it must not touch shared state."""
        hint = _detection_hint(storage, row.id, ruleset)
        try:
            # The one call into Ollama for this image — `engine` is a
            # `VisionEngine` (in practice an `analyzing.OllamaClient` asked
            # the description question), one HTTP request per photo.
            payload = engine.analyze(row.path, hint)
        except (EngineError, OSError) as exc:
            return row.id, str(exc)
        storage.results.finish_analyze(row.id, payload, engine.model)
        return row.id, None

    with ThreadPoolExecutor(max_workers=workers.analyze, thread_name_prefix="analyze") as pool:
        while not stopper.stopped:
            batch_size = _batch_size(workers.analyze * 2, remaining_budget)
            if batch_size is None:
                break

            rows = images.claim_analyze(batch_size)
            if not rows:
                upstream = images.detect_in_flight() or images.adjudicate_in_flight()
                if wait_for_detect and not stopper.stopped and upstream:
                    stopper.wait(2.0)
                    continue
                break
            remaining_budget = _spend_budget(remaining_budget, len(rows))

            futures = [pool.submit(process, row) for row in rows]
            for future in as_completed(futures):
                image_id, error = future.result()
                if error:
                    images.fail(image_id, Stage.ANALYZE, error)
                    stats.errors += 1
                    log.warning("analysis failed on image %s: %s", image_id, error)
                else:
                    stats.processed += 1

            _report(on_progress, Stage.ANALYZE, len(rows), images)
    return stats


def _detection_hint(storage: Storage, image_id: int, ruleset: RuleSet) -> str | None:
    """What the detector already knows, phrased for a prompt."""
    bands = ruleset.class_bands()
    hint = ", ".join(
        f"{d['class']} ({d['confidence']:.0%})"
        for d in storage.detections.for_image(image_id)
        if d["confidence"] >= bands.get(d["class"], _DEFAULT_BAND).confidence
    )
    return hint or None
