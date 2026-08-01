"""Use cases that run the heavy stages, plus the cheap re-run of the rules.

The engine factories live here because this is the layer that is allowed to know
both what the pipeline needs (a `DetectorEngine`, a `VisionEngine`) and which
concrete adapter provides it. The pipeline itself never finds out.
"""

from __future__ import annotations

from typing import Callable

from ..analyzing import OllamaClient
from ..domain.adjudication import adjudicated
from ..domain.decision import decide
from ..domain.detection import VIDEO_CLASS
from ..domain.rules import RuleSet
from ..pipeline import (AdjudicatorEngine, DetectorEngine, StageStats, Stopper, VisionEngine,
                        run_adjudicate_stage, run_analyze_stage, run_detect_stage, run_pipeline)
from ..storage import Stage
from ..pipeline.stages import ProgressFn
from .context import AppContext

#: How long a stage waits for the vision engine to come up before giving up.
STARTUP_WAIT_ALONE = 30.0
STARTUP_WAIT_PARALLEL = 60.0


def detector_factory(ctx: AppContext, classes: list[str]) -> Callable[[], DetectorEngine]:
    """A callable that builds a `Detector` (loads YOLO weights) when called —
    deferred so the stage thread that owns it pays the load cost, not the
    thread that merely set the run up."""
    def build() -> DetectorEngine:
        """Load the model. Called once, inside the stage's own thread."""
        from ..detecting import Detector

        return Detector(ctx.settings.detect, classes, decode_workers=ctx.settings.workers.detect)

    return build


def _ollama_client_factory(ctx: AppContext, startup_wait: float,
                           classes: list[str] = ()) -> Callable[[], OllamaClient]:
    """A ready `OllamaClient`: constructed with `classes` (empty for
    adjudication) and blocked on until the model is pulled and warm."""
    def build() -> OllamaClient:
        """Connect and resolve the model tag. Called once, inside the stage's
        own thread."""
        client = OllamaClient(ctx.settings.analyze, classes)
        client.ensure_model(wait=startup_wait)
        return client

    return build


def vision_factory(ctx: AppContext, classes: list[str],
                   startup_wait: float) -> Callable[[], VisionEngine]:
    """A callable that builds an `OllamaClient` for the semantic pass, asking
    about `classes`."""
    return _ollama_client_factory(ctx, startup_wait, classes)


def adjudicator_factory(ctx: AppContext,
                        startup_wait: float) -> Callable[[], AdjudicatorEngine]:
    """The same client, asked the other question.

    No classes are passed: what is in doubt is decided per image by the stage,
    from what the detector actually saw in that picture.
    """
    return _ollama_client_factory(ctx, startup_wait)


# --------------------------------------------------------------------- stages


def _real_classes(ruleset: RuleSet) -> list[str]:
    """`ruleset.detector_classes()` minus the `video` pseudo-class: YOLO and
    Ollama only ever hear about classes a model could actually be asked for."""
    return [c for c in ruleset.detector_classes() if c != VIDEO_CLASS]


def detect(ctx: AppContext, ruleset: RuleSet, stopper: Stopper,
           on_progress: ProgressFn | None = None, limit: int | None = None) -> StageStats:
    """Run the detect stage alone, for whatever classes `ruleset` requires."""
    classes = _real_classes(ruleset)
    return run_detect_stage(
        ctx.storage, ctx.settings.detect, ruleset, detector_factory(ctx, classes), stopper,
        on_progress, limit=limit,
    )


def adjudicate(ctx: AppContext, ruleset: RuleSet, stopper: Stopper,
               on_progress: ProgressFn | None = None, limit: int | None = None,
               startup_wait: float = STARTUP_WAIT_ALONE,
               wait_for_detect: bool = False) -> StageStats:
    """Run the adjudicate stage alone (or as part of `run_all`, via
    `wait_for_detect`)."""
    return run_adjudicate_stage(
        ctx.storage, ctx.settings.detect, ruleset, ctx.settings.workers,
        adjudicator_factory(ctx, startup_wait), stopper, on_progress, limit=limit,
        wait_for_detect=wait_for_detect,
    )


def analyze(ctx: AppContext, ruleset: RuleSet, stopper: Stopper,
            on_progress: ProgressFn | None = None, limit: int | None = None,
            startup_wait: float = STARTUP_WAIT_ALONE,
            wait_for_detect: bool = False) -> StageStats:
    """Run the analyze stage alone (or as part of `run_all`, via
    `wait_for_detect`)."""
    classes = _real_classes(ruleset)
    return run_analyze_stage(
        ctx.storage, ctx.settings.analyze, ctx.settings.detect, ctx.settings.workers,
        vision_factory(ctx, classes, startup_wait), stopper, on_progress, limit=limit,
        wait_for_detect=wait_for_detect,
    )


def run_all(ctx: AppContext, ruleset: RuleSet, stopper: Stopper,
            on_progress: ProgressFn | None = None,
            with_analyze: bool = True) -> dict[str, StageStats]:
    """Every enabled stage in parallel, feeding the next one through the index.

    Only detection is fatal. The two Ollama stages degrade to exactly the
    behaviour the tool had before they existed: without a second opinion an
    uncertain photo goes to the review folder, and without the semantic pass it
    is sorted but not searchable. Neither is worth throwing away a detection run
    for.
    """
    stages: dict[str, tuple[Callable[[], StageStats], bool]] = {
        Stage.DETECT.value: (
            lambda: detect(ctx, ruleset, stopper, on_progress),
            True,  # nothing else can happen without detections
        )
    }
    if ctx.settings.analyze.adjudicate:
        stages[Stage.ADJUDICATE.value] = (
            lambda: adjudicate(
                ctx, ruleset, stopper, on_progress,
                startup_wait=STARTUP_WAIT_PARALLEL, wait_for_detect=True,
            ),
            False,
        )
    if with_analyze and ctx.settings.analyze.enabled:
        stages[Stage.ANALYZE.value] = (
            lambda: analyze(
                ctx, ruleset, stopper, on_progress,
                startup_wait=STARTUP_WAIT_PARALLEL, wait_for_detect=True,
            ),
            False,  # an enrichment: losing it must not lose the detection work
        )
    return run_pipeline(stages, stopper)


# ------------------------------------------------------------------- decisions


def recheck(ctx: AppContext, ruleset: RuleSet, on_progress: Callable[[int], None] | None = None,
            persist: bool = True) -> dict[str, dict[str, int]]:
    """Re-apply the rules to the evidence already in the index.

    No detector, no Ollama, no GPU: the stored boxes and the stored verdicts are
    re-evaluated together, in that order, which is the same sequence a live run
    puts them through. Replaying the verdicts is what keeps a rule edit from
    silently throwing away work the second opinion already paid for.
    `persist=False` makes it a projection ("what would these rules do?"), which
    is what the preview endpoints use — same code path, so the preview cannot
    drift from the real thing.
    """
    detect_settings = ctx.settings.detect
    categories: dict[str, int] = {}
    actions: dict[str, int] = {}

    for index, image_id in enumerate(ctx.storage.images.iter_decided_ids(), 1):
        detections = adjudicated(
            ctx.storage.detections.objects_for_image(image_id),
            ctx.storage.adjudications.objects_for_image(image_id),
            detect_settings.confidence,
        )
        decision = decide(
            detections, ruleset, detect_settings.confidence, detect_settings.review_confidence
        )
        if persist:
            ctx.storage.images.set_decision(
                image_id, decision.category, decision.action, decision.needs_review
            )
        categories[decision.category] = categories.get(decision.category, 0) + 1
        actions[decision.action] = actions.get(decision.action, 0) + 1
        if on_progress:
            on_progress(index)

    if persist:
        # A re-decision can raise the review flag on an image that was settled,
        # or clear it on one that was not, so both queues are re-derived.
        ctx.storage.images.sync_adjudication_queue()
        ctx.storage.images.skip_analysis_for_ignored()
    return {"categories": categories, "actions": actions}
