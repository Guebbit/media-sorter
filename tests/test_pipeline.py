"""The stage loops, driven by fake engines.

Possible at all because the pipeline depends on the protocols in
`pipeline.ports` rather than on ultralytics and requests — so the batching,
claiming and failure handling can be tested without a GPU or a server.
"""

from __future__ import annotations

import pytest

from photosort.domain.adjudication import ABSENT, PRESENT, UNSURE, Adjudication
from photosort.domain.detection import Detection
from photosort.domain.rules import RuleSet
from photosort.errors import VisionError
from photosort.pipeline import (Stopper, run_adjudicate_stage, run_analyze_stage,
                                run_detect_stage, run_pipeline)
from photosort.services import library
from photosort.storage import DONE, ERROR, SKIPPED, Stage

CAT = Detection("cat", 0.95, 0, 0, 10, 10)
#: In the review band (0.35–0.65): kept, flagged, but not a match on its own.
MAYBE_CAT = Detection("cat", 0.5, 0, 0, 10, 10)


class FakeDetector:
    """Returns the queued detections for every path; None means unreadable."""

    model_name = "fake.pt"

    def __init__(self, per_path=None, unreadable=frozenset(), explode=False, default=None):
        self.per_path = per_path or {}
        self.unreadable = unreadable
        self.explode = explode
        self.default = [CAT] if default is None else default
        self.batches: list[list[str]] = []
        self.closed = False

    def detect_batch(self, paths):
        self.batches.append(list(paths))
        if self.explode:
            raise RuntimeError("cuda is on fire")
        return {
            path: None if path.endswith(tuple(self.unreadable))
            else self.per_path.get(path, self.default)
            for path in paths
        }

    def close(self):
        self.closed = True


class FakeVision:
    model = "fake-vlm"

    def __init__(self, fail_on=frozenset()):
        self.fail_on = fail_on
        self.seen: list[tuple[str, str | None]] = []

    def analyze(self, path, hint=None):
        self.seen.append((path, hint))
        if path.endswith(tuple(self.fail_on) or ("\0",)):
            raise VisionError("model refused")
        return {"cat_count": 1, "notes": "fake"}


@pytest.fixture
def indexed(ctx):
    library.scan_library(ctx)
    return ctx


@pytest.fixture
def cats() -> RuleSet:
    return RuleSet.starter(["cat"])


# --------------------------------------------------------------------- detect


def test_detect_stage_processes_everything_and_records_decisions(indexed, cats):
    engine = FakeDetector()
    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, cats, lambda: engine, Stopper()
    )
    assert stats.processed == 5
    assert stats.categories == {"cat": 5}
    assert indexed.storage.images.count("detect_state = ?", (DONE,)) == 5
    assert engine.closed


def test_detect_stage_honours_a_limit(indexed, cats):
    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, cats, FakeDetector, Stopper(), limit=2
    )
    assert stats.processed == 2
    assert indexed.storage.images.count_pending(Stage.DETECT) == 3


def test_an_unreadable_image_is_an_error_not_a_crash(indexed, cats):
    engine = FakeDetector(unreadable={"a.jpg"})
    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, cats, lambda: engine, Stopper()
    )
    assert stats.errors == 1
    assert stats.processed == 4
    assert indexed.storage.images.count("detect_state = ?", (ERROR,)) == 1


def test_a_failed_batch_fails_its_rows_and_keeps_going(indexed, cats):
    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, cats,
        lambda: FakeDetector(explode=True), Stopper(),
    )
    assert stats.processed == 0
    assert stats.errors == 5
    assert indexed.storage.images.count("detect_state = ?", (ERROR,)) == 5


def test_a_stopped_run_leaves_the_rest_pending(indexed, cats):
    stopper = Stopper()
    stopper.stop()
    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, cats, FakeDetector, stopper
    )
    assert stats.processed == 0
    assert indexed.storage.images.count_pending(Stage.DETECT) == 5


# ----------------------------------------------------------------------- video


def _insert_video(storage, path="/lib/clip.mp4"):
    """A video row, without a real file on disk — `is_video` only looks at the
    extension, and nothing in the detect stage ever opens the path."""
    storage.images.upsert([{
        "path": path, "filename": path.rsplit("/", 1)[-1], "root": "/lib", "hash": "deadbeef",
        "size": 100, "mtime": 0.0, "width": None, "height": None, "format": None, "taken_at": None,
    }])


def test_a_video_gets_a_synthetic_class_and_never_reaches_the_detector(indexed, cats):
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])
    engine = FakeDetector()

    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, ruleset, lambda: engine, Stopper()
    )

    assert stats.processed == 6
    assert stats.categories == {"cat": 5, "video": 1}
    assert all("/lib/clip.mp4" not in batch for batch in engine.batches)
    # No vision model in this pipeline can look at a video either, so the
    # semantic pass is already settled rather than left pending on the
    # strength of the video rule's `move` action.
    row = indexed.storage.engine.conn.execute(
        "SELECT analyze_state FROM images WHERE path = ?", ("/lib/clip.mp4",)
    ).fetchone()
    assert row["analyze_state"] == SKIPPED


def test_an_all_video_batch_never_builds_the_detector(ctx):
    """A library (or a batch) with no photos at all must not pay to load YOLO."""
    _insert_video(ctx.storage)
    ruleset = RuleSet.starter(["cat", "video"])

    def factory():
        raise AssertionError("the detector should never be built for an all-video batch")

    stats = run_detect_stage(ctx.storage, ctx.settings.detect, ruleset, factory, Stopper())
    assert stats.processed == 1
    assert stats.categories == {"video": 1}


def test_analyze_never_sees_a_video(indexed, cats):
    """A video's action is `move`, not `ignore` — the ordinary ignore-skip
    would not catch it, so this checks the video-specific skip instead."""
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])
    run_detect_stage(indexed.storage, indexed.settings.detect, ruleset, FakeDetector, Stopper())

    engine = FakeVision()
    stats = run_analyze_stage(
        indexed.storage, indexed.settings.analyze, ruleset,
        indexed.settings.workers, lambda: engine, Stopper(),
    )
    assert stats.processed == 5
    assert all(path != "/lib/clip.mp4" for path, _ in engine.seen)


# -------------------------------------------------------------------- analyze


def _detect_all(ctx, ruleset, **kwargs):
    return run_detect_stage(
        ctx.storage, ctx.settings.detect, ruleset, lambda: FakeDetector(**kwargs), Stopper()
    )


def test_analysis_only_sees_images_the_rules_kept(indexed, cats):
    # Two cats, three that no rule claims -> ignored, so never analysed.
    paths = sorted(r["path"] for r in indexed.storage.images.iter_live_paths())
    _detect_all(indexed, cats, per_path={p: ([CAT] if i < 2 else []) for i, p in enumerate(paths)})

    engine = FakeVision()
    stats = run_analyze_stage(
        indexed.storage, indexed.settings.analyze, cats,
        indexed.settings.workers, lambda: engine, Stopper(),
    )
    assert stats.processed == 2
    assert len(engine.seen) == 2


def test_the_detector_result_is_passed_to_the_vision_engine_as_a_hint(indexed, cats):
    _detect_all(indexed, cats)
    engine = FakeVision()
    run_analyze_stage(
        indexed.storage, indexed.settings.analyze, cats,
        indexed.settings.workers, lambda: engine, Stopper(),
    )
    assert all(hint == "cat (95%)" for _, hint in engine.seen)


def test_a_refusing_engine_records_an_error_per_image(indexed, cats):
    _detect_all(indexed, cats)
    stats = run_analyze_stage(
        indexed.storage, indexed.settings.analyze, cats,
        indexed.settings.workers, lambda: FakeVision(fail_on={".jpg", ".png", ".jpeg", ".webp"}),
        Stopper(),
    )
    assert stats.processed == 0
    assert stats.errors == 5
    assert indexed.storage.images.count("analyze_state = ?", (ERROR,)) == 5


# --------------------------------------------------------------------- runner


def test_a_fatal_stage_stops_the_others():
    stopper = Stopper()

    def boom():
        raise RuntimeError("no weights")

    with pytest.raises(RuntimeError, match="no weights"):
        run_pipeline({"detect": (boom, True)}, stopper)
    assert stopper.stopped


def test_a_non_fatal_stage_failing_keeps_the_others_results():
    from photosort.pipeline import StageStats

    stopper = Stopper()

    def fine() -> StageStats:
        return StageStats(processed=3)

    def boom() -> StageStats:
        raise RuntimeError("engine is down")

    results = run_pipeline({"detect": (fine, True), "analyze": (boom, False)}, stopper)
    assert results["detect"].processed == 3
    assert "analyze" not in results
    assert not stopper.stopped


# ---------------------------------------------------------------- adjudicate


class FakeAdjudicator:
    """Answers with a fixed verdict per class, and records what it was asked."""

    model = "fake-vlm"

    def __init__(self, verdict=PRESENT, confidence=0.9, fail_on=frozenset()):
        self.verdict = verdict
        self.confidence = confidence
        self.fail_on = fail_on
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    def adjudicate(self, path, classes):
        self.asked.append((path, tuple(classes)))
        if path.endswith(tuple(self.fail_on) or ("\0",)):
            raise VisionError("ollama is down")
        return [Adjudication(cls, self.verdict, self.confidence) for cls in classes]


def _detect_as(indexed, cats, detection):
    """Detect every image as `detection`, so the review flag lands exactly as a
    real run would set it — the queue for the next stage is that flag."""
    return _detect_all(indexed, cats, default=[detection])


def test_a_confident_detection_is_never_escalated(indexed, cats):
    """Half the point of the cascade: the expensive question is asked only where
    the cheap answer was in doubt."""
    _detect_as(indexed, cats, CAT)
    assert indexed.storage.images.count_pending(Stage.ADJUDICATE) == 0
    assert indexed.storage.images.count("adjudicate_state = ?", (SKIPPED,)) == 5

    engine = FakeAdjudicator()
    stats = run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: engine, Stopper(),
    )
    assert stats.processed == 0
    assert engine.asked == []


def test_an_uncertain_detection_is_escalated(indexed, cats):
    _detect_as(indexed, cats, MAYBE_CAT)
    assert indexed.storage.images.count_pending(Stage.ADJUDICATE) == 5


def test_a_confirmed_verdict_promotes_the_photo_out_of_the_catch_all(indexed, cats):
    """0.5 is below the 0.65 threshold, so the rules saw nothing. The second
    opinion says the cat is there, and the photo becomes a cat."""
    _detect_as(indexed, cats, MAYBE_CAT)
    assert indexed.storage.images.count("category = 'none'") == 5

    engine = FakeAdjudicator(PRESENT)
    stats = run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: engine, Stopper(),
    )
    assert stats.processed == 5
    assert stats.categories == {"cat": 5}
    assert stats.verdicts == {PRESENT: 5}
    assert indexed.storage.images.count("category = 'cat' AND action = 'copy'") == 5
    # Settled, so no longer anyone's problem to look at.
    assert indexed.storage.images.count("needs_review = 1") == 0
    assert {classes for _, classes in engine.asked} == {("cat",)}


def test_a_rejected_verdict_clears_the_review_flag(indexed, cats):
    _detect_as(indexed, cats, MAYBE_CAT)
    assert indexed.storage.images.count("needs_review = 1") == 5

    run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: FakeAdjudicator(ABSENT), Stopper(),
    )
    assert indexed.storage.images.count("needs_review = 1") == 0
    assert indexed.storage.images.count("category = 'none'") == 5


def test_an_unsure_verdict_leaves_the_photo_for_a_human(indexed, cats):
    """`unsure` is the answer the review folder exists for."""
    _detect_as(indexed, cats, MAYBE_CAT)
    run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: FakeAdjudicator(UNSURE), Stopper(),
    )
    assert indexed.storage.images.count("needs_review = 1") == 5
    assert indexed.storage.images.count("adjudicate_state = ?", (DONE,)) == 5


def test_a_failed_verdict_leaves_the_photo_exactly_as_it_was(indexed, cats):
    """Degrading to the old behaviour is the requirement: an unreachable Ollama
    must not lose the review flag that sends the photo to a person."""
    _detect_as(indexed, cats, MAYBE_CAT)
    stats = run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: FakeAdjudicator(fail_on={".jpg", ".png", ".jpeg", ".webp"}), Stopper(),
    )
    assert stats.errors == 5
    assert stats.processed == 0
    assert indexed.storage.images.count("needs_review = 1") == 5
    assert indexed.storage.images.count("adjudicate_state = ?", (ERROR,)) == 5


def test_the_semantic_pass_waits_for_a_verdict_before_writing_a_photo_off(indexed, cats):
    """An image the second opinion is about to rescue from `ignore` must not be
    marked "not applicable" for description before the verdict lands."""
    _detect_as(indexed, cats, MAYBE_CAT)
    indexed.storage.images.skip_analysis_for_ignored()
    assert indexed.storage.images.count("analyze_state = ?", (SKIPPED,)) == 0

    run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: FakeAdjudicator(PRESENT), Stopper(),
    )
    # Promoted to `link`, so it is now a candidate for description rather than skipped.
    assert indexed.storage.images.count_pending(Stage.ANALYZE) == 5


def test_a_verdict_survives_a_rule_edit(indexed, cats):
    """`recheck` replays stored verdicts, so escalation is paid for once."""
    from photosort.services import processing

    _detect_as(indexed, cats, MAYBE_CAT)
    run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: FakeAdjudicator(PRESENT), Stopper(),
    )
    outcome = processing.recheck(indexed, cats)
    assert outcome["categories"] == {"cat": 5}
    assert indexed.storage.images.count("needs_review = 1") == 0
