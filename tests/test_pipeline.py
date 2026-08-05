"""The stage loops, driven by fake engines.

Possible at all because the pipeline depends on the protocols in
`pipeline.ports` rather than on ultralytics and requests — so the batching,
claiming and failure handling can be tested without a GPU or a server.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.domain.adjudication import ABSENT, PRESENT, UNSURE, Adjudication
from app.domain.detection import Detection
from app.domain.rules import RuleSet
from app.errors import VisionError
from app.pipeline import (Stopper, run_adjudicate_stage, run_analyze_stage,
                                run_detect_stage, run_pipeline)
from app.services import library
from app.storage import DONE, ERROR, SKIPPED, Stage

CAT = Detection("cat", 0.95, 0, 0, 10, 10)
#: In the review band (0.35–0.65): kept, flagged, but not a match on its own.
MAYBE_CAT = Detection("cat", 0.5, 0, 0, 10, 10)


class FakeDetector:
    """Returns the queued detections for every path; None means unreadable.

    `in_video` is the same thing for `detect_video`, which the stage calls
    instead of `detect_batch` for a video: None (the default) stands for a file
    no frame could be read from, `[]` for frames that showed nothing.
    """

    model_name = "fake.pt"

    def __init__(self, per_path=None, unreadable=frozenset(), explode=False, default=None,
                 in_video=None, video_explodes=False):
        self.per_path = per_path or {}
        self.unreadable = unreadable
        self.explode = explode
        self.default = [CAT] if default is None else default
        self.in_video = in_video
        self.video_explodes = video_explodes
        self.batches: list[list[str]] = []
        self.videos: list[str] = []
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

    def detect_video(self, path):
        self.videos.append(path)
        if self.video_explodes:
            raise RuntimeError("ffmpeg is on fire")
        return self.in_video

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
        "phash": None, "size": 100, "mtime": 0.0, "width": None, "height": None,
        "format": None, "taken_at": None,
    }])


def _analyze_state(storage, path="/lib/clip.mp4"):
    return storage.engine.conn.execute(
        "SELECT analyze_state FROM images WHERE path = ?", (path,)
    ).fetchone()["analyze_state"]


def _stored_detections(storage, path="/lib/clip.mp4"):
    """`{class: model}` for one file — what the detect stage wrote about it."""
    rows = storage.engine.conn.execute(
        "SELECT d.class, d.model FROM detections d JOIN images i ON i.id = d.image_id "
        "WHERE i.path = ?",
        (path,),
    ).fetchall()
    return {row["class"]: row["model"] for row in rows}


def test_a_video_is_sorted_by_what_the_frames_showed(indexed, cats):
    """The point of sampling frames: a video of a cat sorts as a cat, because
    the cat rule sits above the video rule."""
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])
    engine = FakeDetector(in_video=[CAT])

    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, ruleset, lambda: engine, Stopper()
    )

    assert stats.processed == 6
    assert stats.categories == {"cat": 6}
    assert engine.videos == ["/lib/clip.mp4"]
    # A video is never part of an image batch: it takes the other route.
    assert all("/lib/clip.mp4" not in batch for batch in engine.batches)
    # Both classes are recorded — `video` is the fallback that did not have to
    # be used, not something the cat replaces.
    assert _stored_detections(indexed.storage) == {"cat": "fake.pt", "video": "fake.pt"}


def test_a_video_the_frames_showed_nothing_in_falls_back_to_the_video_rule(indexed, cats):
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])
    engine = FakeDetector(in_video=[])

    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, ruleset, lambda: engine, Stopper()
    )

    assert stats.categories == {"cat": 5, "video": 1}
    # Frames were read, so the model that read them is what gets recorded.
    assert _stored_detections(indexed.storage) == {"video": "fake.pt"}


def test_an_unreadable_video_is_sorted_by_extension_rather_than_failed(indexed, cats):
    """A container nothing can open is still a video — the one thing known
    about it stays true, so it sorts on that instead of erroring."""
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])

    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, ruleset,
        lambda: FakeDetector(in_video=None), Stopper(),
    )

    assert stats.errors == 0
    assert stats.categories == {"cat": 5, "video": 1}
    assert _stored_detections(indexed.storage) == {"video": "file-extension"}
    # Nothing looked inside, so the semantic pass has nothing to describe.
    assert _analyze_state(indexed.storage) == SKIPPED


def test_a_video_that_breaks_the_detector_still_gets_its_pseudo_class(indexed, cats):
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])

    stats = run_detect_stage(
        indexed.storage, indexed.settings.detect, ruleset,
        lambda: FakeDetector(video_explodes=True), Stopper(),
    )

    assert stats.errors == 0
    assert stats.categories == {"cat": 5, "video": 1}


def test_no_frames_configured_means_no_video_is_ever_opened(ctx):
    """`DETECT_VIDEO_FRAMES=0` goes back to sorting videos by extension alone —
    and a library of nothing but videos then never pays to load YOLO."""
    _insert_video(ctx.storage)
    ruleset = RuleSet.starter(["cat", "video"])

    def factory():
        raise AssertionError("nothing should be loaded when no frame will be read")

    stats = run_detect_stage(
        ctx.storage, replace(ctx.settings.detect, video_frames=0), ruleset, factory, Stopper()
    )
    assert stats.processed == 1
    assert stats.categories == {"video": 1}
    assert _analyze_state(ctx.storage) == SKIPPED


def test_analyze_describes_a_video_it_could_read_frames_from(indexed, cats):
    """Ollama gets a frame like any photo (`imaging.to_jpeg_bytes`), so a video
    the detector could see into is describable too."""
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])
    run_detect_stage(
        indexed.storage, indexed.settings.detect, ruleset,
        lambda: FakeDetector(in_video=[CAT]), Stopper(),
    )

    engine = FakeVision()
    stats = run_analyze_stage(
        indexed.storage, ruleset, indexed.settings.workers, lambda: engine, Stopper(),
    )
    assert stats.processed == 6
    assert "/lib/clip.mp4" in [path for path, _ in engine.seen]


def test_analyze_never_sees_a_video_nothing_looked_inside(indexed, cats):
    """The other way round: no frame, nothing to describe. A video's action is
    `move`, not `ignore`, so the ordinary ignore-skip would not catch it."""
    _insert_video(indexed.storage)
    ruleset = RuleSet.starter(["cat", "video"])
    run_detect_stage(indexed.storage, indexed.settings.detect, ruleset, FakeDetector, Stopper())

    engine = FakeVision()
    stats = run_analyze_stage(
        indexed.storage, ruleset, indexed.settings.workers, lambda: engine, Stopper(),
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
        indexed.storage, cats, indexed.settings.workers, lambda: engine, Stopper(),
    )
    assert stats.processed == 2
    assert len(engine.seen) == 2


def test_the_detector_result_is_passed_to_the_vision_engine_as_a_hint(indexed, cats):
    _detect_all(indexed, cats)
    engine = FakeVision()
    run_analyze_stage(
        indexed.storage, cats, indexed.settings.workers, lambda: engine, Stopper(),
    )
    assert all(hint == "cat (95%)" for _, hint in engine.seen)


def test_a_refusing_engine_records_an_error_per_image(indexed, cats):
    _detect_all(indexed, cats)
    stats = run_analyze_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: FakeVision(fail_on={".jpg", ".png", ".jpeg", ".webp"}), Stopper(),
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
    from app.pipeline import StageStats

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
    from app.services import processing

    _detect_as(indexed, cats, MAYBE_CAT)
    run_adjudicate_stage(
        indexed.storage, cats, indexed.settings.workers,
        lambda: FakeAdjudicator(PRESENT), Stopper(),
    )
    outcome = processing.recheck(indexed, cats)
    assert outcome["categories"] == {"cat": 5}
    assert indexed.storage.images.count("needs_review = 1") == 0
