"""Sampling stills out of a video, and merging what they showed.

The videos here are real ones, written frame by frame with OpenCV: seeking is
the part worth testing, and a fake that hands back a list of arrays would test
nothing. Each frame is a flat grey whose value *is* its index times eight, so a
sampled frame can say where in the file it came from.

MJPG in an AVI, because that encoder ships in the opencv wheel on every
platform and every one of its frames is a keyframe — a seek lands where it was
asked to, which is what makes the positions assertable at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app import imaging, video
from app.domain.detection import Detection, merge_frames
from app.errors import VideoError

#: How far a decoded grey may drift from the one that was written. MJPG is
#: lossy; a flat frame survives it well, but not exactly.
TOLERANCE = 6


def make_video(path: Path, frames: int = 30, fps: int = 10, size=(64, 48)) -> Path:
    """A video of `frames` flat grey frames, frame `i` filled with `i * 8`."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "no MJPG encoder available to write the fixture"
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), index * 8, np.uint8))
    writer.release()
    return path


def frame_index(frame: np.ndarray) -> int:
    """Which frame of `make_video`'s output this is, read back off its grey."""
    return int(round(float(frame.mean()) / 8))


# -------------------------------------------------------------------- sampling


def test_frames_are_spread_across_the_whole_video(tmp_path):
    frames = video.sample_frames(make_video(tmp_path / "clip.avi", frames=30), 8)

    assert len(frames) == 8
    indices = [frame_index(f) for f in frames]
    assert indices == sorted(indices)
    # The midpoint of each of eight equal slices of thirty frames.
    assert indices == pytest.approx([1, 5, 9, 13, 16, 20, 24, 28], abs=1)


def test_neither_the_first_nor_the_last_frame_is_sampled(tmp_path):
    """The two a naive even spacing would always take, and the two most likely
    to be black, a fade or a title card."""
    frames = video.sample_frames(make_video(tmp_path / "clip.avi", frames=30), 4)

    indices = [frame_index(f) for f in frames]
    assert min(indices) > 0
    assert max(indices) < 29


def test_a_video_shorter_than_the_sample_count_gives_what_it_has(tmp_path):
    frames = video.sample_frames(make_video(tmp_path / "short.avi", frames=3), 8)
    assert 0 < len(frames) <= 3


def test_the_middle_frame_comes_from_the_middle(tmp_path):
    frame = video.middle_frame(make_video(tmp_path / "clip.avi", frames=30))
    assert frame_index(frame) == pytest.approx(15, abs=1)


def test_a_file_that_is_not_a_video_raises(tmp_path):
    broken = tmp_path / "clip.mp4"
    broken.write_bytes(b"not a video, just bytes")
    with pytest.raises(VideoError):
        video.sample_frames(broken, 4)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(VideoError):
        video.sample_frames(tmp_path / "nothing.mp4", 4)


def test_frames_come_back_in_opencv_order(tmp_path):
    """BGR, because that is what ultralytics assumes of a numpy array — the
    detector hands frames straight to `predict`. `to_rgb` is the way back."""
    import cv2

    path = tmp_path / "blue.avi"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    blue = np.zeros((48, 64, 3), np.uint8)
    blue[:, :, 0] = 255  # channel 0 is blue in OpenCV's order
    for _ in range(10):
        writer.write(blue)
    writer.release()

    frame = video.middle_frame(path)
    assert frame[..., 0].mean() > 250 and frame[..., 2].mean() < TOLERANCE
    rgb = video.to_rgb(frame)
    assert rgb[..., 2].mean() > 250 and rgb[..., 0].mean() < TOLERANCE
    assert rgb.flags["C_CONTIGUOUS"], "Pillow cannot read a reversed view"


def test_a_thumbnail_of_a_video_is_a_frame_of_it(tmp_path):
    """`imaging.to_jpeg_bytes` answers for a video too — it is what the web UI's
    thumbnail route and the Ollama request both go through."""
    data = imaging.to_jpeg_bytes(make_video(tmp_path / "clip.avi"), max_edge=32)
    assert data.startswith(b"\xff\xd8")  # JPEG SOI


# --------------------------------------------------------------------- merging


def box(cls: str, confidence: float) -> Detection:
    return Detection(cls, confidence, 0, 0, 10, 10)


def test_the_busiest_frame_wins_a_class():
    """Counts have to survive the merge — `min_count` conditions read them —
    so the frame that saw two cats represents the video, not the one that saw
    a single more confident cat."""
    merged = merge_frames([
        [box("cat", 0.99)],
        [box("cat", 0.7), box("cat", 0.6)],
        [],
    ])
    assert [d.confidence for d in merged] == [0.7, 0.6]


def test_the_more_confident_frame_wins_a_tie():
    merged = merge_frames([[box("cat", 0.4)], [box("cat", 0.8)]])
    assert [d.confidence for d in merged] == [0.8]


def test_a_class_only_one_frame_saw_still_counts():
    """A dog that walks into shot halfway through is in the video."""
    merged = merge_frames([[box("cat", 0.9)], [box("cat", 0.9), box("dog", 0.8)]])
    assert {d.cls for d in merged} == {"cat", "dog"}


def test_classes_come_out_strongest_first():
    merged = merge_frames([[box("dog", 0.5), box("cat", 0.9), box("cat", 0.8)]])
    assert [d.cls for d in merged] == ["cat", "cat", "dog"]


def test_frames_that_showed_nothing_merge_to_nothing():
    assert merge_frames([[], [], []]) == []
    assert merge_frames([]) == []
