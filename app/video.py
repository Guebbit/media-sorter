"""Video files: which ones they are, and how to get stills out of them.

The only module that knows about OpenCV — and it loads it lazily, so a library
of nothing but photos never imports it at all. That is also why `is_video`
lives here rather than in `domain`: everything that has to *ask* the question
(the scanner's extension list, the detect stage, `imaging`) would otherwise
pull the answer out of a layer that is meant to stay pure.

Nothing here decodes a whole video, and nothing here plays one. Two callers
want stills out of one: the detector, which samples a handful of frames spread
across the file and runs them through YOLO as an ordinary batch, and `imaging`,
which needs a single picture whenever something has to *show* the file (a
thumbnail) or send it somewhere (an Ollama request). Both come through
`sample_frames`, so seeking, containers OpenCV cannot open, and headers that
lie about their own length are handled once.

Frames come out in BGR channel order, exactly as OpenCV read them — the order
ultralytics assumes for a numpy array, so the detector can pass them straight
through. `to_rgb` is for the other direction, where a human eventually looks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import VideoError

if TYPE_CHECKING:  # numpy arrives with ultralytics; nothing here needs it to import
    import numpy as np

log = logging.getLogger(__name__)

#: How far into a video the sequential fallback is willing to walk. Seeking
#: needs a length; without one the only way forward is to read frame by frame,
#: and this is what stops that from becoming "decode the whole film".
_SEQUENTIAL_LIMIT = 900

#: Assumed frame rate when the container will not state one, used only to space
#: the fallback's samples about a second apart.
_ASSUMED_FPS = 25

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".3g2", ".mpg", ".mpeg", ".wmv",
})


def is_video(path: str | Path) -> bool:
    """Whether `path` names a video rather than a photo, by extension alone.

    Not by probing it: the scanner asks this of every file it walks, and the
    answer decides whether a file is even worth opening.
    """
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def sample_frames(path: str | Path, count: int) -> list[np.ndarray]:
    """`count` frames spread across the whole video, in BGR.

    Taken at the midpoint of `count` equal slices — (i + ½)/count of the way in
    — rather than at 0, ½, 1: the first and last frames of a real video are
    disproportionately a fade, a title card or plain black, and those are the
    two an evenly-spaced `linspace` would always pick.

    Fewer than `count` come back for a video shorter than that, or when a seek
    lands somewhere undecodable — the caller gets whatever was readable, since
    six good frames answer the question as well as eight. Raises `VideoError`
    only when the file yielded nothing at all.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    import cv2  # lazy: ships with ultralytics, and an image-only run never needs it

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(f"cannot open video {path}")
    try:
        # Not every container states a frame count, and some state zero. Only
        # the ones that do can be sampled by seeking.
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = _seek_evenly(capture, total, count) if total > 0 else _walk(capture, count)
    finally:
        capture.release()

    if not frames:
        raise VideoError(f"no readable frame in {path}")
    return frames


def middle_frame(path: str | Path) -> np.ndarray:
    """One frame from halfway in — the still that stands for the whole file
    wherever a picture of it is wanted rather than its contents."""
    return sample_frames(path, 1)[0]


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """A BGR frame in the RGB order Pillow expects. Contiguous, because a
    reversed view is not something every C library will accept."""
    import numpy as np

    return np.ascontiguousarray(frame[:, :, ::-1])


def _seek_evenly(capture, total: int, count: int) -> list[np.ndarray]:
    """One frame per equal slice of a video whose length the header states.

    A seek can land on the nearest keyframe rather than the exact position, and
    a failed read is skipped rather than raised: neither matters when the point
    is "roughly this far into the file".
    """
    import cv2

    frames = []
    for index in range(min(count, total)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(total * (index + 0.5) / count))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
        else:
            log.debug("unreadable frame at slice %d/%d", index + 1, count)
    return frames


def _walk(capture, count: int) -> list[np.ndarray]:
    """The fallback for a container that reports no frame count.

    Reads forward from the start keeping roughly one frame per second, which
    gives the same "spread out" sampling for the front `_SEQUENTIAL_LIMIT`
    frames of the file — the whole of a short clip, the opening of a long one.
    """
    import cv2

    stride = int(capture.get(cv2.CAP_PROP_FPS) or 0) or _ASSUMED_FPS
    frames = []
    for index in range(_SEQUENTIAL_LIMIT):
        ok, frame = capture.read()
        if not ok:
            break
        # Offset by half a stride for the same reason `_seek_evenly` is: frame
        # zero of a video is the one least likely to show anything.
        if index % stride == stride // 2:
            frames.append(frame)
            if len(frames) == count:
                break
    return frames
