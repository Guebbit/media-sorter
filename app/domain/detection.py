"""The one value type shared by the detector, the rules and the index.

It lives on its own so `rules.py` can be precise about what it matches against
without importing the decision engine, and the decision engine can import the
rules — no cycle, no untyped tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Detection:
    """One box a detector found, in the coordinates of the original image."""

    cls: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_row(cls, row) -> "Detection":
        """Rebuild from a stored row — used whenever rules are re-applied to
        images the detector already processed."""
        return cls(
            cls=row["class"], confidence=row["confidence"],
            x1=row["x1"], y1=row["y1"], x2=row["x2"], y2=row["y2"],
        )


#: The pseudo-class every video file gets, on top of whatever the detector
#: found in its frames. No model looks for it — it is true of the file by its
#: extension (`video.is_video`) — so a rule matching `{"class": "video"}`
#: catches a video whatever is in it, which makes it the fallback *below* the
#: real classes rather than a verdict of its own. It is also all a video gets
#: when frame sampling is off, or when no frame of it could be read.
VIDEO_CLASS = "video"

#: What `finish_detect` records as the "model" for a video nothing looked
#: inside of, in place of a YOLO weights filename.
VIDEO_MODEL = "file-extension"


def merge_frames(per_frame: Iterable[Sequence[Detection]]) -> list[Detection]:
    """Several frames' detections collapsed into one verdict for the file.

    The index stores detections per *file*, and a video is one row, so the
    frames have to agree on a single answer. Per class, the frame that saw the
    most of it wins, ties going to the more confident one — the busiest frame,
    not the average. Two reasons it is a maximum rather than a mean:

    * a cat that walks into shot halfway through is still a cat in the video,
      and averaging over the frames it is absent from would hide it;
    * counts have to survive, because `min_count` conditions read them. Taking
      the best box per class instead of the best *frame* per class would cap
      every video at one of everything, and "two cats" would never match.

    Boxes are the winning frame's own, in that frame's coordinates. Ordered by
    descending confidence, so a stored list reads as what the video is mostly
    of. An empty input gives an empty list — a video nothing was found in.
    """
    best: dict[str, list[Detection]] = {}
    for detections in per_frame:
        by_class: dict[str, list[Detection]] = {}
        for detection in detections:
            by_class.setdefault(detection.cls, []).append(detection)
        for cls, boxes in by_class.items():
            if cls not in best or _strength(boxes) > _strength(best[cls]):
                best[cls] = boxes
    return [box for boxes in sorted(best.values(), key=_strength, reverse=True) for box in boxes]


def _strength(boxes: Sequence[Detection]) -> tuple[int, float]:
    """How good a case one frame makes for one class: how many it saw, then
    how sure it was of the best of them."""
    return len(boxes), max(box.confidence for box in boxes)
