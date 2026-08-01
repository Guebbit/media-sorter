"""The one value type shared by the detector, the rules and the index.

It lives on its own so `rules.py` can be precise about what it matches against
without importing the decision engine, and the decision engine can import the
rules — no cycle, no untyped tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


#: The pseudo-class a video file gets instead of a YOLO detection. Nothing
#: decodes a video's frames — there is no model for that here — so a rule
#: matches a video by name (`{"class": "video"}`) the same way it matches a
#: real object, and the detector never sees the file at all.
VIDEO_CLASS = "video"

#: What `finish_detect` records as the "model" for a video's synthetic
#: detection, in place of a YOLO weights filename.
VIDEO_MODEL = "file-extension"

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".3g2", ".mpg", ".mpeg", ".wmv",
})


def is_video(path: str | Path) -> bool:
    """Whether `path` names a video rather than a photo, by extension alone."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
