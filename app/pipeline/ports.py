"""What the pipeline needs from an inference engine — and nothing more.

Stated as protocols so this package never imports ultralytics or requests. A
test can drive a whole stage with a dozen lines of fake, and swapping either
engine for something else is a change in `services`, not here.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..domain.adjudication import Adjudication
from ..domain.detection import Detection


class DetectorEngine(Protocol):
    """Turns image paths into boxes, a batch at a time."""

    model_name: str

    def detect_batch(self, paths: list[str]) -> dict[str, list[Detection] | None]:
        """A None value means that image could not be read."""
        ...

    def detect_video(self, path: str) -> list[Detection] | None:
        """Boxes for one video, already merged across the frames the engine
        chose to look at — the pipeline stores one row per file, not per frame.

        None means no frame could be read. Deliberately not the same as an
        empty list, which means the engine looked and found nothing: only the
        first sends the caller back to matching the file by extension alone.
        """
        ...

    def close(self) -> None:
        """Release whatever `detect_batch` was holding (a model, a thread pool)."""
        ...


class AdjudicatorEngine(Protocol):
    """Settles whether named classes are really in one image.

    Separate from `VisionEngine` even though one adapter implements both: the
    two stages ask opposite questions, and a stage should not be handed the
    ability to ask the other one.
    """

    model: str

    def adjudicate(self, path: str, classes: Sequence[str]) -> list[Adjudication]:
        """A verdict for each of `classes`, for the image at `path`."""
        ...


class VisionEngine(Protocol):
    """Describes one image, given an optional hint about what is in it."""

    model: str

    def analyze(self, path: str, hint: str | None = None) -> dict[str, Any]:
        """The semantic-pass result for the image at `path`."""
        ...
