"""A second opinion on what the detector was unsure about.

Pure data and pure list surgery — this module calls nothing and knows nothing
about vision models. A verdict arrives as a value, and the only thing done with
it is to promote, drop or leave alone boxes the detector already produced. The
ordinary decision engine then runs over the result, unchanged and still AI-free:
`decide` cannot tell an adjudicated detection from a confident one, which is
exactly the point. Everything that follows — the category, the action, the
review flag — falls out of the rules the same way it always did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .detection import Detection

#: The three answers a second opinion may give. `UNSURE` is a real answer, not a
#: failure: an image nobody can settle is precisely the one worth a human.
PRESENT = "present"
ABSENT = "absent"
UNSURE = "unsure"
VERDICTS = (PRESENT, ABSENT, UNSURE)


@dataclass(frozen=True, slots=True)
class Adjudication:
    """One class, settled or explicitly not settled."""

    cls: str
    verdict: str
    #: How sure the second opinion was, 0-1. Self-reported and uncalibrated, so
    #: it is only ever used to *raise* a promoted detection above the threshold,
    #: never to lower one.
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Reject a verdict string outside `VERDICTS` at construction time,
        rather than let a typo travel silently into storage or a decision."""
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"unknown verdict {self.verdict!r}; expected one of {', '.join(VERDICTS)}"
            )

    @classmethod
    def from_row(cls, row) -> "Adjudication":
        """Rebuild from a stored row — the read side of `storage.adjudications`."""
        return cls(cls=row["class"], verdict=row["verdict"], confidence=row["confidence"])


def uncertain_classes(detections: Sequence[Detection], confidence: float,
                      review_confidence: float) -> list[str]:
    """The classes worth escalating: seen in the review band, never above it.

    The same set `decide` uses to raise `needs_review`, which is deliberate — the
    second opinion is asked about exactly the images that would otherwise have
    gone straight to a human, and about nothing else.
    """
    confident = {d.cls for d in detections if d.confidence >= confidence}
    borderline = {
        d.cls for d in detections if review_confidence <= d.confidence < confidence
    }
    return sorted(borderline - confident)


def adjudicated(detections: Sequence[Detection], adjudications: Iterable[Adjudication],
                confidence: float) -> tuple[Detection, ...]:
    """`detections` as the second opinion leaves them.

    * `present` promotes that class's best borderline box to the decision
      threshold, so the rules see a real hit. Only the best one: a verdict
      establishes that the subject is *there*, not how many there are, so a rule
      asking for `min_count: 2` still needs the detector to have been sure twice.
    * `absent` drops the class entirely, which is what stops a rejected guess
      from flagging the image for review.
    * `unsure` changes nothing, leaving the image exactly as uncertain as it
      was — and an uncertain image is what the review folder is for.

    Boxes above the threshold are never touched, whatever the verdict says: the
    detector was confident, and this pass exists to settle doubt, not to
    overrule certainty.
    """
    by_class = {a.cls: a for a in adjudications}
    if not by_class:
        return tuple(detections)

    kept: list[Detection] = []
    promoted: set[str] = set()
    # Highest confidence first, so "the best borderline box" is simply the first
    # one of its class we meet.
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        verdict = by_class.get(detection.cls)
        if verdict is None or detection.confidence >= confidence:
            kept.append(detection)
            continue
        if verdict.verdict == ABSENT:
            continue
        if verdict.verdict == PRESENT and detection.cls not in promoted:
            promoted.add(detection.cls)
            kept.append(
                # Floored at the threshold so it counts at all, raised to the
                # verdict's own confidence so a rule with a stricter
                # `min_confidence` can still respond to a very sure second look.
                Detection(
                    cls=detection.cls,
                    confidence=max(confidence, verdict.confidence),
                    x1=detection.x1, y1=detection.y1, x2=detection.x2, y2=detection.y2,
                )
            )
            continue
        kept.append(detection)
    return tuple(kept)
