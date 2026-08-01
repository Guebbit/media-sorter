"""The decision engine.

Deliberately zero AI: it evaluates the user's rules against detections someone
else produced. The engine knows nothing about cats, dogs or folders — only
"first matching rule wins" and how to spot a borderline call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .detection import Detection
from .rules import (ClassBand, DEFAULT_CONFIDENCE, DEFAULT_OLLAMA_REVIEW,
                    DEFAULT_REVIEW_CONFIDENCE, MatchContext, RuleSet)

#: What an image gets when no rule claims it.
UNMATCHED_CATEGORY = "none"
UNMATCHED_ACTION = "ignore"

#: What a class with no band at all gets — one a rule no longer mentions
#: (deleted since the detection was made), but the row still carries.
_DEFAULT_BAND = ClassBand(DEFAULT_CONFIDENCE, DEFAULT_REVIEW_CONFIDENCE, DEFAULT_OLLAMA_REVIEW)


@dataclass(frozen=True, slots=True)
class Decision:
    category: str
    action: str
    needs_review: bool


def decide(detections: Sequence[Detection], ruleset: RuleSet) -> Decision:
    """Return the matched rule's name and action, plus a review flag.

    Review is intentionally independent of the rules: it reports detector
    uncertainty, which is a property of the image, not of the user's policy.
    A detection between its class's review confidence and confidence flags
    the image only when it could have changed the outcome.
    """
    rule = ruleset.evaluate(MatchContext(tuple(detections), DEFAULT_CONFIDENCE))

    bands = ruleset.class_bands()

    def band(cls: str) -> ClassBand:
        return bands.get(cls, _DEFAULT_BAND)

    confident = {d.cls for d in detections if d.confidence >= band(d.cls).confidence}
    uncertain = {
        d.cls for d in detections
        if band(d.cls).review_confidence <= d.confidence < band(d.cls).confidence
    }
    needs_review = bool(uncertain - confident)

    if rule is None:
        return Decision(UNMATCHED_CATEGORY, UNMATCHED_ACTION, needs_review)
    return Decision(rule.name, rule.action, needs_review)
