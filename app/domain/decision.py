"""The decision engine.

Deliberately zero AI: it evaluates the user's rules against detections someone
else produced. The engine knows nothing about cats, dogs or folders — only
"first matching rule wins" and how to spot a borderline call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .detection import Detection
from .rules import DEFAULT_CONFIDENCE, MatchContext, RuleSet

#: What an image gets when no rule claims it.
UNMATCHED_CATEGORY = "none"
UNMATCHED_ACTION = "ignore"


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

    # Every class the detector left in its review band. The `ollama_review`
    # toggle is deliberately ignored: a class nobody will ask Ollama about is
    # still one a person should look at, which is the whole point of the flag.
    needs_review = bool(ruleset.class_bands().unsettled(detections))

    if rule is None:
        return Decision(UNMATCHED_CATEGORY, UNMATCHED_ACTION, needs_review)
    return Decision(rule.name, rule.action, needs_review)
