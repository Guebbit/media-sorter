"""User-defined matching rules.

The pipeline does not know what "cats" or "dogs" mean. It evaluates an ordered
list of rules against the detections of an image; the first rule that matches
decides both the category and what should happen to the file.

Split in two because there are two independent things to get right: the
condition language (`conditions`) and rule priority and serialisation
(`ruleset`). Reading and writing the rules *file* is not here at all — that is
storage's job, which is what keeps this package free of I/O.
"""

from __future__ import annotations

from ...errors import RuleError
from .conditions import (AllOf, Always, AnyDetection, AnyOf, ClassBand, Condition,
                         DEFAULT_CONFIDENCE, DEFAULT_OLLAMA_REVIEW, DEFAULT_REVIEW_CONFIDENCE,
                         HasClass, InDoubt, MatchContext, NoneOf, Not, parse_condition,
                         register_condition)
from .ruleset import (DOUBT_ACTIONS, DOUBT_RULE_NAME, RULES_VERSION, Rule, RuleSet,
                      reorder)

__all__ = [
    "AllOf", "Always", "AnyDetection", "AnyOf", "ClassBand", "Condition",
    "DEFAULT_CONFIDENCE", "DEFAULT_OLLAMA_REVIEW", "DEFAULT_REVIEW_CONFIDENCE",
    "DOUBT_ACTIONS", "DOUBT_RULE_NAME", "HasClass", "InDoubt", "MatchContext", "NoneOf", "Not",
    "parse_condition", "register_condition",
    "RULES_VERSION", "Rule", "RuleError", "RuleSet", "reorder",
]
