"""The decision engine: rule outcome plus the review flag."""

from __future__ import annotations

from app.domain.decision import decide
from app.domain.rules import Always, HasClass, Rule, RuleSet
from tests.conftest import detection

def run(detections, ruleset):
    return decide(list(detections), ruleset)


def test_matches_the_expected_category(ruleset):
    assert run([detection("cat", 0.9), detection("dog", 0.8)], ruleset).category == "cat-dog"
    assert run([detection("cat", 0.9)], ruleset).category == "cat"
    assert run([detection("dog", 0.7)], ruleset).category == "dog"
    assert run([], ruleset).category == "none"


def test_action_comes_from_the_matched_rule(ruleset):
    assert run([detection("cat", 0.9)], ruleset).action == "copy"
    assert run([], ruleset).action == "ignore"


def test_below_threshold_does_not_count_towards_a_category(ruleset):
    decision = run([detection("cat", 0.5)], ruleset)
    assert decision.category == "none"
    assert decision.needs_review is True


def test_review_flags_a_borderline_class_that_would_have_changed_the_outcome(ruleset):
    decision = run([detection("cat", 0.9), detection("dog", 0.4)], ruleset)
    assert decision.category == "cat"
    assert decision.needs_review is True


def test_no_review_when_the_borderline_class_already_won(ruleset):
    decision = run([detection("cat", 0.9), detection("cat", 0.4)], ruleset)
    assert decision.category == "cat"
    assert decision.needs_review is False


def test_noise_below_the_review_threshold_is_ignored(ruleset):
    decision = run([detection("cat", 0.9), detection("dog", 0.1)], ruleset)
    assert decision.needs_review is False


def test_no_matching_rule_falls_back_to_none():
    empty = RuleSet((Rule("cats", HasClass("cat")),))
    decision = decide([detection("dog", 0.9)], empty)
    assert (decision.category, decision.action) == ("none", "ignore")


def test_rule_specific_confidence_is_honoured():
    ruleset = RuleSet((
        Rule("maybe-cat", HasClass("cat", min_confidence=0.3), action="review"),
        Rule("none", Always(), action="ignore"),
    ))
    decision = decide([detection("cat", 0.4)], ruleset)
    assert decision.category == "maybe-cat"
    assert decision.action == "review"


def test_min_count_rule():
    ruleset = RuleSet((
        Rule("crowd", HasClass("cat", min_count=3)),
        Rule("none", Always(), action="ignore"),
    ))
    two = [detection("cat", 0.9), detection("cat", 0.9)]
    assert decide(two, ruleset).category == "none"
    assert decide(two + [detection("cat", 0.8)], ruleset).category == "crowd"
