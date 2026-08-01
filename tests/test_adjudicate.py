"""The second opinion: what is escalated, what a verdict does, how it is parsed.

Every test here is plain values — the point of keeping the adjudication logic in
`domain/` is that "a confirmed cat becomes a cat" needs no model, no server and
no database to state.
"""

from __future__ import annotations

import pytest

from photosort.analyzing.contract import (build_verdict_prompt, build_verdict_schema,
                                          parse_verdict_reply)
from photosort.domain.adjudication import (ABSENT, PRESENT, UNSURE, Adjudication, adjudicated,
                                           uncertain_classes)
from photosort.domain.decision import decide
from photosort.domain.detection import Detection
from photosort.domain.rules import HasClass, Rule, RuleSet

CONF, REVIEW = 0.65, 0.35

SURE_CAT = Detection("cat", 0.9, 0, 0, 10, 10)
MAYBE_CAT = Detection("cat", 0.5, 0, 0, 10, 10)
FAINT_CAT = Detection("cat", 0.4, 0, 0, 10, 10)
MAYBE_DOG = Detection("dog", 0.5, 0, 0, 10, 10)


@pytest.fixture
def cats() -> RuleSet:
    # The starter ruleset leaves every band unset, so it resolves to the
    # hardcoded defaults (0.65 / 0.35 / ollama_review=True) — the same
    # CONF/REVIEW this file already asserts against.
    return RuleSet.starter(["cat"])


# ------------------------------------------------------------ what is escalated


def test_only_the_borderline_classes_are_worth_asking_about(cats):
    assert uncertain_classes([MAYBE_CAT], cats) == ["cat"]
    assert uncertain_classes([SURE_CAT], cats) == []
    assert uncertain_classes([], cats) == []


def test_a_class_seen_confidently_elsewhere_in_the_frame_is_not_in_doubt(cats):
    """One sure cat and one blurry one is not an open question about cats."""
    assert uncertain_classes([SURE_CAT, MAYBE_CAT], cats) == []


def test_a_detection_below_the_review_band_is_not_escalated(cats):
    """Below the review threshold it was never stored as doubt, only as noise."""
    assert uncertain_classes([Detection("cat", 0.2, 0, 0, 1, 1)], cats) == []


def test_a_class_with_ollama_review_off_is_never_escalated():
    """Toggled off on the condition, a borderline hit stays out of the
    escalation queue even though it is squarely inside the review band."""
    off = RuleSet((Rule("cat", HasClass("cat", ollama_review=False)),))
    assert uncertain_classes([MAYBE_CAT], off) == []


def test_the_escalation_queue_is_exactly_what_raises_the_review_flag(cats):
    """These two have to agree, or photos are either asked about twice or
    flagged for a human without anyone having looked."""
    for detections in ([MAYBE_CAT], [SURE_CAT], [], [FAINT_CAT]):
        flagged = decide(detections, cats).needs_review
        assert flagged == bool(uncertain_classes(detections, cats))


# --------------------------------------------------------- what a verdict does


def test_present_promotes_the_photo_to_a_real_match(cats):
    result = adjudicated([MAYBE_CAT], [Adjudication("cat", PRESENT, 0.9)], cats)
    assert decide(result, cats).category == "cat"
    assert decide(result, cats).needs_review is False


def test_absent_drops_the_class_and_the_doubt_with_it(cats):
    result = adjudicated([MAYBE_CAT], [Adjudication("cat", ABSENT, 0.9)], cats)
    assert result == ()
    decision = decide(result, cats)
    assert decision.category == "none"
    assert decision.needs_review is False


def test_unsure_changes_nothing_at_all(cats):
    result = adjudicated([MAYBE_CAT], [Adjudication("cat", UNSURE, 0.1)], cats)
    assert result == (MAYBE_CAT,)
    assert decide(result, cats).needs_review is True


def test_a_verdict_never_overrules_a_confident_detection(cats):
    """The stage exists to settle doubt, not to second-guess certainty."""
    result = adjudicated([SURE_CAT], [Adjudication("cat", ABSENT, 1.0)], cats)
    assert result == (SURE_CAT,)


def test_presence_is_not_a_count(cats):
    """A verdict says the subject is there, not how many — so only the best box
    is promoted and a `min_count: 2` rule still needs the detector."""
    result = adjudicated(
        [MAYBE_CAT, FAINT_CAT], [Adjudication("cat", PRESENT, 0.9)], cats
    )
    assert [d.confidence for d in result] == [0.9, 0.4]


def test_a_promoted_detection_carries_the_verdicts_own_confidence(cats):
    """Floored at the threshold, so it counts; raised to what the model said, so
    a rule with a stricter `min_confidence` can respond to a very sure answer."""
    (promoted,) = adjudicated([MAYBE_CAT], [Adjudication("cat", PRESENT, 0.95)], cats)
    assert promoted.confidence == 0.95
    (floored,) = adjudicated([MAYBE_CAT], [Adjudication("cat", PRESENT, 0.1)], cats)
    assert floored.confidence == CONF


def test_classes_nobody_was_asked_about_are_left_alone(cats):
    result = adjudicated([MAYBE_CAT, MAYBE_DOG], [Adjudication("cat", ABSENT, 0.9)], cats)
    assert result == (MAYBE_DOG,)


def test_no_verdicts_is_the_identity(cats):
    assert adjudicated([MAYBE_CAT], [], cats) == (MAYBE_CAT,)


def test_an_unknown_verdict_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown verdict"):
        Adjudication("cat", "probably")


# ------------------------------------------------------------------- the prompt


def test_the_schema_asks_only_about_the_classes_in_doubt():
    schema = build_verdict_schema(["cat"])
    assert set(schema["required"]) == {"cat_verdict", "cat_confidence"}
    assert schema["properties"]["cat_verdict"]["enum"] == [PRESENT, ABSENT, UNSURE]


def test_the_prompt_does_not_quote_the_detectors_guess():
    """Handing over the number the model is being asked to check turns a second
    opinion into an echo of the first."""
    prompt = build_verdict_prompt(["cat"])
    assert "cat" in prompt
    assert "0.5" not in prompt and "%" not in prompt


def test_a_clean_reply_is_read_as_given():
    verdicts = parse_verdict_reply('{"cat_verdict": "absent", "cat_confidence": 0.8}', ["cat"])
    assert verdicts == [Adjudication("cat", ABSENT, 0.8)]


def test_a_model_that_answers_the_question_instead_of_the_schema():
    verdicts = parse_verdict_reply('{"cat_verdict": "yes", "cat_confidence": 1}', ["cat"])
    assert verdicts[0].verdict == PRESENT


def test_an_unreadable_reply_is_unsure_not_absent():
    """A model we cannot understand has told us nothing, and nothing means a
    human looks — inventing `absent` from a parse failure would drop the photo."""
    for reply in ("", "I think there might be a cat?", "{not json"):
        assert parse_verdict_reply(reply, ["cat"]) == [Adjudication("cat", UNSURE, 0.0)]


def test_a_missing_class_in_the_reply_is_unsure():
    verdicts = parse_verdict_reply('{"dog_verdict": "present"}', ["cat", "dog"])
    assert {v.cls: v.verdict for v in verdicts} == {"cat": UNSURE, "dog": PRESENT}


def test_a_confidence_outside_the_range_is_clamped():
    verdicts = parse_verdict_reply('{"cat_verdict": "present", "cat_confidence": 42}', ["cat"])
    assert verdicts[0].confidence == 1.0
