"""The rules engine: parsing, matching, serialisation and validation."""

from __future__ import annotations

import json

import pytest

from app.actions.registry import default_registry
from app.domain.rules import (AllOf, Always, AnyDetection, AnyOf, ClassBand,
                                    DEFAULT_CONFIDENCE, DEFAULT_OLLAMA_REVIEW,
                                    DEFAULT_REVIEW_CONFIDENCE, HasClass, MatchContext, NoneOf,
                                    Not, Rule, RuleError, RuleSet, parse_condition, reorder)
from app.storage import RulesStore
from tests.conftest import detection


def ctx(*pairs, confidence: float = 0.65) -> MatchContext:
    return MatchContext(
        detections=tuple(detection(cls, conf) for cls, conf in pairs),
        default_confidence=confidence,
    )


# ------------------------------------------------------------------ conditions


def test_has_class_respects_default_threshold():
    condition = HasClass("cat")
    assert condition.matches(ctx(("cat", 0.9)))
    assert not condition.matches(ctx(("cat", 0.5)))


def test_has_class_min_count():
    condition = HasClass("cat", min_count=2)
    assert not condition.matches(ctx(("cat", 0.9)))
    assert condition.matches(ctx(("cat", 0.9), ("cat", 0.8)))


def test_has_class_confidence_override_beats_global():
    lenient = HasClass("cat", min_confidence=0.3)
    assert lenient.matches(ctx(("cat", 0.4)))
    strict = HasClass("cat", min_confidence=0.95)
    assert not strict.matches(ctx(("cat", 0.9)))


def test_boolean_operators():
    both = AllOf([HasClass("cat"), HasClass("dog")])
    either = AnyOf([HasClass("cat"), HasClass("dog")])
    neither = NoneOf([HasClass("cat"), HasClass("dog")])

    assert both.matches(ctx(("cat", 0.9), ("dog", 0.9)))
    assert not both.matches(ctx(("cat", 0.9)))
    assert either.matches(ctx(("cat", 0.9)))
    assert not either.matches(ctx(("horse", 0.9)))
    assert neither.matches(ctx(("horse", 0.9)))
    assert not neither.matches(ctx(("cat", 0.9)))


def test_not_and_always():
    assert Not(HasClass("cat")).matches(ctx(("dog", 0.9)))
    assert not Not(HasClass("cat")).matches(ctx(("cat", 0.9)))
    assert Always().matches(ctx())


def test_any_detection_counts_everything():
    condition = AnyDetection(min_count=2)
    assert condition.matches(ctx(("cat", 0.9), ("horse", 0.8)))
    assert not condition.matches(ctx(("cat", 0.9)))


# -------------------------------------------------------------------- parsing


def test_string_shorthand_is_a_class_condition():
    assert parse_condition("cat") == HasClass("cat")


def test_parse_nested_condition():
    node = {"all_of": [{"class": "cat", "min_count": 2}, {"not": {"class": "dog"}}]}
    condition = parse_condition(node)
    assert condition.matches(ctx(("cat", 0.9), ("cat", 0.7)))
    assert not condition.matches(ctx(("cat", 0.9), ("cat", 0.7), ("dog", 0.9)))


def test_condition_round_trips_through_json():
    node = {"any_of": [{"class": "cat", "min_confidence": 0.4}, {"class": "dog", "min_count": 3}]}
    assert parse_condition(node).to_json() == node


def test_review_band_and_ollama_toggle_round_trip_through_json():
    node = {"class": "cat", "min_confidence": 0.8, "min_review_confidence": 0.6,
            "ollama_review": False}
    condition = parse_condition(node)
    assert condition.min_review_confidence == 0.6
    assert condition.ollama_review is False
    assert condition.to_json() == node


@pytest.mark.parametrize("node, message", [
    ({"class": ""}, "non-empty string"),
    ({"class": "cat", "min_count": 0}, "min_count"),
    ({"class": "cat", "min_confidence": 5}, "min_confidence"),
    ({"class": "cat", "min_review_confidence": 5}, "min_review_confidence"),
    ({"class": "cat", "min_confidence": 0.3, "min_review_confidence": 0.5}, "min_review_confidence"),
    ({"class": "cat", "ollama_review": "yes"}, "ollama_review"),
    ({"all_of": "cat"}, "must be a list"),
    ({"nonsense": 1}, "unknown condition"),
    (42, "must be an object"),
])
def test_bad_conditions_are_rejected(node, message):
    with pytest.raises(RuleError, match=message):
        parse_condition(node)


def test_classes_are_collected_from_the_whole_tree():
    condition = parse_condition({"all_of": [{"class": "cat"}, {"any_of": ["dog", "bird"]}]})
    assert condition.classes() == {"cat", "dog", "bird"}


# ---------------------------------------------------------------------- rules


def test_first_matching_rule_wins():
    ruleset = RuleSet((
        Rule("specific", AllOf([HasClass("cat"), HasClass("dog")])),
        Rule("general", HasClass("cat")),
    ))
    assert ruleset.evaluate(ctx(("cat", 0.9), ("dog", 0.9))).name == "specific"
    assert ruleset.evaluate(ctx(("cat", 0.9))).name == "general"
    assert ruleset.evaluate(ctx()) is None


def test_disabled_rules_are_skipped():
    ruleset = RuleSet((
        Rule("off", HasClass("cat"), enabled=False),
        Rule("on", HasClass("cat")),
    ))
    assert ruleset.evaluate(ctx(("cat", 0.9))).name == "on"


# ------------------------------------------------------------- class bands


def test_an_unmentioned_field_falls_back_to_the_hardcoded_default():
    ruleset = RuleSet((Rule("cat", HasClass("cat")),))
    assert ruleset.class_bands()["cat"] == ClassBand(
        DEFAULT_CONFIDENCE, DEFAULT_REVIEW_CONFIDENCE, DEFAULT_OLLAMA_REVIEW
    )


def test_an_uncustomised_mention_does_not_shadow_a_later_rules_override():
    """A rule that references a class without overriding anything (`cat`
    inside an `all_of` with `dog`, say) must not win that class's band just
    because it comes first in priority order — only an explicit value should
    ever claim a field. Regression for a bug where the whole band tuple was
    claimed on first sight, silently discarding a later rule's override."""
    ruleset = RuleSet((
        Rule("cat-dog", AllOf([HasClass("cat"), HasClass("dog")])),
        Rule("cat", HasClass("cat", min_confidence=0.2)),
    ))
    assert ruleset.class_bands()["cat"].confidence == 0.2


def test_each_field_resolves_independently_from_whichever_rule_set_it():
    """Not merely "first rule wins" — first rule to set *that field* wins,
    so two different rules can each contribute one half of a class's band."""
    ruleset = RuleSet((
        Rule("a", HasClass("cat", min_confidence=0.8)),
        Rule("b", HasClass("cat", min_review_confidence=0.5, ollama_review=False)),
    ))
    band = ruleset.class_bands()["cat"]
    assert band == ClassBand(confidence=0.8, review_confidence=0.5, ollama_review=False)


def test_needs_adjudication_is_true_when_any_class_opts_in():
    assert not RuleSet(
        (Rule("cat", HasClass("cat", ollama_review=False)),)
    ).needs_adjudication()
    assert RuleSet((Rule("cat", HasClass("cat")),)).needs_adjudication()  # default is on
    assert RuleSet((
        Rule("cat", HasClass("cat", ollama_review=False)),
        Rule("dog", HasClass("dog", ollama_review=True)),
    )).needs_adjudication()


def test_validate_rejects_a_band_left_inverted_by_its_defaults():
    """`min_confidence` alone, set low enough to fall under the default
    `min_review_confidence`, produces an inverted band once defaults fill the
    other field in — validation must catch that, not just the case where both
    are set explicitly on the same condition."""
    ruleset = RuleSet((Rule("cat", HasClass("cat", min_confidence=0.2)),))
    with pytest.raises(RuleError, match="review confidence"):
        ruleset.validate(default_registry().names())


def test_target_folder_defaults_to_a_title_cased_name():
    assert Rule("cat-dog", Always()).target_folder() == "Cat-Dog"
    assert Rule("cats", Always(), folder="Feline").target_folder() == "Feline"


def test_ruleset_round_trips():
    original = RuleSet.starter(["cat", "dog"])
    assert RuleSet.from_json(original.to_json()).to_json() == original.to_json()


def test_duplicate_rule_names_are_rejected():
    payload = {"rules": [
        {"name": "same", "when": {"class": "cat"}},
        {"name": "same", "when": {"class": "dog"}},
    ]}
    with pytest.raises(RuleError, match="duplicate"):
        RuleSet.from_json(payload)


def test_validate_rejects_unknown_action():
    ruleset = RuleSet((Rule("x", Always(), action="teleport"),))
    with pytest.raises(RuleError, match="unknown action"):
        ruleset.validate(default_registry().names())


def test_validate_rejects_classes_the_model_cannot_detect():
    ruleset = RuleSet((Rule("x", HasClass("dragon")),))
    with pytest.raises(RuleError, match="dragon"):
        ruleset.validate(default_registry().names(), known_classes=["cat", "dog"])


def test_required_classes_drives_the_detector():
    ruleset = RuleSet((Rule("x", AllOf([HasClass("cat"), HasClass("bird")])),))
    assert ruleset.required_classes() == {"cat", "bird"}


# ------------------------------------------------------- the default ruleset


def test_starter_ruleset_covers_combined_then_individual_classes():
    """Most specific first, because the first match wins."""
    ruleset = RuleSet.starter(["cat", "dog"])
    assert ruleset.names() == ["cat-dog", "cat", "dog", "none", "in-doubt"]

    assert ruleset.evaluate(ctx(("cat", 0.9), ("dog", 0.9))).name == "cat-dog"
    assert ruleset.evaluate(ctx(("cat", 0.9))).name == "cat"
    assert ruleset.evaluate(ctx(("dog", 0.9))).name == "dog"
    assert ruleset.evaluate(ctx()).name == "none"

    actions = {rule.name: rule.action for rule in ruleset.rules}
    assert actions == {"cat-dog": "copy", "cat": "copy", "dog": "copy", "none": "ignore",
                       "in-doubt": "move"}


def test_starter_ruleset_stays_linear_as_classes_are_added():
    """One combined rule plus one per class — not 2^n rules."""
    ruleset = RuleSet.starter(["cat", "dog", "bird"])
    assert ruleset.names() == ["bird-cat-dog", "bird", "cat", "dog", "none", "in-doubt"]

    many = RuleSet.starter(["cat", "dog", "bird", "horse", "sheep"])
    assert len(many.rules) == 8  # 1 combined + 5 classes + catch-all + doubt

    # A partial combination falls through to the first single-class rule.
    assert ruleset.evaluate(ctx(("bird", 0.9), ("cat", 0.9))).name == "bird"


def test_starter_ruleset_gives_video_its_own_move_rule():
    """A video paired with a cat in the same photo means nothing, so `video`
    stays out of the combinatorial `all_of` rule and gets `move` rather than
    the ordinary `copy`."""
    ruleset = RuleSet.starter(["cat", "dog", "video"])
    assert ruleset.names() == ["cat-dog", "cat", "dog", "video", "none", "in-doubt"]

    video_rule = ruleset.by_name("video")
    assert video_rule.action == "move"
    assert video_rule.target_folder() == "video"

    # Still evaluated as an ordinary detection: whatever else is in frame wins
    # first, same as any other class.
    assert ruleset.evaluate(ctx(("cat", 0.9), ("dog", 0.9), ("video", 1.0))).name == "cat-dog"
    assert ruleset.evaluate(ctx(("video", 1.0))).name == "video"


def test_starter_ruleset_of_video_alone():
    ruleset = RuleSet.starter(["video"])
    assert ruleset.names() == ["video", "none", "in-doubt"]
    assert ruleset.evaluate(ctx(("video", 1.0))).name == "video"



# ------------------------------------- the detector classes the rules imply


def test_required_classes_become_the_detector_list():
    ruleset = RuleSet((Rule("x", AllOf([HasClass("cat"), HasClass("bird")])),))
    assert ruleset.detector_classes() == ["bird", "cat"]


def test_a_ruleset_that_asks_for_nothing_is_an_error():
    """Detecting nothing is never what someone meant."""
    ruleset = RuleSet((Rule("everything", Always()),))
    with pytest.raises(RuleError, match="nothing to detect"):
        ruleset.detector_classes()


# ------------------------------------------------------- the store, not the rules


def test_store_load_after_save_round_trips(tmp_path):
    store = RulesStore(tmp_path / "rules.json")
    assert not store.exists()
    store.save(RuleSet.starter(["cat", "dog"]))
    assert store.exists()
    assert store.load().names() == ["cat-dog", "cat", "dog", "none", "in-doubt"]

    # A later edit must be honoured rather than overwritten by anything else.
    store.path.write_text(json.dumps({"rules": [{"name": "only", "when": {"class": "cat"}}]}))
    assert store.load().names() == ["only", "in-doubt"]


def test_rule_validate_rejects_unknown_action():
    ruleset = RuleSet.from_json(
        {"rules": [{"name": "x", "when": {"class": "cat"}, "action": "teleport"}]})
    with pytest.raises(RuleError, match="unknown action"):
        ruleset.validate(default_registry().names())


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "nested" / "rules.json"
    RulesStore(path).save(RuleSet.starter(["cat"]))
    assert path.exists()
    assert list(path.parent.iterdir()) == [path]


def test_invalid_json_reports_the_file(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text("{ not json")
    with pytest.raises(RuleError, match="not valid JSON"):
        RulesStore(path).load()


def test_reorder_moves_a_rule_and_clamps():
    ruleset = RuleSet.starter(["cat", "dog"])
    assert reorder(ruleset, "dog", -1).names() == ["cat-dog", "dog", "cat", "none", "in-doubt"]
    assert reorder(ruleset, "cat-dog", -5).names() == ruleset.names()
    with pytest.raises(RuleError, match="no rule named"):
        reorder(ruleset, "ghost", 1)


# ------------------------------------------------------- retired action names


# ------------------------------------------------------------- the doubt rule


def test_the_doubt_rule_is_added_to_a_file_that_has_none():
    """Every library has an answer to "what about the ones you are unsure of".
    A file predating the rule gets the default rather than losing the answer."""
    ruleset = RuleSet.from_json({"version": 1, "rules": [
        {"name": "cat", "when": {"class": "cat"}, "action": "copy"},
    ]})
    assert ruleset.names() == ["cat", "in-doubt"]
    assert ruleset.for_doubt().action == "move"
    assert ruleset.for_doubt().target_folder() == "_Review"


def test_the_doubt_rule_is_kept_last_and_only_once():
    ruleset = RuleSet.from_json({"version": 1, "rules": [
        {"name": "in-doubt", "when": {"in_doubt": True}, "action": "copy"},
        {"name": "cat", "when": {"class": "cat"}, "action": "copy"},
    ]})
    assert ruleset.names() == ["cat", "in-doubt"]
    assert ruleset.for_doubt().action == "copy"   # the chosen action survives


def test_the_doubt_rule_never_wins_an_ordinary_match():
    """Its condition is about the decision, not the detections, so it must not
    be able to swallow photos in the priority chain."""
    ruleset = RuleSet.starter(["cat"])
    assert ruleset.evaluate(ctx(("cat", 0.9))).name == "cat"
    assert ruleset.evaluate(ctx()).name == "none"


def test_the_doubt_rule_refuses_to_delete():
    from app.actions.registry import default_registry

    ruleset = RuleSet.from_json({"version": 1, "rules": [
        {"name": "in-doubt", "when": {"in_doubt": True}, "action": "delete"},
    ]})
    with pytest.raises(RuleError, match="must be one of"):
        ruleset.validate(default_registry().names())


def test_the_doubt_rule_cannot_be_reordered():
    ruleset = RuleSet.starter(["cat", "dog"])
    with pytest.raises(RuleError, match="always comes last"):
        reorder(ruleset, "in-doubt", -1)


def test_an_ordinary_rule_cannot_be_pushed_past_the_doubt_rule():
    ruleset = RuleSet.starter(["cat", "dog"])
    assert reorder(ruleset, "cat", 99).names() == ["cat-dog", "dog", "none", "cat", "in-doubt"]


def test_the_doubt_rule_survives_a_round_trip():
    ruleset = RuleSet.starter(["cat"])
    reloaded = RuleSet.from_json(ruleset.to_json())
    assert reloaded.names() == ruleset.names()
    assert reloaded.for_doubt() == ruleset.for_doubt()
