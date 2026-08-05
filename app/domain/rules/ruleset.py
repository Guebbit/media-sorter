"""Rules and their priority order.

What a rule *does* is just an action name; it is resolved against the action
registry, which this layer deliberately knows nothing about beyond "here is a
list of names that exist" (see `validate`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from ...errors import RuleError
from ..detection import VIDEO_CLASS
from .conditions import (AllOf, Always, ClassBand, ClassBands, Condition, DEFAULT_CONFIDENCE,
                         DEFAULT_OLLAMA_REVIEW, DEFAULT_REVIEW_CONFIDENCE, HasClass, InDoubt,
                         MatchContext, merge_overrides, parse_condition, reject_unknown)

#: The rule for photos the detector could not settle. Always present, always
#: last, and not deletable: "what happens to the ones I am unsure about" is a
#: question every library has an answer to, and burying that answer in a setting
#: is what made it invisible before.
DOUBT_RULE_NAME = "in-doubt"
DOUBT_FOLDER = "_Review"
#: `delete` is deliberately absent. Throwing away what you were unsure about is
#: the one thing doubt should never lead to unattended.
DOUBT_ACTIONS = ("move", "copy", "ignore")
DOUBT_DEFAULT_ACTION = "move"


@dataclass(frozen=True, slots=True)
class Rule:
    """One line of the ruleset: a condition, and what to do when it matches."""

    name: str
    condition: Condition
    action: str = "copy"
    folder: str | None = None
    enabled: bool = True

    def target_folder(self) -> str:
        """Where `copy` and `move` put the file. Defaults to the rule's name."""
        return self.folder or "-".join(part.capitalize() for part in self.name.split("-"))

    def to_json(self) -> dict[str, Any]:
        """This rule as the dict shape `RuleSet.to_json`/the rules file expects."""
        node: dict[str, Any] = {
            "name": self.name,
            "when": self.condition.to_json(),
            "action": self.action,
        }
        if self.folder:
            node["folder"] = self.folder
        if not self.enabled:
            node["enabled"] = False
        return node

    @classmethod
    def from_json(cls, node: Any) -> "Rule":
        """Parse one rule from the rules file's dict shape."""
        if not isinstance(node, dict):
            raise RuleError(f"a rule must be an object, got {type(node).__name__}")
        reject_unknown(node, ("name", "when", "action", "folder", "enabled"), "a rule")
        name = node.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuleError(f"rule 'name' must be a non-empty string, got {name!r}")
        if "when" not in node:
            raise RuleError(f"rule {name!r} has no 'when' condition")
        action = node.get("action", "copy")
        if not isinstance(action, str) or not action.strip():
            raise RuleError(f"rule {name!r} has an invalid 'action': {action!r}")
        folder = node.get("folder")
        if folder is not None and (not isinstance(folder, str) or not folder.strip()):
            raise RuleError(f"rule {name!r} has an invalid 'folder': {folder!r}")
        action = action.strip().lower()
        folder = folder.strip() if folder else None
        return cls(
            name=name.strip(),
            condition=parse_condition(node["when"]),
            action=action,
            folder=folder,
            enabled=bool(node.get("enabled", True)),
        )


def doubt_rule(action: str = DOUBT_DEFAULT_ACTION, folder: str = DOUBT_FOLDER) -> Rule:
    """A fresh doubt rule — what a ruleset gets when its file has none."""
    return Rule(name=DOUBT_RULE_NAME, condition=InDoubt(), action=action, folder=folder)


def with_doubt_rule(rules: Sequence[Rule]) -> tuple[Rule, ...]:
    """The rules, with the doubt rule present exactly once and last.

    Applied on every load, so a file a person edited it out of comes back
    with it rather than silently losing the answer. Its condition is forced
    too: `in_doubt` is the only one that means anything in that slot, whatever
    the file says.
    """
    ordinary = [r for r in rules if r.name != DOUBT_RULE_NAME]
    existing = next((r for r in rules if r.name == DOUBT_RULE_NAME), None)
    if existing is None:
        return (*ordinary, doubt_rule())
    return (*ordinary, replace(existing, condition=InDoubt(),
                               folder=existing.folder or DOUBT_FOLDER))


@dataclass(frozen=True, slots=True)
class RuleSet:
    """The rules, in priority order, plus everything that reads or writes
    them as a whole: matching, serialization, validation, and the starter set."""

    rules: tuple[Rule, ...] = ()

    # ------------------------------------------------------------- evaluation

    def evaluate(self, ctx: MatchContext) -> Rule | None:
        """First enabled rule that matches wins; order is the priority.

        The doubt rule is not in this race. It answers a different question —
        what to do when the *decision* was uncertain — which no condition over
        detections can express, so the planner applies it separately.
        """
        for rule in self.rules:
            if rule.name == DOUBT_RULE_NAME:
                continue
            if rule.enabled and rule.condition.matches(ctx):
                return rule
        return None

    def for_doubt(self) -> Rule:
        """What to do with a photo nobody could settle. Always answerable."""
        for rule in self.rules:
            if rule.name == DOUBT_RULE_NAME:
                return rule
        return doubt_rule()

    def by_name(self, name: str) -> Rule | None:
        """The rule called `name`, or None if there is no such rule."""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def required_classes(self) -> set[str]:
        """The union of every rule condition's `classes()` — every detector
        class any rule in this set could possibly need."""
        return set().union(*(r.condition.classes() for r in self.rules)) if self.rules else set()

    def detector_classes(self) -> list[str]:
        """What the detector should look for: whatever the rules mention.

        There is deliberately no fallback to a configured list — one source of
        truth means the detector and the rules can never drift apart.
        """
        required = self.required_classes()
        if not required:
            raise RuleError(
                "no rule references a detector class, so there is nothing to detect. "
                "Add at least one condition like {\"class\": \"cat\"}."
            )
        return sorted(required)

    def names(self) -> list[str]:
        """Every rule's name, in priority order."""
        return [rule.name for rule in self.rules]

    def class_bands(self) -> ClassBands:
        """Every mentioned class's resolved confidence policy.

        Resolved one *field* at a time by `merge_overrides` — the same
        resolution a single condition's children get, applied across whole
        rules in priority order. Anything no rule sets at all falls back to
        `DEFAULT_BAND` rather than to a global setting.
        """
        merged = merge_overrides(
            rule.condition.class_bands()
            for rule in self.rules if rule.name != DOUBT_RULE_NAME
        )
        return ClassBands(
            (cls, ClassBand(
                confidence=DEFAULT_CONFIDENCE if confidence is None else confidence,
                review_confidence=(DEFAULT_REVIEW_CONFIDENCE if review_confidence is None
                                   else review_confidence),
                ollama_review=DEFAULT_OLLAMA_REVIEW if ollama_review is None else ollama_review,
            ))
            for cls, (confidence, review_confidence, ollama_review) in merged.items()
        )

    def needs_adjudication(self) -> bool:
        """Whether any class in this ruleset is banded for a second opinion at
        all. Not a settings flag: it is true the moment any rule opts a class
        in, so the answer follows the rules rather than shadowing them."""
        return any(band.ollama_review for band in self.class_bands().values())

    # ---------------------------------------------------------- serialisation

    def to_json(self) -> dict[str, Any]:
        """This ruleset as the dict shape the rules file stores."""
        return {"rules": [rule.to_json() for rule in self.rules]}

    @classmethod
    def from_json(cls, data: Any) -> "RuleSet":
        """Parse a rules file's contents, rejecting duplicate rule names, and
        make sure the doubt rule is present exactly once and last."""
        if not isinstance(data, dict):
            raise RuleError(f"the rules file must contain an object, got {type(data).__name__}")
        reject_unknown(data, ("rules",), "the rules file")
        raw = data.get("rules")
        if not isinstance(raw, list):
            raise RuleError("the rules file must have a 'rules' list")
        rules = tuple(Rule.from_json(node) for node in raw)
        seen: set[str] = set()
        for rule in rules:
            if rule.name in seen:
                raise RuleError(f"duplicate rule name {rule.name!r}")
            seen.add(rule.name)
        return cls(with_doubt_rule(rules))

    # ------------------------------------------------------------- validation

    def validate(self, known_actions: Iterable[str],
                 known_classes: Iterable[str] | None = None) -> None:
        """Raise `RuleError` if any rule uses an unregistered action, if the
        doubt rule uses an action outside `DOUBT_ACTIONS`, or (when
        `known_classes` is given) if a rule references a class the detector
        cannot find."""
        known_actions = set(known_actions)
        for rule in self.rules:
            if rule.action not in known_actions:
                raise RuleError(
                    f"rule {rule.name!r} uses unknown action {rule.action!r}; "
                    f"available: {', '.join(sorted(known_actions))}"
                )
            if rule.name == DOUBT_RULE_NAME and rule.action not in DOUBT_ACTIONS:
                raise RuleError(
                    f"the {DOUBT_RULE_NAME!r} rule must be one of "
                    f"{', '.join(DOUBT_ACTIONS)}, got {rule.action!r}"
                )
        if known_classes is not None:
            known = {c.lower() for c in known_classes}
            unknown = {c for c in self.required_classes() if c not in known}
            if unknown:
                raise RuleError(
                    f"rules reference classes the detector cannot find: {sorted(unknown)}"
                )
        # A condition validates its own two fields in isolation, but leaving
        # only one of them set can still produce an inverted band once the
        # other falls back to its default — e.g. `min_confidence: 0.2` alone
        # pairs with the default `min_review_confidence` of 0.35.
        for cls, band in self.class_bands().items():
            if band.review_confidence > band.confidence:
                raise RuleError(
                    f"class {cls!r} has a review confidence ({band.review_confidence}) "
                    f"above its confidence ({band.confidence}) once defaults are applied; "
                    "set both explicitly on the same condition"
                )

    # ---------------------------------------------------------------- default

    @classmethod
    def starter(cls, classes: Sequence[str]) -> "RuleSet":
        """A sensible first ruleset: everything together, then one rule each.

        Linear in the number of classes. Enumerating every combination would be
        2^n rules — 31 of them for five classes — and nobody wants to read that.
        Whatever is missing is one click away in the editor.

        `video` is pulled out of that combinatorial treatment rather than
        folded in with the rest, and placed *below* every real class: it is
        true of a file by extension, so a rule matching it would win every
        video the moment it came first, and a video of a cat would never reach
        the cat rule. Last, it is the fallback it should be — everything the
        detector found in the frames sorts by what it found, and whatever it
        found nothing in still lands in `video/` instead of the catch-all. It
        gets `move` rather than the ordinary `copy` for the same reason it
        always did: a library keeps one copy of a two-gigabyte clip.
        """
        requested = {c.lower() for c in classes if c.strip()}
        wants_video = VIDEO_CLASS in requested
        ordered = sorted(requested - {VIDEO_CLASS})
        if not ordered and not wants_video:
            raise RuleError("cannot build a starter ruleset from an empty class list")

        rules: list[Rule] = []
        if len(ordered) > 1:
            rules.append(Rule(
                name="-".join(ordered),
                condition=AllOf([HasClass(c) for c in ordered]),
                action="copy",
            ))
        rules += [Rule(name=c, condition=HasClass(c), action="copy") for c in ordered]
        if wants_video:
            rules.append(Rule(name=VIDEO_CLASS, condition=HasClass(VIDEO_CLASS),
                              action="move", folder=VIDEO_CLASS))
        rules.append(Rule(name="none", condition=Always(), action="ignore"))
        return cls(with_doubt_rule(rules))


def reorder(ruleset: RuleSet, name: str, offset: int) -> RuleSet:
    """Move a rule up or down. Kept next to the rules because priority *is*
    order, and only this layer gets to say so."""
    names = ruleset.names()
    if name not in names:
        raise RuleError(f"no rule named {name!r}")
    if name == DOUBT_RULE_NAME:
        # It is not in the priority race, so a position for it would mean
        # nothing — and one moved off the end would look like it had been lost.
        raise RuleError(f"the {DOUBT_RULE_NAME!r} rule always comes last")
    index = names.index(name)
    # Ordinary rules move among themselves; the last slot is not theirs to take.
    last = len(names) - 2 if DOUBT_RULE_NAME in names else len(names) - 1
    target = max(0, min(last, index + offset))
    rules = list(ruleset.rules)
    rules.insert(target, rules.pop(index))
    return replace(ruleset, rules=tuple(rules))
