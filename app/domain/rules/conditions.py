"""The condition language: what a rule is allowed to ask about an image.

A condition is a small composable tree, so new operators are added by writing a
new `Condition` subclass and registering its parser — nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ...errors import RuleError
from ..detection import Detection

#: What a class condition gets when it does not set its own band. Replaces the
#: old global `DETECT_CONFIDENCE`/`REVIEW_CONFIDENCE`/`ADJUDICATE_ENABLED`
#: settings — there is no machine-wide right answer for these any more than
#: there is for the class itself, so the fallback lives in code, not config.
DEFAULT_CONFIDENCE = 0.65
DEFAULT_REVIEW_CONFIDENCE = 0.35
DEFAULT_OLLAMA_REVIEW = True


@dataclass(frozen=True, slots=True)
class ClassBand:
    """One class's resolved confidence policy: the auto-pass threshold, the
    review-band floor below it, and whether a borderline hit gets asked to
    Ollama at all."""

    confidence: float
    review_confidence: float
    ollama_review: bool


@dataclass(frozen=True, slots=True)
class MatchContext:
    """Everything a condition is allowed to look at.

    Conditions receive raw detections rather than pre-aggregated counts so that
    a single rule can use its own confidence threshold without the engine
    having to guess which thresholds will be needed.
    """

    detections: tuple[Detection, ...]
    default_confidence: float

    def count(self, cls: str, min_confidence: float | None = None) -> int:
        """How many detections of `cls` meet `min_confidence` (or the
        rule-wide default confidence, if not given)."""
        threshold = self.default_confidence if min_confidence is None else min_confidence
        return sum(1 for d in self.detections if d.cls == cls and d.confidence >= threshold)

    def best(self, cls: str) -> float:
        """The highest confidence seen for `cls`, or 0.0 if it wasn't detected."""
        return max((d.confidence for d in self.detections if d.cls == cls), default=0.0)

    def total(self, min_confidence: float | None = None) -> int:
        """How many detections of any class meet `min_confidence`."""
        threshold = self.default_confidence if min_confidence is None else min_confidence
        return sum(1 for d in self.detections if d.confidence >= threshold)


class Condition(ABC):
    """A boolean test over a MatchContext."""

    @abstractmethod
    def matches(self, ctx: MatchContext) -> bool:
        """Whether this condition holds for `ctx`."""

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """This condition as the dict shape the rules file stores."""

    def classes(self) -> set[str]:
        """Which detector classes this condition needs. Drives what the detector
        looks for."""
        return set()

    def class_bands(self) -> dict[str, tuple[float | None, float | None, bool | None]]:
        """Which classes this condition sets a confidence policy for, and what
        it asked for — `None` in a slot means "use the default." Only `HasClass`
        actually answers; everything else unions or delegates to its children,
        so a rule's per-class overrides surface regardless of how deep in an
        `all_of`/`any_of` tree the `class` node sits."""
        return {}

    def describe(self) -> str:
        """Human-readable summary, for `rules show` and the editor."""
        return self.__class__.__name__.lower()


@dataclass(frozen=True, slots=True)
class HasClass(Condition):
    """At least `min_count` detections of `cls`, each at or above `min_confidence`.

    `min_review_confidence` and `ollama_review` do not affect matching at
    all — `matches` only ever asks about `min_confidence`, same as before.
    They are read separately, by `RuleSet.class_bands()`, to decide what
    happens to a detection of `cls` that falls short of `min_confidence`:
    whether it is worth a second opinion, and from how low a score.
    """

    cls: str
    min_count: int = 1
    min_confidence: float | None = None
    min_review_confidence: float | None = None
    ollama_review: bool | None = None

    def matches(self, ctx: MatchContext) -> bool:
        return ctx.count(self.cls, self.min_confidence) >= self.min_count

    def to_json(self) -> dict[str, Any]:
        node: dict[str, Any] = {"class": self.cls}
        if self.min_count != 1:
            node["min_count"] = self.min_count
        if self.min_confidence is not None:
            node["min_confidence"] = self.min_confidence
        if self.min_review_confidence is not None:
            node["min_review_confidence"] = self.min_review_confidence
        if self.ollama_review is not None:
            node["ollama_review"] = self.ollama_review
        return node

    def classes(self) -> set[str]:
        return {self.cls}

    def class_bands(self) -> dict[str, tuple[float | None, float | None, bool | None]]:
        return {self.cls: (self.min_confidence, self.min_review_confidence, self.ollama_review)}

    def describe(self) -> str:
        text = f"{self.min_count}+ {self.cls}" if self.min_count > 1 else self.cls
        if self.min_confidence is not None:
            text = f"{text} @{self.min_confidence:.2f}"
        if self.min_review_confidence is not None:
            text = f"{text} (review @{self.min_review_confidence:.2f})"
        if self.ollama_review is not None:
            text = f"{text} [ollama {'on' if self.ollama_review else 'off'}]"
        return text


@dataclass(frozen=True, slots=True)
class AnyDetection(Condition):
    """Matches when anything at all was detected — useful for a catch-all."""

    min_count: int = 1
    min_confidence: float | None = None

    def matches(self, ctx: MatchContext) -> bool:
        return ctx.total(self.min_confidence) >= self.min_count

    def to_json(self) -> dict[str, Any]:
        node: dict[str, Any] = {"any_detection": True}
        if self.min_count != 1:
            node["min_count"] = self.min_count
        if self.min_confidence is not None:
            node["min_confidence"] = self.min_confidence
        return node

    def describe(self) -> str:
        return f"any {self.min_count}+ detection"


class _Composite(Condition):
    """Shared plumbing for the boolean operators."""

    key = ""

    def __init__(self, children: Sequence[Condition]):
        """Reject an empty operator at construction time — an `all_of` with
        nothing to require, or an `any_of` with nothing that could satisfy
        it, is a rule author's mistake, not a valid ruleset."""
        if not children:
            raise RuleError(f"{self.key!r} needs at least one child condition")
        self.children = tuple(children)

    def to_json(self) -> dict[str, Any]:
        return {self.key: [child.to_json() for child in self.children]}

    def classes(self) -> set[str]:
        return set().union(*(child.classes() for child in self.children))

    def class_bands(self) -> dict[str, tuple[float | None, float | None, bool | None]]:
        """Every child's bands, merged left to right, one field at a time.

        A child that mentions a class without overriding a field must not
        shadow a later child's override of that same field — see
        `RuleSet.class_bands()`, which resolves the same way across whole
        rules for the identical reason.
        """
        confidences: dict[str, float] = {}
        reviews: dict[str, float] = {}
        ollama_reviews: dict[str, bool] = {}
        classes: set[str] = set()
        for child in self.children:
            for cls, (confidence, review_confidence, ollama_review) in child.class_bands().items():
                classes.add(cls)
                if confidence is not None:
                    confidences.setdefault(cls, confidence)
                if review_confidence is not None:
                    reviews.setdefault(cls, review_confidence)
                if ollama_review is not None:
                    ollama_reviews.setdefault(cls, ollama_review)
        return {
            cls: (confidences.get(cls), reviews.get(cls), ollama_reviews.get(cls))
            for cls in classes
        }

    def describe(self) -> str:
        joiner = {"all_of": " and ", "any_of": " or ", "none_of": " nor "}[self.key]
        inner = joiner.join(child.describe() for child in self.children)
        return f"no {inner}" if self.key == "none_of" else inner

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.children == self.children

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.children))


class AllOf(_Composite):
    """Matches only when every child condition matches."""

    key = "all_of"

    def matches(self, ctx: MatchContext) -> bool:
        return all(child.matches(ctx) for child in self.children)


class AnyOf(_Composite):
    """Matches when at least one child condition matches."""

    key = "any_of"

    def matches(self, ctx: MatchContext) -> bool:
        return any(child.matches(ctx) for child in self.children)


class NoneOf(_Composite):
    """Matches only when no child condition matches."""

    key = "none_of"

    def matches(self, ctx: MatchContext) -> bool:
        return not any(child.matches(ctx) for child in self.children)


@dataclass(frozen=True, slots=True)
class Not(Condition):
    """Inverts `child`."""

    child: Condition

    def matches(self, ctx: MatchContext) -> bool:
        return not self.child.matches(ctx)

    def to_json(self) -> dict[str, Any]:
        return {"not": self.child.to_json()}

    def classes(self) -> set[str]:
        return self.child.classes()

    def class_bands(self) -> dict[str, tuple[float | None, float | None, bool | None]]:
        return self.child.class_bands()

    def describe(self) -> str:
        return f"not ({self.child.describe()})"


@dataclass(frozen=True, slots=True)
class Always(Condition):
    """Matches every image — the catch-all rule's condition."""

    def matches(self, ctx: MatchContext) -> bool:
        return True

    def to_json(self) -> dict[str, Any]:
        return {"always": True}

    def describe(self) -> str:
        return "anything"


@dataclass(frozen=True, slots=True)
class InDoubt(Condition):
    """The detector was not sure — and neither was the second opinion.

    Never true here: doubt is a property of the *decision*, recorded on the image
    row as `needs_review`, not something a condition can read off the detections.
    It is a condition at all so that the doubt rule can live in the rules list
    and be edited like any other; the planner is what actually applies it, and
    `RuleSet.evaluate` skips it so it can never win the ordinary match.
    """

    def matches(self, ctx: MatchContext) -> bool:
        return False

    def to_json(self) -> dict[str, Any]:
        return {"in_doubt": True}

    def describe(self) -> str:
        return "the detector was unsure"


# ------------------------------------------------------------------- parsing

# Registry keyed by the JSON field that identifies the node type. Adding an
# operator means adding one entry here.
_PARSERS: dict[str, Callable[[dict[str, Any]], Condition]] = {}


def register_condition(key: str, parser: Callable[[dict[str, Any]], Condition]) -> None:
    """Make `parser` handle any condition node containing the field `key`."""
    _PARSERS[key] = parser


def _parse_class(node: dict[str, Any]) -> Condition:
    """A `{"class": ...}` node (or its string shorthand) into a `HasClass`."""
    name = node.get("class")
    if not isinstance(name, str) or not name.strip():
        raise RuleError(f"'class' must be a non-empty string, got {name!r}")
    min_count = node.get("min_count", 1)
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise RuleError(f"'min_count' must be an integer >= 1, got {min_count!r}")
    confidence = node.get("min_confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise RuleError(f"'min_confidence' must be between 0 and 1, got {confidence!r}")
        confidence = float(confidence)
    review_confidence = node.get("min_review_confidence")
    if review_confidence is not None:
        if (not isinstance(review_confidence, (int, float))
                or not 0.0 <= float(review_confidence) <= 1.0):
            raise RuleError(
                f"'min_review_confidence' must be between 0 and 1, got {review_confidence!r}"
            )
        review_confidence = float(review_confidence)
        if confidence is not None and review_confidence > confidence:
            raise RuleError(
                f"'min_review_confidence' must be <= 'min_confidence', "
                f"got {review_confidence!r} > {confidence!r}"
            )
    ollama_review = node.get("ollama_review")
    if ollama_review is not None and not isinstance(ollama_review, bool):
        raise RuleError(f"'ollama_review' must be a boolean, got {ollama_review!r}")
    return HasClass(name.strip().lower(), min_count, confidence, review_confidence, ollama_review)


def _parse_any_detection(node: dict[str, Any]) -> Condition:
    """A `{"any_detection": true, ...}` node into an `AnyDetection`, reusing
    `_parse_class`'s `min_count`/`min_confidence` validation."""
    base = _parse_class({"class": "_", **{k: v for k, v in node.items() if k != "any_detection"}})
    assert isinstance(base, HasClass)
    return AnyDetection(base.min_count, base.min_confidence)


def _composite_parser(factory: Callable[[Sequence[Condition]], Condition], key: str):
    """A parser for a boolean-operator node: reads the list under `key` and
    recursively parses each child before handing them to `factory`."""
    def parse(node: dict[str, Any]) -> Condition:
        """Build the composite condition `factory` produces from `node`."""
        children = node.get(key)
        if not isinstance(children, list):
            raise RuleError(f"{key!r} must be a list of conditions")
        return factory([parse_condition(child) for child in children])

    return parse


register_condition("class", _parse_class)
register_condition("any_detection", _parse_any_detection)
register_condition("all_of", _composite_parser(AllOf, "all_of"))
register_condition("any_of", _composite_parser(AnyOf, "any_of"))
register_condition("none_of", _composite_parser(NoneOf, "none_of"))
register_condition("not", lambda node: Not(parse_condition(node["not"])))
register_condition("always", lambda node: Always())
register_condition("in_doubt", lambda node: InDoubt())


def parse_condition(node: Any) -> Condition:
    """A JSON condition node (or its bare-string shorthand) into a `Condition`,
    dispatching on whichever registered key the node contains."""
    if isinstance(node, str):  # shorthand: "cat" == {"class": "cat"}
        return _parse_class({"class": node})
    if not isinstance(node, dict):
        raise RuleError(f"a condition must be an object or a class name, got {type(node).__name__}")
    for key, parser in _PARSERS.items():
        if key in node:
            return parser(node)
    raise RuleError(
        f"unknown condition {sorted(node)!r}; expected one of {sorted(_PARSERS)}"
    )
