"""What we ask a vision model for, and what we accept back.

Pure functions over data: no HTTP, no configuration. Separating the contract
from the transport means the prompt and the schema can be checked directly, and
that "the model returned something sloppy" is handled in one place rather than
wherever a request happens to be made.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ..domain.adjudication import ABSENT, PRESENT, UNSURE, VERDICTS, Adjudication
from ..errors import VisionError

PROMPT_TEMPLATE = """Analyze this image. It has been detected as containing: {subjects}.

Rules:
- Return JSON only. No prose, no markdown, no explanation.
- Counts are integers. If unsure, use your best estimate.
- kinds: the specific type or breed, only when you are reasonably confident,
  otherwise an empty list.
- quality is an integer 0-10 judging sharpness, framing and lighting.
- is_main_subject is true when {subjects} is what the photo is *of*, false when
  it merely happens to be in frame.
"""

#: The fields every analysis reply carries regardless of which classes were
#: asked about: each entry's JSON-schema fragment and its coercion function, so
#: adding or changing a shared field means editing exactly one line instead of
#: three (the schema, the `required` list and the `coerce` assignment used to
#: be kept in sync by hand).
_SHARED_FIELD_SPECS: tuple[tuple[str, dict[str, Any], Any], ...] = (
    ("activities", {"type": "array", "items": {"type": "string"}},
     lambda v: _as_str_list(v)),
    ("environment", {"type": "string"}, lambda v: str(v or "").strip()),
    ("is_main_subject", {"type": "boolean"}, lambda v: bool(v)),
    ("quality", {"type": "integer", "minimum": 0, "maximum": 10},
     lambda v: max(0, min(10, _as_int(v)))),
    ("notes", {"type": "string"}, lambda v: str(v or "").strip()),
)
SHARED_FIELDS = tuple(name for name, _, _ in _SHARED_FIELD_SPECS)

#: The adjudication prompt asks the opposite question to the one above: not
#: "describe what is there", but "is it there at all". The detector's own guess
#: is deliberately withheld — quoting a number the model is being asked to check
#: turns a second opinion into an echo.
VERDICT_TEMPLATE = """Look at this image and decide, for each subject below, whether it is there.

Subjects: {subjects}

An object detector thought it might have seen one, but was not confident enough
to say. You are the second opinion, and your answer decides what happens to this
photo.

Rules:
- Return JSON only. No prose, no markdown, no explanation.
- "{present}" only if you can actually see at least one, clearly enough to name it.
- "{absent}" only if you are confident there is none in the picture.
- "{unsure}" whenever you genuinely cannot tell — a bad angle, too dark, too
  small, partly hidden, or it could just as easily be something else.
- "{unsure}" is a real answer and the right one when in doubt. An unsure photo is
  set aside for a person to look at, which is far better than a confident guess
  that is wrong.
- confidence is 0.0 to 1.0: how sure you are of the verdict you just gave.
"""


def _list_subjects(classes: Sequence[str], sep: str) -> str:
    """Human-readable subject list, falling back to a generic noun when the
    rules mention no classes at all (still a valid, if useless, ruleset)."""
    return sep.join(classes) or "a subject"


def build_prompt(classes: Sequence[str]) -> str:
    """The user-message text for the semantic pass — `PROMPT_TEMPLATE` with
    `classes` filled in as the subjects the detector already found."""
    return PROMPT_TEMPLATE.format(subjects=_list_subjects(classes, " or "))


def build_verdict_prompt(classes: Sequence[str]) -> str:
    """The user-message text for adjudication — `VERDICT_TEMPLATE` with
    `classes` (the ones in doubt for this image) and the three verdict
    strings filled in, so the model echoes back exactly the words
    `parse_verdict_reply` knows how to read."""
    return VERDICT_TEMPLATE.format(
        subjects=_list_subjects(classes, ", "),
        present=PRESENT, absent=ABSENT, unsure=UNSURE,
    )


def build_verdict_schema(classes: Sequence[str]) -> dict[str, Any]:
    """One verdict and one confidence per class under question.

    Built per image rather than per client, because the classes in doubt are a
    property of what the detector saw in *that* picture.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for cls in classes:
        properties[f"{cls}_verdict"] = {"type": "string", "enum": list(VERDICTS)}
        properties[f"{cls}_confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
        required += [f"{cls}_verdict", f"{cls}_confidence"]
    return {"type": "object", "properties": properties, "required": required}


def parse_verdict_reply(text: str, classes: Sequence[str]) -> list[Adjudication]:
    """Read a verdict reply, defaulting anything unusable to `unsure`.

    A model that answers in a shape we cannot read has told us nothing, and
    "nothing" is `unsure` — which routes the photo to a human. Inventing
    `absent` from a parse failure would silently drop it instead.
    """
    try:
        data = _load_object(text)
    except VisionError:
        data = {}
    return [_verdict_for(data, cls) for cls in classes]


def _verdict_for(data: dict[str, Any], cls: str) -> Adjudication:
    """One class's verdict out of a parsed reply, tolerating the model
    answering `"yes"`/`"no"` instead of the schema's own vocabulary."""
    raw = str(data.get(f"{cls}_verdict") or "").strip().lower()
    # Small models like to answer the question rather than the schema.
    aliases = {"yes": PRESENT, "true": PRESENT, "no": ABSENT, "false": ABSENT,
               "none": ABSENT, "maybe": UNSURE, "unknown": UNSURE, "": UNSURE}
    verdict = raw if raw in VERDICTS else aliases.get(raw, UNSURE)
    confidence = _as_float(data.get(f"{cls}_confidence"))
    return Adjudication(cls=cls, verdict=verdict, confidence=max(0.0, min(1.0, confidence)))


def _as_float(value: Any) -> float:
    """`value` as a float, or 0.0 for anything a model returned that isn't one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_schema(classes: Sequence[str]) -> dict[str, Any]:
    """The schema follows whatever classes the rules care about.

    Adding `bird` to a rule automatically asks the model for `bird_count` and
    `bird_kinds` — there is no second list to maintain. Field names stay
    subject-neutral so the tool works for anything the detector can find.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for cls in classes:
        properties[f"{cls}_count"] = {"type": "integer", "minimum": 0}
        properties[f"{cls}_kinds"] = {"type": "array", "items": {"type": "string"}}
        required += [f"{cls}_count", f"{cls}_kinds"]

    for name, schema, _coerce in _SHARED_FIELD_SPECS:
        properties[name] = schema
    required += list(SHARED_FIELDS)
    return {"type": "object", "properties": properties, "required": required}


def parse_reply(text: str, classes: Sequence[str]) -> dict[str, Any]:
    """Read a reply, salvaging a model that ignored "JSON only"."""
    return coerce(_load_object(text), classes)


def _load_object(text: str) -> dict[str, Any]:
    """The first JSON object in a reply, however much commentary surrounds it."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise VisionError(f"model returned no JSON: {text[:200]!r}") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VisionError(f"model returned invalid JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise VisionError(f"expected a JSON object, got {type(data).__name__}")
    return data


def coerce(data: dict[str, Any], classes: Sequence[str]) -> dict[str, Any]:
    """Normalise the shape so downstream queries can trust it."""
    result: dict[str, Any] = {}
    for cls in classes:
        result[f"{cls}_count"] = _as_int(data.get(f"{cls}_count"))
        result[f"{cls}_kinds"] = _as_str_list(data.get(f"{cls}_kinds"))
    for name, _schema, coerce_value in _SHARED_FIELD_SPECS:
        result[name] = coerce_value(data.get(name))
    return result


def _as_int(value: Any) -> int:
    """`value` as an int (via float, so a model's `"3.0"` still counts), or 0
    for anything unusable."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_str_list(value: Any) -> list[str]:
    """`value` coerced to a list of non-empty strings: a single string becomes
    a one-item list, anything else unusable becomes an empty one — a model
    asked for a list sometimes answers with a bare string instead."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]
