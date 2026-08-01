"""The optional semantic pass, served by a vision-language engine."""

from __future__ import annotations

from .client import OllamaClient
from .contract import (build_prompt, build_schema, build_verdict_prompt, build_verdict_schema,
                       coerce, parse_reply, parse_verdict_reply)

__all__ = [
    "OllamaClient", "build_prompt", "build_schema", "build_verdict_prompt",
    "build_verdict_schema", "coerce", "parse_reply", "parse_verdict_reply",
]
