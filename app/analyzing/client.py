"""Talking to Ollama.

Transport only: build a request, survive a server that is still starting, hand
the reply to `contract` to make sense of. Structured output is requested via
Ollama's JSON-schema `format`, with a text-salvage fallback for models that
ignore it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import requests

from .. import imaging
from ..config import AnalyzeSettings
from ..domain.adjudication import Adjudication
from ..errors import VisionError
from .contract import (build_prompt, build_schema, build_verdict_prompt, build_verdict_schema,
                       parse_reply, parse_verdict_reply)

log = logging.getLogger(__name__)


class OllamaClient:
    """A `requests.Session` pointed at one Ollama server, plus the state
    (resolved model tag, per-run prompt/schema) that would otherwise have to be
    threaded through every call. See `_ask` for the one method that actually
    puts a request on the wire."""

    def __init__(self, settings: AnalyzeSettings, classes: Sequence[str] = ()):
        self.settings = settings
        # The rules decide which classes matter, so the schema follows them.
        self.classes = list(classes)
        self.schema = build_schema(self.classes)
        self.prompt = build_prompt(self.classes)
        self.session = requests.Session()
        # Resolved to the exact installed tag by ensure_model(); the API rejects
        # a bare name like "llava-llama3" when the store holds "llava-llama3:8b".
        self.model = settings.model

    # ------------------------------------------------------------------ setup

    def health(self) -> list[str]:
        """`GET /api/tags` — the names of every model this Ollama has pulled.
        Raises if the server cannot be reached at all; returns an empty list
        (never raises) if it is up but has nothing installed."""
        response = self.session.get(f"{self.settings.url}/api/tags", timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def wait_ready(self, timeout: float = 60.0) -> list[str]:
        """Poll `health()` every two seconds until it succeeds or `timeout`
        runs out. The engine is often still starting when the pipeline is not
        — Ollama takes a few seconds to bind its port on a cold start."""
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.health()
            except requests.RequestException as exc:
                last = exc
                log.debug("waiting for ollama at %s: %s", self.settings.url, exc)
                time.sleep(2.0)
        raise VisionError(f"ollama not reachable at {self.settings.url}: {last}")

    def ensure_model(self, wait: float = 0.0) -> str:
        """Resolve the configured model name to a tag Ollama actually has
        installed, and pin `self.model` to it — every later `_ask` call sends
        this exact string as the `model` field, and Ollama's `/api/chat`
        rejects a bare name like `"llava-llama3"` when the store holds
        `"llava-llama3:8b"`. Tries the configured name, then `:latest`, then
        any installed tag sharing the same name before the colon. Raises
        `VisionError` if nothing matches."""
        available = self.wait_ready(wait) if wait else self.health()
        wanted = self.settings.model

        for candidate in (wanted, f"{wanted}:latest"):
            if candidate in available:
                self.model = candidate
                return candidate

        # "llava-llama3" configured, "llava-llama3:8b" installed.
        matches = sorted(name for name in available if name.split(":")[0] == wanted.split(":")[0])
        if matches:
            self.model = matches[0]
            if self.model != wanted:
                log.info("resolved ollama model %r to installed tag %r", wanted, self.model)
            return self.model

        raise VisionError(
            f"model {wanted!r} not found in Ollama. Available: {', '.join(sorted(available)) or 'none'}"
        )

    # ------------------------------------------------------------------ work

    def encode_image(self, path: str) -> str:
        """The image at `path` as base64 JPEG, resized to `max_edge` — the
        exact string that goes in the request's `images` array. Downscaling
        here (not left to Ollama) keeps the upload small and the request fast
        for a model that would resize it internally anyway."""
        return imaging.to_jpeg_base64(path, self.settings.max_edge)

    def _ask(self, path: str, content: str, schema: dict[str, Any]) -> str:
        """The one place a request reaches Ollama: `POST /api/chat` with one
        user message pairing `content` (the prompt) with the base64-encoded
        image, `format` set to `schema` so Ollama constrains the model's
        output to that JSON shape, and `temperature: 0` for the most
        repeatable answer a given model can give. Returns the raw text of the
        reply — still a string, not yet parsed or validated as JSON; that is
        `contract.py`'s job. Raises `VisionError` on any transport failure
        (timeout, connection refused, non-2xx status)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content, "images": [self.encode_image(path)]}],
            "stream": False,
            "format": schema,
            "keep_alive": self.settings.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.settings.num_ctx},
        }
        try:
            response = self.session.post(
                f"{self.settings.url}/api/chat", json=payload, timeout=self.settings.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise VisionError(f"ollama request failed: {exc}") from exc

        return response.json().get("message", {}).get("content", "")

    def analyze(self, path: str, hint: str | None = None) -> dict[str, Any]:
        """The semantic pass: ask what the model sees in `path`, using the
        per-class description schema built in `__init__` from `self.classes`.
        `hint` — the detector's own boxes, already turned into text like
        `"cat (82%)"` — is appended to the prompt as a prior, not sent as
        structured data: the model is meant to describe the photo in its own
        words, not to confirm or repeat a number back."""
        content = self.prompt
        if hint:
            content += f"\nA detector already found: {hint}. Use that as a strong prior.\n"
        return parse_reply(self._ask(path, content, self.schema), self.classes)

    def adjudicate(self, path: str, classes: Sequence[str]) -> list[Adjudication]:
        """Settle whether each of `classes` is actually in this image.

        The prompt and schema are built here rather than in `__init__` because
        what is in doubt is a property of the picture, not of the ruleset — a
        photo with one borderline cat is asked one question, not five.
        """
        classes = list(classes)
        if not classes:
            return []
        text = self._ask(
            path, build_verdict_prompt(classes), build_verdict_schema(classes)
        )
        return parse_verdict_reply(text, classes)
