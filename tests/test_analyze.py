"""The Ollama client, against a fake server — no model required."""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from app.analyzing import OllamaClient, build_schema
from app.config import load_settings
from app.domain.adjudication import ABSENT, PRESENT, Adjudication
from app.errors import VisionError


class FakeOllama:
    """Records requests and replies with whatever the test queued up."""

    def __init__(self):
        self.requests: list[dict] = []
        self.replies: list[str] = []
        self.models = ["llava-llama3:8b", "llava:13b"]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._json({"models": [{"name": name} for name in outer.models]})

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                reply = outer.replies.pop(0) if outer.replies else "{}"
                self._json({"message": {"content": reply}})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.shutdown()


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    yield server
    server.stop()


@pytest.fixture
def client(env, fake_ollama, monkeypatch):
    monkeypatch.setenv("MEDIASORT_OLLAMA_URL", fake_ollama.url)
    return OllamaClient(load_settings().analyze, ["cat", "dog"])


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), (20, 40, 60)).save(path)
    return path


GOOD = json.dumps({
    "cat_count": 2, "dog_count": 1, "cat_kinds": ["Maine Coon"], "dog_kinds": ["Husky"],
    "activities": ["sleeping"], "environment": "living room", "is_main_subject": True,
    "quality": 9, "notes": "cosy",
})


# --------------------------------------------------------------------- schema


def test_schema_follows_the_classes():
    schema = build_schema(["cat", "dog"])
    assert {"cat_count", "cat_kinds", "dog_count", "dog_kinds"} <= set(schema["properties"])
    assert "bird_count" in build_schema(["bird"])["properties"]


def test_schema_always_asks_for_the_shared_fields():
    properties = build_schema(["cat"])["properties"]
    assert {"activities", "environment", "is_main_subject", "quality", "notes"} <= set(properties)


# ------------------------------------------------------------- model resolution


def test_bare_name_resolves_to_the_installed_tag(client):
    """Ollama 404s on 'llava-llama3' when the store holds 'llava-llama3:8b'."""
    assert client.ensure_model() == "llava-llama3:8b"
    assert client.model == "llava-llama3:8b"


def test_exact_tag_is_kept(env, fake_ollama, monkeypatch):
    monkeypatch.setenv("MEDIASORT_OLLAMA_URL", fake_ollama.url)
    monkeypatch.setenv("MEDIASORT_OLLAMA_MODEL", "llava:13b")
    assert OllamaClient(load_settings().analyze).ensure_model() == "llava:13b"


def test_missing_model_is_reported(env, fake_ollama, monkeypatch):
    monkeypatch.setenv("MEDIASORT_OLLAMA_URL", fake_ollama.url)
    monkeypatch.setenv("MEDIASORT_OLLAMA_MODEL", "qwen2.5-vl")
    with pytest.raises(VisionError, match="not found"):
        OllamaClient(load_settings().analyze).ensure_model()


def test_unreachable_server_raises(env, monkeypatch):
    monkeypatch.setenv("MEDIASORT_OLLAMA_URL", "http://127.0.0.1:1")
    with pytest.raises(VisionError, match="not reachable"):
        OllamaClient(load_settings().analyze).wait_ready(timeout=0.1)


# ------------------------------------------------------------------- encoding


def test_large_images_are_downscaled(client, photo):
    encoded = client.encode_image(str(photo))
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert max(image.size) == client.settings.max_edge
    assert abs(image.width / image.height - 4 / 3) < 0.01


# ------------------------------------------------------------------ responses


def test_clean_json_is_parsed(client, fake_ollama, photo):
    fake_ollama.replies.append(GOOD)
    client.ensure_model()
    result = client.analyze(str(photo))
    assert result["cat_count"] == 2
    assert result["cat_kinds"] == ["Maine Coon"]
    assert result["quality"] == 9


def test_the_request_carries_image_schema_and_hint(client, fake_ollama, photo):
    fake_ollama.replies.append(GOOD)
    client.ensure_model()
    client.analyze(str(photo), hint="cat (95%)")
    request = fake_ollama.requests[-1]
    assert request["format"]["type"] == "object"
    assert len(request["messages"][0]["images"]) == 1
    assert "cat (95%)" in request["messages"][0]["content"]
    assert request["model"] == "llava-llama3:8b"


def test_prose_wrapped_json_is_salvaged(client, fake_ollama, photo):
    fake_ollama.replies.append("Sure!\n```json\n" + GOOD + "\n```\nHope that helps!")
    client.ensure_model()
    assert client.analyze(str(photo))["cat_count"] == 2


def test_sloppy_types_are_coerced(client, fake_ollama, photo):
    fake_ollama.replies.append(json.dumps({
        "cat_count": "3", "dog_count": None, "cat_kinds": "Siamese", "dog_kinds": None,
        "activities": None, "environment": None, "quality": 42,
    }))
    client.ensure_model()
    result = client.analyze(str(photo))
    assert result["cat_count"] == 3            # string -> int
    assert result["dog_count"] == 0            # None -> 0
    assert result["cat_kinds"] == ["Siamese"]  # string -> list
    assert result["quality"] == 10             # clamped
    assert result["is_main_subject"] is False     # missing -> default


def test_non_json_reply_raises(client, fake_ollama, photo):
    fake_ollama.replies.append("I cannot analyze images.")
    client.ensure_model()
    with pytest.raises(VisionError, match="no JSON"):
        client.analyze(str(photo))


def test_json_array_reply_raises(client, fake_ollama, photo):
    fake_ollama.replies.append("[1, 2, 3]")
    client.ensure_model()
    with pytest.raises(VisionError):
        client.analyze(str(photo))


# ---------------------------------------------------------------- adjudicate


def test_adjudicate_asks_only_about_the_classes_handed_to_it(client, fake_ollama, photo):
    fake_ollama.replies.append('{"cat_verdict": "present", "cat_confidence": 0.8}')
    verdicts = client.adjudicate(str(photo), ["cat"])

    assert verdicts == [Adjudication("cat", PRESENT, 0.8)]
    request = fake_ollama.requests[-1]
    # The schema follows the question, not the ruleset: `dog` is in the client's
    # classes and must not be asked about here.
    assert set(request["format"]["required"]) == {"cat_verdict", "cat_confidence"}
    assert "dog" not in request["messages"][0]["content"]


def test_adjudicate_with_nothing_in_doubt_never_calls_the_model(client, fake_ollama, photo):
    assert client.adjudicate(str(photo), []) == []
    assert fake_ollama.requests == []


def test_a_model_that_rambles_still_yields_a_verdict(client, fake_ollama, photo):
    fake_ollama.replies.append(
        'Looking closely, I can see:\n{"cat_verdict": "absent", "cat_confidence": 0.7}\nHope that helps!'
    )
    assert client.adjudicate(str(photo), ["cat"])[0].verdict == ABSENT
