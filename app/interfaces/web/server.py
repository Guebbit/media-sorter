"""HTTP plumbing: sockets, routing, status codes.

Standard library only — no extra dependency, and no asset fetched from a CDN,
which keeps the "runs completely offline" promise intact. The routing table maps
a path to a `WebApi` method and stops there; nothing in this file knows what any
of those methods do.

The server is deliberately unauthenticated: it is a control panel for your own
machine, not a service. Keep it on a trusted network.
"""

from __future__ import annotations

import json
import logging
import traceback
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ...errors import MediaSortError
from ...services import AppContext, duplicates
from .api import WebApi

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
#: The two pages this server hosts. Separate documents rather than tabs of one:
#: sorting a library and de-duplicating a folder are different jobs over
#: different folders, and neither should load the other's state to be used.
PAGE = STATIC / "webui.html"
DUPES_PAGE = STATIC / "dupes.html"


class _Handler(BaseHTTPRequestHandler):
    api: WebApi  # injected by build_server()
    server_version = "mediasort"
    # Keep-alive: the UI polls every couple of seconds, and every response below
    # sets an explicit Content-Length, which HTTP/1.1 requires.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        """Route the base class's access log through our own logger instead
        of stderr."""
        log.debug("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------ utils

    def _send(self, payload: Any, status: int = 200) -> None:
        """Write `payload` as a JSON response."""
        self._respond(
            json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status
        )

    def _send_html(self, html: str) -> None:
        """Write `html` as a 200 HTML response — used for the single-page app."""
        self._respond(html.encode("utf-8"), "text/html; charset=utf-8", 200)

    def _send_image(self, body: bytes, content_type: str) -> None:
        """Write `body` as a 200 image response — the original or a thumbnail."""
        self._respond(body, content_type, 200)

    def _respond(self, body: bytes, content_type: str, status: int) -> None:
        """Write the status line, headers and body of one HTTP response."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Any:
        """The request body, parsed as JSON, or None if there wasn't one."""
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return None
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _dispatch(self, build_routes: Callable[[], dict[str, Callable[[], None]]], path: str,
                 catch: tuple[type[Exception], ...] = (MediaSortError,)) -> None:
        """Build the path->handler mapping and run whichever one matches.

        `build_routes` is called *inside* the try, not before, so a route that
        needs to read the request body (every POST route) still has its
        failures — bad JSON, a missing field — turned into the same 400
        response as everything else, rather than an uncaught traceback.
        """
        try:
            handler = build_routes().get(path)
            if handler is None:
                self._send({"error": "not found"}, 404)
                return
            handler()
        except catch as exc:
            self._send({"error": str(exc)}, 400)
        except BrokenPipeError:
            pass  # the browser navigated away mid-response
        except Exception as exc:  # noqa: BLE001
            log.error("%s %s failed: %s", self.command, path, traceback.format_exc())
            self._send({"error": str(exc)}, 500)

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:  # noqa: N802
        """Every read-only route: the page itself, plus each `WebApi` getter."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        api = self.api

        def routes() -> dict[str, Callable[[], None]]:
            """This request's path->handler mapping, built fresh each call so
            `query` is captured from the URL actually being served."""
            return {
                "/": lambda: self._send_html(_page(PAGE)),
                "/index.html": lambda: self._send_html(_page(PAGE)),
                "/duplicates": lambda: self._send_html(_page(DUPES_PAGE)),
                "/api/rules": lambda: self._send(api.get_rules()),
                "/api/meta": lambda: self._send(api.get_meta()),
                "/api/stats": lambda: self._send(api.get_stats()),
                "/api/preview": lambda: self._send(api.get_preview()),
                "/api/recheck": lambda: self._send(api.get_recheck()),
                "/api/samples": lambda: self._send(
                    api.get_samples((query.get("category") or [None])[0])
                ),
                "/api/settings": lambda: self._send(api.get_settings()),
                "/api/ollama/models": lambda: self._send(
                    api.get_vision_models((query.get("url") or [None])[0])
                ),
                "/api/duplicates": lambda: self._send(api.get_duplicate_groups(
                    int((query.get("distance") or [duplicates.DEFAULT_MAX_DISTANCE])[0]),
                    int((query.get("limit") or [200])[0]),
                )),
                "/api/dupes/config": lambda: self._send(api.get_dupes_config()),
                "/api/job": lambda: self._send(api.get_job()),
                "/api/dupes/image": lambda: self._send_image(
                    *api.dupe_image_bytes(int(query["id"][0]))
                ),
                "/api/dupes/thumbnail": lambda: self._send_image(
                    *api.dupe_thumbnail_bytes(
                        int(query["id"][0]), int((query.get("size") or [240])[0])
                    )
                ),
                "/api/image": lambda: self._send_image(
                    *api.image_bytes(int(query["id"][0]))
                ),
                "/api/thumbnail": lambda: self._send_image(
                    *api.thumbnail_bytes(
                        int(query["id"][0]), int((query.get("size") or [240])[0])
                    )
                ),
            }

        # `/api/image` and `/api/thumbnail` need a required `id` query param —
        # the first GET routes for which a missing/bad param is a client
        # mistake (400), not a server bug (500), so this catches the same
        # exceptions POST already does for a malformed body.
        self._dispatch(routes, parsed.path, catch=(MediaSortError, KeyError, ValueError))

    def do_POST(self) -> None:  # noqa: N802
        """Every mutating route: rule edits, settings edits, and starting a job."""
        parsed = urlparse(self.path)
        api = self.api

        def routes() -> dict[str, Callable[[], None]]:
            """This request's path->handler mapping, built fresh each call so
            the body is read exactly once, regardless of which route matches
            (or none does — HTTP/1.1 keep-alive still needs it consumed)."""
            payload = self._body() or {}
            return {
                "/api/rules": lambda: self._send(api.put_rules(payload)),
                "/api/rules/move": lambda: self._send(
                    api.move_rule(payload["name"], int(payload["offset"]))
                ),
                "/api/run/scan": lambda: self._send({"started": api.start_scan()}),
                "/api/run/pipeline": lambda: self._send(
                    {"started": api.start_pipeline(bool(payload.get("analyze", True)))}
                ),
                "/api/run/apply": lambda: self._send({"started": api.start_apply()}),
                "/api/run/all": lambda: self._send({"started": api.start_everything(
                    bool(payload.get("analyze", True)),
                )}),
                "/api/run/recheck": lambda: self._send({"started": api.start_recheck()}),
                "/api/dupes/scan": lambda: self._send({"started": api.start_dupes_scan()}),
                "/api/dupes/discard": lambda: self._send(
                    api.discard_marked_duplicates(bool(payload.get("delete", False)))
                ),
                "/api/duplicates/mark": lambda: self._send(
                    api.mark_duplicates(payload["image_ids"], payload.get("mark"))
                ),
                "/api/duplicates/dismiss": lambda: self._send(
                    api.dismiss_duplicates(payload["image_ids"])
                ),
                "/api/settings": lambda: self._send(api.put_settings(payload)),
                "/api/settings/reset": lambda: self._send(api.reset_settings(payload)),
            }

        self._dispatch(routes, parsed.path, catch=(MediaSortError, KeyError, ValueError))


def build_server(ctx: AppContext) -> ThreadingHTTPServer:
    """A ready-to-serve HTTP server bound to `ctx.settings.web`'s host/port,
    with one `WebApi` instance shared by every request handler."""
    api = WebApi(ctx)
    handler = type("BoundHandler", (_Handler,), {"api": api})
    server = ThreadingHTTPServer((ctx.settings.web.host, ctx.settings.web.port), handler)
    server.daemon_threads = True
    return server


def serve(ctx: AppContext) -> None:
    """Build a server and run it until Ctrl-C."""
    server = build_server(ctx)
    host, port = server.server_address[0], server.server_address[1]
    log.info("mediasort web UI on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


@lru_cache(maxsize=4)
def _page(path: Path) -> str:
    """Read a page once, on first request rather than at import time."""
    return path.read_text(encoding="utf-8")
