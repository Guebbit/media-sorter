"""The local web UI: a rules editor and a control panel.

Two halves, on purpose. `api` is the operations, as methods returning data;
`server` is sockets and routing. The split is what lets the whole interface be
tested without opening a port.
"""

from __future__ import annotations

from .api import WebApi
from .jobs import JobRunner
from .server import build_server, serve

__all__ = ["JobRunner", "WebApi", "build_server", "serve"]
