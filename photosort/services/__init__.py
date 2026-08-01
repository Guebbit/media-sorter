"""Use cases: everything the tool can do, stated once.

This is the layer both front ends call. A command like "apply the rules" exists
here exactly once and returns plain data; the CLI renders it as a table and the
web UI serialises it as JSON. Neither contains a decision, and adding a third
front end would not need a fourth copy.

No module here imports rich, typer or http.server — if a service ever needs to
know how it is being displayed, something has been put in the wrong place.
"""

from __future__ import annotations

from .context import AppContext

__all__ = ["AppContext"]
