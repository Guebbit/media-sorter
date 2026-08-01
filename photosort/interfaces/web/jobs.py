"""Running one long task at a time, in the background.

A browser cannot wait ten minutes for a POST, so the UI starts a job and polls.
This class is only about *that* — concurrency and reporting. What the job does
is a service call handed in from outside, and the work itself keeps its state in
the index, so a job dying loses nothing.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Callable

log = logging.getLogger(__name__)


class JobRunner:
    """One background job at a time, with enough state for a polling client
    to render a status line without ever blocking on the job itself."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state = "idle"
        # Which job the message belongs to, still set once it has finished:
        # `state` goes back to "idle" and a result with nothing to attach it to
        # is how a button comes to look like it did nothing.
        self.name = ""
        self.message = ""
        self.error = ""
        self.started_at = 0.0
        self.finished_at = 0.0

    @property
    def busy(self) -> bool:
        """Whether a job is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, name: str, target: Callable[[], str]) -> bool:
        """False when something else is already running."""
        with self._lock:
            if self.busy:
                return False
            self.state = name
            self.name = name
            self.message = f"{name} started"
            self.error = ""
            self.started_at = time.time()
            self.finished_at = 0.0

            def wrapper() -> None:
                """Run `target` on its own thread, capturing its result or
                exception into this runner's state instead of letting either
                escape onto a thread nothing is watching."""
                try:
                    self.message = target() or f"{name} finished"
                except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                    self.error = str(exc)
                    self.message = f"{name} failed"
                    log.error("job %s failed: %s", name, traceback.format_exc())
                finally:
                    self.finished_at = time.time()
                    self.state = "idle"

            self._thread = threading.Thread(target=wrapper, name=f"job-{name}", daemon=True)
            self._thread.start()
            return True

    def snapshot(self) -> dict[str, Any]:
        """The current job state as plain data, for `WebApi.get_stats`."""
        busy = self.busy
        started = self.started_at
        return {
            "busy": busy, "state": self.state, "name": self.name,
            "message": self.message, "error": self.error,
            "started_at": started,
            "finished_at": self.finished_at,
            # Server-side, because a browser clock does not agree with ours and a
            # job that outlives the page still has to report how long it has run.
            "elapsed": (time.time() if busy else self.finished_at) - started if started else 0.0,
        }

    def join(self, timeout: float = 30.0) -> None:
        """Block until the current job finishes. For tests and shutdown."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
