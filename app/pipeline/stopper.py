"""Cooperative shutdown.

One flag shared by every stage: Ctrl-C, or one stage failing fatally, means the
others finish their current batch and stop. Nothing is lost because nothing is
in flight that is not also a claimed row.
"""

from __future__ import annotations

import threading


class Stopper:
    """A shared, thread-safe "please wind down" flag, checked between batches."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def stop(self) -> None:
        """Ask every stage watching this stopper to finish its current batch
        and return."""
        self._event.set()

    @property
    def stopped(self) -> bool:
        """Whether `stop()` has been called."""
        return self._event.is_set()

    def wait(self, seconds: float) -> bool:
        """Sleep, but wake immediately if someone asks us to stop."""
        return self._event.wait(seconds)
