"""Running both stages at once.

Detection streams work to analysis through the index, so the two only ever meet
as committed rows. The asymmetry in `guard` is deliberate: detection is the
source of everything, so its failure stops the run, while analysis is an
enrichment whose absence must not throw away detection work.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from .stages import StageStats
from .stopper import Stopper

log = logging.getLogger(__name__)


def run_pipeline(stages: dict[str, tuple[Callable[[], StageStats], bool]],
                 stopper: Stopper) -> dict[str, StageStats]:
    """Run named stages concurrently.

    `stages` maps a name to the work and whether its failure is fatal. Building
    that mapping is the caller's job — this function only owns the threading.
    """
    results: dict[str, StageStats] = {}
    errors: list[BaseException] = []

    def guard(name: str, fn: Callable[[], StageStats], fatal: bool) -> None:
        """Run one stage's `fn` in its own thread; on failure, stop the whole
        run if `fatal`, otherwise just log and let the others keep going."""
        try:
            results[name] = fn()
        except BaseException as exc:  # noqa: BLE001 - report it rather than dying silently
            errors.append(exc)
            log.error("%s stage aborted: %s", name, exc)
            if fatal:
                stopper.stop()
            else:
                log.warning("continuing without the %s stage", name)

    threads = [
        threading.Thread(target=guard, args=(name, fn, fatal), name=f"{name}-stage")
        for name, (fn, fatal) in stages.items()
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stopper.stop()
        for thread in threads:
            thread.join()
        raise

    if errors and not results:
        raise errors[0]
    return results
