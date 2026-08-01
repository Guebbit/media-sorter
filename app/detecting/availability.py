"""Is the detector usable in this environment?

A separate question from "does it work", and one the front ends need to answer
before doing anything expensive. The message names the extra to install and
nothing about how the user chose to run the tool: whether that means a pip
install, an image rebuild or something else is not this layer's business.
"""

from __future__ import annotations

from ..errors import EngineUnavailable


def is_available() -> bool:
    """Whether `ultralytics` (and therefore `torch`) can be imported at all.

    Only an import probe — this never loads a model or touches a GPU, so it is
    cheap enough to call before every command that might need the detector.
    """
    try:
        import ultralytics  # noqa: F401  # probe only
    except ImportError:
        return False
    return True


def ensure_available() -> None:
    """Like `is_available`, but raises with an actionable message instead of
    returning False — for a command that is about to fail anyway if the
    dependency is missing, and would rather say why up front."""
    try:
        import ultralytics  # noqa: F401  # probe only
    except ImportError as exc:
        raise EngineUnavailable(
            f"the detection dependencies are not installed ({exc}). "
            "They are the `detect` extra: ultralytics and torch."
        ) from None
