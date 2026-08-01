"""The object detector, behind an adapter.

Ultralytics and torch are a heavy, optional dependency: every import of them is
lazy and inside a function, so the commands that do not detect anything still
start instantly on a machine that does not have them.
"""

from __future__ import annotations

from .availability import ensure_available, is_available
from .detector import Detector, available_classes, describe_device, resolve_model

__all__ = [
    "Detector", "available_classes", "describe_device", "ensure_available", "is_available",
    "resolve_model",
]
