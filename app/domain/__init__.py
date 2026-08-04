"""The problem, described without reference to how anything is stored or run.

Nothing in this package performs I/O, reads the environment or imports a
third-party library. That is the point: what a rule means, and what should
happen to a photo given a set of detections, is decided here and is testable
with plain values. Every other layer depends on this one; this one depends on
nobody.
"""

from __future__ import annotations

from .adjudication import (ABSENT, PRESENT, UNSURE, VERDICTS, Adjudication, adjudicated,
                           uncertain_classes)
from .decision import Decision, decide
from .detection import Detection
from .similarity import cluster, hamming

__all__ = [
    "ABSENT", "PRESENT", "UNSURE", "VERDICTS", "Adjudication", "Decision", "Detection",
    "adjudicated", "cluster", "decide", "hamming", "uncertain_classes",
]
