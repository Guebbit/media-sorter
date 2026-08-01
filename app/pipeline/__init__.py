"""Driving the three heavy stages.

They are independent loops over the index rather than in-memory queues. That
gives the same "detection never waits for anything downstream" behaviour while
surviving a kill -9: all handoff state is committed rows, so resuming is just
running the same command again. Work flows detect → adjudicate → analyze, and
each stage claims what the one before it has committed.

The stages talk to inference engines through the protocols in `ports`, and are
handed factories rather than engines — which is why an unavailable Ollama can
fail inside its own thread without taking the detection work down with it.
"""

from __future__ import annotations

from .ports import AdjudicatorEngine, DetectorEngine, VisionEngine
from .runner import run_pipeline
from .stages import StageStats, run_adjudicate_stage, run_analyze_stage, run_detect_stage
from .stopper import Stopper

__all__ = [
    "AdjudicatorEngine", "DetectorEngine", "StageStats", "Stopper", "VisionEngine",
    "run_adjudicate_stage", "run_analyze_stage", "run_detect_stage", "run_pipeline",
]
