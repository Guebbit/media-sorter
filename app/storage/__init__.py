"""The index — everything the tool remembers between runs.

One SQLite file, one `Engine` that owns connections and transactions, and a
repository per table on top of it. `Storage` is only a bundle: it wires them to
a shared engine so a caller can be handed a single object and still reach the
narrow piece it needs (`storage.images`, `storage.links`, …).
"""

from __future__ import annotations

from pathlib import Path

from .activity import ActivityLog
from .adjudications import AdjudicationRepository
from .detections import DetectionRepository
from .engine import Engine
from .images import ImageRepository, ImageRow
from .links import LinkRepository
from .metadata import MetadataRepository
from .reporting import Reporting
from .results import ResultsWriter
from .rules_store import RulesStore
from .schema import SCHEMA_VERSION
from .state import DONE, ERROR, PENDING, RUNNING, SKIPPED, Stage

__all__ = [
    "ActivityLog", "AdjudicationRepository", "DetectionRepository", "Engine",
    "ImageRepository", "ImageRow", "LinkRepository", "MetadataRepository", "Reporting",
    "ResultsWriter", "RulesStore", "SCHEMA_VERSION", "Stage", "Storage",
    "DONE", "ERROR", "PENDING", "RUNNING", "SKIPPED",
]


class Storage:
    """Every repository, wired to one shared `Engine` — the single object a
    caller holds to reach any table."""

    def __init__(self, path: Path):
        self.engine = Engine(path)
        self.images = ImageRepository(self.engine)
        self.detections = DetectionRepository(self.engine)
        self.adjudications = AdjudicationRepository(self.engine)
        self.metadata = MetadataRepository(self.engine)
        self.results = ResultsWriter(
            self.engine, self.detections, self.adjudications, self.metadata
        )
        self.links = LinkRepository(self.engine)
        self.activity = ActivityLog(self.engine)
        self.reporting = Reporting(self.engine, self.images)

    @property
    def path(self) -> Path:
        """Where the SQLite file lives."""
        return self.engine.path

    def init(self) -> None:
        """Create or verify the schema. Raises `IncompatibleIndex` on an old file."""
        self.engine.init_schema()

    def close(self) -> None:
        """Close this thread's connection."""
        self.engine.close()
