"""Recording the outcome of a stage — the one place that writes two tables at once.

"Detection finished for this image" means new detection rows *and* a new
decision on the image row, and a run killed between the two would leave the
index lying. Composing the repositories inside a single transaction keeps that
guarantee in one obvious place instead of duplicating it per caller.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..domain.adjudication import Adjudication
from ..domain.decision import Decision
from ..domain.detection import Detection
from .adjudications import AdjudicationRepository
from .detections import DetectionRepository
from .engine import Engine
from .metadata import MetadataRepository
from .state import DONE, PENDING, SKIPPED


class ResultsWriter:
    """Writes a stage's evidence and the image row's derived decision
    together, atomically — see the module docstring for why."""

    def __init__(self, engine: Engine, detections: DetectionRepository,
                 adjudications: AdjudicationRepository, metadata: MetadataRepository):
        self._engine = engine
        self._detections = detections
        self._adjudications = adjudications
        self._metadata = metadata

    def finish_detect(self, image_id: int, detections: Sequence[Detection], decision: Decision,
                      model: str, skip_analyze: bool = False) -> None:
        """Detection's own verdict, and whether anyone needs to second-guess it.

        Queueing the image for adjudication happens here rather than in a
        sweep afterwards, because the decision that raises the doubt is written
        in this transaction: the queue is then correct the moment the row is
        committed, instead of correct once some housekeeping call gets round to
        it. An image with no doubt skips the stage rather than sitting pending,
        which is what lets the description pass tell "settled" from "waiting".

        `skip_analyze` marks the semantic pass not applicable here and now,
        rather than pending — for a video no frame could be read from, say,
        which leaves nothing for a vision model to describe.
        `skip_analysis_for_ignored` would only catch it later, and only if its
        rule's action happens to be `ignore`.
        """
        with self._engine.write() as conn:
            self._detections.replace_for_image(image_id, detections, model)
            analyze_column = ", analyze_state=?" if skip_analyze else ""
            conn.execute(
                f"UPDATE images SET detect_state=?, adjudicate_state=?, category=?, action=?, "
                f"needs_review=?{analyze_column}, error=NULL, updated_at=? WHERE id=?",
                (
                    DONE, PENDING if decision.needs_review else SKIPPED,
                    decision.category, decision.action, int(decision.needs_review),
                    *((SKIPPED,) if skip_analyze else ()),
                    time.time(), image_id,
                ),
            )

    def finish_adjudicate(self, image_id: int, adjudications: Sequence[Adjudication],
                          decision: Decision, model: str) -> None:
        """The verdicts, and the decision they lead to, in one transaction.

        The image row is rewritten because that is the point of the stage: a
        settled `present` can move a photo out of the catch-all rule, and a
        settled `absent` clears the review flag that sent it to a human.
        """
        with self._engine.write() as conn:
            self._adjudications.replace_for_image(image_id, adjudications, model)
            conn.execute(
                "UPDATE images SET adjudicate_state=?, category=?, action=?, needs_review=?, "
                "error=NULL, updated_at=? WHERE id=?",
                (DONE, decision.category, decision.action, int(decision.needs_review),
                 time.time(), image_id),
            )

    def finish_analyze(self, image_id: int, payload: dict[str, Any], model: str) -> None:
        """Store the semantic-pass result and mark the stage done. No
        decision to rewrite — the description is search metadata, not a rule
        input, so nothing about the image's category or action changes."""
        with self._engine.write() as conn:
            self._metadata.upsert(image_id, payload, model)
            conn.execute(
                "UPDATE images SET analyze_state=?, updated_at=? WHERE id=?",
                (DONE, time.time(), image_id),
            )
