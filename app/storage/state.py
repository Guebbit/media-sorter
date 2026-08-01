"""The vocabulary of resumability: what state a row is in, per stage."""

from __future__ import annotations

import enum

# Per-stage processing state.
PENDING = 0
RUNNING = 1
DONE = 2
ERROR = 3
SKIPPED = 4  # deliberately not applicable, e.g. an ignored image needs no analysis


class Stage(str, enum.Enum):
    """The three resumable processing stages, in the order work flows through.

    Defined once so a stage name can never be a loose string that happens to
    match a column — the column is derived from the member, not typed by hand.
    """

    DETECT = "detect"
    #: The second opinion, on the images DETECT was not sure about.
    ADJUDICATE = "adjudicate"
    ANALYZE = "analyze"

    @property
    def column(self) -> str:
        """This stage's state column in the `images` table."""
        return f"{self.value}_state"

    @classmethod
    def parse(cls, value: str) -> "Stage":
        """`value` (case-insensitive) as a `Stage`, or a `ValueError` naming
        the valid options."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"unknown stage {value!r}; expected one of {', '.join(s.value for s in cls)}"
            ) from None
