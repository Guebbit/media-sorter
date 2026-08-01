"""What an action is: its inputs, its plan, and the contract it implements.

Actions are planned before anything touches the filesystem, so `--dry-run`
reports exactly what a real run would do rather than guessing along a second
code path.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import OutputSettings
from ..domain.rules import Rule

if TYPE_CHECKING:
    from ..storage import ImageRepository


@dataclass(frozen=True, slots=True)
class ActionContext:
    """Everything an action is allowed to know about the world.

    Narrower than the full settings on purpose: an action can decide where to
    put a file and how, and cannot reach the detector, the index or the network.
    """

    output: OutputSettings
    #: Used to mirror the library's structure inside the trash folder.
    input_roots: tuple[Path, ...] = ()


@dataclass(slots=True)
class ImageTarget:
    """The subject of an action: one indexed image plus its rule outcome."""

    image_id: int
    path: str
    filename: str
    category: str | None
    needs_review: bool


@dataclass(slots=True)
class PlannedAction:
    """One concrete, not-yet-executed step: what to do, to which file, and
    (for actions that produce output) where. The unit both `--dry-run` and a
    real `apply` work from."""

    image_id: int
    action: str
    source: str
    target: str | None = None
    detail: str = ""

    def describe(self) -> str:
        """One-line human summary, for `--dry-run` output."""
        return f"{self.action}: {self.source}" + (f" -> {self.target}" if self.target else "")


class NameAllocator:
    """Hands out collision-free output paths.

    Deliberately ignores what is already on disk: names must depend only on the
    index, otherwise a second run would rename the links it created itself.
    """

    def __init__(self) -> None:
        self._taken: set[str] = set()

    def allocate(self, directory: Path, filename: str, image_id: int) -> str:
        """`filename` in `directory`, or a name disambiguated with `image_id`
        if this allocator already handed that path out."""
        candidate = str(directory / filename)
        if candidate in self._taken:
            stem, suffix = os.path.splitext(filename)
            candidate = str(directory / f"{stem}_{image_id}{suffix}")
        self._taken.add(candidate)
        return candidate


class Action(ABC):
    """A named strategy: plan first, execute second."""

    name: str = ""
    #: True when the original does not stay where it is. Such actions need an
    #: explicit opt-in before they run, and are never mirrored into the review
    #: folder — see `planner.plan_all`.
    #:
    #: Not called "destructive": that is true of `delete` and false of `move`,
    #: which loses nothing, and a user asked to confirm a destructive action
    #: they know to be safe learns to distrust the warning.
    consumes_original: bool = False
    #: What this action does to the original, in the words shown to whoever has
    #: to confirm it. Each action says it for itself, because "destructive" is
    #: the wrong summary of at least one of them.
    consequence: str = ""
    #: Whether this action's output is recorded and pruned when it no longer
    #: belongs. Lets the executor ask the action instead of comparing action
    #: names to the string "link".
    tracked: bool = False
    description: str = ""

    @abstractmethod
    def plan(self, ctx: ActionContext, target: ImageTarget, rule: Rule,
             namer: NameAllocator) -> list[PlannedAction]:
        """What this action would do to `target`, without doing it yet.
        Zero or more steps — `ignore` plans nothing, `move` plans one."""

    @abstractmethod
    def execute(self, ctx: ActionContext, planned: PlannedAction) -> bool:
        """Carry out one step `plan` produced. True on success, False when
        there was nothing to do (already done, or the source vanished)."""

    def settle(self, planned: PlannedAction, images: ImageRepository) -> None:
        """Reconcile the index with whatever `execute` did to the original.

        Only actions that touch the original need this; the default is to do
        nothing, which is right for anything that merely adds to the output tree.
        """
