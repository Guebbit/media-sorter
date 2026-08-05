"""The four actions that ship with the tool.

Each is self-contained: it decides what it wants done and expresses it through
`app.filesystem`, so none of them contains an `os` call and all of them
can be planned without side effects.

Each is also named for exactly what it does to the photo — copy, move, delete,
ignore. There is no separate setting deciding what one of them means, because a
rule that reads `copy` and then hardlinks is a rule nobody can check by reading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .. import filesystem
from ..domain.rules import Rule
from .base import Action, ActionContext, ImageTarget, NameAllocator, PlannedAction

if TYPE_CHECKING:
    from ..storage import ImageRepository

log = logging.getLogger(__name__)


def plan_copy(target: ImageTarget, directory: Path, namer: NameAllocator) -> PlannedAction:
    """One copy into a given folder. Also used by the planner's review mirror."""
    return PlannedAction(
        image_id=target.image_id,
        action=CopyAction.name,
        source=target.path,
        target=namer.allocate(directory, target.filename, target.image_id),
    )


# ------------------------------------------------------------------ copy


class CopyAction(Action):
    """Duplicate the original into the rule's folder; nothing consumed."""

    name = "copy"
    tracked = True
    description = "Duplicate the photo into a folder. The original stays where it is."

    def plan(self, ctx: ActionContext, target: ImageTarget, rule: Rule,
             namer: NameAllocator) -> list[PlannedAction]:
        """One copy, into the rule's folder."""
        # Review mirroring is added once by the planner, for every action.
        return [plan_copy(target, ctx.output.folder / rule.target_folder(), namer)]

    def execute(self, ctx: ActionContext, planned: PlannedAction) -> bool:
        """Write the copy to disk. Always succeeds unless the filesystem itself fails."""
        filesystem.copy(planned.target or "", planned.source)
        return True


# ------------------------------------------------------------------ move


class MoveAction(Action):
    """Relocate the original into the rule's folder; nothing left behind."""

    name = "move"
    consumes_original = True
    consequence = "moves the original"
    description = "Relocate the original into the folder. Nothing is left behind."

    def plan(self, ctx: ActionContext, target: ImageTarget, rule: Rule,
             namer: NameAllocator) -> list[PlannedAction]:
        """One move into the rule's folder, or none if it is already there."""
        destination = namer.allocate(
            ctx.output.folder / rule.target_folder(), target.filename, target.image_id
        )
        # A second run finds the photo already filed. There is no link record to
        # recognise it by — the file *is* the output — so the check is the path
        # itself. The name stays reserved either way, so nothing else claims it.
        if Path(destination) == Path(target.path):
            return []
        return [PlannedAction(target.image_id, self.name, target.path, destination)]

    def execute(self, ctx: ActionContext, planned: PlannedAction) -> bool:
        """Move the file. False if the source is already gone."""
        source = Path(planned.source)
        if not source.exists():
            return False
        # `move` may disambiguate against a file that was already there, so the
        # path it returns — not the planned one — is where the photo now lives.
        planned.target = str(
            filesystem.move(source, planned.target or "", disambiguate_with=planned.image_id)
        )
        return True

    def settle(self, planned: PlannedAction, images: ImageRepository) -> None:
        """Update the index to the file's new location."""
        images.relocate(planned.image_id, planned.target or "")


# ------------------------------------------------------------------ delete


def plan_delete(image_id: int, path: str, ctx: ActionContext) -> PlannedAction:
    """One deletion: a mirrored path in the trash folder.

    Mirrors the original folder structure inside the trash so two files with
    the same name cannot overwrite each other. Module-level like `plan_copy`,
    for the same reason: a manual, one-off discard (`services.duplicates`)
    needs to plan the same move without a `Rule` to plan it against.
    """
    relative = filesystem.relative_to_any(path, ctx.input_roots)
    # No `detail`: it is only ever read as a fallback for a missing `target`
    # (see `executor._execute`), and a deletion always has one — the trash path
    # the file is about to occupy, which is the more useful thing to log anyway.
    return PlannedAction(
        image_id, DeleteAction.name, path, str(ctx.output.trash_folder / relative)
    )


class DeleteAction(Action):
    """Remove the original — to the trash folder, so it stays recoverable."""

    name = "delete"
    consumes_original = True
    consequence = "removes the original"
    description = "Remove the original. Moves it to a trash folder."

    def plan(self, ctx: ActionContext, target: ImageTarget, rule: Rule,
             namer: NameAllocator) -> list[PlannedAction]:
        """One deletion, via `plan_delete`."""
        return [plan_delete(target.image_id, target.path, ctx)]

    def execute(self, ctx: ActionContext, planned: PlannedAction) -> bool:
        """Move to trash. False if already gone."""
        source = Path(planned.source)
        if not source.exists():
            return False
        filesystem.move(source, planned.target or "", disambiguate_with=planned.image_id)
        return True

    def settle(self, planned: PlannedAction, images: ImageRepository) -> None:
        """Flag the image deleted in the index."""
        images.mark_deleted(planned.image_id)


# ------------------------------------------------------------------ ignore


class IgnoreAction(Action):
    """Do nothing to the original; a no-op plan and a no-op execute."""

    name = "ignore"
    description = "Do nothing. The image stays indexed and searchable."

    def plan(self, ctx: ActionContext, target: ImageTarget, rule: Rule,
             namer: NameAllocator) -> list[PlannedAction]:
        """Nothing to do."""
        return []

    def execute(self, ctx: ActionContext, planned: PlannedAction) -> bool:
        """Nothing to do; never called since `plan` never produces a step."""
        return True
