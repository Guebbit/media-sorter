"""The composition root: the one place that decides which pieces are in play.

Every service takes a context as its first argument, so a caller (a front end, a
test) chooses the settings and the index once and then never passes them again. Wiring lives here and nowhere else — no module reaches
for a global to find its collaborators.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..actions import ActionContext, ActionRegistry
from ..actions.registry import default_registry
from ..config import Settings, load_settings
from ..storage import RulesStore, Storage


@dataclass(frozen=True, slots=True)
class AppContext:
    settings: Settings
    storage: Storage
    #: The duplicate finder's own index, over its own folders. A second
    #: `Storage` rather than a second table: it answers a different question
    #: about a different folder set, so nothing it records should ever reach
    #: the sorting pipeline's counters, and nothing the pipeline scans should
    #: turn up in a duplicate review.
    dupes: Storage
    rules: RulesStore
    actions: ActionRegistry

    @classmethod
    def create(cls, settings: Settings | None = None, init_storage: bool = True) -> "AppContext":
        """Assemble everything. Raises `ConfigError` on an unusable setting; an
        index from an older schema is rebuilt rather than refused."""
        settings = settings or load_settings()
        storage = Storage(settings.paths.database)
        dupes = Storage(settings.dupes.database)
        if init_storage:
            storage.init()
            dupes.init()
        return cls(
            settings=settings,
            storage=storage,
            dupes=dupes,
            rules=RulesStore(settings.paths.rules),
            actions=default_registry(),
        )

    def reloaded(self, settings: Settings) -> "AppContext":
        """This context rebuilt around `settings` — what a front end swaps in
        after a settings edit.

        Here rather than in the front end because it is the same wiring
        `create` does, and a caller that assembled it itself would be a second
        place that has to know a changed `paths.rules` means a new
        `RulesStore`. Front ends decide *when* to reload; what a reload
        consists of is this module's business.

        The index is kept rather than reopened unless it has actually moved.
        It can: turning "remember this library" on or off, or changing the
        photo folders, changes where the index belongs — so the comparison is
        the real test, not a formality.
        """
        storage = self.storage
        if settings.paths.database != storage.path:
            storage = Storage(settings.paths.database)
            storage.init()
            self.storage.close()
        # Same test for the duplicate index, which moves for its own reasons:
        # its folder changed, or "remember the duplicate scan" was toggled.
        dupes = self.dupes
        if settings.dupes.database != dupes.path:
            dupes = Storage(settings.dupes.database)
            dupes.init()
            self.dupes.close()
        return replace(
            self, settings=settings, storage=storage, dupes=dupes,
            rules=RulesStore(settings.paths.rules),
        )

    @property
    def action_context(self) -> ActionContext:
        """The narrow view of the world that actions are allowed to see."""
        return ActionContext(
            output=self.settings.output,
            input_roots=tuple(self.settings.library.input_folders),
        )
