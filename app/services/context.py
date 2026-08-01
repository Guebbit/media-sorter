"""The composition root: the one place that decides which pieces are in play.

Every service takes a context as its first argument, so a caller (a front end, a
test) chooses the settings and the index once and then never passes them again. Wiring lives here and nowhere else — no module reaches
for a global to find its collaborators.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import ActionContext, ActionRegistry
from ..actions.registry import default_registry
from ..config import Settings, load_settings
from ..storage import RulesStore, Storage


@dataclass(frozen=True, slots=True)
class AppContext:
    settings: Settings
    storage: Storage
    rules: RulesStore
    actions: ActionRegistry

    @classmethod
    def create(cls, settings: Settings | None = None, init_storage: bool = True) -> "AppContext":
        """Assemble everything. Raises `IncompatibleIndex` on an index from an
        older schema, and `ConfigError` on an unusable setting."""
        settings = settings or load_settings()
        storage = Storage(settings.paths.database)
        if init_storage:
            storage.init()
        return cls(
            settings=settings,
            storage=storage,
            rules=RulesStore(settings.paths.rules),
            actions=default_registry(),
        )

    @property
    def action_context(self) -> ActionContext:
        """The narrow view of the world that actions are allowed to see."""
        return ActionContext(
            output=self.settings.output,
            input_roots=tuple(self.settings.library.input_folders),
        )
