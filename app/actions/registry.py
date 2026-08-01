"""The catalogue of available actions.

A registry rather than a hard-coded list because three separate things need to
enumerate actions — rule validation, the CLI's `rules actions`, the web UI's
picker — and none of them should carry its own copy.
"""

from __future__ import annotations

from .base import Action


class ActionRegistry:
    """A lookup of `Action`s by name — what the planner, the executor, rule
    validation and both front ends all share instead of hardcoding the list."""

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        """Add `action` under its own name, replacing any previous entry."""
        self._actions[action.name] = action

    def get(self, name: str) -> Action:
        """The action called `name`, or `KeyError` naming what is available."""
        try:
            return self._actions[name]
        except KeyError:
            raise KeyError(
                f"unknown action {name!r}; available: {', '.join(sorted(self._actions))}"
            ) from None

    def names(self) -> list[str]:
        """Every registered action's name, alphabetical."""
        return sorted(self._actions)

    def catalog(self) -> list[dict]:
        """Everything a UI needs to offer a choice of action."""
        return [
            {
                "name": a.name,
                "consumes_original": a.consumes_original,
                "consequence": a.consequence,
                "description": a.description,
            }
            for a in sorted(self._actions.values(), key=lambda a: a.name)
        ]

    def consuming_names(self) -> set[str]:
        """The actions after which the original is no longer where it was."""
        return {a.name for a in self._actions.values() if a.consumes_original}


def default_registry() -> ActionRegistry:
    """A fresh registry holding the four built-in actions — the one place a
    deployment adding a custom action would also register it."""
    from .builtin import CopyAction, DeleteAction, IgnoreAction, MoveAction

    registry = ActionRegistry()
    for action in (CopyAction(), MoveAction(), DeleteAction(), IgnoreAction()):
        registry.register(action)
    return registry
