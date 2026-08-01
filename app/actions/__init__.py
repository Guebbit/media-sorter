"""What happens to an image once a rule matches it.

Three separable things, one per module:

  `base`      what an action *is* — the interface, its inputs, its plan
  `builtin`   the four that ship, each a self-contained strategy
  `planner`   turning the whole index into a list of intended actions
  `executor`  carrying that list out, idempotently

Adding "move", "tag" or "upload" means writing one class in (or outside) this
package and registering it: the rules engine, the CLI and the web UI all pick it
up without modification.
"""

from __future__ import annotations

from .base import Action, ActionContext, ImageTarget, NameAllocator, PlannedAction
from .builtin import CopyAction, DeleteAction, IgnoreAction, MoveAction
from .executor import ApplyStats, apply_actions
from .planner import plan_all
from .registry import ActionRegistry, default_registry

__all__ = [
    "Action", "ActionContext", "ActionRegistry", "ApplyStats", "DeleteAction", "IgnoreAction",
    "CopyAction", "ImageTarget", "MoveAction", "NameAllocator", "PlannedAction",
    "apply_actions", "default_registry", "plan_all",
]
