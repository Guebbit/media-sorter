"""Where the rules file lives.

The rules themselves are domain logic and know nothing about files; persisting
them is this module's job. That separation is why the CLI and the web UI cannot
disagree about what "the rules" are: both go through one store.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain.rules import RuleSet
from ..errors import RuleError


class RulesStore:
    """File-backed persistence for one rules file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def exists(self) -> bool:
        """Whether the rules file has been written yet."""
        return self.path.exists()

    def load(self) -> RuleSet:
        """Read and parse the rules file. `RuleError` on bad JSON or bad rules."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuleError(f"{self.path} is not valid JSON: {exc}") from exc
        return RuleSet.from_json(data)

    def save(self, ruleset: RuleSet) -> None:
        """Write `ruleset` out atomically, via a temp file plus rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written via a temp file so a crash mid-write cannot leave the editor
        # with an unparseable rules file.
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(ruleset.to_json(), indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)
