"""Assembling the command line.

Commands are declared in `commands/*`, one module per group; this module only
decides which exist and in what order they appear in `--help`. Registration is
by splicing each module's declarations in, so the commands stay flat
(`photosort scan`) while their code stays grouped.
"""

from __future__ import annotations

import typer

from .commands import apply, config, insight, library, maintain, process, rules, web

app = typer.Typer(
    name="photosort",
    help="Offline photo sorter: scan -> detect -> your rules -> optional AI analysis -> actions.",
    no_args_is_help=True,
    add_completion=False,
)

#: In the order a user meets them.
COMMAND_MODULES = (library, process, apply, web, insight, maintain)

for module in COMMAND_MODULES:
    app.registered_commands += module.app.registered_commands

app.add_typer(rules.app, name="rules")
app.add_typer(config.app, name="config")


def main() -> None:
    """The `photosort` console-script entry point."""
    app()


if __name__ == "__main__":
    main()
