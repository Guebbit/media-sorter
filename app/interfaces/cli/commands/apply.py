"""`apply` — execute the rule actions."""

from __future__ import annotations

import typer

from ....services import applying
from .. import render
from ..runtime import OUTPUT_OPTION, console, load_rules, setup

app = typer.Typer()


@app.command()
def apply(
    output_folder: str = OUTPUT_OPTION,
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Show what would happen."),
    no_prune: bool = typer.Option(False, "--no-prune", help="Keep links that no longer match."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Execute the rule actions: build the link tree, run moves and deletions."""
    ctx = setup(verbose, output_folder=output_folder)
    ruleset = load_rules(ctx)

    with console.status("planning actions"):
        stats, planned = applying.apply(
            ctx, ruleset, prune=not no_prune, dry_run=dry_run
        )

    if dry_run:
        render.dry_run(stats, planned)
    else:
        render.apply_result(stats, ctx.settings.output.folder)
