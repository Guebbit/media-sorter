"""The `rules` sub-app: inspect, seed and check the ruleset."""

from __future__ import annotations

import typer

from ....errors import RuleError
from ....services import rules as rules_service
from .. import render
from ..runtime import console, load_rules, require_detector, setup

app = typer.Typer(help="Inspect and manage the matching rules.", no_args_is_help=True)


def _show(ctx, ruleset) -> None:
    """Render `ruleset`'s table, shared by `show` and `init`."""
    try:
        classes = ruleset.detector_classes()
    except RuleError as exc:
        # Still worth printing the table: seeing the rules is how you fix this.
        classes = [f"[red]{exc}[/red]"]
    render.ruleset(ruleset, ctx.actions.consuming_names(), ctx.rules.path, classes)


@app.command("show")
def show():
    """Print the active rules in evaluation order."""
    ctx = setup()
    _show(ctx, load_rules(ctx))


@app.command("init")
def init(
    classes: str = typer.Option(None, "--classes", help="Comma-separated, e.g. cat,dog,bird."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing rules file."),
):
    """Write a starter rules file from a list of classes."""
    ctx = setup()
    try:
        ruleset = rules_service.init_ruleset(ctx, classes, force=force)
    except RuleError as exc:
        console.print(f"[yellow]{exc} — pass --force to replace it[/yellow]")
        raise typer.Exit(code=1) from None
    console.print(f"wrote [bold]{len(ruleset.rules)}[/bold] rules to {ctx.rules.path}")
    _show(ctx, ruleset)


@app.command("validate")
def validate(
    check_classes: bool = typer.Option(
        False, "--check-classes", help="Also confirm the model can detect every class used."
    ),
):
    """Check the rules file parses and refers to real actions (and classes)."""
    ctx = setup()
    if check_classes:
        require_detector()
    try:
        ruleset = rules_service.validate_ruleset(ctx, check_classes=check_classes)
    except RuleError as exc:
        console.print(f"[red]invalid:[/red] {exc}")
        raise typer.Exit(code=1) from None
    console.print(f"[green]ok[/green] — {len(ruleset.rules)} rules")


@app.command("actions")
def actions():
    """List the actions a rule can use."""
    ctx = setup()
    render.action_catalog(ctx.actions.catalog())


@app.command("classes")
def classes():
    """List every class a rule can match: the configured detector's, plus `video`."""
    ctx = setup()
    require_detector()
    names = rules_service.detector_classes(ctx)
    console.print(f"[bold]{len(names)}[/bold] classes: {ctx.settings.detect.model} plus `video`:")
    console.print("  " + ", ".join(names))
