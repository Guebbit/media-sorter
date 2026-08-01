"""Commands that only look: `stats`, `doctor`, `verify`, `duplicates`, `history`,
`search`, `export`."""

from __future__ import annotations

from pathlib import Path

import typer

from ....services import diagnostics, exporting, insights, maintenance
from .. import render
from ..runtime import console, load_rules, setup

app = typer.Typer()


@app.command()
def stats(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Show pipeline progress and category counts."""
    ctx = setup(verbose)
    render.overview(insights.overview(ctx))


@app.command()
def doctor():
    """Check configuration, GPU, model weights, rules and engine connectivity."""
    ctx = setup()
    render.checks(diagnostics.doctor(ctx))


@app.command()
def verify(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Check that indexed originals and generated links still exist."""
    ctx = setup(verbose)
    with console.status("verifying"):
        report = maintenance.verify(ctx)
    render.verify_report(report)


@app.command()
def duplicates(limit: int = typer.Option(20, "--limit", "-n")):
    """List files sharing a SHA256 hash."""
    ctx = setup()
    groups = insights.duplicates(ctx, limit)
    if not groups:
        console.print("[green]no duplicates[/green]")
        return
    render.duplicates(groups)


@app.command()
def history(limit: int = typer.Option(30, "--limit", "-n")):
    """Show recent actions that moved or removed an original."""
    ctx = setup()
    entries = insights.history(ctx, limit)
    if not entries:
        console.print("[green]no originals have been moved or removed[/green]")
        return
    render.history(entries)


@app.command()
def search(
    term: str = typer.Argument(..., help="Substring matched against the analysis metadata."),
    limit: int = typer.Option(50, "--limit", "-n"),
):
    """Free-text search over stored analysis metadata."""
    ctx = setup()
    ruleset = load_rules(ctx)
    results = insights.search(ctx, term, ruleset, limit)
    if not results:
        console.print(f"[yellow]no matches for {term!r}[/yellow]")
        return
    render.search_results(results)


@app.command()
def export(
    output: Path = typer.Option(None, "--out", "-o", help="Defaults to <output folder>/export.json."),
    fmt: str = typer.Option("json", "--format", "-f", help="json | csv"),
    category: str = typer.Option(None, "--category", help="Only this category."),
):
    """Dump the index (including analysis metadata) to JSON or CSV."""
    ctx = setup()
    if fmt not in exporting.FORMATS:
        raise typer.BadParameter(f"format must be one of {', '.join(exporting.FORMATS)}")
    destination = output or ctx.settings.output.folder / f"export.{fmt}"
    written = exporting.export(ctx, destination, fmt, category)
    console.print(f"wrote [bold]{written}[/bold] records to {destination}")
