"""Commands that repair or throw away derived state: `retry`, `reset`, `clean`."""

from __future__ import annotations

import typer

from ....services import maintenance, processing
from ....storage import Stage
from ..runtime import console, load_rules, setup

app = typer.Typer()

ALL = "all"


def _stage(value: str | None) -> Stage | None:
    """`None` and "all" both mean every stage."""
    if not value or value.lower() == ALL:
        return None
    try:
        return Stage.parse(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


@app.command()
def retry(stage: str = typer.Option(
        None, "--stage", help="detect | adjudicate | analyze (default: all).")):
    """Put failed images back in the queue."""
    ctx = setup()
    requeued = maintenance.retry_errors(ctx, _stage(stage))
    console.print(f"requeued [bold]{requeued}[/bold] images")


@app.command()
def reset(
    stage: str = typer.Argument(..., help="detect | adjudicate | analyze | all"),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Mark a stage as pending again so it reruns on every image."""
    ctx = setup()
    member = _stage(stage)
    label = member.value if member else ALL
    if not yes and not typer.confirm(f"reset {label}? this reprocesses every image"):
        raise typer.Abort()
    maintenance.reset(ctx, member)

    # Dropping verdicts is not enough to requeue the images they settled: an
    # `absent` cleared the review flag, and the flag is the queue. Re-deciding
    # from the raw detections puts it back exactly as the detector left it.
    if member in (None, Stage.ADJUDICATE):
        processing.recheck(ctx, load_rules(ctx))
    console.print(f"[green]{label} reset[/green]")


@app.command()
def clean(
    links: bool = typer.Option(False, "--links", help="Delete the whole output tree."),
    missing: bool = typer.Option(False, "--missing", help="Drop rows whose file is gone."),
    vacuum: bool = typer.Option(False, "--vacuum", help="Compact the index."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
):
    """Remove generated output and stale rows. Never touches originals."""
    ctx = setup()
    if not (links or missing or vacuum):
        raise typer.BadParameter("choose at least one of --links, --missing, --vacuum")

    if links:
        target = ctx.settings.output.folder
        if not yes and not typer.confirm(f"delete everything under {target}?"):
            raise typer.Abort()
        report = maintenance.clean_links(ctx)
        for problem in report.problems:
            console.print(f"[yellow]skip {problem}[/yellow]")
        console.print(f"removed [bold]{report.removed}[/bold] entries from {target}")

    if missing:
        purged = maintenance.purge_missing(ctx)
        console.print(f"purged [bold]{purged}[/bold] rows for files no longer on disk")

    if vacuum:
        with console.status("vacuuming"):
            maintenance.vacuum(ctx)
        console.print("index compacted")
