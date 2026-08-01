"""`scan` — index new or changed images."""

from __future__ import annotations

import typer

from ....services import library
from .. import render
from ..runtime import INPUT_OPTION, console, progress, require_folders, setup

app = typer.Typer()


@app.command()
def scan(
    input_folders: list[str] = INPUT_OPTION,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Walk the input folders and index new or changed images."""
    ctx = setup(verbose, input_folders=input_folders)
    require_folders(ctx)
    folders = ", ".join(str(p) for p in ctx.settings.library.input_folders)
    console.print(f"[dim]inputs:[/dim] {folders}")

    with progress() as bar:
        task = bar.add_task("scanning", total=None)
        stats = library.scan_library(
            ctx, lambda s: bar.update(task, completed=s.seen), authoritative=not input_folders
        )
        bar.update(task, total=stats.seen, completed=stats.seen)

    render.scan_stats(stats)
