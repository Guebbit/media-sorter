"""The stage commands: `detect`, `adjudicate`, `analyze`, `run`, `recheck`."""

from __future__ import annotations

from typing import Callable

import typer

from ....pipeline import Stopper
from ....pipeline.stages import StageStats
from ....services import applying, insights, library, processing
from ....storage import Stage
from .. import render
from ..runtime import (INPUT_OPTION, OUTPUT_OPTION, console, install_stop_handlers, load_rules,
                       progress, require_detector, require_folders, setup)

app = typer.Typer()


def _run_stage(label: str, pending: int, empty_message: str,
               run_fn: Callable[[Stopper, Callable[[str, int, int], None]], StageStats],
               ) -> StageStats | None:
    """The shape every single-stage command shares: bail out with a friendly
    message if there is nothing to do, otherwise wire up a `Stopper` and a
    progress bar around `run_fn` and print the result. Returns the stats (or
    None, if nothing ran) so a caller like `adjudicate` can still act on
    `stats.verdicts` afterwards."""
    if not pending:
        console.print(f"[green]{empty_message}[/green]")
        return None
    stopper = Stopper()
    install_stop_handlers(stopper)
    with progress() as bar:
        task = bar.add_task(label, total=pending)
        stats = run_fn(stopper, lambda stage, delta, remaining: bar.advance(task, delta))
    render.stage_result(label, stats.processed, stats.errors, stats.categories)
    return stats


@app.command()
def detect(
    limit: int = typer.Option(0, "--limit", "-n", help="Stop after N images (0 = all)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run detection on every image that has not been through the detector yet."""
    ctx = setup(verbose)
    require_detector()
    ruleset = load_rules(ctx)

    # Deferred: importing `detecting` pulls in torch, which every command that
    # never touches the detector would otherwise pay to load at startup.
    from ....detecting import describe_device

    console.print(
        f"[dim]device:[/dim] {describe_device(ctx.settings.detect)}  "
        f"[dim]model:[/dim] {ctx.settings.detect.model}  "
        f"[dim]looking for:[/dim] {', '.join(ruleset.detector_classes())}"
    )

    pending = ctx.storage.images.count_pending(Stage.DETECT)
    _run_stage(
        "detect", pending, "nothing to detect",
        lambda stopper, on_progress: processing.detect(
            ctx, ruleset, stopper, on_progress, limit=limit or None
        ),
    )


@app.command()
def adjudicate(
    limit: int = typer.Option(0, "--limit", "-n", help="Stop after N images (0 = all)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ask the vision model about the images the detector was unsure of.

    Confirms or rejects each borderline detection and re-decides the photo. What
    neither model can settle keeps its review flag, which is what puts it in
    front of a person.
    """
    ctx = setup(verbose)
    ruleset = load_rules(ctx)
    if not ruleset.needs_adjudication():
        console.print("[yellow]no rule asks for a second opinion — nothing to do[/yellow]")
        return
    console.print(
        f"[dim]engine:[/dim] {ctx.settings.analyze.url}  "
        f"[dim]model:[/dim] {ctx.settings.analyze.model}"
    )

    ctx.storage.images.sync_adjudication_queue()
    pending = ctx.storage.images.count_pending(Stage.ADJUDICATE)
    stats = _run_stage(
        "adjudicate", pending, "nothing to adjudicate — the detector was sure every time",
        lambda stopper, on_progress: processing.adjudicate(
            ctx, ruleset, stopper, on_progress, limit=limit or None
        ),
    )
    if stats and stats.verdicts:
        render.verdicts(stats.verdicts)


@app.command()
def analyze(
    limit: int = typer.Option(0, "--limit", "-n", help="Stop after N images (0 = all)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the semantic pass on images the rules did not ignore."""
    ctx = setup(verbose)
    ruleset = load_rules(ctx)
    console.print(
        f"[dim]engine:[/dim] {ctx.settings.analyze.url}  "
        f"[dim]model:[/dim] {ctx.settings.analyze.model}"
    )

    pending = ctx.storage.images.count_pending(Stage.ANALYZE)
    _run_stage(
        "analyze", pending, "nothing to analyze",
        lambda stopper, on_progress: processing.analyze(
            ctx, ruleset, stopper, on_progress, limit=limit or None
        ),
    )


@app.command()
def run(
    input_folders: list[str] = INPUT_OPTION,
    output_folder: str = OUTPUT_OPTION,
    no_analyze: bool = typer.Option(False, "--no-analyze", help="Detection only."),
    no_apply: bool = typer.Option(False, "--no-apply", help="Skip running the rule actions."),
    skip_scan: bool = typer.Option(False, "--skip-scan", help="Reuse the existing index."),
    yes: bool = typer.Option(False, "--yes", "-y",
                             help="Also run actions that move or remove originals."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Full pipeline: scan, then detect and analyze in parallel, then apply the rules."""
    ctx = setup(verbose, input_folders=input_folders, output_folder=output_folder)
    if not skip_scan:
        require_folders(ctx)
    require_detector()
    ruleset = load_rules(ctx)
    stopper = Stopper()
    install_stop_handlers(stopper)

    if not skip_scan:
        with progress() as bar:
            task = bar.add_task("scanning", total=None)
            stats = library.scan_library(
                ctx, lambda s: bar.update(task, completed=s.seen), authoritative=not input_folders
            )
            bar.update(task, total=stats.seen, completed=stats.seen)
        render.scan_summary(stats)

    counts = ctx.storage.reporting.pending_counts()
    with_analyze = not no_analyze

    with progress() as bar:
        tasks = {Stage.DETECT.value: bar.add_task("detect", total=counts["detect_pending"])}
        # Both downstream totals grow as the detector finds things; seed them
        # with what we already know and correct them as the stages report.
        if ruleset.needs_adjudication():
            tasks[Stage.ADJUDICATE.value] = bar.add_task(
                "adjudicate", total=counts["adjudicate_pending"] or None)
        if with_analyze:
            tasks[Stage.ANALYZE.value] = bar.add_task(
                "analyze", total=counts["analyze_pending"] or None)

        def on_progress(stage: str, delta: int, remaining: int) -> None:
            """Advance the named stage's bar; downstream bars also grow as
            the upstream stage discovers more work for them (detect's own
            total is fixed up front, so it never needs correcting)."""
            task = tasks.get(stage)
            if task is None:
                return
            bar.advance(task, delta)
            if stage != Stage.DETECT.value:
                bar.update(task, total=bar.tasks[task].completed + remaining)

        results = processing.run_all(ctx, ruleset, stopper, on_progress, with_analyze=with_analyze)

    for name, stats in results.items():
        render.stage_result(name, stats.processed, stats.errors)
        if stats.verdicts:
            render.verdicts(stats.verdicts)

    if not no_apply and not stopper.stopped:
        stats, planned = applying.apply(ctx, ruleset, confirmed=yes)
        render.apply_result(stats, ctx.settings.output.folder)
    render.overview(insights.overview(ctx))


@app.command()
def recheck(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Re-apply the rules to the detections and verdicts already in the index.

    Instant and GPU-free — the right command after editing rules.
    """
    ctx = setup(verbose)
    ruleset = load_rules(ctx)
    total = len(ctx.storage.images.iter_decided_ids())

    with progress() as bar:
        task = bar.add_task("re-evaluating", total=total)
        outcome = processing.recheck(
            ctx, ruleset, on_progress=lambda done: bar.update(task, completed=done)
        )

    console.print(f"re-evaluated [bold]{sum(outcome['categories'].values())}[/bold] images")
    render.overview(insights.overview(ctx))
