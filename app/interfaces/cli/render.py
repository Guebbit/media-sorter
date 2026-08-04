"""Turning service results into terminal output.

The only module in the CLI that knows what a table looks like. Every function
here takes plain data and returns nothing — so a command is always "call a
service, hand the result to a renderer".
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from rich.table import Table

from ...actions import ApplyStats, PlannedAction
from ...domain.rules import RuleSet
from ...scanning import ScanStats
from ...services.diagnostics import Check
from ...services.maintenance import VerifyReport
from .runtime import console

DRY_RUN_SAMPLE = 15

STATUS_STYLE = {"ok": "green", "fail": "red", "warn": "yellow", "info": ""}


def _pairs_table(rows: Iterable[tuple[str, Any]], **kwargs) -> Table:
    """A borderless two-column label/value table — the shape most of the
    simpler renderers below share."""
    table = Table(box=None, **kwargs)
    table.add_column("", style="dim")
    table.add_column("", justify="right")
    for label, value in rows:
        table.add_row(label, str(value))
    return table


def scan_stats(stats: ScanStats) -> None:
    """The `scan` command's full breakdown table."""
    console.print(_pairs_table([
        ("files seen", stats.seen), ("new", stats.added), ("changed", stats.changed),
        ("unchanged", stats.unchanged), ("unreadable", stats.errors),
        ("missing (gone from disk)", stats.missing),
    ]))


def scan_summary(stats: ScanStats) -> None:
    """The one-line version of `scan_stats`, printed before `run`'s pipeline."""
    console.print(
        f"indexed [bold]{stats.seen}[/bold] files "
        f"({stats.added} new, {stats.changed} changed, {stats.unchanged} unchanged)"
    )


def stage_result(name: str, processed: int, errors: int,
                 categories: Mapping[str, int] | None = None) -> None:
    """One stage command's summary line, plus its category tally if it has one."""
    console.print(f"{name}: processed [bold]{processed}[/bold], errors {errors}")
    if categories:
        console.print("  " + _tally(dict(sorted(categories.items()))))


def verdicts(counts: Mapping[str, int]) -> None:
    """What the second opinion actually settled — the only honest measure of
    whether escalating bought anything."""
    console.print("  [dim]second opinion:[/dim] " + _tally(dict(sorted(counts.items()))))


def dry_run(stats: ApplyStats, planned: Sequence[PlannedAction]) -> None:
    """`apply --dry-run`'s output: a per-action tally, a sample of the actual
    planned steps, and a note about anything whose original is already gone."""
    table = Table(title="dry run — nothing was changed", box=None, title_justify="left")
    table.add_column("action", style="bold")
    table.add_column("images", justify="right")
    for name, count in sorted(stats.by_action.items()):
        table.add_row(name, str(count))
    console.print(table)
    for item in planned[:DRY_RUN_SAMPLE]:
        console.print(f"  [dim]{item.describe()}[/dim]")
    if len(planned) > DRY_RUN_SAMPLE:
        console.print(f"  [dim]… and {len(planned) - DRY_RUN_SAMPLE} more[/dim]")
    if stats.skipped:
        console.print(
            f"[yellow]{stats.skipped} action(s) would be skipped — their original is "
            f"already gone from where the index expects it[/yellow]"
        )


def apply_result(stats: ApplyStats, output_folder: Any) -> None:
    """`apply`'s one-line, color-coded summary of what actually happened."""
    parts = [
        f"copies: [bold]{stats.created}[/bold] new", f"{stats.existing} unchanged",
        f"{stats.pruned} pruned",
    ]
    if stats.moved:
        parts.append(f"[cyan]{stats.moved} moved[/cyan]")
    if stats.deleted:
        parts.append(f"[red]{stats.deleted} deleted[/red]")
    if stats.skipped:
        parts.append(f"[yellow]{stats.skipped} skipped[/yellow]")
    if stats.errors:
        parts.append(f"[red]{stats.errors} errors[/red]")
    console.print(", ".join(parts) + f"  [dim]-> {output_folder}[/dim]")


def overview(data: Mapping[str, Any]) -> None:
    """`stats`'s full picture: per-stage progress, category/action tallies,
    detection counts, and the index-wide totals."""
    counts = data["pipeline"]
    table = Table(title="pipeline", box=None, title_justify="left")
    table.add_column("stage", style="bold")
    for column in ("done", "pending", "errors"):
        table.add_column(column, justify="right")
    for stage in ("detect", "adjudicate", "analyze"):
        table.add_row(
            stage,
            str(counts[f"{stage}_done"]),
            str(counts[f"{stage}_pending"]),
            str(counts[f"{stage}_error"]),
        )
    console.print(table)

    categories = Table(title="categories", box=None, title_justify="left")
    categories.add_column("category", style="bold")
    categories.add_column("images", justify="right")
    for name, value in data["categories"].items():
        categories.add_row(name, str(value))
    console.print(categories)

    if data["actions"]:
        console.print("actions: " + _tally(data["actions"]))
    if data["classes"]:
        console.print("detections: " + _tally(data["classes"]))
    console.print(
        f"[dim]indexed {counts['total']} images, {counts['missing']} missing, "
        f"{counts['deleted']} deleted, {counts['review']} flagged for review[/dim]"
    )


def _tally(counts: Mapping[str, int]) -> str:
    """`counts` as one `key=value  key=value` line."""
    return "  ".join(f"{k}=[bold]{v}[/bold]" for k, v in counts.items())


def ruleset(rules: RuleSet, consuming: set[str], path: Any, classes: Sequence[str]) -> None:
    """`rules show`'s table: every rule in priority order, with consuming
    actions (`move`/`delete`) highlighted."""
    table = Table(box=None)
    table.add_column("#", style="dim")
    table.add_column("name", style="bold")
    table.add_column("matches")
    table.add_column("action")
    table.add_column("target")
    for index, rule in enumerate(rules.rules, 1):
        style = "red" if rule.action in consuming else "cyan"
        table.add_row(
            str(index),
            rule.name if rule.enabled else f"[dim]{rule.name} (off)[/dim]",
            rule.condition.describe(),
            f"[{style}]{rule.action}[/{style}]",
            rule.target_folder() if rule.action in {"copy", "move"} else "-",
        )
    console.print(table)
    console.print(f"[dim]file: {path}[/dim]")
    console.print(f"[dim]detector will look for: {', '.join(classes)}[/dim]")


def action_catalog(entries: Sequence[Mapping[str, Any]]) -> None:
    """`rules actions`'s table: every registered action and what it does to
    the original."""
    table = Table(box=None)
    table.add_column("action", style="bold")
    table.add_column("the original")
    table.add_column("description")
    for entry in entries:
        table.add_row(
            entry["name"],
            f"[red]{entry['consequence']}[/red]" if entry["consumes_original"]
            else "stays put",
            entry["description"],
        )
    console.print(table)


#: The same words the Settings tab puts on a field's badge.
SOURCE_LABEL = {
    "settings": "saved", "environment": "from .env", "default": "default", "unset": "not set",
}


def settings_fields(described: Mapping[str, Any]) -> None:
    """`config show`'s table: every editable setting, its value, and which
    layer (saved / .env / default) supplied it."""
    table = Table(box=None)
    table.add_column("setting", style="bold")
    table.add_column("value")
    table.add_column("from", style="dim")
    for field in described["fields"]:
        value = field["value"]
        if isinstance(value, bool):
            shown = "on" if value else "off"
        elif isinstance(value, list):
            shown = ", ".join(str(item) for item in value) or "[red]—[/red]"
        else:
            shown = str(value)
        table.add_row(field["key"], shown, SOURCE_LABEL.get(field["source"], field["source"]))
    console.print(table)
    console.print(f"[dim]saved settings: {described['file']}[/dim]")
    console.print(f"[dim].env: {described['env_file'] or 'none found'}[/dim]")


def checks(results: Sequence[Check]) -> None:
    """`doctor`'s table: every diagnostic check, colored by status."""
    table = Table(box=None)
    table.add_column("check", style="bold")
    table.add_column("result")
    for check in results:
        style = STATUS_STYLE[check.status]
        detail = f"[{style}]{check.detail}[/{style}]" if style else check.detail
        table.add_row(check.name, detail)
    console.print(table)


def verify_report(report: VerifyReport) -> None:
    """`verify`'s output: one line per count, green unless it flags a problem."""
    for key, value in report.as_dict().items():
        style = "red" if value and key != "copies_ok" else "green"
        console.print(f"{key:18} [{style}]{value}[/{style}]")


def duplicates(groups: Sequence[Mapping[str, Any]]) -> None:
    """`duplicates`'s output: each group of identical files, hash then paths."""
    for group in groups:
        console.print(f"[bold]{group['hash'][:16]}[/bold]  x{group['count']}")
        for path in group["paths"]:
            console.print(f"  [dim]{path}[/dim]")


def history(entries: Sequence[Mapping[str, Any]]) -> None:
    """`history`'s table: recent irreversible actions, newest first."""
    table = Table(box=None)
    table.add_column("action", style="bold")
    table.add_column("source")
    table.add_column("moved to", style="dim")
    for entry in entries:
        table.add_row(entry["action"], entry["path"], entry["detail"])
    console.print(table)


def search_results(results: Sequence[Mapping[str, Any]]) -> None:
    """`search`'s output: each match's path, category and highlighted terms."""
    for row in results:
        bits = [row["category"], *row["highlights"]]
        console.print(f"[bold]{row['path']}[/bold]\n  [dim]{' | '.join(bits)}[/dim]")
    console.print(f"[dim]{len(results)} result(s)[/dim]")
