"""Everything a CLI command needs before it can do any work.

Logging, the console, the shared context, signal handling and the mapping from
an application error to an exit code. Commands stay free of all of it.
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import Callable, NoReturn, TypeVar

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from ...config import load_settings, overrides
from ...domain.rules import RuleSet
from ...errors import ConfigError, IncompatibleIndex, PhotosortError, RuleError
from ...pipeline import Stopper
from ...services import AppContext
from ...services import rules as rules_service

console = Console()

#: Shared so the two folders are spelled and described the same way wherever they
#: can be given. Both are one-run overrides; `photosort config set` is what
#: persists them.
INPUT_OPTION = typer.Option(
    None, "--input", "-i", metavar="FOLDER",
    help="Folder to index, instead of the saved ones. Repeatable, this run only.",
)
OUTPUT_OPTION = typer.Option(
    None, "--output", metavar="FOLDER",
    help="Where to write the sorted tree, instead of the saved one. This run only.",
)


def fail(message: str, code: int = 2) -> NoReturn:
    """Print `message` in red and exit — the CLI's one way to fail cleanly,
    with no traceback."""
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=code)


T = TypeVar("T")


def run_or_fail(fn: Callable[[], T], catch: type[BaseException] = ConfigError) -> T:
    """Call `fn`, turning a `catch` exception into `fail`'s red one-liner
    instead of a traceback — every command-level error surfaces the same way,
    however many places check for one."""
    try:
        return fn()
    except catch as exc:
        fail(str(exc))


def setup(verbose: bool = False, input_folders: list[str] | None = None,
          output_folder: str | None = None) -> AppContext:
    """Build the context every command works through.

    `input_folders` and `output_folder` are the `--input` / `--output` flags:
    validated by the same code the Settings tab uses, applied to this run only,
    and never written anywhere.
    """
    settings = run_or_fail(
        lambda: overrides.with_folders(load_settings(), input_folders, output_folder)
    )

    logging.basicConfig(
        level="DEBUG" if verbose else settings.log_level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return run_or_fail(lambda: AppContext.create(settings), catch=IncompatibleIndex)


def load_rules(ctx: AppContext) -> RuleSet:
    """The active rules, or a clear error naming the file."""
    try:
        return rules_service.active_ruleset(ctx)
    except RuleError as exc:
        console.print(f"[red]rules error:[/red] {exc}")
        console.print(f"[dim]file: {ctx.rules.path}[/dim]")
        raise typer.Exit(code=2) from None


def require_folders(ctx: AppContext) -> None:
    """Stop before the progress bar when nothing has said where the photos are.

    The scanner enforces this too, but a `ConfigError` escaping a command body
    reaches the user as a traceback; here it reaches them as the sentence it is.
    """
    run_or_fail(ctx.settings.library.require_folders)


def require_detector() -> None:
    """Stop before the progress bar when the detection dependencies are missing."""
    # Deferred: `detecting` imports ultralytics/torch, which only the commands
    # that actually touch the detector need to pay to load.
    from ...detecting import ensure_available

    run_or_fail(ensure_available, catch=PhotosortError)


def install_stop_handlers(stopper: Stopper) -> None:
    """First interrupt asks the stages to wind down; a second one is impatient."""
    def handler(signum, frame):  # noqa: ARG001
        """SIGINT/SIGTERM handler: stop cooperatively once, force-exit on a second signal."""
        if stopper.stopped:
            console.print("[red]forced exit[/red]")
            sys.exit(130)
        console.print("\n[yellow]stopping after the current batch — progress is saved[/yellow]")
        stopper.stop()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def progress() -> Progress:
    """A `rich.Progress` bar configured the same way for every stage command."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
