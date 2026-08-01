"""The `config` sub-app: the CLI half of the web UI's Settings tab.

Both front ends go through `services.configuring`, so a folder saved here and a
folder saved in the browser are the same write to the same overlay, validated the
same way. Nothing in here parses the environment or touches the file itself.
"""

from __future__ import annotations

import typer

from ....config import overrides
from ....services import configuring
from .. import render
from ..runtime import console, fail, run_or_fail, setup

app = typer.Typer(help="Show and save the settings the CLI and the UI share.",
                  no_args_is_help=True)


@app.command("show")
def show():
    """Print each editable setting, its value, and which layer supplied it."""
    ctx = setup()
    render.settings_fields(configuring.describe(ctx))


@app.command("set")
def set_(
    input_folders: list[str] = typer.Option(
        None, "--input", "-i", metavar="FOLDER",
        help="Folder to index. Repeatable; replaces the saved list.",
    ),
    output_folder: str = typer.Option(
        None, "--output", metavar="FOLDER", help="Where to write the sorted tree.",
    ),
    ollama_url: str = typer.Option(None, "--ollama-url", metavar="URL"),
    ollama_model: str = typer.Option(None, "--ollama-model", metavar="NAME"),
    confidence: float = typer.Option(
        None, "--confidence", metavar="0-1",
        help="Score a detection needs to satisfy a rule condition.",
    ),
    review_confidence: float = typer.Option(
        None, "--review-confidence", metavar="0-1",
        help="Below --confidence but at or above this, a detection is flagged for review.",
    ),
    analyze: bool = typer.Option(
        None, "--analyze/--no-analyze", help="Run the semantic pass at all.",
    ),
    adjudicate: bool = typer.Option(
        None, "--adjudicate/--no-adjudicate",
        help="Ask the vision model about detections the detector was unsure of.",
    ),
):
    """Save settings for every later command, and for the web UI.

    Writes the same overlay the Settings tab writes, which wins over `.env`.
    """
    ctx = setup()
    values = {
        "INPUT_FOLDERS": input_folders,
        "OUTPUT_FOLDER": output_folder,
        "OLLAMA_URL": ollama_url,
        "OLLAMA_MODEL": ollama_model,
        "DETECT_CONFIDENCE": confidence,
        "REVIEW_CONFIDENCE": review_confidence,
        "ANALYZE_ENABLED": analyze,
        "ADJUDICATE_ENABLED": adjudicate,
    }
    given = {key: value for key, value in values.items() if value not in (None, [], ())}
    if not given:
        fail("nothing to set — pass at least one of --input, --output, "
             "--ollama-url, --ollama-model, --confidence, --review-confidence, "
             "--analyze/--no-analyze, --adjudicate/--no-adjudicate")

    run_or_fail(lambda: configuring.update(ctx, given))

    console.print(f"saved to [bold]{ctx.settings.paths.settings}[/bold]")
    # Re-read rather than echo what was submitted: what is on disk now is the
    # answer, and it is what the next command will see.
    render.settings_fields(configuring.describe(setup()))


@app.command("unset")
def unset(
    keys: list[str] = typer.Argument(
        ..., metavar="KEY...", help=f"One or more of: {', '.join(sorted(overrides.BY_KEY))}.",
    ),
):
    """Forget saved settings, handing them back to `.env` or the default."""
    ctx = setup()
    run_or_fail(lambda: configuring.reset(ctx, [key.strip().upper() for key in keys]))
    render.settings_fields(configuring.describe(setup()))
