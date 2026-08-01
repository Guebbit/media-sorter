"""`web` — serve the rules editor and control panel."""

from __future__ import annotations

from dataclasses import replace

import typer

from ...web import serve
from ..runtime import console, load_rules, setup

app = typer.Typer()


@app.command()
def web(
    port: int = typer.Option(None, "--port", "-p", help="Override the configured port."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Serve the rules editor and control panel."""
    ctx = setup(verbose)
    if port:
        ctx = replace(ctx, settings=replace(ctx.settings, web=replace(ctx.settings.web, port=port)))
    load_rules(ctx)  # fail fast on a broken rules file

    settings = ctx.settings.web
    # A bind address of 0.0.0.0 is not a browsable URL, and on hosts that resolve
    # localhost to ::1 first an IPv4-only listener looks dead — so print a
    # loopback address the user can actually click.
    shown = "127.0.0.1" if settings.host in {"0.0.0.0", "::", ""} else settings.host
    console.print(f"[green]mediasort web UI[/green] on http://{shown}:{settings.port}")
    console.print("[dim]unauthenticated by design — keep it on a trusted network[/dim]")
    serve(ctx)
