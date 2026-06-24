from __future__ import annotations

import typer

from osmosis_ai.cli.commands.dev.server import app as server_app

app: typer.Typer = typer.Typer(help="Internal developer tooling.", no_args_is_help=True)
app.add_typer(server_app, name="server")
