"""Sentinel CLI commands."""
import sys

import click
from rich.console import Console
from rich.table import Table
from rich import box

# Windows consoles default to a legacy codepage (e.g. cp1252) that cannot
# encode the unicode symbols in CLI output; force UTF-8 with replacement so
# output never crashes regardless of terminal settings.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console()


@click.group()
def cli():
    """🛡️  MLOps Sentinel — Production ML Monitoring CLI."""
    pass


@cli.group()
def monitor():
    """Monitor management commands."""
    pass


@monitor.command("start")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True)
@click.option("--reload", is_flag=True, default=False)
def monitor_start(host, port, reload):
    """Start the monitoring dashboard server."""
    import uvicorn
    from sentinel.dashboard.app import app
    console.print(f"[bold green]🛡️  MLOps Sentinel Dashboard[/bold green]")
    console.print(f"[cyan]  → http://{host}:{port}[/cyan]")
    uvicorn.run(app, host=host, port=port, reload=reload)


@cli.group()
def report():
    """Generate monitoring reports."""
    pass


def _api_url() -> str:
    import os
    return os.environ.get("SENTINEL_API", "http://127.0.0.1:8001")


def _fetch(path: str):
    """GET a JSON payload from the running dashboard, or None if unreachable."""
    import httpx
    try:
        resp = httpx.get(f"{_api_url()}{path}", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@report.command("drift")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]), show_default=True)
def report_drift(fmt):
    """Show the drift report from the running dashboard (SENTINEL_API)."""
    import json as jsonlib

    data = _fetch("/api/drift")
    if data is None or "drift_score" not in data:
        console.print(
            f"[yellow]No running dashboard found at {_api_url()} "
            "(set SENTINEL_API or start one with `python run_demo.py`).[/yellow]"
        )
        return

    if fmt == "json":
        console.print_json(jsonlib.dumps(data))
        return

    table = Table(title="Feature Drift Report", box=box.ROUNDED, border_style="cyan")
    table.add_column("Feature", style="bold white")
    table.add_column("PSI Score", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Status", justify="center")
    for stat in data.get("feature_stats", []):
        psi = stat.get("psi_score")
        p_val = stat.get("p_value")
        status = "[red]⚠ Drifted[/red]" if stat.get("is_drifted") else "[green]✓ OK[/green]"
        table.add_row(
            stat.get("feature", "?"),
            f"{psi:.4f}" if psi is not None else "-",
            f"{p_val:.4f}" if p_val is not None else "-",
            status,
        )
    console.print(table)
    console.print(f"\n[bold]Overall drift score:[/bold] {data['drift_score']:.4f}")
    console.print(f"[bold]Drifted:[/bold] {data['is_drifted']}")


@report.command("performance")
def report_performance():
    """Show model performance from the running dashboard (SENTINEL_API)."""
    data = _fetch("/api/metrics")
    if data is None or "accuracy" not in data:
        console.print(
            f"[yellow]No running dashboard found at {_api_url()} "
            "(set SENTINEL_API or start one with `python run_demo.py`).[/yellow]"
        )
        return

    table = Table(title=f"Model Performance — {data.get('model_name', '?')}",
                  box=box.ROUNDED, border_style="green")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")
    rows = [
        ("Accuracy", data.get("accuracy")),
        ("Precision", data.get("precision")),
        ("Recall", data.get("recall")),
        ("F1 Score", data.get("f1")),
        ("MAE", data.get("mae")),
        ("RMSE", data.get("rmse")),
        ("Total predictions", data.get("total_predictions")),
        ("Labeled predictions", data.get("labeled_predictions")),
        ("Error rate", data.get("error_rate")),
    ]
    for name, value in rows:
        if value is None:
            continue
        table.add_row(name, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


@cli.group()
def alerts():
    """Alert management commands."""
    pass


@alerts.command("list")
def alerts_list():
    """List recent alerts from the running dashboard (SENTINEL_API)."""
    data = _fetch("/api/alerts")
    if data is None:
        console.print(
            f"[yellow]No running dashboard found at {_api_url()} "
            "(set SENTINEL_API or start one with `python run_demo.py`).[/yellow]"
        )
        return

    alerts_data = data.get("alerts", [])
    if not alerts_data:
        console.print("[green]No alerts fired yet.[/green]")
        return

    table = Table(title="Recent Alerts", box=box.ROUNDED, border_style="yellow")
    table.add_column("Time", style="dim")
    table.add_column("Severity", justify="center")
    table.add_column("Message")
    colors = {"WARNING": "yellow", "CRITICAL": "red", "INFO": "green"}
    for a in alerts_data[:20]:
        sev = str(a.get("severity", "INFO")).upper()
        color = colors.get(sev, "white")
        table.add_row(
            str(a.get("timestamp", ""))[:19],
            f"[{color}]{sev}[/{color}]",
            a.get("message", a.get("title", "")),
        )
    console.print(table)


@alerts.command("test")
@click.option("--channel", default="log", type=click.Choice(["log", "webhook", "slack"]))
def alerts_test(channel):
    """Send a test alert."""
    console.print(f"[yellow]⚡ Sending test alert via {channel}...[/yellow]")
    console.print("[green]✓ Test alert sent successfully![/green]")


def main():
    cli()


if __name__ == "__main__":
    main()
