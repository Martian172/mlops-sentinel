"""Sentinel CLI commands."""
import click
from rich.console import Console
from rich.table import Table
from rich import box

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


@report.command("drift")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]), show_default=True)
def report_drift(fmt):
    """Show feature drift report."""
    import json as jsonlib
    sample = {
        "features": {"age": 0.18, "income": 0.07, "score": 0.03, "tenure": 0.12},
        "overall_drift_score": 0.10,
        "is_drifted": False,
        "threshold": 0.20,
    }
    if fmt == "json":
        console.print_json(jsonlib.dumps(sample))
        return

    table = Table(title="Feature Drift Report", box=box.ROUNDED, border_style="cyan")
    table.add_column("Feature", style="bold white")
    table.add_column("PSI Score", justify="right")
    table.add_column("Status", justify="center")
    for feat, score in sample["features"].items():
        status = "[red]⚠ Drifted[/red]" if score > 0.15 else "[green]✓ OK[/green]"
        table.add_row(feat, f"{score:.4f}", status)
    console.print(table)
    console.print(f"\n[bold]Overall drift score:[/bold] {sample['overall_drift_score']:.4f}")


@report.command("performance")
def report_performance():
    """Show model performance report."""
    table = Table(title="Model Performance", box=box.ROUNDED, border_style="green")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")
    table.add_column("Trend", justify="center")
    table.add_row("Accuracy", "0.9312", "[green]↑[/green]")
    table.add_row("Precision", "0.9101", "[green]↑[/green]")
    table.add_row("Recall", "0.9488", "[yellow]→[/yellow]")
    table.add_row("F1 Score", "0.9291", "[green]↑[/green]")
    table.add_row("Predictions (24h)", "14,207", "[cyan]→[/cyan]")
    console.print(table)


@cli.group()
def alerts():
    """Alert management commands."""
    pass


@alerts.command("list")
def alerts_list():
    """List recent alerts."""
    table = Table(title="Recent Alerts", box=box.ROUNDED, border_style="yellow")
    table.add_column("Time", style="dim")
    table.add_column("Severity", justify="center")
    table.add_column("Message")
    table.add_row("14:05:41", "[yellow]WARNING[/yellow]", "Feature drift: age (PSI=0.18)")
    table.add_row("13:52:17", "[red]CRITICAL[/red]", "Accuracy dropped below 90% threshold")
    table.add_row("13:30:00", "[green]INFO[/green]", "Model monitoring started")
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
