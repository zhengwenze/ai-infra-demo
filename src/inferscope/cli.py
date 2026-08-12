"""InferScope command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console

from inferscope.config import load_config, validate_results_directory
from inferscope.environment import environment_as_json
from inferscope.errors import ConfigurationError
from inferscope.models import ValidationStatus
from inferscope.runner import ExperimentRunner

app = typer.Typer(no_args_is_help=True, help="SLO-aware LLM inference benchmarking.")
env_app = typer.Typer(no_args_is_help=True)
server_app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
app.add_typer(env_app, name="env")
app.add_typer(server_app, name="server")
app.add_typer(benchmark_app, name="benchmark")
console = Console()
DEFAULT_PROJECT_DIR = Path.cwd()


@env_app.command("capture")
def env_capture(
    project_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = (
        DEFAULT_PROJECT_DIR
    ),
) -> None:
    """Print a secret-free reproducibility fingerprint."""
    typer.echo(environment_as_json(project_dir))


@server_app.command("check")
def server_check(
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible base URL")],
    timeout_seconds: Annotated[float, typer.Option(min=0.1)] = 5.0,
) -> None:
    """Check model endpoint readiness without generating tokens."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout_seconds)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[red]target unavailable:[/red] {exc}")
        raise typer.Exit(3) from exc
    console.print(f"[green]ready[/green] HTTP {response.status_code}")


@benchmark_app.command("run")
def benchmark_run(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
    repeat: Annotated[int | None, typer.Option("--repeat", min=1)] = None,
) -> None:
    """Run a workload matrix and persist evidence-first artifacts."""
    project_dir = Path.cwd().resolve()
    try:
        config = load_config(config_path)
        destination = validate_results_directory(results_dir, project_dir)
    except ConfigurationError as exc:
        console.print(f"[red]{exc.code.value}:[/red] {exc}")
        raise typer.Exit(2) from exc

    def progress(done: int, total: int) -> None:
        console.print(f"requests {done}/{total}", end="\r")

    runner = ExperimentRunner(
        config,
        results_dir=destination,
        project_dir=project_dir,
        progress=progress,
    )
    try:
        outcomes = asyncio.run(runner.run(repeats=repeat))
    except KeyboardInterrupt as exc:
        console.print("\n[yellow]interrupted[/yellow]")
        raise typer.Exit(130) from exc
    except httpx.HTTPError as exc:
        console.print(f"\n[red]target unavailable:[/red] {exc}")
        raise typer.Exit(3) from exc
    console.print()
    for outcome in outcomes:
        console.print(
            f"{outcome.run_id}: {outcome.validation_status.value} -> {outcome.report_path}"
        )
    if any(outcome.validation_status is not ValidationStatus.VALID for outcome in outcomes):
        raise typer.Exit(4)


@benchmark_app.command("show")
def benchmark_show(
    run_id: Annotated[str, typer.Option("--run-id")],
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("results"),
) -> None:
    """Print persisted validation and aggregate JSON for one run."""
    raw = results_dir / "raw" / run_id / "validation.json"
    aggregate = results_dir / "processed" / run_id / "aggregate.json"
    if not raw.is_file() or not aggregate.is_file():
        console.print("[red]run artifacts were not found[/red]")
        raise typer.Exit(2)
    typer.echo(
        json.dumps(
            {
                "validation": json.loads(raw.read_text(encoding="utf-8")),
                "aggregate": json.loads(aggregate.read_text(encoding="utf-8")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
