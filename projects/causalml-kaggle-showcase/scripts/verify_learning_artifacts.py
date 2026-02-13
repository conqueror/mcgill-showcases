from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from causal_showcase.config import PROJECT_ROOT
from causal_showcase.verification import verify_learning_artifacts

app = typer.Typer(help="Verify that learning artifacts were generated correctly.")


@app.command()
def main(
    project_root: Annotated[
        Path,
        typer.Option(file_okay=False, dir_okay=True, help="Project root path."),
    ] = PROJECT_ROOT,
    require_notebook_artifacts: Annotated[
        bool,
        typer.Option(help="Require notebook-generated figures (Qini + SHAP)."),
    ] = False,
) -> None:
    result = verify_learning_artifacts(
        project_root=project_root,
        require_notebook_artifacts=require_notebook_artifacts,
    )

    if result.passed:
        typer.echo("Artifact verification PASSED.")
    else:
        typer.echo("Artifact verification FAILED.")

    if result.errors:
        typer.echo("\nErrors:")
        for err in result.errors:
            typer.echo(f"- {err}")

    if result.warnings:
        typer.echo("\nWarnings:")
        for warn in result.warnings:
            typer.echo(f"- {warn}")

    if not result.passed:
        typer.echo("\nSuggested commands to regenerate artifacts:")
        typer.echo("- uv run python scripts/run_pipeline.py")
        typer.echo("- uv run python scripts/policy_simulator.py")
        typer.echo("- uv run python scripts/confounding_checks.py")
        typer.echo(
            "- Run notebooks 04 and 07 if you require notebook artifacts "
            "(with --require-notebook-artifacts)."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
