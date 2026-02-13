"""Command line interface for the learning showcase."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from .config import PathsConfig, ShowcaseConfig
from .pipeline import run_full_showcase

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    dataset: Literal["digits", "business"] = typer.Option(
        "digits",
        help="Dataset mode: digits (sklearn) or business (loan.csv).",
    ),
    labeled_fraction: float = typer.Option(0.1, help="Fraction of labeled training examples."),
    random_state: int = typer.Option(42, help="Random seed for reproducibility."),
    contrastive_epochs: int = typer.Option(8, help="Epochs for self-supervised pretraining."),
    dec_pretrain_epochs: int = typer.Option(8, help="DEC autoencoder pretrain epochs."),
    dec_finetune_epochs: int = typer.Option(10, help="DEC fine-tune epochs."),
    active_learning_rounds: int = typer.Option(6, help="Active learning rounds."),
    active_learning_query_size: int = typer.Option(25, help="New labels acquired each round."),
    business_read_rows: int = typer.Option(
        60_000,
        help="Rows to read from loan.csv in business mode.",
    ),
    business_sample_size: int = typer.Option(
        3_000,
        help="Stratified sample size used in business mode.",
    ),
    business_csv_path: Path | None = None,
) -> None:
    """Run the full educational pipeline and export CSV/PNG outputs."""

    project_root = Path(__file__).resolve().parents[2]
    config = ShowcaseConfig(
        dataset=dataset,
        labeled_fraction=labeled_fraction,
        random_state=random_state,
        contrastive_epochs=contrastive_epochs,
        dec_pretrain_epochs=dec_pretrain_epochs,
        dec_finetune_epochs=dec_finetune_epochs,
        active_learning_rounds=active_learning_rounds,
        active_learning_query_size=active_learning_query_size,
        business_read_rows=business_read_rows,
        business_sample_size=business_sample_size,
        business_csv_path=business_csv_path,
    )
    paths = PathsConfig.from_project_root(project_root=project_root)

    outputs = run_full_showcase(config=config, paths=paths)
    final_round = int(outputs["active_learning"]["round"].max())
    best_active = (
        outputs["active_learning"][outputs["active_learning"]["round"] == final_round]
        .sort_values("accuracy", ascending=False)
        .iloc[0]["strategy"]
    )

    typer.echo("Pipeline complete. Top methods by module:")
    typer.echo(f"- Clustering: {outputs['clustering'].iloc[0]['algorithm']}")
    typer.echo(f"- Semi-supervised: {outputs['semi_supervised'].iloc[0]['method']}")
    typer.echo(f"- Active learning: {best_active}")
    typer.echo(f"- Self-supervised: {outputs['self_supervised'].iloc[0]['method']}")


if __name__ == "__main__":
    app()
