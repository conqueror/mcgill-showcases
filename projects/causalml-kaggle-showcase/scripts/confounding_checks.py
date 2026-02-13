from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from causal_showcase.config import ARTIFACTS_DIR, FIGURES_DIR, RAW_DATA_PATH
from causal_showcase.data import load_marketing_ab_data
from causal_showcase.diagnostics import covariate_balance_table, propensity_diagnostics
from causal_showcase.plots import plot_propensity_overlap

app = typer.Typer(help="Run covariate balance and propensity-overlap checks.")


@app.command()
def main(
    data_path: Annotated[
        Path,
        typer.Option(
            exists=False,
            dir_okay=False,
            file_okay=True,
            help="Path to marketing_ab.csv",
        ),
    ] = RAW_DATA_PATH,
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at `{data_path}`. "
            "Run `uv run python scripts/download_kaggle_dataset.py` first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prepared = load_marketing_ab_data(data_path)

    balance_df = covariate_balance_table(prepared.X, prepared.treatment)
    prop = propensity_diagnostics(prepared.X, prepared.treatment)

    balance_csv = ARTIFACTS_DIR / "covariate_balance.csv"
    scores_csv = ARTIFACTS_DIR / "propensity_scores.csv"
    summary_txt = ARTIFACTS_DIR / "confounding_summary.txt"
    overlap_png = FIGURES_DIR / "propensity_overlap.png"

    balance_df.to_csv(balance_csv, index=False)
    pd.DataFrame({"propensity_score": prop.scores, "treatment": prepared.treatment}).to_csv(
        scores_csv,
        index=False,
    )

    plot_propensity_overlap(prop.scores, prepared.treatment, overlap_png)

    flagged = balance_df.loc[balance_df["abs_smd"] > 0.1]
    summary_lines = [
        "Confounding diagnostics summary",
        f"Propensity AUC: {prop.auc:.4f}",
        f"Overlap share in [0.05, 0.95]: {prop.overlap_share:.4f}",
        f"Features with |SMD| > 0.10: {len(flagged)}",
    ]
    summary_txt.write_text("\n".join(summary_lines) + "\n")

    typer.echo("Confounding diagnostics completed.")
    typer.echo(f"Covariate balance: {balance_csv}")
    typer.echo(f"Propensity scores: {scores_csv}")
    typer.echo(f"Overlap figure: {overlap_png}")
    typer.echo(f"Summary: {summary_txt}")


if __name__ == "__main__":
    app()
