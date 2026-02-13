from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from causal_showcase.config import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    RAW_DATA_PATH,
    REPORT_PATH,
    TREE_SUMMARY_PATH,
)
from causal_showcase.data import load_marketing_ab_data, train_test_split_prepared
from causal_showcase.evaluation import estimate_empirical_ate, qini_auc, qini_curve, uplift_at_k
from causal_showcase.modeling import fit_meta_learners, fit_uplift_tree
from causal_showcase.plots import plot_qini_curves, plot_uplift_distribution

app = typer.Typer(help="Run the end-to-end CausalML learning pipeline.")


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
    train_data, test_data = train_test_split_prepared(prepared)

    baseline_ate = estimate_empirical_ate(test_data.outcome, test_data.treatment)

    learner_results = fit_meta_learners(train_data, test_data)
    tree_result = fit_uplift_tree(train_data, test_data)

    summary_rows: list[dict[str, float | str]] = []
    curves: dict[str, pd.DataFrame] = {}

    for learner_name, result in learner_results.items():
        curve = qini_curve(test_data.outcome, test_data.treatment, result.uplift_scores)
        curves[learner_name] = curve

        summary_rows.append(
            {
                "model": learner_name,
                "baseline_empirical_ate": baseline_ate,
                "estimated_ate": result.ate,
                "ate_ci_low": result.ate_ci_low,
                "ate_ci_high": result.ate_ci_high,
                "uplift_at_30pct": uplift_at_k(
                    test_data.outcome,
                    test_data.treatment,
                    result.uplift_scores,
                    top_fraction=0.30,
                ),
                "qini_auc": qini_auc(curve),
            }
        )

    tree_curve = qini_curve(test_data.outcome, test_data.treatment, tree_result.uplift_scores)
    curves["Uplift Tree (KL)"] = tree_curve
    summary_rows.append(
        {
            "model": "Uplift Tree (KL)",
            "baseline_empirical_ate": baseline_ate,
            "estimated_ate": float("nan"),
            "ate_ci_low": float("nan"),
            "ate_ci_high": float("nan"),
            "uplift_at_30pct": uplift_at_k(
                test_data.outcome,
                test_data.treatment,
                tree_result.uplift_scores,
                top_fraction=0.30,
            ),
            "qini_auc": qini_auc(tree_curve),
        }
    )

    summary_df = pd.DataFrame(summary_rows).sort_values("qini_auc", ascending=False)
    summary_df.to_csv(REPORT_PATH, index=False)

    plot_qini_curves(curves, FIGURES_DIR / "qini_curves.png")

    best_model_name = str(summary_df.iloc[0]["model"])
    if best_model_name in learner_results:
        best_scores = learner_results[best_model_name].uplift_scores
    else:
        best_scores = tree_result.uplift_scores
    plot_uplift_distribution(
        best_scores,
        best_model_name,
        FIGURES_DIR / "best_model_uplift_distribution.png",
    )

    TREE_SUMMARY_PATH.write_text(tree_result.tree_summary)

    typer.echo("Pipeline completed.")
    typer.echo(f"Metrics: {REPORT_PATH}")
    typer.echo(f"Figures: {FIGURES_DIR}")
    typer.echo(f"Tree summary: {TREE_SUMMARY_PATH}")


if __name__ == "__main__":
    app()
