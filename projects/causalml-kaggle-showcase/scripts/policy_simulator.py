from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import seaborn as sns
import typer

from causal_showcase.config import ARTIFACTS_DIR, FIGURES_DIR, RAW_DATA_PATH
from causal_showcase.data import load_marketing_ab_data, train_test_split_prepared
from causal_showcase.modeling import fit_meta_learners, fit_uplift_tree
from causal_showcase.policy import select_best_model_per_budget, simulate_policy_table

app = typer.Typer(help="Simulate targeting policy outcomes across budget levels.")


def _parse_budgets(raw: str) -> list[float]:
    values = [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("No budget values provided.")

    unique_sorted = sorted(set(values))
    for budget in unique_sorted:
        if not 0 < budget <= 1:
            raise ValueError(f"Budget must be in (0, 1], got {budget}.")
    return unique_sorted


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
    budgets: Annotated[
        str,
        typer.Option(help="Comma-separated budget fractions, e.g., 0.1,0.2,0.3"),
    ] = "0.1,0.2,0.3,0.4,0.5",
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at `{data_path}`. "
            "Run `uv run python scripts/download_kaggle_dataset.py` first."
        )

    budget_values = _parse_budgets(budgets)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prepared = load_marketing_ab_data(data_path)
    train_data, test_data = train_test_split_prepared(prepared)

    learner_results = fit_meta_learners(train_data, test_data)
    tree_result = fit_uplift_tree(train_data, test_data)

    score_by_model = {name: result.uplift_scores for name, result in learner_results.items()}
    score_by_model["Uplift Tree (KL)"] = tree_result.uplift_scores

    policy_df = simulate_policy_table(
        y=test_data.outcome,
        treatment=test_data.treatment,
        score_by_model=score_by_model,
        budgets=budget_values,
    )
    best_df = select_best_model_per_budget(policy_df)

    policy_csv = ARTIFACTS_DIR / "policy_simulation.csv"
    best_csv = ARTIFACTS_DIR / "policy_best_models.csv"
    plot_path = FIGURES_DIR / "policy_incremental_conversions.png"

    policy_df.to_csv(policy_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=policy_df,
        x="budget_fraction",
        y="expected_incremental_conversions",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.set_title("Expected Incremental Conversions by Budget")
    ax.set_xlabel("Budget fraction (top-k treated)")
    ax.set_ylabel("Expected incremental conversions")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    typer.echo("Policy simulation completed.")
    typer.echo(f"All model/budget outcomes: {policy_csv}")
    typer.echo(f"Best model per budget: {best_csv}")
    typer.echo(f"Figure: {plot_path}")


if __name__ == "__main__":
    app()
