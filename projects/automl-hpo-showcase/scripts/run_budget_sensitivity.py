#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from automl_hpo_showcase.search_bayes_tpe import run_tpe_search
from automl_hpo_showcase.search_hyperopt import run_hyperopt_search
from automl_hpo_showcase.search_random_grid import run_grid_search, run_random_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run budget sensitivity for HPO strategies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--with-hyperopt", action="store_true")
    return parser.parse_args()


def _best_score_for(df: pd.DataFrame) -> float:
    return float(df["score"].max()) if not df.empty else float("nan")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    budgets = [4, 8] if args.quick else [4, 8, 16]
    rows: list[dict[str, float | int | str]] = []

    for budget in budgets:
        grid = run_grid_search(budget=budget, random_state=args.seed)
        random = run_random_search(budget=budget, seed=args.seed, random_state=args.seed)
        tpe = run_tpe_search(budget=budget, seed=args.seed)
        hyperopt = (
            run_hyperopt_search(budget=budget, seed=args.seed)
            if args.with_hyperopt
            else pd.DataFrame()
        )

        rows.append({"strategy": "grid", "budget": budget, "best_score": _best_score_for(grid)})
        rows.append({"strategy": "random", "budget": budget, "best_score": _best_score_for(random)})
        rows.append({"strategy": "tpe", "budget": budget, "best_score": _best_score_for(tpe)})
        if args.with_hyperopt:
            rows.append(
                {
                    "strategy": "hyperopt_tpe",
                    "budget": budget,
                    "best_score": _best_score_for(hyperopt),
                }
            )

    frame = pd.DataFrame(rows)

    fig_path = root / "artifacts/hpo/cost_vs_score.png"
    csv_path = root / "artifacts/hpo/cost_vs_score.csv"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for strategy, subset in frame.groupby("strategy"):
        plt.plot(subset["budget"], subset["best_score"], marker="o", label=strategy)
    plt.xlabel("Budget (trials)")
    plt.ylabel("Best ROC-AUC")
    plt.title("HPO Cost vs Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)

    frame.to_csv(csv_path, index=False)

    manifest_path = root / "artifacts/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "required_files": []}

    required = set(manifest.get("required_files", []))
    required.update(["artifacts/hpo/cost_vs_score.csv", "artifacts/hpo/cost_vs_score.png"])
    manifest["required_files"] = sorted(required)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
