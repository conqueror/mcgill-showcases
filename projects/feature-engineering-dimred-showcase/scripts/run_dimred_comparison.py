#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

from feature_dimred_showcase.dimensionality_reduction import embedding_quality, run_embeddings
from feature_dimred_showcase.preprocessing import make_split, transform_split


def _dataset() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    raw = load_iris(as_frame=True)
    x_df = raw.data.copy()
    y = raw.target.copy()
    x_df["petal_bucket"] = pd.cut(
        x_df["petal length (cm)"],
        bins=[0.0, 2.5, 4.5, 8.0],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype(str)
    numeric = [col for col in x_df.columns if col != "petal_bucket"]
    categorical = ["petal_bucket"]
    return x_df, y, numeric, categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dimensionality reduction comparison")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    x_df, y, numeric, categorical = _dataset()
    split = make_split(x_df, y, random_state=42)
    x_train, _, _, _ = transform_split(
        split,
        numeric_features=numeric,
        categorical_features=categorical,
        encoding="onehot",
    )

    embeddings = run_embeddings(x_train, random_state=42, quick=args.quick)
    quality = embedding_quality(embeddings, split.y_train)

    quality_path = root / "artifacts/dimred/embedding_quality_metrics.csv"
    plot_path = root / "artifacts/dimred/embedding_plots.png"

    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(quality_path, index=False)

    cols = len(embeddings)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    for axis, item in zip(axes, embeddings, strict=True):
        scatter = axis.scatter(
            item.embedding[:, 0],
            item.embedding[:, 1],
            c=split.y_train,
            cmap="viridis",
            s=18,
        )
        axis.set_title(item.method.upper())
        axis.set_xlabel("dim_1")
        axis.set_ylabel("dim_2")

    fig.colorbar(scatter, ax=axes, shrink=0.7)
    fig.tight_layout()
    fig.savefig(plot_path)

    manifest_path = root / "artifacts/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "required_files": []}

    required = set(manifest.get("required_files", []))
    required.update(
        {
            "artifacts/dimred/embedding_quality_metrics.csv",
            "artifacts/dimred/embedding_plots.png",
        }
    )
    manifest["required_files"] = sorted(required)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
