#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def _base_frame() -> tuple[pd.DataFrame, pd.Series]:
    raw = load_iris(as_frame=True)
    frame = raw.data.copy()
    frame["sample_id"] = np.arange(len(frame), dtype=int)
    frame["category"] = pd.cut(
        frame["petal length (cm)"],
        bins=[0.0, 2.5, 4.5, 8.0],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype(str)
    target = raw.target.copy()
    return frame, target


def _entity_embedding_proxy(frame: pd.DataFrame, output_path: Path) -> None:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    onehot = encoder.fit_transform(frame[["category"]])
    svd = TruncatedSVD(n_components=2, random_state=42)
    embeddings = svd.fit_transform(onehot)
    out = pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "embedding_0": embeddings[:, 0],
            "embedding_1": embeddings[:, 1],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def _maybe_featuretools(frame: pd.DataFrame, target: pd.Series, output_path: Path) -> str:
    try:
        import featuretools as ft
    except Exception:
        return "skipped_missing_dependency"

    entity_frame = frame.copy()
    entity_frame["target"] = target.to_numpy()
    es = ft.EntitySet(id="iris")
    es = es.add_dataframe(
        dataframe_name="samples",
        dataframe=entity_frame,
        index="sample_id",
    )
    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="samples",
        max_depth=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.reset_index(drop=True).to_csv(output_path, index=False)
    return "written"


def _maybe_tsfresh(frame: pd.DataFrame, output_path: Path) -> str:
    try:
        from tsfresh import extract_features
    except Exception:
        return "skipped_missing_dependency"

    long_df = pd.DataFrame(
        {
            "id": np.repeat(frame["sample_id"].to_numpy(), 4),
            "time": np.tile(np.arange(4), len(frame)),
            "value": frame[
                [
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ]
            ]
            .to_numpy()
            .reshape(-1),
        }
    )
    extracted = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        disable_progressbar=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted.reset_index(drop=True).to_csv(output_path, index=False)
    return "written"


def _maybe_autofeat(frame: pd.DataFrame, target: pd.Series, output_path: Path) -> str:
    try:
        from autofeat import AutoFeatClassifier
    except Exception:
        return "skipped_missing_dependency"

    numeric_frame = frame[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ]
    model = AutoFeatClassifier(verbose=0, random_state=42)
    transformed = model.fit_transform(numeric_frame, target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(output_path, index=False)
    return "written"


def main() -> None:
    from ml_core.contracts import merge_required_files

    root = Path(__file__).resolve().parents[1]
    frame, target = _base_frame()

    required = ["artifacts/advanced/entity_embeddings.csv"]
    _entity_embedding_proxy(frame, root / required[0])

    statuses = {
        "featuretools": _maybe_featuretools(
            frame,
            target,
            root / "artifacts/advanced/featuretools_features.csv",
        ),
        "tsfresh": _maybe_tsfresh(frame, root / "artifacts/advanced/tsfresh_features.csv"),
        "autofeat": _maybe_autofeat(
            frame,
            target,
            root / "artifacts/advanced/autofeat_features.csv",
        ),
    }

    for key, status in statuses.items():
        status_path = root / f"artifacts/advanced/{key}_status.txt"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(f"{status}\n", encoding="utf-8")
        required.append(f"artifacts/advanced/{key}_status.txt")
        if status == "written":
            required.append(f"artifacts/advanced/{key}_features.csv")

    merge_required_files(root / "artifacts/manifest.json", required)
    (root / "artifacts/advanced/summary.json").write_text(
        json.dumps(statuses, indent=2),
        encoding="utf-8",
    )
    merge_required_files(root / "artifacts/manifest.json", ["artifacts/advanced/summary.json"])


if __name__ == "__main__":
    main()
