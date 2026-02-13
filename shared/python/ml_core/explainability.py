"""Optional explainability helpers for SHAP and LIME exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_shap_importance(
    model: object,
    x_eval: pd.DataFrame,
    *,
    output_path: Path,
) -> str:
    """Write global SHAP importance CSV when SHAP dependency is available."""

    try:
        import shap
    except Exception:
        return "skipped_missing_dependency"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model, x_eval)
    shap_values = explainer(x_eval)
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        values = values[:, :, 0]
    mean_abs = np.abs(values).mean(axis=0)

    out = pd.DataFrame(
        {
            "feature": x_eval.columns,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    out.to_csv(output_path, index=False)
    return "written"


def run_lime_local_explanations(
    model_predict_proba: callable,
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    *,
    output_path: Path,
    class_names: list[str] | None = None,
    n_rows: int = 10,
) -> str:
    """Write sample-level LIME explanation weights for classification models."""

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception:
        return "skipped_missing_dependency"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = ["0", "1"]

    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=[str(c) for c in x_train.columns],
        class_names=class_names,
        mode="classification",
    )

    rows: list[dict[str, float | int | str]] = []
    for idx in range(min(n_rows, len(x_eval))):
        exp = explainer.explain_instance(
            x_eval.iloc[idx].to_numpy(),
            model_predict_proba,
            num_features=min(8, x_eval.shape[1]),
        )
        for feature_name, weight in exp.as_list():
            rows.append(
                {
                    "sample_id": idx,
                    "feature": feature_name,
                    "weight": float(weight),
                    "abs_weight": float(abs(weight)),
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return "written"
