from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def subgroup_metrics(
    y_true: pd.Series,
    probas: npt.NDArray[np.float64],
    group: pd.Series,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    if y_true.empty or len(y_true) != len(probas) or len(y_true) != len(group):
        raise ValueError("Labels, probabilities, and groups must have equal nonzero length.")
    if not y_true.index.equals(group.index):
        raise ValueError("Labels and groups must have aligned indexes.")
    if not y_true.isin([0, 1]).all() or group.isna().any():
        raise ValueError("Labels must be binary and groups must not be missing.")
    if probas.ndim != 1 or not np.isfinite(probas).all() or ((probas < 0) | (probas > 1)).any():
        raise ValueError("Probabilities must be a finite vector in [0, 1].")
    preds = (probas >= threshold).astype(int)

    rows: list[dict[str, object]] = []
    for group_value in group.unique():
        idx = group == group_value
        y = y_true[idx].to_numpy()
        p = preds[idx]

        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())

        rows.append(
            {
                "group": group_value,
                "count": int(idx.sum()),
                "positive_count": tp + fn,
                "negative_count": fp + tn,
                "predicted_positive_count": tp + fp,
                "selection_rate": _safe_rate(int((p == 1).sum()), len(p)),
                "tpr": _safe_rate(tp, tp + fn),
                "fpr": _safe_rate(fp, fp + tn),
                "fnr": _safe_rate(fn, tp + fn),
                "precision": _safe_rate(tp, tp + fp),
            }
        )

    return pd.DataFrame(rows)


def disparity_table(group_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["selection_rate", "tpr", "fpr", "fnr", "precision"]
    rows: list[dict[str, float | str]] = []
    for metric in numeric_cols:
        values = group_metrics[metric].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "max": float(values.max()),
                "min": float(values.min()),
                "gap": float(values.max() - values.min()),
                "supported_groups": int(np.isfinite(values).sum()),
                "total_groups": len(values),
            }
        )
    return pd.DataFrame(rows)
