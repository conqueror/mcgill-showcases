from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def subgroup_metrics(
    y_true: pd.Series,
    probas: npt.NDArray[np.float64],
    group: pd.Series,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    preds = (probas >= threshold).astype(int)

    rows: list[dict[str, float | int]] = []
    for group_value in sorted(group.unique()):
        idx = group == group_value
        y = y_true[idx].to_numpy()
        p = preds[idx]

        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())

        rows.append(
            {
                "group": int(group_value),
                "count": int(idx.sum()),
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
            }
        )
    return pd.DataFrame(rows)
