from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xai_fairness_showcase.fairness import disparity_table, subgroup_metrics
from xai_fairness_showcase.modeling import evaluate_binary, predict_probabilities, train_logistic


def _reweigh_samples(group: pd.Series) -> npt.NDArray[np.float64]:
    counts = group.value_counts().to_dict()
    total = float(len(group))
    return np.asarray([total / (len(counts) * counts[g]) for g in group], dtype=float)


def _postprocess_group_thresholds(
    probas: npt.NDArray[np.float64],
    group: pd.Series,
    *,
    thresholds: pd.Series,
) -> npt.NDArray[np.int64]:
    preds = np.zeros_like(probas, dtype=int)
    for g in group.unique():
        idx = group == g
        group_probs = probas[idx]
        if g not in thresholds.index:
            raise ValueError(f"No validation threshold for group {g!r}.")
        threshold = float(thresholds.loc[g])
        preds[idx] = (group_probs >= threshold).astype(int)
    return preds


def run_mitigation_benchmark(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    g_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    g_test: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    x_fit, x_val, y_fit, _, g_fit, g_val = train_test_split(
        x_train, y_train, g_train, test_size=0.25, random_state=42, stratify=g_train,
    )

    baseline_model = train_logistic(x_fit, y_fit)
    baseline_probs = predict_probabilities(baseline_model, x_test)
    rows.append(_score_strategy("baseline", baseline_probs, y_test, g_test))

    pre_weights = _reweigh_samples(g_fit)
    pre_model = train_logistic(x_fit, y_fit, sample_weight=pre_weights)
    pre_probs = predict_probabilities(pre_model, x_test)
    rows.append(_score_strategy("pre_processing_reweight", pre_probs, y_test, g_test))

    in_model = train_logistic(x_fit, y_fit, class_weight="balanced")
    in_probs = predict_probabilities(in_model, x_test)
    rows.append(_score_strategy("in_processing_balanced", in_probs, y_test, g_test))

    val_probs = predict_probabilities(baseline_model, x_val)
    target_rate = float((val_probs >= 0.5).mean())
    thresholds = pd.Series({g: float(np.quantile(val_probs[g_val == g], 1.0 - target_rate))
                            for g in g_val.unique()})
    post_preds = _postprocess_group_thresholds(baseline_probs, g_test, thresholds=thresholds)
    rows.append(
        _score_postprocessing(
            "post_processing_threshold",
            baseline_probs,
            post_preds,
            y_test,
            g_test,
        )
    )

    return pd.DataFrame(rows)


def _score_strategy(
    strategy_name: str,
    probs: npt.NDArray[np.float64],
    y_true: pd.Series,
    group: pd.Series,
) -> dict[str, float | str]:
    scores = evaluate_binary(y_true, probs)
    g_metrics = subgroup_metrics(y_true, probs, group)
    disparities = disparity_table(g_metrics).set_index("metric")
    return {
        "strategy": strategy_name,
        "roc_auc": scores["roc_auc"],
        "accuracy": scores["accuracy"],
        "selection_rate_gap": float(disparities.loc["selection_rate", "gap"]),
        "tpr_gap": float(disparities.loc["tpr", "gap"]),
        "fpr_gap": float(disparities.loc["fpr", "gap"]),
    }


def _score_postprocessing(
    strategy_name: str,
    probs: npt.NDArray[np.float64],
    preds: npt.NDArray[np.int64],
    y_true: pd.Series,
    group: pd.Series,
) -> dict[str, float | str]:
    accuracy = float((preds == y_true.to_numpy()).mean())
    group_metrics = subgroup_metrics(y_true, preds.astype(np.float64), group)
    disparities = disparity_table(group_metrics).set_index("metric")
    return {
        "strategy": strategy_name,
        "roc_auc": float("nan"),
        "accuracy": accuracy,
        "selection_rate_gap": float(disparities.loc["selection_rate", "gap"]),
        "tpr_gap": float(disparities.loc["tpr", "gap"]),
        "fpr_gap": float(disparities.loc["fpr", "gap"]),
    }


def train_baseline_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    return train_logistic(x_train, y_train)
