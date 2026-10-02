from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


def _require_file(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"Missing file: {path}")
        return
    if path.is_file() and path.stat().st_size == 0:
        errors.append(f"File is empty: {path}")
    if path.suffix == ".png":
        try:
            image = mpimg.imread(path)
            if image.size == 0 or not np.isfinite(image).all():
                errors.append(f"Invalid image: {path}")
        except (OSError, ValueError, SyntaxError) as exc:
            errors.append(f"Invalid rendered image: {path}: {exc}")


def _require_columns(path: Path, expected: set[str], errors: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        errors.append(f"Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        errors.append(f"Cannot read {path}: {exc}")
        return None
    missing = sorted(expected - set(df.columns))
    if missing:
        errors.append(f"{path} missing columns: {missing}")
        return None
    if len(df) == 0:
        errors.append(f"{path} has no rows")
        return None
    return df


def _require_finite(df: pd.DataFrame, columns: list[str], name: str, errors: list[str]) -> None:
    values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append(f"{name} must contain finite numeric values in {columns}.")


def verify_learning_artifacts(
    project_root: Path,
    *,
    require_notebook_artifacts: bool = False,
) -> VerificationResult:
    artifacts = project_root / "artifacts"
    figures = artifacts / "figures"

    errors: list[str] = []
    warnings: list[str] = []

    required_files = [
        artifacts / "metrics_summary.csv",
        artifacts / "uplift_tree.txt",
        figures / "qini_curves.png",
        figures / "best_model_uplift_distribution.png",
        artifacts / "policy_simulation.csv",
        artifacts / "policy_best_models.csv",
        figures / "policy_incremental_conversions.png",
        artifacts / "covariate_balance.csv",
        artifacts / "propensity_scores.csv",
        artifacts / "confounding_summary.txt",
        figures / "propensity_overlap.png",
    ]

    if require_notebook_artifacts:
        required_files.extend(
            [
                figures / "notebook_qini_curves.png",
                figures / "notebook_shap_summary.png",
            ]
        )

    for path in required_files:
        _require_file(path, errors)

    metrics_df = _require_columns(
        artifacts / "metrics_summary.csv",
        {
            "model",
            "baseline_empirical_ate",
            "estimated_ate",
            "ate_ci_low",
            "ate_ci_high",
            "uplift_at_30pct",
            "qini_auc",
        },
        errors,
    )

    policy_df = _require_columns(
        artifacts / "policy_simulation.csv",
        {
            "model",
            "budget_fraction",
            "targeted_users",
            "uplift_rate",
            "expected_incremental_conversions",
        },
        errors,
    )

    best_df = _require_columns(
        artifacts / "policy_best_models.csv",
        {
            "model",
            "budget_fraction",
            "targeted_users",
            "uplift_rate",
            "expected_incremental_conversions",
        },
        errors,
    )

    balance_df = _require_columns(
        artifacts / "covariate_balance.csv",
        {
            "feature",
            "treated_mean",
            "control_mean",
            "standardized_mean_difference",
            "abs_smd",
        },
        errors,
    )

    propensity_df = _require_columns(
        artifacts / "propensity_scores.csv",
        {
            "propensity_score",
            "treatment",
        },
        errors,
    )

    summary_path = artifacts / "confounding_summary.txt"
    if summary_path.exists():
        summary_text = summary_path.read_text()
        expected_lines = [
            "Propensity AUC:",
            "Overlap share",
            "Features with |SMD|",
        ]
        for marker in expected_lines:
            if marker not in summary_text:
                errors.append(f"{summary_path} missing expected text: `{marker}`")

    tree_path = artifacts / "uplift_tree.txt"
    if tree_path.exists() and len(tree_path.read_text().strip()) < 20:
        warnings.append("Uplift tree summary looks unusually short; verify model training output.")

    if metrics_df is not None and metrics_df["model"].nunique() < 3:
        warnings.append("metrics_summary.csv has fewer than 3 unique models.")
    if metrics_df is not None:
        columns = ["baseline_empirical_ate", "uplift_at_30pct", "qini_auc"]
        if "validation_qini_auc" in metrics_df.columns:
            columns.append("validation_qini_auc")
        _require_finite(metrics_df, columns, "metrics_summary.csv", errors)
        # A single uplift tree has no library ATE/CI estimate; its missing values are explicit.
        estimated = metrics_df.loc[metrics_df["model"] != "Uplift Tree (KL)"]
        _require_finite(estimated, ["estimated_ate", "ate_ci_low", "ate_ci_high"],
                        "metrics_summary.csv", errors)

    if policy_df is not None:
        _require_finite(policy_df, ["budget_fraction", "targeted_users", "uplift_rate",
                                   "expected_incremental_conversions"],
                        "policy_simulation.csv", errors)
        if policy_df["budget_fraction"].nunique() < 3:
            warnings.append("policy_simulation.csv has fewer than 3 budget levels.")
        if policy_df["model"].nunique() < 3:
            warnings.append("policy_simulation.csv has fewer than 3 models.")

    if best_df is not None and policy_df is not None:
        _require_finite(best_df, ["budget_fraction", "targeted_users", "uplift_rate",
                                 "expected_incremental_conversions"],
                        "policy_best_models.csv", errors)
        unique_budgets = policy_df["budget_fraction"].nunique()
        if len(best_df) != unique_budgets:
            warnings.append(
                "policy_best_models.csv row count does not match number of budget levels "
                "in policy_simulation.csv."
            )

    if balance_df is not None:
        _require_finite(balance_df, ["treated_mean", "control_mean"], "covariate_balance.csv",
                        errors)
        # Infinite SMD denotes constant but separated arms; missing SMD is unsupported.
        if balance_df[["standardized_mean_difference", "abs_smd"]].apply(
            pd.to_numeric, errors="coerce",
        ).isna().any().any():
            errors.append("covariate_balance.csv contains unsupported SMD values.")
        high_imbalance = int((balance_df["abs_smd"] > 0.1).sum())
        if high_imbalance > 0:
            warnings.append(
                f"{high_imbalance} features have |SMD| > 0.10; investigate confounding risk."
            )

    if propensity_df is not None:
        _require_finite(propensity_df, ["propensity_score", "treatment"], "propensity_scores.csv",
                        errors)
        out_of_bounds = propensity_df[
            (propensity_df["propensity_score"] < 0.0)
            | (propensity_df["propensity_score"] > 1.0)
        ]
        if len(out_of_bounds) > 0:
            errors.append("propensity_scores.csv has values outside [0, 1].")

    passed = len(errors) == 0
    return VerificationResult(passed=passed, errors=errors, warnings=warnings)
