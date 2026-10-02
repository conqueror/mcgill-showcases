from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from causal_showcase.verification import verify_learning_artifacts


def _write_file(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def _create_minimal_artifacts(
    project_root: Path,
    *,
    include_notebook_outputs: bool = False,
) -> None:
    artifacts = project_root / "artifacts"
    figures = artifacts / "figures"

    _write_file(artifacts / "uplift_tree.txt", "root -> split -> leaf\n")

    figures.mkdir(parents=True, exist_ok=True)
    for name in ["qini_curves", "best_model_uplift_distribution",
                 "policy_incremental_conversions", "propensity_overlap"]:
        mpimg.imsave(figures / f"{name}.png", np.ones((2, 2, 3)))

    if include_notebook_outputs:
        mpimg.imsave(figures / "notebook_qini_curves.png", np.ones((2, 2, 3)))
        mpimg.imsave(figures / "notebook_shap_summary.png", np.ones((2, 2, 3)))

    pd.DataFrame(
        {
            "model": ["S", "T", "X"],
            "baseline_empirical_ate": [0.02, 0.02, 0.02],
            "estimated_ate": [0.01, 0.02, 0.03],
            "ate_ci_low": [0.0, 0.01, 0.02],
            "ate_ci_high": [0.02, 0.03, 0.04],
            "uplift_at_30pct": [0.04, 0.05, 0.06],
            "qini_auc": [1.0, 1.1, 1.2],
            "validation_qini_auc": [1.0, 1.1, 1.2],
        }
    ).to_csv(artifacts / "metrics_summary.csv", index=False)

    pd.DataFrame(
        {
            "model": ["S", "T", "X", "S", "T", "X"],
            "budget_fraction": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
            "targeted_users": [10, 10, 10, 20, 20, 20],
            "uplift_rate": [0.01, 0.02, 0.03, 0.02, 0.03, 0.04],
            "expected_incremental_conversions": [0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
        }
    ).to_csv(artifacts / "policy_simulation.csv", index=False)

    pd.DataFrame(
        {
            "model": ["X", "T"],
            "budget_fraction": [0.1, 0.2],
            "targeted_users": [10, 20],
            "uplift_rate": [0.03, 0.03],
            "expected_incremental_conversions": [0.3, 0.6],
        }
    ).to_csv(artifacts / "policy_best_models.csv", index=False)

    pd.DataFrame(
        {
            "feature": ["f1", "f2"],
            "treated_mean": [1.0, 2.0],
            "control_mean": [1.1, 1.8],
            "standardized_mean_difference": [-0.1, 0.2],
            "abs_smd": [0.1, 0.2],
        }
    ).to_csv(artifacts / "covariate_balance.csv", index=False)

    pd.DataFrame(
        {
            "propensity_score": [0.1, 0.2, 0.8, 0.7],
            "treatment": [0, 0, 1, 1],
        }
    ).to_csv(artifacts / "propensity_scores.csv", index=False)

    summary = "\n".join(
        [
            "Confounding diagnostics summary",
            "Propensity AUC: 0.61",
            "Overlap share in [0.05, 0.95]: 0.98",
            "Features with |SMD| > 0.10: 1",
        ]
    )
    _write_file(artifacts / "confounding_summary.txt", summary)


def test_verify_learning_artifacts_passes_with_complete_outputs(tmp_path: Path) -> None:
    _create_minimal_artifacts(tmp_path)

    result = verify_learning_artifacts(tmp_path)

    assert result.passed is True
    assert result.errors == []


def test_verify_learning_artifacts_fails_when_required_files_missing(tmp_path: Path) -> None:
    _create_minimal_artifacts(tmp_path)
    (tmp_path / "artifacts" / "metrics_summary.csv").unlink()

    result = verify_learning_artifacts(tmp_path)

    assert result.passed is False
    assert any("metrics_summary.csv" in err for err in result.errors)


def test_verify_learning_artifacts_requires_notebook_outputs_when_requested(tmp_path: Path) -> None:
    _create_minimal_artifacts(tmp_path, include_notebook_outputs=False)

    result = verify_learning_artifacts(tmp_path, require_notebook_artifacts=True)

    assert result.passed is False
    assert any("notebook_qini_curves.png" in err for err in result.errors)


def test_verifier_rejects_placeholder_png(tmp_path: Path) -> None:
    _create_minimal_artifacts(tmp_path)
    (tmp_path / "artifacts/figures/qini_curves.png").write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
    result = verify_learning_artifacts(tmp_path)
    assert not result.passed
    assert any("qini_curves.png" in error for error in result.errors)


@pytest.mark.parametrize("filename,column", [("metrics_summary.csv", "estimated_ate"),
    ("metrics_summary.csv", "ate_ci_low"), ("metrics_summary.csv", "ate_ci_high"),
    ("metrics_summary.csv", "validation_qini_auc"),
    ("policy_simulation.csv", "uplift_rate"), ("propensity_scores.csv", "propensity_score")])
def test_verifier_rejects_nonfinite_estimates(tmp_path: Path, filename: str, column: str) -> None:
    _create_minimal_artifacts(tmp_path)
    path = tmp_path / "artifacts" / filename
    frame = pd.read_csv(path)
    frame.loc[0, column] = float("nan")
    frame.to_csv(path, index=False)
    result = verify_learning_artifacts(tmp_path)
    assert not result.passed
    assert any(filename in error for error in result.errors)


def test_verifier_rejects_missing_ate_interval_columns(tmp_path: Path) -> None:
    _create_minimal_artifacts(tmp_path)
    path = tmp_path / "artifacts/metrics_summary.csv"
    pd.read_csv(path).drop(columns=["ate_ci_low", "ate_ci_high"]).to_csv(path, index=False)
    result = verify_learning_artifacts(tmp_path)
    assert not result.passed
    assert any("ate_ci_low" in error for error in result.errors)
