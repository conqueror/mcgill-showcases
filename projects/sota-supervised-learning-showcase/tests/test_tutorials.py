import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sota_supervised_showcase import classification
from sota_supervised_showcase.data import (
    load_digits_split,
    rebalance_binary_training_data,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "notebooks").glob("*.ipynb")))
def test_notebook_code_is_python(path: Path) -> None:
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert not any(line.endswith("\\n") for line in cell["source"])
            ast.parse(source, filename=str(path))


def test_notebook_validator_rejects_invalid_python(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    notebooks = tmp_path / "notebooks"
    scripts.mkdir()
    notebooks.mkdir()
    validator = scripts / "validate_notebooks.sh"
    validator.write_text((ROOT / "scripts/validate_notebooks.sh").read_text())
    (notebooks / "bad.ipynb").write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "code", "source": ["def broken(:\n"]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    # Keep the existing uv command offline by forwarding only `uv run python`.
    uv = tmp_path / "uv"
    uv.write_text(f'#!/bin/sh\nshift 2\nexec "{sys.executable}" "$@"\n')
    uv.chmod(0o755)
    import os

    result = subprocess.run(
        ["bash", str(validator)],
        env=dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}"),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert "SyntaxError" in result.stderr


def test_domain_notebook_grades_correct_and_incorrect_answers() -> None:
    notebook = json.loads((ROOT / "notebooks/02_domain_case_studies.ipynb").read_text())
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "domain_notebook", "exec"), namespace)
    correct = namespace["check_case"]("C1", "binary", "recall", "high_recall")
    incorrect = namespace["check_case"]("C1", "regression", "rmse", "balanced")
    assert correct["score_out_of_3"] == 3
    assert incorrect["score_out_of_3"] == 0


@pytest.mark.parametrize(
    ("strategy", "expected_rows"),
    [("upsample_minority", 18), ("downsample_majority", 6)],
)
def test_resampling_uses_actual_minority(strategy: str, expected_rows: int) -> None:
    features = np.arange(12).reshape(-1, 1)
    labels = np.array([1] * 9 + [0] * 3)
    result_x, result_y = rebalance_binary_training_data(features, labels, strategy)
    assert len(result_x) == len(result_y) == expected_rows
    assert np.bincount(result_y).tolist() == [expected_rows // 2] * 2


def test_multiclass_comparison_uses_identical_base_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimators: list[Any] = []
    for name in ["OneVsRestClassifier", "OneVsOneClassifier"]:
        constructor = getattr(classification, name)

        def observe(estimator: Any, constructor: Any = constructor) -> Any:
            estimators.append(estimator)
            return constructor(estimator)

        monkeypatch.setattr(classification, name, observe)
    classification.evaluate_multiclass_strategies(load_digits_split())
    for estimator in estimators:
        assert isinstance(estimator, Pipeline)
        assert isinstance(estimator.steps[0][1], StandardScaler)
        assert isinstance(estimator.steps[1][1], LogisticRegression)
    assert (
        estimators[0].steps[1][1].get_params() == estimators[1].steps[1][1].get_params()
    )


def test_documented_manual_boosting_runs() -> None:
    document = (ROOT / "docs/code-examples.md").read_text()
    code = re.findall(r"```python\n(.*?)```", document, re.DOTALL)[-1]
    namespace: dict[str, Any] = {}
    exec(compile(code, "manual_boosting_example", "exec"), namespace)
    assert namespace["y_pred"].shape == namespace["y_test"].shape
    assert np.isfinite(namespace["y_pred"]).all()
