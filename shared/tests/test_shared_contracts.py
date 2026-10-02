"""Regression checks for the shared supervised artifacts and split helpers."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/python"))
sys.path.insert(0, str(ROOT / "shared/scripts"))

import verify_supervised_contract as verifier  # noqa: E402
from ml_core.contracts import (  # noqa: E402
    merge_required_files,
    write_supervised_contract_artifacts,
)
from ml_core.explainability import (  # noqa: E402
    run_lime_local_explanations,
    run_shap_importance,
)
from ml_core.splits import build_supervised_split  # noqa: E402


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contents = {
            "artifacts/eda/univariate_summary.csv": (
                "feature,missing_ratio,n_unique,mean,std,min,p50,max,mode\nx,0,3,2,1,1,2,3,\n"
            ),
            "artifacts/eda/bivariate_vs_target.csv": (
                "feature,analysis_type,stat,value\nx,numeric_corr,pearson_corr,1\n"
            ),
            "artifacts/eda/missingness_summary.csv": (
                "feature,missing_count,missing_ratio\nx,0,0\n"
            ),
            "artifacts/eda/correlation_matrix.csv": ",x\nx,1\n",
            "artifacts/leakage/leakage_report.csv": (
                "check,feature,severity,value\nexact_target_match,x,high,1\n"
            ),
            "artifacts/eval/metrics_summary.csv": "metric,value\ntest_f1,0.75\n",
            "artifacts/experiments/experiment_log.csv": (
                "run_name,timestamp_utc,split_strategy,primary_metric,primary_metric_value\n"
                "demo,2026-10-01T00:00:00+00:00,stratified,test_f1,0.75\n"
            ),
        }
        for name, content in self.contents.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.split = {
            "task_type": "classification",
            "strategy": "stratified",
            "train_rows": 6,
            "val_rows": 2,
            "test_rows": 2,
            "random_state": 42,
            "no_overlap_checks_passed": True,
        }
        split_path = self.root / "artifacts/splits/split_manifest.json"
        split_path.parent.mkdir(parents=True)
        split_path.write_text(json.dumps(self.split), encoding="utf-8")
        self.source = self.root / "src/demo.py"
        self.source.parent.mkdir()
        self.source.write_text("SEED = 42\n", encoding="utf-8")
        self.refresh_manifest()

    def refresh_manifest(self) -> None:
        merge_required_files(
            self.root / "artifacts/manifest.json", verifier.REQUIRED_SUPERVISED_FILES
        )

    def test_valid_artifacts_and_intentional_leakage_findings_are_accepted(self) -> None:
        self.assertEqual(verifier.verify_project(self.root), [])

    def test_a_categorical_only_dataset_keeps_the_writers_empty_correlation_marker(self) -> None:
        frame = pd.DataFrame({"category": ["a", "b", "c"] * 10})
        target = pd.Series([0, 1] * 15)
        split = build_supervised_split(frame, target)
        required = write_supervised_contract_artifacts(
            project_root=self.root,
            frame=frame,
            target=target,
            split=split,
            task_type="classification",
            strategy="stratified",
            random_state=42,
            metrics={"f1": 0.75},
            run_name="categorical_demo",
        )
        merge_required_files(self.root / "artifacts/manifest.json", required)
        self.assertEqual(verifier.verify_project(self.root), [])

    def test_each_diagnostic_rejects_empty_headers_and_corrupt_values(self) -> None:
        for name, content in self.contents.items():
            with self.subTest(file=name):
                path = self.root / name
                for bad in ("", content.splitlines()[0] + "\n", "unrelated\nvalue\n"):
                    path.write_text(bad, encoding="utf-8")
                    self.refresh_manifest()
                    self.assertTrue(verifier.verify_project(self.root), (name, bad))
                path.write_text(content, encoding="utf-8")
        for name, bad in {
            "artifacts/eval/metrics_summary.csv": "metric,value\nf1,nan\n",
            "artifacts/eda/missingness_summary.csv": (
                "feature,missing_count,missing_ratio\nx,1,1.5\n"
            ),
            "artifacts/eda/correlation_matrix.csv": ",x\nx,2\n",
            "artifacts/leakage/leakage_report.csv": (
                "check,feature,severity,value\nexact_target_match,x,unknown,1\n"
            ),
            "artifacts/experiments/experiment_log.csv": self.contents[
                "artifacts/experiments/experiment_log.csv"
            ].replace("0.75", "not-a-number"),
        }.items():
            with self.subTest(file=name, corruption=bad):
                path = self.root / name
                original = path.read_text(encoding="utf-8")
                path.write_text(bad, encoding="utf-8")
                self.refresh_manifest()
                self.assertTrue(verifier.verify_project(self.root))
                path.write_text(original, encoding="utf-8")

    def test_ragged_csv_is_rejected(self) -> None:
        (self.root / "artifacts/eval/metrics_summary.csv").write_text(
            "metric,value\nf1,0.75,extra\n", encoding="utf-8"
        )
        self.refresh_manifest()
        self.assertTrue(verifier.verify_project(self.root))

    def test_negative_standard_deviation_is_rejected(self) -> None:
        path = self.root / "artifacts/eda/univariate_summary.csv"
        path.write_text(
            self.contents[str(path.relative_to(self.root))].replace(
                "x,0,3,2,1,1,2,3,", "x,0,3,2,-1,1,2,3,"
            ),
            encoding="utf-8",
        )
        self.refresh_manifest()
        self.assertTrue(verifier.verify_project(self.root))

    def test_split_manifest_rejects_invalid_types(self) -> None:
        path = self.root / "artifacts/splits/split_manifest.json"
        for key, value in (("train_rows", True), ("random_state", "42")):
            with self.subTest(key=key):
                path.write_text(json.dumps({**self.split, key: value}), encoding="utf-8")
                self.refresh_manifest()
                self.assertTrue(verifier.verify_project(self.root))

    def test_corrupt_json_returns_errors(self) -> None:
        for name in ("artifacts/splits/split_manifest.json", "artifacts/manifest.json"):
            with self.subTest(file=name):
                path = self.root / name
                original = path.read_text(encoding="utf-8")
                for bad in ("{", "[]"):
                    path.write_text(bad, encoding="utf-8")
                    self.assertTrue(verifier.verify_project(self.root))
                path.write_text(original, encoding="utf-8")

    def test_manifest_requires_version_and_a_list_of_paths(self) -> None:
        path = self.root / "artifacts/manifest.json"
        for payload in (
            {"required_files": verifier.REQUIRED_SUPERVISED_FILES},
            {"version": True, "required_files": verifier.REQUIRED_SUPERVISED_FILES},
            {"version": 1, "required_files": {}},
        ):
            with self.subTest(payload=payload):
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertTrue(verifier.verify_project(self.root))

    def test_random_strategy_is_in_the_schema(self) -> None:
        schema = json.loads((ROOT / "shared/contracts/split_manifest.schema.json").read_text())
        self.assertIn("random", schema["properties"]["strategy"]["enum"])

    def test_source_edits_make_artifacts_stale(self) -> None:
        self.source.write_text("SEED = 99\n", encoding="utf-8")
        self.assertTrue(verifier.verify_project(self.root))
        self.assertFalse(verifier._has_all_required_files(self.root))

    def test_stale_sources_trigger_the_cli_bootstrap(self) -> None:
        script = self.root / "scripts/bootstrap.py"
        script.parent.mkdir()
        script.write_text(
            "from pathlib import Path\n"
            "from ml_core.contracts import merge_required_files\n"
            "root = Path(__file__).resolve().parents[1]\n"
            "(root / 'bootstrapped').touch()\n"
            f"merge_required_files(root / 'artifacts/manifest.json', {verifier.REQUIRED_SUPERVISED_FILES!r})\n",
            encoding="utf-8",
        )
        config = self.root / "projects.json"
        config.write_text(
            json.dumps(
                {
                    "projects": [
                        {
                            "path": str(self.root),
                            "bootstrap_cmd": (
                                f"PYTHONPATH={shlex.quote(str(ROOT / 'shared/python'))} "
                                f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
                            ),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.refresh_manifest()
        self.source.write_text("SEED = 99\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "shared/scripts/verify_supervised_contract.py"),
                "--config",
                str(config),
                "--bootstrap-missing",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.root / "bootstrapped").is_file())

    def test_artifact_edits_invalidate_recorded_hashes(self) -> None:
        (self.root / "artifacts/eval/metrics_summary.csv").write_text(
            "metric,value\ntest_f1,0.99\n", encoding="utf-8"
        )
        self.assertTrue(verifier.verify_project(self.root))

    def test_other_manifest_outputs_are_also_fingerprinted(self) -> None:
        extra = self.root / "artifacts/summary.md"
        extra.write_text("original summary", encoding="utf-8")
        merge_required_files(self.root / "artifacts/manifest.json", ["artifacts/summary.md"])
        self.assertEqual(verifier.verify_project(self.root), [])
        extra.write_text("changed summary", encoding="utf-8")
        self.assertTrue(verifier.verify_project(self.root))

    def test_missing_listed_outputs_trigger_bootstrap(self) -> None:
        extra = self.root / "artifacts/optional.csv"
        extra.write_text("value\n1\n", encoding="utf-8")
        merge_required_files(self.root / "artifacts/manifest.json", ["artifacts/optional.csv"])
        extra.unlink()
        self.assertFalse(verifier._has_all_required_files(self.root))


class SplitAndExplanationTests(unittest.TestCase):
    def test_tied_times_stay_together_and_targets_remain_aligned(self) -> None:
        times = pd.Series([0] * 7 + [1] * 3 + [2] * 4 + [3] * 6)
        frame = pd.DataFrame({"time": times, "row": np.arange(20)})
        target = pd.Series(np.arange(20), name="target")
        order = np.random.default_rng(42).permutation(20)
        split = build_supervised_split(
            frame.iloc[order],
            target.iloc[order],
            strategy="timeseries",
            time_values=times.iloc[order],
        )
        self.assertLess(split.x_train.time.max(), split.x_val.time.min())
        self.assertLess(split.x_val.time.max(), split.x_test.time.min())
        # Requested cuts are 12 and 16; the nearest viable time-bucket cuts are 10 and 14.
        self.assertEqual(list(map(len, (split.x_train, split.x_val, split.x_test))), [10, 4, 6])
        for x, y in (
            (split.x_train, split.y_train),
            (split.x_val, split.y_val),
            (split.x_test, split.y_test),
        ):
            self.assertEqual(x.row.tolist(), y.tolist())
        self.assertEqual(sum(map(len, (split.x_train, split.x_val, split.x_test))), 20)

    def test_held_out_time_buckets_cannot_change_training_fit(self) -> None:
        from sklearn.impute import SimpleImputer

        times = pd.Series([0] * 7 + [1] * 3 + [2] * 4 + [3] * 6)
        frame = pd.DataFrame({"value": np.arange(20, dtype=float)})
        changed = frame.copy()
        changed.loc[times >= 2, "value"] = -1000.0
        fitted_means = []
        for data in (frame, changed):
            split = build_supervised_split(
                data, pd.Series(range(20)), strategy="timeseries", time_values=times
            )
            fitted_means.append(SimpleImputer().fit(split.x_train).statistics_[0])
        # Training buckets contain values 0 through 9: (0 + ... + 9) / 10 = 4.5.
        self.assertEqual(fitted_means, [4.5, 4.5])

    def test_unique_times_keep_the_existing_row_counts(self) -> None:
        frame = pd.DataFrame({"time": range(20)})
        split = build_supervised_split(
            frame, pd.Series(range(20)), strategy="timeseries", time_values=frame.time
        )
        self.assertEqual(list(map(len, (split.x_train, split.x_val, split.x_test))), [12, 4, 4])

    def test_time_split_rejects_missing_times_or_too_few_time_buckets(self) -> None:
        frame = pd.DataFrame({"x": range(10)})
        for times in (pd.Series([0] * 5 + [1] * 5), pd.Series([0, np.nan] + list(range(8)))):
            with self.subTest(times=times.tolist()), self.assertRaises(ValueError):
                build_supervised_split(
                    frame, pd.Series(range(10)), strategy="timeseries", time_values=times
                )

    def test_lime_uses_a_reproducible_seed_and_only_training_rows_to_fit(self) -> None:
        training_inputs: list[np.ndarray] = []
        seeds: list[int | None] = []

        class FakeExplainer:
            def __init__(self, training_data: np.ndarray, **kwargs: object) -> None:
                training_inputs.append(training_data.copy())
                seeds.append(kwargs.get("random_state"))
                self.rng = np.random.RandomState(kwargs.get("random_state"))

            def explain_instance(self, *args: object, **kwargs: object) -> object:
                return types.SimpleNamespace(as_list=lambda: [("x", float(self.rng.normal()))])

        fake_lime = types.ModuleType("lime.lime_tabular")
        fake_lime.LimeTabularExplainer = FakeExplainer
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(
                sys.modules, {"lime": types.ModuleType("lime"), "lime.lime_tabular": fake_lime}
            ),
        ):
            path = Path(temp) / "lime.csv"

            def predict(x: np.ndarray) -> np.ndarray:
                return np.full((len(x), 2), 0.5)

            run_lime_local_explanations(predict, frame, frame, output_path=path)
            first = path.read_bytes()
            run_lime_local_explanations(predict, frame, frame, output_path=path)
            self.assertEqual(path.read_bytes(), first)
            run_lime_local_explanations(predict, frame, frame * 1000, output_path=path)
            for actual in training_inputs:
                np.testing.assert_array_equal(actual, frame.to_numpy())
            self.assertEqual(seeds, [42, 42, 42])
            self.assertEqual(pd.read_csv(path)["random_state"].tolist(), [42, 42, 42])

            run_lime_local_explanations(predict, frame, frame, output_path=path, random_state=7)
            self.assertEqual(seeds[-1], 7)
            self.assertEqual(pd.read_csv(path)["random_state"].tolist(), [7, 7, 7])

    def test_lime_predictor_has_a_callable_type_annotation(self) -> None:
        self.assertEqual(
            get_type_hints(run_lime_local_explanations)["model_predict_proba"],
            Callable[[np.ndarray], np.ndarray],
        )

    @unittest.skipUnless(importlib.util.find_spec("shap"), "optional SHAP is not installed")
    def test_shap_accepts_the_logistic_pipeline_used_by_the_audit(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        frame = pd.DataFrame({"signal": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "constant": [0.0] * 6})
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(
            frame.to_numpy(), [0, 0, 0, 1, 1, 1]
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shap.csv"
            self.assertEqual(run_shap_importance(model, frame, output_path=path), "written")
            weights = pd.read_csv(path).set_index("feature")["mean_abs_shap"]
            self.assertEqual(set(weights.index), {"signal", "constant"})
            self.assertGreater(weights["signal"], 0)
            self.assertEqual(weights["constant"], 0)

    @unittest.skipUnless(importlib.util.find_spec("shap"), "optional SHAP is not installed")
    def test_shap_keeps_logit_units_for_a_supported_bare_logistic_model(self) -> None:
        from sklearn.linear_model import LogisticRegression

        frame = pd.DataFrame({"signal": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "constant": [0.0] * 6})
        model = LogisticRegression().fit(frame, [0, 0, 0, 1, 1, 1])
        model.coef_ = np.array([[2.0, 0.0]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shap.csv"
            self.assertEqual(run_shap_importance(model, frame, output_path=path), "written")
            weights = pd.read_csv(path).set_index("feature")["mean_abs_shap"]
            # Mean signal is zero: abs(2*x) is [6, 4, 2, 2, 4, 6]; its mean is 4.
            self.assertAlmostEqual(weights["signal"], 4.0)
            self.assertEqual(weights["constant"], 0.0)


if __name__ == "__main__":
    unittest.main()
