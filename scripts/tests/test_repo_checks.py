"""Checks for root command failure handling and public-ledger privacy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_digits_bundle(root: Path) -> dict[str, str]:
    """A tiny valid bundle, specified independently of the verifier's table registry."""
    reports = root / "artifacts/reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "digits",
        "dataset": "sklearn_digits",
        "n_samples": 100,
        "n_features": 4,
        "n_classes": 2,
        "labeled_fraction_train": 0.1,
        "active_learning_gain_vs_random_at_final_round": 0.1,
        "best_clustering_algorithm": "KMeans",
        "best_semisup_method": "demo",
        "best_selfsup_method": "demo",
        "best_active_learning_strategy": "random",
    }
    (reports / "digits_run_summary.json").write_text(json.dumps(summary))
    tables = {
        "clustering_metrics": "algorithm,ari,nmi\nKMeans,1,1\n",
        "kmeans_model_selection": "k,inertia\n2,1\n",
        "anomaly_metrics": "algorithm,precision,recall,f1\ndemo,1,1,1\n",
        "semi_supervised_metrics": "method,accuracy,f1_macro\ndemo,1,1\n",
        "self_supervised_metrics": "method,accuracy,f1_macro\ndemo,1,1\n",
        "dec_metrics": "algorithm,ari,nmi\nDEC,1,1\n",
        "active_learning_metrics": "strategy,accuracy,f1_macro\nrandom,1,1\n",
    }
    for table, content in tables.items():
        (reports / f"digits_{table}.csv").write_text(content)
    return tables


class RootCommandTests(unittest.TestCase):
    def test_verify_runs_a_verifier_even_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Makefile").write_bytes((ROOT / "Makefile").read_bytes())
            result = subprocess.run(
                ["make", "verify"], cwd=root, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("Skipping", result.stdout)

    def test_verify_does_not_just_detect_supervised_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Makefile").write_bytes((ROOT / "Makefile").read_bytes())
            for name in ("sota-supervised-learning-showcase", "sota-unsupervised-semisup-showcase"):
                artifacts = root / "projects" / name / "artifacts"
                artifacts.mkdir(parents=True)
                (artifacts / "summary.md").write_text("corrupt")
                (artifacts / "reports").mkdir()
                (artifacts / "reports/digits_run_summary.json").write_text("{}")
            result = subprocess.run(
                ["make", "verify"], cwd=root, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("artifacts detected", result.stdout.lower())

    def test_unsupervised_corruption_fails_even_when_other_verifiers_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            makefile = (ROOT / "Makefile").read_text()
            (root / "Makefile").write_text(makefile)
            for line in makefile.splitlines():
                if "_DIR := projects/" in line:
                    directory = root / line.split(":=", 1)[1].strip()
                    directory.mkdir(parents=True)
                    (directory / "Makefile").write_text("verify:\n\t@true\n")
            shutil.copytree(ROOT / "shared", root / "shared")
            verifier = root / "shared/scripts/verify_supervised_contract.py"
            verifier.write_text(
                verifier.read_text().replace(
                    'if __name__ == "__main__":\n    main()',
                    'if __name__ == "__main__":\n    pass',
                )
            )
            (root / "scripts").mkdir()
            shutil.copy(ROOT / "scripts/verify_unsupervised_artifacts.py", root / "scripts")
            reports = root / "projects/sota-unsupervised-semisup-showcase/artifacts/reports"
            reports.mkdir(parents=True)
            (reports / "digits_run_summary.json").write_text("{}")
            result = subprocess.run(
                ["make", "verify"], cwd=root, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("summary must identify the digits dataset", result.stdout)
            write_digits_bundle(root / "projects/sota-unsupervised-semisup-showcase")
            valid_result = subprocess.run(
                ["make", "verify"], cwd=root, text=True, capture_output=True, check=False
            )
            self.assertEqual(valid_result.returncode, 0, valid_result.stdout + valid_result.stderr)

    def test_smoke_includes_the_offline_causal_contract_baseline(self) -> None:
        result = subprocess.run(
            ["make", "-n", "smoke"], cwd=ROOT, text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scripts/run_contract_baseline.py", result.stdout)

    def test_preflight_has_no_required_ignored_default_plan(self) -> None:
        env = os.environ.copy()
        env.pop("HARNESS_PLAN_PATH", None)
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/dev/harness-cli-preflight.sh")],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_an_explicit_missing_plan_still_fails(self) -> None:
        env = {**os.environ, "HARNESS_PLAN_PATH": "/nonexistent/explicit-task-plan.md"}
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/dev/harness-cli-preflight.sh")],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing file: /nonexistent/explicit-task-plan.md", result.stdout)


class PublicDocumentationTests(unittest.TestCase):
    def test_run_ledgers_have_no_private_absolute_paths(self) -> None:
        for path in (ROOT / "docs/agents/runs").glob("*.md"):
            with self.subTest(path=path.name):
                self.assertNotRegex(path.read_text(), r"/(?:Users|home|private)/")


class ArtifactCommandTests(unittest.TestCase):
    def test_empty_project_registry_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "projects.json"
            config.write_text('{"projects": []}')
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "shared/scripts/verify_supervised_contract.py"),
                    "--config",
                    str(config),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_unsupervised_bundle_rejects_empty_and_corrupt_metric_tables(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from verify_unsupervised_artifacts import verify_project

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tables = write_digits_bundle(root)
            reports = root / "artifacts/reports"
            self.assertEqual(verify_project(root), [])
            for table in tables:
                path = reports / f"digits_{table}.csv"
                content = path.read_text()
                with self.subTest(table=table):
                    path.write_text(content.splitlines()[0] + "\n")
                    self.assertTrue(verify_project(root))
                    path.write_text(content.replace(",1", ",nan") if ",1" in content else "corrupt")
                    self.assertTrue(verify_project(root))
                    path.write_text(content)

    def test_unsupervised_summary_cannot_name_an_unrelated_dataset(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from verify_unsupervised_artifacts import verify_project

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_digits_bundle(root)
            path = root / "artifacts/reports/digits_run_summary.json"
            payload = json.loads(path.read_text())
            payload["dataset"] = "unrelated"
            path.write_text(json.dumps(payload))
            self.assertTrue(verify_project(root))


if __name__ == "__main__":
    unittest.main()
