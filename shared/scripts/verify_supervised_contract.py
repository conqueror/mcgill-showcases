#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_SUPERVISED_FILES = [
    "artifacts/splits/split_manifest.json",
    "artifacts/eda/univariate_summary.csv",
    "artifacts/eda/bivariate_vs_target.csv",
    "artifacts/eda/missingness_summary.csv",
    "artifacts/eda/correlation_matrix.csv",
    "artifacts/leakage/leakage_report.csv",
    "artifacts/eval/metrics_summary.csv",
    "artifacts/experiments/experiment_log.csv",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_all_required_files(project_root: Path) -> bool:
    return all((project_root / rel_path).exists() for rel_path in REQUIRED_SUPERVISED_FILES)


def _bootstrap_project(project_root: Path, bootstrap_cmd: str) -> None:
    print(f"Bootstrapping supervised artifacts in {project_root} ...")
    subprocess.run(
        ["bash", "-lc", bootstrap_cmd],
        cwd=project_root,
        check=True,
    )


def _validate_split_manifest(path: Path) -> list[str]:
    payload = _load_json(path)
    required_keys = {
        "task_type",
        "strategy",
        "train_rows",
        "val_rows",
        "test_rows",
        "random_state",
        "no_overlap_checks_passed",
    }
    errors: list[str] = []
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        errors.append(f"{path}: missing keys {missing}")
        return errors

    for key in ("train_rows", "val_rows", "test_rows"):
        value = payload.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{path}: {key} must be positive integer")

    if payload.get("strategy") not in {"stratified", "group", "timeseries", "kfold", "random"}:
        errors.append(f"{path}: unsupported strategy `{payload.get('strategy')}`")

    if payload.get("task_type") not in {"classification", "regression"}:
        errors.append(f"{path}: unsupported task_type `{payload.get('task_type')}`")

    if payload.get("no_overlap_checks_passed") is not True:
        errors.append(f"{path}: no_overlap_checks_passed must be true")

    return errors


def _validate_experiment_log(path: Path) -> list[str]:
    required_cols = {
        "run_name",
        "timestamp_utc",
        "split_strategy",
        "primary_metric",
        "primary_metric_value",
    }
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: missing file"]

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    missing = sorted(required_cols - set(fieldnames))
    if missing:
        errors.append(f"{path}: missing columns {missing}")
    if len(rows) == 0:
        errors.append(f"{path}: experiment log is empty")
    return errors


def _validate_manifest_lists_required(project_root: Path) -> list[str]:
    manifest_path = project_root / "artifacts/manifest.json"
    if not manifest_path.exists():
        return [f"{manifest_path}: missing file"]

    payload = _load_json(manifest_path)
    required_files = set(payload.get("required_files", []))

    errors: list[str] = []
    for file_name in REQUIRED_SUPERVISED_FILES:
        if file_name not in required_files:
            errors.append(f"{manifest_path}: missing required_files entry `{file_name}`")
    return errors


def verify_project(project_root: Path) -> list[str]:
    errors: list[str] = []

    for rel_path in REQUIRED_SUPERVISED_FILES:
        file_path = project_root / rel_path
        if not file_path.exists():
            errors.append(f"{project_root}: missing `{rel_path}`")

    split_manifest_path = project_root / "artifacts/splits/split_manifest.json"
    if split_manifest_path.exists():
        errors.extend(_validate_split_manifest(split_manifest_path))

    exp_log_path = project_root / "artifacts/experiments/experiment_log.csv"
    errors.extend(_validate_experiment_log(exp_log_path))

    errors.extend(_validate_manifest_lists_required(project_root))

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate supervised showcase artifact contract")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("shared/config/supervised_projects.json"),
        help="Path to supervised project config",
    )
    parser.add_argument(
        "--bootstrap-missing",
        action="store_true",
        help="Run each project's bootstrap command when required artifacts are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = _load_json(config_path)
    projects = config.get("projects", [])

    all_errors: list[str] = []
    for entry in projects:
        if isinstance(entry, str):
            rel_path = entry
            bootstrap_cmd = None
        elif isinstance(entry, dict):
            rel_path = str(entry.get("path", ""))
            bootstrap_cmd = entry.get("bootstrap_cmd")
        else:
            all_errors.append(f"Unsupported project entry: {entry!r}")
            continue

        if not rel_path:
            all_errors.append(f"Project entry missing `path`: {entry!r}")
            continue

        project_path = repo_root / rel_path
        if not project_path.exists():
            all_errors.append(f"Missing project path: {project_path}")
            continue

        if args.bootstrap_missing and not _has_all_required_files(project_path):
            if not bootstrap_cmd:
                all_errors.append(f"Missing bootstrap_cmd for project: {project_path}")
                continue
            try:
                _bootstrap_project(project_path, bootstrap_cmd)
            except subprocess.CalledProcessError as exc:
                all_errors.append(
                    f"Bootstrap failed for {project_path} with exit code {exc.returncode}"
                )
                continue

        all_errors.extend(verify_project(project_path))

    if all_errors:
        print("Supervised contract verification failed:")
        for err in all_errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("Supervised contract verification passed.")


if __name__ == "__main__":
    main()
