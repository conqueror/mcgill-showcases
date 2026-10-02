#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from ml_core.provenance import artifact_hashes, source_digest  # noqa: E402

SPLIT_STRATEGIES = {"stratified", "group", "timeseries", "kfold", "random"}

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _has_all_required_files(project_root: Path) -> bool:
    if not all((project_root / name).is_file() for name in REQUIRED_SUPERVISED_FILES):
        return False
    try:
        return not _validate_provenance(project_root)
    except (OSError, ValueError, TypeError):
        return False


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
        if type(value) is not int or value <= 0:
            errors.append(f"{path}: {key} must be positive integer")

    if payload.get("strategy") not in SPLIT_STRATEGIES:
        errors.append(f"{path}: unsupported strategy `{payload.get('strategy')}`")

    if payload.get("task_type") not in {"classification", "regression"}:
        errors.append(f"{path}: unsupported task_type `{payload.get('task_type')}`")

    if payload.get("no_overlap_checks_passed") is not True:
        errors.append(f"{path}: no_overlap_checks_passed must be true")

    if type(payload.get("random_state")) is not int:
        errors.append(f"{path}: random_state must be an integer")
    for key in ("group_column", "time_column"):
        if payload.get(key) is not None and not isinstance(payload[key], str):
            errors.append(f"{path}: {key} must be a string or null")

    return errors


def _read_csv(path: Path, required_cols: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Reject missing headers, empty tables, and rows with the wrong field count."""
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    missing = sorted(required_cols - set(fieldnames))
    if missing:
        errors.append(f"{path}: missing columns {missing}")
    if len(fieldnames) != len(set(fieldnames)):
        errors.append(f"{path}: duplicate column names")
    if not rows:
        errors.append(f"{path}: table is empty")
    if any(None in row or None in row.values() for row in rows):
        errors.append(f"{path}: row field count does not match the header")
    return errors, rows


def _number(value: str, *, low: float = -math.inf, high: float = math.inf) -> float:
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"invalid numeric value {value!r}")
    return result


def _validate_table(path: Path) -> list[str]:
    if path.name == "correlation_matrix.csv" and path.read_text(encoding="utf-8").strip() == '""':
        # pandas writes this marker when there are no numeric features to correlate.
        errors, summaries = _read_csv(path.with_name("univariate_summary.csv"), {"feature", "mean"})
        if not errors and all(row.get("mean") == "" for row in summaries):
            return []
    columns = {
        "univariate_summary.csv": {
            "feature",
            "missing_ratio",
            "n_unique",
            "mean",
            "std",
            "min",
            "p50",
            "max",
            "mode",
        },
        "bivariate_vs_target.csv": {"feature", "analysis_type", "stat", "value"},
        "missingness_summary.csv": {"feature", "missing_count", "missing_ratio"},
        "leakage_report.csv": {"check", "feature", "severity", "value"},
        "metrics_summary.csv": {"metric", "value"},
        "experiment_log.csv": {
            "run_name",
            "timestamp_utc",
            "split_strategy",
            "primary_metric",
            "primary_metric_value",
        },
        "correlation_matrix.csv": {""},
    }
    errors, rows = _read_csv(path, columns[path.name])
    if errors:
        return errors
    for line, row in enumerate(rows, start=2):
        try:
            if path.name == "correlation_matrix.csv":
                labels = set(row) - {""}
                if not labels or {r[""] for r in rows} != labels or len(rows) != len(labels):
                    raise ValueError("correlation row and column labels must match")
                for label in labels:
                    if row[label] != "":  # Undefined correlations for constant columns are valid.
                        _number(row[label], low=-1, high=1)
                continue
            if path.name == "leakage_report.csv":
                if not row["check"] or row["severity"] not in {"info", "medium", "high"}:
                    raise ValueError("invalid leakage check or severity")
                _number(row["value"])
                continue
            if path.name == "experiment_log.csv":
                if any(not row[key].strip() for key in columns[path.name]):
                    raise ValueError("experiment fields must not be blank")
                if row["split_strategy"] not in SPLIT_STRATEGIES:
                    raise ValueError("unsupported experiment split strategy")
                timestamp = datetime.fromisoformat(row["timestamp_utc"])
                offset = timestamp.utcoffset()
                if offset is None or offset.total_seconds() != 0:
                    raise ValueError("timestamp_utc must include a UTC offset")
                _number(row["primary_metric_value"])
                continue
            name = "metric" if path.name == "metrics_summary.csv" else "feature"
            if not row[name].strip():
                raise ValueError(f"{name} must not be blank")
            if "missing_ratio" in row:
                _number(row["missing_ratio"], low=0, high=1)
            for key in ("missing_count", "n_unique"):
                if key in row and not _number(row[key], low=0).is_integer():
                    raise ValueError(f"{key} must be a nonnegative integer")
            for key in ("mean", "std", "min", "p50", "max"):
                if key in row and row[key] != "":
                    _number(row[key], low=0 if key == "std" else -math.inf)
            if path.name == "bivariate_vs_target.csv":
                if row["analysis_type"] not in {"numeric_corr", "category_target_mean"}:
                    raise ValueError("unsupported bivariate analysis_type")
                if not row["stat"].strip():
                    raise ValueError("bivariate stat must not be blank")
                if row["value"] != "":  # Constant features can have undefined correlation.
                    value = _number(row["value"])
                    if row["analysis_type"] == "numeric_corr" and not -1 <= value <= 1:
                        raise ValueError("Pearson correlation must be between -1 and 1")
            elif "value" in row:
                _number(row["value"])
        except (ValueError, TypeError) as exc:
            errors.append(f"{path}:{line}: {exc}")
    return errors


def _validate_experiment_log(path: Path) -> list[str]:
    return _validate_table(path)


def _validate_manifest_lists_required(project_root: Path) -> list[str]:
    manifest_path = project_root / "artifacts/manifest.json"
    if not manifest_path.exists():
        return [f"{manifest_path}: missing file"]

    payload = _load_json(manifest_path)
    errors: list[str] = []
    if type(payload.get("version")) is not int or payload["version"] < 1:
        errors.append(f"{manifest_path}: version must be a positive integer")
    paths = payload.get("required_files")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(name, str) or not name for name in paths)
    ):
        return errors + [f"{manifest_path}: required_files must be a nonempty list of paths"]
    required_files = set(paths)
    for file_name in REQUIRED_SUPERVISED_FILES:
        if file_name not in required_files:
            errors.append(f"{manifest_path}: missing required_files entry `{file_name}`")
    return errors


def _validate_provenance(project_root: Path) -> list[str]:
    manifest_path = project_root / "artifacts/manifest.json"
    payload = _load_json(manifest_path)
    errors: list[str] = []
    if payload.get("source_sha256") != source_digest(project_root):
        errors.append(f"{manifest_path}: source/configuration changed; regenerate artifacts")
    paths = payload.get("required_files")
    if not isinstance(paths, list) or any(not isinstance(name, str) for name in paths):
        return errors + [f"{manifest_path}: required_files must be a list of paths"]
    expected_hashes = artifact_hashes(project_root, paths)
    recorded_hashes = payload.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict) or any(
        recorded_hashes.get(name) != digest for name, digest in expected_hashes.items()
    ):
        errors.append(f"{manifest_path}: artifact hashes missing or changed; regenerate artifacts")
    return errors


def verify_project(project_root: Path) -> list[str]:
    errors: list[str] = []

    for rel_path in REQUIRED_SUPERVISED_FILES:
        file_path = project_root / rel_path
        if not file_path.is_file():
            errors.append(f"{project_root}: missing `{rel_path}`")
            continue
        try:
            validator = _validate_split_manifest if file_path.suffix == ".json" else _validate_table
            errors.extend(validator(file_path))
        except (OSError, ValueError, TypeError, csv.Error) as exc:
            errors.append(f"{file_path}: {exc}")

    for validator in (_validate_manifest_lists_required, _validate_provenance):
        try:
            errors.extend(validator(project_root))
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{project_root}: {exc}")

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
        help="Run bootstrap commands when required artifacts are missing or stale.",
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
    if not isinstance(projects, list) or not projects:
        raise SystemExit(
            "Supervised contract verification failed: projects must be a nonempty list"
        )

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
