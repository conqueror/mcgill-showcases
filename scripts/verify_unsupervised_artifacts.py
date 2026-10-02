#!/usr/bin/env python3
"""Validate the digits smoke bundle used by root make verify."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared/scripts"))
from verify_supervised_contract import _load_json, _number, _read_csv  # noqa: E402

TABLES: dict[str, tuple[str | None, dict[str, tuple[float, float]]]] = {
    "clustering_metrics": ("algorithm", {"ari": (-1, 1), "nmi": (0, 1)}),
    "kmeans_model_selection": (None, {"k": (1, float("inf")), "inertia": (0, float("inf"))}),
    "anomaly_metrics": ("algorithm", {"precision": (0, 1), "recall": (0, 1), "f1": (0, 1)}),
    "semi_supervised_metrics": ("method", {"accuracy": (0, 1), "f1_macro": (0, 1)}),
    "self_supervised_metrics": ("method", {"accuracy": (0, 1), "f1_macro": (0, 1)}),
    "dec_metrics": ("algorithm", {"ari": (-1, 1), "nmi": (0, 1)}),
    "active_learning_metrics": ("strategy", {"accuracy": (0, 1), "f1_macro": (0, 1)}),
}


def verify_project(project_root: Path) -> list[str]:
    """Reject missing, empty, malformed, or out-of-range summary/metric artifacts."""
    reports = project_root / "artifacts/reports"
    errors: list[str] = []
    try:
        summary = _load_json(reports / "digits_run_summary.json")
        if summary.get("mode") != "digits" or summary.get("dataset") != "sklearn_digits":
            raise ValueError("summary must identify the digits dataset")
        for key in ("n_samples", "n_features", "n_classes"):
            if type(summary.get(key)) is not int or summary[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        _number(str(summary["labeled_fraction_train"]), low=0, high=1)
        _number(str(summary["active_learning_gain_vs_random_at_final_round"]), low=-1, high=1)
        for key in (
            "best_clustering_algorithm",
            "best_semisup_method",
            "best_selfsup_method",
            "best_active_learning_strategy",
        ):
            if not isinstance(summary.get(key), str) or not summary[key].strip():
                raise ValueError(f"{key} must identify a method")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"{reports / 'digits_run_summary.json'}: {exc}")
    for table, (name, metrics) in TABLES.items():
        path = reports / f"digits_{table}.csv"
        try:
            columns = set(metrics) | ({name} if name else set())
            table_errors, rows = _read_csv(path, columns)
            errors.extend(table_errors)
            if table_errors:
                continue
            for row in rows:
                if name and not row[name].strip():
                    raise ValueError(f"{name} must not be blank")
                for metric, (low, high) in metrics.items():
                    _number(row[metric], low=low, high=high)
        except (OSError, ValueError, TypeError, csv.Error) as exc:
            errors.append(f"{path}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    errors = verify_project(parser.parse_args().project_root)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("Unsupervised digits artifact verification passed.")


if __name__ == "__main__":
    main()
