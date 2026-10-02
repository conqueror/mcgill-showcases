#!/usr/bin/env python
"""Validate required demand API artifacts and OpenAPI contract file."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from demand_api_observability_showcase.api.app import create_app
from demand_api_observability_showcase.model.store import ModelStore
from demand_api_observability_showcase.settings import Settings

REQUIRED_FILES = [
    "artifacts/model.joblib",
    "artifacts/metrics.json",
    "openapi.json",
]


def main() -> None:
    """Fail with missing-path details when required outputs are absent."""

    root = Path(__file__).resolve().parents[1]
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    try:
        for name in REQUIRED_FILES:
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Empty or invalid artifact: {name}")
        store = ModelStore(root / "artifacts/model.joblib")
        store.load()
        bundle = store.bundle
        if bundle is None:
            raise ValueError("Model bundle is missing")
        metrics = json.loads((root / "artifacts/metrics.json").read_text())
        expected_metrics = {
            "model_version": bundle.model_version,
            "trained_at_iso": bundle.trained_at_iso,
            **asdict(bundle.metrics),
        }
        if metrics != expected_metrics:
            raise ValueError("Metrics JSON does not match the model bundle")
        spec = json.loads((root / "openapi.json").read_text())
        if spec != create_app(Settings(otel_enabled=False)).openapi():
            raise ValueError("OpenAPI contract does not match the API")
    except Exception as exc:
        raise SystemExit(f"Invalid artifact contents: {exc}") from exc


if __name__ == "__main__":
    main()
