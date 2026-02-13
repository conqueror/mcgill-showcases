#!/usr/bin/env bash
set -euo pipefail
uv run uvicorn mlops_drift_showcase.serve:app --host 127.0.0.1 --port 8000
