from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def append_run_tracking(
    tracking_path: Path,
    *,
    run_name: str,
    metrics: dict[str, float],
    notes: str,
) -> pd.DataFrame:
    """Append run metadata to a CSV file and return the full run log."""
    tracking_path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "notes": notes,
    }
    row.update(metrics)

    if tracking_path.exists():
        existing = pd.read_csv(tracking_path)
    else:
        existing = pd.DataFrame()

    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(tracking_path, index=False)
    return updated
