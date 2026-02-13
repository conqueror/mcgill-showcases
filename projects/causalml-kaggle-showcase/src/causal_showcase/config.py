from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "marketing_ab.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORT_PATH = ARTIFACTS_DIR / "metrics_summary.csv"
TREE_SUMMARY_PATH = ARTIFACTS_DIR / "uplift_tree.txt"
