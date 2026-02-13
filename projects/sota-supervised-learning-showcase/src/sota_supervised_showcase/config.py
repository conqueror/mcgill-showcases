"""Project-level constants used across demos."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRACE_ID = "sota-supervised-learning-showcase"
TENANT_ID = "learner"
