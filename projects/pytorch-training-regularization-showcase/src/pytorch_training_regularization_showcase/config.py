"""Project-level constants for the PyTorch training showcase."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"
PROJECT_NAME = "pytorch-training-regularization-showcase"
PACKAGE_NAME = "pytorch_training_regularization_showcase"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
