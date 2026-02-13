"""Central configuration for the learning showcase pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ShowcaseConfig(BaseModel):
    """Configuration knobs tuned for practical runtime and reproducibility."""

    dataset: Literal["digits", "business"] = Field(default="digits")
    random_state: int = Field(default=42, ge=0)
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    labeled_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    semisup_kmeans_clusters: int = Field(default=50, ge=2)
    self_training_threshold: float = Field(default=0.85, gt=0.5, lt=1.0)
    contrastive_epochs: int = Field(default=8, ge=1)
    contrastive_batch_size: int = Field(default=128, ge=16)
    contrastive_embedding_dim: int = Field(default=32, ge=4)
    dec_pretrain_epochs: int = Field(default=8, ge=1)
    dec_finetune_epochs: int = Field(default=10, ge=1)
    active_learning_rounds: int = Field(default=6, ge=2, le=30)
    active_learning_query_size: int = Field(default=25, ge=5, le=500)
    business_read_rows: int = Field(default=60_000, ge=1_000)
    business_sample_size: int = Field(default=3_000, ge=500)
    business_csv_path: Path | None = None


class PathsConfig(BaseModel):
    """Filesystem layout for reproducible runs."""

    project_root: Path
    artifacts_dir: Path
    reports_dir: Path
    figures_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> PathsConfig:
        artifacts_dir = project_root / "artifacts"
        return cls(
            project_root=project_root,
            artifacts_dir=artifacts_dir,
            reports_dir=artifacts_dir / "reports",
            figures_dir=artifacts_dir / "figures",
        )

    def ensure(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
