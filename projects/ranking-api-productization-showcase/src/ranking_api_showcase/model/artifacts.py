from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lightgbm import Booster


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    booster: Booster
    feature_names: list[str]
    meta: dict[str, object] | None


def _load_feature_names(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
        raise ValueError(f"Invalid feature schema at {path}; expected JSON list[str].")
    if len(raw) == 0:
        raise ValueError(f"Invalid feature schema at {path}; empty feature list.")
    return list(raw)


def _load_meta(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid model meta at {path}; expected JSON object.")
    return {str(k): v for k, v in raw.items()}


def load_artifacts(model_path: Path, feature_names_path: Path, meta_path: Path) -> ModelArtifacts:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")

    booster = Booster(model_file=str(model_path))
    feature_names = _load_feature_names(feature_names_path)
    meta = _load_meta(meta_path)
    return ModelArtifacts(booster=booster, feature_names=feature_names, meta=meta)
