from __future__ import annotations

from pathlib import Path

import joblib

from demand_api_observability_showcase.model.bundle import ModelBundle


class ModelStore:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._bundle: ModelBundle | None = None

    @property
    def bundle(self) -> ModelBundle | None:
        return self._bundle

    def load(self) -> None:
        if not self._model_path.exists():
            self._bundle = None
            return

        loaded = joblib.load(self._model_path)
        if not isinstance(loaded, ModelBundle):
            raise TypeError(f"Unexpected model bundle type: {type(loaded)}")
        self._bundle = loaded
