from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_drift_showcase.train import load_model


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1)


class PredictResponse(BaseModel):
    prediction: int
    probability: float


def _load_default_model() -> Any:
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/model/model.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return load_model(model_path)


def create_app(model: Any | None = None) -> FastAPI:
    active_model = model
    if active_model is None:
        try:
            active_model = _load_default_model()
        except FileNotFoundError:
            active_model = None
    app = FastAPI(title="MLOps Drift Showcase API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        if active_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifact missing. Run training pipeline first.",
            )
        try:
            prob = float(active_model.predict_proba([payload.features])[0][1])
        except Exception as exc:  # pragma: no cover - defensive conversion
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

        pred = int(prob >= 0.5)
        return PredictResponse(prediction=pred, probability=prob)

    return app


app = create_app()
