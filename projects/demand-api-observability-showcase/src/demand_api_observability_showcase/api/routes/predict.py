from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from demand_api_observability_showcase.model.features import features_from_datetime
from demand_api_observability_showcase.model.store import ModelStore

router = APIRouter()


class PredictRequest(BaseModel):
    pickup_zone_id: int = Field(ge=1, le=263)
    pickup_datetime: datetime


class PredictResponse(BaseModel):
    pickup_zone_id: int
    pickup_datetime: datetime
    predicted_pickups: float
    model_version: str


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    model_store: ModelStore = request.app.state.model_store
    bundle = model_store.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: make train-demo")

    features = features_from_datetime(body.pickup_zone_id, body.pickup_datetime)
    matrix = pd.DataFrame([asdict(features)])
    prediction = float(bundle.model.predict(matrix)[0])

    return PredictResponse(
        pickup_zone_id=body.pickup_zone_id,
        pickup_datetime=body.pickup_datetime,
        predicted_pickups=prediction,
        model_version=bundle.model_version,
    )
