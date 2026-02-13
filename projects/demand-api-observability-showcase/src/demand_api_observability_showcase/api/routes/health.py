from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from demand_api_observability_showcase import __version__

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
