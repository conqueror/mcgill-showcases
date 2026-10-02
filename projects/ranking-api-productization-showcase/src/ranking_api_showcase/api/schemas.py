from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelSchemaResponse(BaseModel):
    feature_names: list[str]
    meta: dict[str, object] | None = None


class PlayerRecord(BaseModel):
    player_id: str = Field(min_length=1)
    features: dict[str, float]


class ScoreRequest(BaseModel):
    records: list[PlayerRecord] = Field(min_length=1, max_length=1000)


class PlayerScore(BaseModel):
    player_id: str
    score: float


class ScoreResponse(BaseModel):
    scores: list[PlayerScore]


class PlayerRank(BaseModel):
    player_id: str
    score: float
    rank: int = Field(ge=1)


class RankResponse(BaseModel):
    rankings: list[PlayerRank]
