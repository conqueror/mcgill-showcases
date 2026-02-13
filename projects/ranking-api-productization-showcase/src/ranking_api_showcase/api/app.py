from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from ranking_api_showcase.api.schemas import (
    HealthResponse,
    ModelSchemaResponse,
    PlayerRank,
    PlayerScore,
    RankResponse,
    ScoreRequest,
    ScoreResponse,
)
from ranking_api_showcase.config import Settings, load_settings
from ranking_api_showcase.logging import configure_logging
from ranking_api_showcase.model.artifacts import ModelArtifacts, load_artifacts
from ranking_api_showcase.model.scoring import argsort_desc, build_feature_matrix, score

logger = logging.getLogger(__name__)


def _get_trace_id(request: Request) -> str:
    header_value = request.headers.get("x-trace-id") or request.headers.get("x-request-id")
    if header_value:
        return header_value
    return str(uuid.uuid4())


def _require_artifacts(app: FastAPI) -> ModelArtifacts:
    artifacts = cast(ModelArtifacts | None, getattr(app.state, "artifacts", None))
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model first.")
    return artifacts


def create_app(settings: Settings | None = None, *, load_model: bool = True) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="ranking-api-productization-showcase", version="0.1.0")
    app.state.settings = settings

    if load_model:
        try:
            app.state.artifacts = load_artifacts(
                model_path=settings.model_path,
                feature_names_path=settings.feature_names_path,
                meta_path=settings.model_meta_path,
            )
            logger.info(
                "model_loaded",
                extra={
                    "model_path": str(settings.model_path),
                    "feature_names_path": str(settings.feature_names_path),
                },
            )
        except Exception:
            app.state.artifacts = None
            logger.exception(
                "model_load_failed",
                extra={
                    "model_path": str(settings.model_path),
                    "feature_names_path": str(settings.feature_names_path),
                },
            )
            if settings.fail_on_model_load:
                raise
    else:
        app.state.artifacts = None
        logger.info("model_load_skipped")

    @app.middleware("http")
    async def request_logging_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        trace_id = _get_trace_id(request)
        tenant_id = request.headers.get("x-tenant-id")
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "request_failed",
                extra={
                    "trace_id": trace_id,
                    "tenant_id": tenant_id,
                    "route": request.url.path,
                    "method": request.method,
                    "status": 500,
                    "latency_ms": latency_ms,
                },
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"},
                headers={"x-trace-id": trace_id},
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request",
            extra={
                "trace_id": trace_id,
                "tenant_id": tenant_id,
                "route": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        response.headers["x-trace-id"] = trace_id
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        artifacts = cast(ModelArtifacts | None, getattr(request.app.state, "artifacts", None))
        return HealthResponse(status="ok", model_loaded=artifacts is not None)

    @app.get("/model/schema", response_model=ModelSchemaResponse)
    async def model_schema(request: Request) -> ModelSchemaResponse:
        artifacts = _require_artifacts(request.app)
        return ModelSchemaResponse(feature_names=artifacts.feature_names, meta=artifacts.meta)

    @app.post("/score", response_model=ScoreResponse)
    async def score_endpoint(request: Request, payload: ScoreRequest) -> ScoreResponse:
        artifacts = _require_artifacts(request.app)
        try:
            matrix = build_feature_matrix(
                feature_dicts=[record.features for record in payload.records],
                feature_names=artifacts.feature_names,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        scores = score(artifacts.booster, matrix)
        return ScoreResponse(
            scores=[
                PlayerScore(player_id=record.player_id, score=float(scores[idx]))
                for idx, record in enumerate(payload.records)
            ]
        )

    @app.post("/predict", response_model=ScoreResponse)
    async def predict_endpoint(request: Request, payload: ScoreRequest) -> ScoreResponse:
        return await score_endpoint(request, payload)

    @app.post("/rank", response_model=RankResponse)
    async def rank_endpoint(request: Request, payload: ScoreRequest) -> RankResponse:
        artifacts = _require_artifacts(request.app)

        try:
            matrix = build_feature_matrix(
                feature_dicts=[record.features for record in payload.records],
                feature_names=artifacts.feature_names,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        scores = score(artifacts.booster, matrix)
        order = argsort_desc([float(value) for value in scores])

        rankings: list[PlayerRank] = []
        for rank_index, record_index in enumerate(order, start=1):
            record = payload.records[record_index]
            rankings.append(
                PlayerRank(
                    player_id=record.player_id,
                    score=float(scores[record_index]),
                    rank=rank_index,
                )
            )

        return RankResponse(rankings=rankings)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app
