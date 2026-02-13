from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DEMAND_API_", extra="ignore")

    log_level: str = Field(default="INFO")
    model_path: Path = Field(default=Path("artifacts/model.joblib"))

    prometheus_enabled: bool = Field(default=True)

    otel_enabled: bool = Field(default=False)
    otel_service_name: str = Field(default="demand-api-observability-showcase")
    otel_exporter_otlp_endpoint: str | None = Field(default=None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
