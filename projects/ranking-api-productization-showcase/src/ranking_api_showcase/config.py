from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RANK_API_",
        extra="ignore",
    )

    log_level: str = Field(default="INFO")
    model_path: Path = Field(default=Path("artifacts/model.txt"))
    feature_names_path: Path = Field(default=Path("artifacts/feature_names.json"))
    model_meta_path: Path = Field(default=Path("artifacts/model_meta.json"))
    fail_on_model_load: bool = Field(default=False)
    service_name: str = Field(default="ranking-api-showcase")


def load_settings() -> Settings:
    return Settings()
