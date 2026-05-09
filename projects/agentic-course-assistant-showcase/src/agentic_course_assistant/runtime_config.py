"""Runtime configuration helpers for optional live SDK examples."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


@dataclass(frozen=True)
class LiveRuntimeConfig:
    """Configuration for optional OpenAI and Gemini-backed examples."""

    openai_api_key: str | None
    gemini_api_key: str | None
    openai_model: str
    gemini_model: str

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)


def load_runtime_config(
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> LiveRuntimeConfig:
    """Load live runtime settings from environment variables and an optional `.env` file."""

    environment = dict(environ or os.environ)
    env_file_values = _load_env_file((project_root or _default_project_root()) / ".env")
    merged = {**env_file_values, **environment}
    return LiveRuntimeConfig(
        openai_api_key=_clean_secret(merged.get("OPENAI_API_KEY")),
        gemini_api_key=_clean_secret(
            merged.get("GEMINI_API_KEY") or merged.get("GOOGLE_API_KEY")
        ),
        openai_model=merged.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        gemini_model=merged.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )


def apply_live_environment(
    project_root: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> LiveRuntimeConfig:
    """Load `.env` settings and expose aliases expected by optional SDKs."""

    target_environ = environ if environ is not None else os.environ
    config = load_runtime_config(project_root=project_root, environ=target_environ)
    if config.openai_api_key:
        target_environ.setdefault("OPENAI_API_KEY", config.openai_api_key)
    if config.gemini_api_key:
        target_environ.setdefault("GEMINI_API_KEY", config.gemini_api_key)
        target_environ.setdefault("GOOGLE_API_KEY", config.gemini_api_key)
    target_environ.setdefault("OPENAI_DEFAULT_MODEL", config.openai_model)
    return config


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        values[key.strip()] = _strip_quotes(raw_value.strip())
    return values


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
