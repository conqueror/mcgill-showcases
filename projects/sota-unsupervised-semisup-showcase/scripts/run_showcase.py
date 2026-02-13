"""Convenience script to run the full showcase without CLI installation."""

from __future__ import annotations

from pathlib import Path

from sota_showcase.config import PathsConfig, ShowcaseConfig
from sota_showcase.pipeline import run_full_showcase


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ShowcaseConfig()
    paths = PathsConfig.from_project_root(project_root=project_root)
    run_full_showcase(config=config, paths=paths)


if __name__ == "__main__":
    main()
