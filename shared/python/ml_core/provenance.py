"""Content fingerprints for the source and files in a supervised artifact manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


def source_digest(project_root: Path) -> str:
    """Hash project code/configuration and the shared supervised implementation."""
    repo_root = Path(__file__).resolve().parents[3]
    files = {
        f"shared/{path.relative_to(repo_root / 'shared')}": path
        for directory in ("python/ml_core", "scripts", "config", "contracts")
        for path in (repo_root / "shared" / directory).rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    files.update(
        {
            f"project/{path.relative_to(project_root)}": path
            for directory in ("src", "scripts")
            for path in (project_root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
    )
    for name in ("pyproject.toml", "uv.lock"):
        if (project_root / name).is_file():
            files[f"project/{name}"] = project_root / name
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def artifact_hashes(project_root: Path, paths: list[str]) -> dict[str, str]:
    """Hash the listed outputs without storing private absolute paths."""
    return {name: hashlib.sha256((project_root / name).read_bytes()).hexdigest() for name in paths}
