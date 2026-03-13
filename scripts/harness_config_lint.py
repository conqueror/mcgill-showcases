#!/usr/bin/env python3
"""Validate the public harness-lite configuration for mcgill-showcases."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Python 3.11+ with tomllib is required.") from exc


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / ".codex" / "config.toml"
ROLES_DIR = ROOT / ".codex" / "agents"
ROUTING_PATH = ROOT / ".codex" / "harness" / "role-skill-matrix.toml"
OPERATING_PACK_PATH = ROOT / "docs" / "agents" / "oodaris-harness-v2-operating-pack.md"
EVAL_README_PATH = ROOT / "docs" / "agents" / "harness-evals" / "README.md"
PREFLIGHT_PATH = ROOT / "scripts" / "dev" / "harness-cli-preflight.sh"

REQUIRED_DEFAULT_KEYS = {
    "model",
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
    "personality",
    "web_search",
    "approval_policy",
    "sandbox_mode",
}
REQUIRED_FEATURE_KEYS = {
    "multi_agent",
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    "runtime_metrics",
}
SUPPORTED_TASK_CLASSES = {
    "standard",
    "review_only",
    "closure_only",
    "harness_change",
}
REQUIRED_PACK_MARKERS = {
    "public-safe",
    "minimal bootstrap",
    "unsupported `high_impact`",
    "private integrations are not bugs",
}
DISALLOWED_TOKENS = {
    " acli",
    " tempo",
    " jira",
    "bd ",
    "bd\n",
    "retail_sme",
    "ml_scientist",
    "data_engineer",
    "or_researcher",
    "agentic_ai_architect",
    "oodaris-agentic-retail",
}


def load_toml(path: Path) -> dict:
    if not path.exists():
      raise FileNotFoundError(path)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def collect_errors() -> list[str]:
    errors: list[str] = []

    try:
        config = load_toml(CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - exercised by command failure
        return [f"failed to read config: {exc}"]

    try:
        routing = load_toml(ROUTING_PATH)
    except Exception as exc:  # pragma: no cover - exercised by command failure
        return [f"failed to read routing manifest: {exc}"]

    defaults = config.get("defaults", {})
    missing_defaults = sorted(REQUIRED_DEFAULT_KEYS - set(defaults))
    if missing_defaults:
        errors.append(f"config missing default keys: {', '.join(missing_defaults)}")

    features = config.get("features", {})
    missing_features = sorted(REQUIRED_FEATURE_KEYS - set(features))
    if missing_features:
        errors.append(f"config missing feature keys: {', '.join(missing_features)}")

    roles = set(routing.get("roles", {}).get("all_supported", []))
    if not roles:
        errors.append("routing manifest declares no supported roles")

    for role in roles:
        role_path = ROLES_DIR / f"{role}.toml"
        if not role_path.exists():
            errors.append(f"missing role file: {role_path.relative_to(ROOT)}")

    task_classes = set(routing.get("task_classes", {}))
    if task_classes != SUPPORTED_TASK_CLASSES:
        errors.append(
            "routing manifest task classes must be exactly: "
            + ", ".join(sorted(SUPPORTED_TASK_CLASSES))
        )

    if not OPERATING_PACK_PATH.exists():
        errors.append(f"missing operating pack: {OPERATING_PACK_PATH.relative_to(ROOT)}")
    if not EVAL_README_PATH.exists():
        errors.append(f"missing eval README: {EVAL_README_PATH.relative_to(ROOT)}")
    if not PREFLIGHT_PATH.exists():
        errors.append(f"missing preflight script: {PREFLIGHT_PATH.relative_to(ROOT)}")

    if OPERATING_PACK_PATH.exists():
        operating_pack = OPERATING_PACK_PATH.read_text(encoding="utf-8").lower()
        for marker in REQUIRED_PACK_MARKERS:
            if marker.lower() not in operating_pack:
                errors.append(f"operating pack missing marker: {marker}")

    for path in (CONFIG_PATH, ROUTING_PATH, OPERATING_PACK_PATH, PREFLIGHT_PATH):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in DISALLOWED_TOKENS:
            if token in text:
                errors.append(
                    f"disallowed private or unsupported reference '{token.strip()}' in "
                    f"{path.relative_to(ROOT)}"
                )

    return errors


def main() -> int:
    errors = collect_errors()
    if not errors:
        print("Harness-lite config lint passed.")
        return 0

    print("Harness-lite config lint failed:")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
