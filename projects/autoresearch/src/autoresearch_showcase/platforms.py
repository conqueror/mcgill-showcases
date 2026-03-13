from __future__ import annotations

from .models import UpstreamProfile

_PROFILES: tuple[UpstreamProfile, ...] = (
    UpstreamProfile(
        key="macos",
        display_name="macOS / Apple Silicon",
        repo="miolini/autoresearch-macos",
        repo_url="https://github.com/miolini/autoresearch-macos",
        repo_commit="783dea5b5092162df13e4f0f36a2df75134218b4",
        hardware="Apple Silicon with MPS; CPU is documented as a fallback path",
        prepare_runtime="prepare.py verifies macOS with Metal/MPS availability before setup",
        attention_backend=(
            "PyTorch SDPA with manual sliding-window masking instead of hard "
            "FlashAttention-3 dependency"
        ),
        compile_policy=(
            "Conservative compile policy on MPS; model compile is CUDA-only in "
            "the inspected fork"
        ),
        training_metric="Validation bits per byte (val_bpb), lower is better",
        mutable_surface="train.py",
        fixed_surface="prepare.py and the evaluation harness",
        notes=(
            "Fork adds explicit MPS support and safer Apple Silicon defaults.",
            "Good choice when the learner wants a real agent run from a Mac laptop.",
        ),
        preflight_commands=(
            "uv sync",
            "uv run prepare.py",
            "uv run train.py",
        ),
    ),
    UpstreamProfile(
        key="unix",
        display_name="Unix / NVIDIA GPU",
        repo="karpathy/autoresearch",
        repo_url="https://github.com/karpathy/autoresearch",
        repo_commit="c2450add72cc80317be1fe8111974b892da10944",
        hardware="Single NVIDIA GPU on a Unix-like environment",
        prepare_runtime=(
            "prepare.py uses the shared data and tokenizer flow without a "
            "macOS-specific guard"
        ),
        attention_backend="FlashAttention-3 loaded through the kernels package",
        compile_policy="torch.compile is part of the CUDA-oriented training path",
        training_metric="Validation bits per byte (val_bpb), lower is better",
        mutable_surface="train.py",
        fixed_surface="prepare.py and the evaluation harness",
        notes=(
            "Original repo assumes CUDA-centric execution and GPU memory reporting.",
            "Good choice when the learner has a Linux box or remote NVIDIA workstation.",
        ),
        preflight_commands=(
            "uv sync",
            "uv run prepare.py",
            "uv run train.py",
        ),
    ),
)


def list_profiles() -> tuple[UpstreamProfile, ...]:
    """Return all supported platform profiles."""

    return _PROFILES


def get_profile(key: str) -> UpstreamProfile:
    """Return one platform profile by key."""

    for profile in _PROFILES:
        if profile.key == key:
            return profile
    raise KeyError(f"Unknown profile: {key}")


def platform_rows() -> list[dict[str, str]]:
    """Return CSV-friendly rows for the platform comparison artifact."""

    return [
        {
            "platform": profile.display_name,
            "repo": profile.repo,
            "repo_url": profile.repo_url,
            "repo_commit": profile.repo_commit,
            "hardware": profile.hardware,
            "prepare_runtime": profile.prepare_runtime,
            "attention_backend": profile.attention_backend,
            "compile_policy": profile.compile_policy,
            "training_metric": profile.training_metric,
            "mutable_surface": profile.mutable_surface,
            "fixed_surface": profile.fixed_surface,
            "notes": " | ".join(profile.notes),
        }
        for profile in _PROFILES
    ]


def upstream_snapshot() -> dict[str, object]:
    """Return a JSON-friendly snapshot of the upstream references used in this showcase."""

    return {
        "sources_checked_on": "2026-03-13",
        "profiles": [
            {
                "key": profile.key,
                "display_name": profile.display_name,
                "repo": profile.repo,
                "repo_url": profile.repo_url,
                "repo_commit": profile.repo_commit,
            }
            for profile in _PROFILES
        ],
    }
