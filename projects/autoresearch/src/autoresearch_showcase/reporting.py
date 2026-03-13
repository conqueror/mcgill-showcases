from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .agent_brief import render_agent_brief
from .decision_policy import default_scenarios, recommend_status
from .platforms import list_profiles, platform_rows, upstream_snapshot

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "artifacts/overview/platform_comparison.csv",
    "artifacts/overview/upstream_snapshot.json",
    "artifacts/overview/research_loop_summary.md",
    "artifacts/analysis/decision_scenarios.csv",
    "artifacts/analysis/simulated_results.tsv",
    "artifacts/analysis/decision_summary.json",
    "artifacts/agent/codex_macos.md",
    "artifacts/agent/codex_unix.md",
    "artifacts/agent/claude_macos.md",
    "artifacts/agent/claude_unix.md",
    "artifacts/summary.md",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _decision_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "scenario": "baseline",
            "baseline_bpb": "n/a",
            "candidate_bpb": 0.9989,
            "delta_bpb": "n/a",
            "complexity_delta": 0,
            "memory_gb": 44.0,
            "status": "keep",
            "description": "Baseline run used only to establish the comparison point.",
            "rationale": (
                "You always keep the baseline because it defines the branch "
                "starting point."
            ),
        }
    ]

    for scenario in default_scenarios():
        status, rationale = recommend_status(
            baseline_bpb=scenario.baseline_bpb,
            candidate_bpb=scenario.candidate_bpb,
            complexity_delta=scenario.complexity_delta,
            crashed=scenario.crashed,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "baseline_bpb": scenario.baseline_bpb,
                "candidate_bpb": scenario.candidate_bpb if not scenario.crashed else "crash",
                "delta_bpb": scenario.delta_bpb if not scenario.crashed else "crash",
                "complexity_delta": scenario.complexity_delta,
                "memory_gb": scenario.memory_gb,
                "status": status,
                "description": scenario.description,
                "rationale": rationale,
            }
        )
    return rows


def _simulated_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "commit": "base000",
            "val_bpb": "0.998900",
            "memory_gb": "44.0",
            "status": "keep",
            "description": "baseline (simulated)",
        }
    ]
    status_rows = _decision_rows()[1:]
    commit_names = ["lr0001", "deep002", "act0003", "simp004", "oom0005"]
    for commit_name, status_row in zip(commit_names, status_rows, strict=True):
        if status_row["status"] == "crash":
            val_bpb = "0.000000"
            memory_gb = "0.0"
        else:
            candidate_bpb = status_row["candidate_bpb"]
            memory_value = status_row["memory_gb"]
            assert isinstance(candidate_bpb, (int, float))
            assert isinstance(memory_value, (int, float))
            val_bpb = f"{candidate_bpb:.6f}"
            memory_gb = f"{memory_value:.1f}"
        rows.append(
            {
                "commit": commit_name,
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": status_row["status"],
                "description": f"{status_row['scenario']} (simulated)",
            }
        )
    return rows


def _decision_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    keep_count = sum(1 for row in rows if row["status"] == "keep")
    discard_count = sum(1 for row in rows if row["status"] == "discard")
    crash_count = sum(1 for row in rows if row["status"] == "crash")
    best_candidates = [
        row
        for row in rows
        if row["status"] == "keep" and isinstance(row["candidate_bpb"], (int, float))
    ]
    best = min(best_candidates, key=_candidate_bpb)
    return {
        "simulated": True,
        "keep_count": keep_count,
        "discard_count": discard_count,
        "crash_count": crash_count,
        "best_keep_scenario": best["scenario"],
        "best_keep_bpb": best["candidate_bpb"],
    }


def _candidate_bpb(row: Mapping[str, object]) -> float:
    value = row["candidate_bpb"]
    assert isinstance(value, (int, float))
    return float(value)


def _research_loop_summary() -> str:
    return """
# Research Loop Summary

## Fixed surfaces

- `prepare.py` owns setup, tokenization, data loading, and evaluation.
- The 300-second time budget is fixed.
- `val_bpb` is the main keep-or-discard score.

## Mutable surface

- `train.py` is the file the agent experiments on.

## Human-owned operating guide

- `program.md` tells the agent what is in scope and how to behave.

## Core loop

1. Establish a baseline.
2. Edit `train.py`.
3. Run one fixed-budget experiment.
4. Read `val_bpb` and memory.
5. Log the result in `results.tsv`.
6. Keep or discard the change.

## Teaching warning

These artifacts are educational and deterministic. The real overnight research
loop runs in the upstream repo on the platform you actually have.
"""


def _summary_markdown() -> str:
    return """
# Autoresearch Showcase Summary

This run generated:

- a grounded comparison of the macOS and Unix upstream repos,
- a teaching version of the keep/discard policy,
- a simulated `results.tsv` trace,
- Codex and Claude Code launch briefs for both platforms.

Recommended next step:

1. choose your platform,
2. open the matching brief in `artifacts/agent/`,
3. clone the upstream repo,
4. launch the real agent workflow there.
"""


def build_showcase(project_root: Path) -> list[Path]:
    """Generate all educational artifacts for the showcase."""

    artifacts_root = project_root / "artifacts"
    written: list[Path] = []

    platform_path = artifacts_root / "overview/platform_comparison.csv"
    _write_csv(platform_path, platform_rows())
    written.append(platform_path)

    snapshot_path = artifacts_root / "overview/upstream_snapshot.json"
    _write_json(snapshot_path, upstream_snapshot())
    written.append(snapshot_path)

    loop_summary_path = artifacts_root / "overview/research_loop_summary.md"
    _write_markdown(loop_summary_path, _research_loop_summary())
    written.append(loop_summary_path)

    decision_rows = _decision_rows()
    decision_scenarios_path = artifacts_root / "analysis/decision_scenarios.csv"
    _write_csv(decision_scenarios_path, decision_rows)
    written.append(decision_scenarios_path)

    simulated_results_path = artifacts_root / "analysis/simulated_results.tsv"
    _write_tsv(simulated_results_path, _simulated_results())
    written.append(simulated_results_path)

    decision_summary_path = artifacts_root / "analysis/decision_summary.json"
    _write_json(decision_summary_path, _decision_summary(decision_rows))
    written.append(decision_summary_path)

    for profile in list_profiles():
        for agent in ("codex", "claude"):
            brief_path = artifacts_root / f"agent/{agent}_{profile.key}.md"
            _write_markdown(brief_path, render_agent_brief(profile, agent))
            written.append(brief_path)

    summary_path = artifacts_root / "summary.md"
    _write_markdown(summary_path, _summary_markdown())
    written.append(summary_path)

    manifest_path = artifacts_root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "project": "autoresearch",
            "required_files": list(REQUIRED_ARTIFACTS),
        },
    )
    written.append(manifest_path)

    return written
