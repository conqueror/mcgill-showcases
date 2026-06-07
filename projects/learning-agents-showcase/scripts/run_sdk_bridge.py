#!/usr/bin/env python3
"""Drive an agent loop with the learned policy and write the OpenAI Agents SDK bridge artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from learning_agents.policies import QTablePolicy
from learning_agents.q_learning import train_q_learning
from learning_agents.reporting import write_csv_artifact, write_text_artifact
from learning_agents.sdk_bridge import bridge_report_markdown, run_bridged_episode, sdk_available


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the SDK-bridge runner's command-line flags.

    What + why: the bridge demonstrates the *learned* orchestration policy driving an agent loop, so
    the flags govern how long to train that policy. Exposes ``--output-dir``, ``--episodes``
    (Q-learning training length; ``None`` defers to the default), and ``--quick`` (a shorter run for
    CI that also pins the trace's start-state seed for reproducibility).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir``, ``episodes``, and ``quick``.

    RL concept:
        Locus of learning A -- RL learns the orchestration policy that the agent framework executes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Trace the learned policy across scenarios as an SDK agent loop and write the bridge files.

    What + why: trains a tabular Q-learning orchestration policy, then rolls it out across each
    scenario, annotating every decision with the OpenAI Agents SDK construct it drives (a
    function-tool call, a handoff, or a final output). Writes
    ``artifacts/sdk_bridge/orchestration_trace.csv`` (the
    learned policy driving the agent loop) and ``artifacts/sdk_bridge/bridge_report.md`` (the
    locus-of-learning-A narrative plus whether the live SDK is installed here). The demonstration is
    pure-Python and needs no SDK or network; the report explains how to enable the live bridge.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        The learned orchestration policy driving an agent framework's tool/handoff/output loop.
    """
    args = parse_args(argv)
    episodes = args.episodes if args.episodes is not None else (120 if args.quick else 400)
    q_result = train_q_learning(episodes=episodes)
    learned = QTablePolicy(q_table=q_result.q_table, name="q_learning")

    # Roll the learned policy out across every scenario; concatenate the SDK-annotated traces so the
    # artifact shows the full range of constructs (tool calls, handoffs, final outputs).
    trace_rows: list[dict[str, int | float | str]] = []
    for scenario_id in range(5):
        trace_rows.extend(
            run_bridged_episode(
                policy=learned,
                scenario_id=scenario_id,
                seed=0 if args.quick else None,
            )
        )
    write_csv_artifact(args.output_dir / "sdk_bridge" / "orchestration_trace.csv", trace_rows)
    write_text_artifact(
        args.output_dir / "sdk_bridge" / "bridge_report.md",
        bridge_report_markdown(sdk_present=sdk_available()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
