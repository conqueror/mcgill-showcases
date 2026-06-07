#!/usr/bin/env python3
"""Log a behaviour policy and learn offline from it by Fitted-Q Iteration; write the evidence."""

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

from learning_agents.dynamic_programming import reachable_acting_states
from learning_agents.offline_rl import collect_logged_dataset, fitted_q_iteration
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the offline-RL runner's command-line flags.

    What + why: offline RL learns from a *fixed* behaviour-policy log, so the flags govern how much
    log to collect and how exploratory the behaviour policy is (which sets coverage). Exposes
    ``--output-dir``, ``--episodes`` (logged episodes; ``None`` defers to the default in ``main``),
    ``--epsilon`` (the behaviour policy's exploration rate, which drives coverage), and ``--quick``
    (a smaller log for CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir``, ``episodes``, ``epsilon``, ``quick``.

    RL concept:
        Offline / batch RL -- the data budget and behaviour exploration that set the log's coverage.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Collect a behaviour log, run Fitted-Q Iteration, and write the offline-RL artifacts.

    What + why: rolls out an epsilon-soft heuristic-router behaviour policy to produce a fixed log,
    then learns from it with tabular Fitted-Q Iteration (no environment interaction). Writes
    ``artifacts/offline_rl/training_curve.csv`` (the batch convergence curve) and
    ``artifacts/offline_rl/dataset_summary.csv`` (log size and state-space coverage -- the honest
    limit on what offline RL can learn). The learned policy's *value* is shown in the main policy
    comparison (the ``offline_fqi`` row).

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Offline / batch RL by Fitted-Q Iteration, plus the data-coverage constraint.
    """
    args = parse_args(argv)
    # --quick shrinks the logged dataset for CI; otherwise honour --episodes or the 600 default.
    episodes = args.episodes if args.episodes is not None else (200 if args.quick else 600)
    dataset = collect_logged_dataset(episodes=episodes, epsilon=args.epsilon, seed=7)
    result = fitted_q_iteration(dataset, gamma=0.9)

    write_csv_artifact(
        args.output_dir / "offline_rl" / "training_curve.csv",
        result.training_curve,
    )

    # Coverage: the log's distinct decision states vs the full reachable acting-state space. The gap
    # is the out-of-distribution region offline RL has no evidence for.
    decision_states = {transition.state.as_tuple() for transition in dataset.transitions}
    reachable = reachable_acting_states()
    coverage_fraction = round(len(decision_states) / len(reachable), 4) if reachable else 0.0
    write_csv_artifact(
        args.output_dir / "offline_rl" / "dataset_summary.csv",
        [
            {
                "num_transitions": len(dataset),
                "num_decision_states": len(decision_states),
                "num_reachable_states": len(reachable),
                "coverage_fraction": coverage_fraction,
                "behavior_policy": dataset.behavior_policy_name,
                "epsilon": dataset.epsilon,
            }
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
