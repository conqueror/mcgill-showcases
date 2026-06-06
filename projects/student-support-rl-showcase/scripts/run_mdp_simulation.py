#!/usr/bin/env python3
"""Write a sample heuristic-policy episode trace.

This is the MDP rung: once actions move the student between states, the bandit picture is
replaced by a Markov decision process with transitions and discounted return
G_t = sum_k gamma^k R_{t+k+1}. The script rolls out a fixed heuristic policy for one scenario
and also emits the concept-map artifacts (MDP spec, algorithm progression, concept map) that
frame the rest of the showcase. See docs/mdp-and-environment.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.evaluation import simulate_episode
from student_support_rl.policies import HeuristicPolicy
from student_support_rl.reporting import (
    algorithm_progression_markdown,
    concept_map_rows,
    mdp_spec_markdown,
    write_csv_artifact,
    write_text_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the MDP-simulation runner's command-line flags.

    Exposes ``--output-dir``, ``--scenario-id`` (which fixed student scenario to roll out), and
    ``--quick`` (which pins the reset seed to ``0``, jittering the start state). The trace is
    reproducible either way -- the environment's transitions are always deterministic and the
    reset seed (``0`` with ``--quick``, ``None`` otherwise) is fixed; the flag only swaps a
    jittered start state for the scenario's noise-free baseline.

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--scenario-id", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Roll out the heuristic policy and write the MDP and concept-map artifacts.

    Writes ``artifacts/mdp/sample_episodes.csv`` (one episode trace of states, actions, and
    rewards) plus the framing docs ``artifacts/concepts/mdp_spec.md``,
    ``artifacts/concepts/algorithm_progression.md``, and ``artifacts/concepts/concept_map.csv``.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Markov decision process and episodic return. See docs/mdp-and-environment.md.
    """
    args = parse_args(argv)
    # --quick pins the reset seed to 0 (a jittered start state); seed=None uses the scenario's
    # noise-free baseline. Either way the trace is deterministic -- step() never uses the RNG.
    rows = simulate_episode(
        policy=HeuristicPolicy(),
        scenario_id=args.scenario_id,
        seed=0 if args.quick else None,
    )
    write_text_artifact(args.output_dir / "concepts" / "mdp_spec.md", mdp_spec_markdown())
    write_text_artifact(
        args.output_dir / "concepts" / "algorithm_progression.md",
        algorithm_progression_markdown(),
    )
    write_csv_artifact(args.output_dir / "concepts" / "concept_map.csv", concept_map_rows())
    write_csv_artifact(args.output_dir / "mdp" / "sample_episodes.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
