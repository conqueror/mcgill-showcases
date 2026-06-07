#!/usr/bin/env python3
"""Write the MDP concept artifacts and a sample heuristic-router episode trace."""

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

from learning_agents.evaluation import simulate_episode
from learning_agents.policies import HeuristicRouterPolicy
from learning_agents.reporting import (
    algorithm_progression_markdown,
    concept_map_rows,
    mdp_spec_markdown,
    write_csv_artifact,
    write_text_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the MDP-simulation runner's command-line flags.

    Exposes ``--output-dir``, ``--scenario-id`` (which fixed agent scenario to roll out), and
    ``--quick`` (which pins the reset seed to ``0``, jittering the start state). The trace is
    reproducible either way -- the environment's transitions are always deterministic and the
    reset seed (``0`` with ``--quick``, ``None`` otherwise) is fixed; the flag only swaps a
    jittered start state for the scenario's noise-free baseline.

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept:
        Choosing the scenario and reset seed that seed a single MDP rollout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--scenario-id", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Roll out the heuristic router policy and write the MDP and concept-map artifacts.

    Writes ``artifacts/mdp/sample_episodes.csv`` (one episode trace of states, actions, and
    rewards) plus the framing docs ``artifacts/concepts/mdp_spec.md``,
    ``artifacts/concepts/algorithm_progression.md``, and ``artifacts/concepts/concept_map.csv``.
    This is the MDP rung: once actions move the agent between states, the bandit picture is
    replaced by a Markov decision process with transitions and discounted return
    G_t = sum_k gamma^k R_{t+k+1}. Scenario 3 ("hard_debug") gives a good multi-step trace.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Markov decision process and episodic return G_t = sum_k gamma^k R_{t+k+1}.
    """
    args = parse_args(argv)
    # --quick pins the reset seed to 0 (a jittered start state); seed=None uses the scenario's
    # noise-free baseline. Either way the trace is deterministic -- step() never uses the RNG.
    rows = simulate_episode(
        policy=HeuristicRouterPolicy(),
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
