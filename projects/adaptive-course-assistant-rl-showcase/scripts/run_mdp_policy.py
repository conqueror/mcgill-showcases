#!/usr/bin/env python3
"""Write the MDP, sample-episode, and assistant-context artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_course_assistant_rl.evaluation import simulate_episode
from adaptive_course_assistant_rl.policies import RuleBasedPolicy
from adaptive_course_assistant_rl.reporting import (
    agentic_rl_bridge_markdown,
    algorithm_progression_markdown,
    interpretation_prompts_markdown,
    mdp_spec_markdown,
    resource_match_rows,
    state_action_reward_rows,
    write_csv_artifact,
    write_text_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episode_rows = simulate_episode(
        policy=RuleBasedPolicy(),
        scenario_id=1 if args.quick else 2,
        seed=0 if args.quick else None,
    )
    write_text_artifact(args.output_dir / "concepts" / "mdp_spec.md", mdp_spec_markdown())
    write_text_artifact(
        args.output_dir / "concepts" / "algorithm_progression.md",
        algorithm_progression_markdown(),
    )
    write_text_artifact(
        args.output_dir / "concepts" / "agentic_rl_bridge.md",
        agentic_rl_bridge_markdown(),
    )
    write_text_artifact(
        args.output_dir / "concepts" / "interpretation_prompts.md",
        interpretation_prompts_markdown(),
    )
    write_csv_artifact(
        args.output_dir / "concepts" / "state_action_reward_schema.csv",
        state_action_reward_rows(),
    )
    write_csv_artifact(args.output_dir / "mdp" / "sample_episodes.csv", episode_rows)
    (args.output_dir / "assistant").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assistant" / "episode_trace.json").write_text(
        json.dumps(episode_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv_artifact(
        args.output_dir / "assistant" / "resource_matches.csv",
        resource_match_rows(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
