#!/usr/bin/env python3
"""Generate the full core artifact set for the adaptive course assistant RL showcase."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_contextual_bandit import main as run_contextual_bandit_main
from run_learning_agent_story import main as run_learning_agent_story_main
from run_mdp_policy import main as run_mdp_policy_main
from run_policy_export import main as run_policy_export_main
from run_policy_gradient import main as run_policy_gradient_main
from run_q_learning import main as run_q_learning_main
from run_reward_audit import main as run_reward_audit_main
from run_rl_family_comparison import main as run_rl_family_comparison_main
from run_rule_policy import main as run_rule_policy_main
from run_sarsa import main as run_sarsa_main

from adaptive_course_assistant_rl.reporting import write_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    write_manifest(args.output_dir / "manifest.json")
    sub_argv = ["--output-dir", str(args.output_dir)]
    if args.quick:
        sub_argv.append("--quick")
    for runner in (
        run_mdp_policy_main,
        run_contextual_bandit_main,
        run_rule_policy_main,
        run_q_learning_main,
        run_sarsa_main,
        run_policy_gradient_main,
        run_rl_family_comparison_main,
        run_reward_audit_main,
        run_learning_agent_story_main,
        run_policy_export_main,
    ):
        runner(sub_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
