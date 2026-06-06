#!/usr/bin/env python3
"""Generate the full non-optional artifact set for the student-support RL showcase."""

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

from student_support_rl.bandit import run_bandit_experiment
from student_support_rl.drl import OptionalDRLError, run_drl_comparison
from student_support_rl.dynamic_programming import (
    gap_rows,
    optimal_action_value_rows,
    optimal_action_values,
)
from student_support_rl.evaluation import evaluate_policies, simulate_episode
from student_support_rl.policies import AdvisorHeavyPolicy, HeuristicPolicy, RandomPolicy
from student_support_rl.policy_gradient import train_reinforce
from student_support_rl.q_learning import q_table_rows, train_q_learning
from student_support_rl.reporting import (
    algorithm_progression_markdown,
    concept_map_rows,
    governance_artifacts,
    mdp_spec_markdown,
    recommendation_from_summary,
    write_csv_artifact,
    write_text_artifact,
)
from student_support_rl.reward_design import (
    compare_reward_models,
    reward_hacking_report,
    reward_model_specs,
)
from student_support_rl.sarsa import train_sarsa


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the full-showcase runner's command-line flags.

    Exposes ``--output-dir``, ``--episodes`` (tabular-control training length, default ``5000``;
    note this runner defaults to a concrete value rather than ``None``), and ``--quick`` (which
    shrinks every training and evaluation budget so the whole pipeline runs fast in CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every rung of the ladder once and write the complete artifact set.

    Orchestrates the whole showcase in dependency order: the bandit warm-up; the MDP and
    concept-map docs; tabular Q-learning, the exact DP optimum and the learned-vs-optimal gap;
    SARSA; REINFORCE; the offline policy comparison and governance docs; the reward-hacking study;
    the deploy/shadow/reject memo; and finally the optional DQN/PPO bridge (which degrades to a
    note if its extras are missing). Writes all non-optional artifacts under ``artifacts/`` so a
    single command reproduces everything ``verify_artifacts`` later checks.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success (also when the optional DRL extras are absent).

    RL concept:
        End-to-end RL ladder, contextual bandit -> MDP -> Q-learning -> DP -> SARSA ->
        policy gradient -> DQN/PPO. See docs/showcase-architecture.md.
    """
    args = parse_args(argv)
    # --quick shrinks every budget below so the full pipeline finishes quickly in CI.
    bandit_steps = 120 if args.quick else 600
    eval_episodes = 3 if args.quick else 12
    train_episodes = 600 if args.quick else args.episodes

    bandit_result = run_bandit_experiment(steps=bandit_steps)
    q_result = train_q_learning(episodes=train_episodes)
    evaluation_summary, evaluation_rows = evaluate_policies(
        policies=[
            RandomPolicy(seed=7),
            HeuristicPolicy(),
            q_result.greedy_policy(),
        ],
        scenario_ids=(0, 1, 2, 3, 4),
        episodes_per_scenario=eval_episodes,
    )
    reward_rows = compare_reward_models(
        policies=[AdvisorHeavyPolicy(), HeuristicPolicy()],
        scenario_ids=(0, 1, 2, 3, 4),
    )
    specs = reward_model_specs()
    governance = governance_artifacts()
    recommendation, rationale = recommendation_from_summary(evaluation_summary)

    write_text_artifact(args.output_dir / "concepts" / "mdp_spec.md", mdp_spec_markdown())
    write_text_artifact(
        args.output_dir / "concepts" / "algorithm_progression.md",
        algorithm_progression_markdown(),
    )
    write_csv_artifact(args.output_dir / "concepts" / "concept_map.csv", concept_map_rows())
    write_csv_artifact(args.output_dir / "bandit" / "reward_trace.csv", bandit_result.reward_trace)
    write_csv_artifact(args.output_dir / "bandit" / "regret_trace.csv", bandit_result.regret_trace)
    write_csv_artifact(
        args.output_dir / "mdp" / "sample_episodes.csv",
        simulate_episode(policy=HeuristicPolicy(), scenario_id=2, seed=0 if args.quick else None),
    )
    write_csv_artifact(
        args.output_dir / "q_learning" / "training_curve.csv",
        q_result.training_curve,
    )
    write_csv_artifact(
        args.output_dir / "q_learning" / "q_table.csv",
        q_table_rows(q_result.q_table),
    )
    optimal_values = optimal_action_values()
    write_csv_artifact(
        args.output_dir / "dp" / "optimal_action_values.csv",
        optimal_action_value_rows(optimal_values),
    )
    write_csv_artifact(
        args.output_dir / "dp" / "q_learning_gap.csv",
        gap_rows(q_result.q_table, optimal_values),
    )
    sarsa_result = train_sarsa(episodes=train_episodes)
    write_csv_artifact(
        args.output_dir / "sarsa" / "training_curve.csv",
        sarsa_result.training_curve,
    )
    write_csv_artifact(
        args.output_dir / "sarsa" / "q_table.csv",
        q_table_rows(sarsa_result.q_table),
    )
    reinforce_result = train_reinforce(episodes=400 if args.quick else 2000)
    write_csv_artifact(
        args.output_dir / "policy_gradient" / "training_curve.csv",
        reinforce_result.training_curve,
    )
    write_csv_artifact(args.output_dir / "eval" / "policy_comparison.csv", evaluation_summary)
    write_csv_artifact(args.output_dir / "eval" / "scenario_results.csv", evaluation_rows)
    write_text_artifact(
        args.output_dir / "reward" / "reward_hacking_report.md",
        reward_hacking_report(reward_rows),
    )
    write_text_artifact(args.output_dir / "reward" / "reward_spec_good.md", specs["good"])
    write_text_artifact(args.output_dir / "reward" / "reward_spec_bad.md", specs["bad"])
    write_text_artifact(
        args.output_dir / "governance" / "safety_controls.md",
        governance["safety_controls"],
    )
    write_text_artifact(
        args.output_dir / "governance" / "offline_eval_plan.md",
        governance["offline_eval_plan"],
    )
    memo = (
        "# Deploy, Shadow, or Reject Memo\n\n"
        f"Recommendation: {recommendation}.\n\n"
        "## Why\n\n"
        f"{rationale}\n\n"
        "## What This Means\n\n"
        "- `deploy`: rare in this teaching repo and only appropriate when offline risk is low.\n"
        "- `shadow`: collect more evidence with human review and no automated actioning.\n"
        "- `reject`: redesign the reward, policy, or safety controls before moving further.\n"
    )
    write_text_artifact(args.output_dir / "business" / "deploy_shadow_reject_memo.md", memo)
    try:
        drl_result = run_drl_comparison(
            timesteps=1200 if args.quick else 3600,
            output_dir=args.output_dir / "drl_optional",
            quick=args.quick,
        )
    except OptionalDRLError as exc:
        write_text_artifact(
            args.output_dir / "drl_optional" / "bridge_report.md",
            (
                "# Optional DRL Bridge\n\n"
                "The optional Gymnasium/SB3 path could not run, so DQN and PPO "
                "artifacts were skipped.\n\n"
                f"Import error: {exc}\n"
            ),
        )
    else:
        write_csv_artifact(
            args.output_dir / "drl_optional" / "rl_family_comparison.csv",
            drl_result.comparison_rows,
        )
        write_csv_artifact(
            args.output_dir / "drl_optional" / "scenario_rollups.csv",
            drl_result.scenario_rows,
        )
        write_csv_artifact(
            args.output_dir / "drl_optional" / "training_summary.csv",
            drl_result.training_rows,
        )
        write_text_artifact(
            args.output_dir / "drl_optional" / "policy_gradient_notes.md",
            drl_result.policy_gradient_notes,
        )
        write_text_artifact(
            args.output_dir / "drl_optional" / "bridge_report.md",
            drl_result.bridge_report,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
