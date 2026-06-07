#!/usr/bin/env python3
"""Generate the full non-optional artifact set for the learning-agents RL showcase."""

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

from learning_agents.bandit import run_bandit_experiment
from learning_agents.cost_cascade import cost_cascade_curve
from learning_agents.dynamic_programming import (
    gap_rows,
    optimal_action_value_rows,
    optimal_action_values,
    reachable_acting_states,
)
from learning_agents.evaluation import evaluate_policies, simulate_episode
from learning_agents.marl import (
    CLIMBING_GAME,
    marl_comparison_rows,
    train_independent_q_learning,
    train_joint_action_learner,
)
from learning_agents.offline_rl import collect_logged_dataset, fitted_q_iteration
from learning_agents.ope import ope_report_rows
from learning_agents.policies import (
    AlwaysEscalatePolicy,
    HeuristicRouterPolicy,
    Policy,
    QTablePolicy,
    RandomPolicy,
)
from learning_agents.policy_gradient import train_reinforce
from learning_agents.preference_optimization import compare_preference_methods
from learning_agents.q_learning import q_table_rows, train_q_learning
from learning_agents.reporting import (
    algorithm_progression_markdown,
    concept_map_rows,
    governance_artifacts,
    mdp_spec_markdown,
    recommendation_from_summary,
    write_csv_artifact,
    write_text_artifact,
)
from learning_agents.reward_study import (
    compare_reward_models,
    reward_hacking_report,
    reward_model_specs,
)
from learning_agents.sarsa import train_sarsa
from learning_agents.sdk_bridge import (
    bridge_report_markdown,
    run_bridged_episode,
    sdk_available,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the full-showcase runner's command-line flags.

    This is the all-in-one orchestrator's entry point, so its flags govern the whole pipeline
    rather than a single rung. It exposes ``--output-dir`` (root for every artifact),
    ``--episodes`` (the tabular-control training length, default ``400``; this runner defaults to a
    concrete value rather than ``None`` so one command reproduces a fixed artifact set), and
    ``--quick`` (which shrinks every training and evaluation budget so the full ladder runs fast in
    CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept:
        Sample budgets (episodes, steps) are the core experiment knob in RL; ``--episodes`` and
        ``--quick`` trade wall-clock cost against how well each learner converges.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every rung of the ladder once and write the complete artifact set.

    Orchestrates the whole showcase in dependency order in a single process: the contextual-bandit
    warm-up; the MDP and concept-map docs; tabular Q-learning, the exact DP optimum and the
    learned-vs-optimal gap; SARSA; REINFORCE; offline RL (Fitted-Q Iteration from a behaviour log)
    and its coverage; off-policy evaluation graded against truth; the cost-aware cascade frontier;
    the offline policy comparison (the full ladder including offline FQI and the DP optimum); the
    reward-hacking study; the governance docs; and finally the deploy/shadow/reject memo. Writes
    every generated required artifact under ``artifacts/`` (the checked-in ``manifest.json`` is the
    one required artifact not regenerated here) so a single command reproduces what
    ``verify_artifacts`` checks. There is no optional DRL bridge in this phase, so nothing lands
    under ``artifacts/drl_optional/``.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        End-to-end RL ladder, contextual bandit -> MDP -> Q-learning -> DP -> SARSA ->
        policy gradient -> offline RL / OPE -> cost cascade. See docs/showcase-architecture.md.
    """
    args = parse_args(argv)
    # --quick shrinks every budget below so the full pipeline finishes quickly in CI.
    bandit_steps = 120 if args.quick else 600
    eval_episodes = 3 if args.quick else 12
    train_episodes = 120 if args.quick else args.episodes
    reinforce_episodes = 400 if args.quick else 1500
    offline_episodes = 200 if args.quick else 600
    ope_episodes = 300 if args.quick else 800
    ope_truth_episodes = 12 if args.quick else 40

    bandit_result = run_bandit_experiment(steps=bandit_steps)
    q_result = train_q_learning(episodes=train_episodes)
    learned = QTablePolicy(q_table=q_result.q_table, name="q_learning")
    # Exact backward-induction Q* -- reused as both the planning-ceiling policy in the comparison
    # and the ground truth for the learned-vs-optimal gap artifact below.
    optimal_values = optimal_action_values()
    dp_optimal = QTablePolicy(q_table=optimal_values, name="dp_optimal")
    # Offline RL: learn a policy from a fixed behaviour log (no new interaction) so the comparison
    # shows the full ladder -- online Q-learning vs offline FQI vs the planning optimum.
    offline_dataset = collect_logged_dataset(episodes=offline_episodes, epsilon=0.6, seed=7)
    offline_result = fitted_q_iteration(offline_dataset, gamma=0.9)
    offline_fqi = QTablePolicy(q_table=offline_result.q_table, name="offline_fqi")
    evaluation_summary, evaluation_rows = evaluate_policies(
        policies=[
            RandomPolicy(seed=7),
            HeuristicRouterPolicy(),
            learned,
            offline_fqi,
            dp_optimal,
        ],
        scenario_ids=(0, 1, 2, 3, 4),
        episodes_per_scenario=eval_episodes,
    )
    comparison_rows = compare_reward_models(
        policies=[AlwaysEscalatePolicy(), HeuristicRouterPolicy()],
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
        simulate_episode(
            policy=HeuristicRouterPolicy(),
            scenario_id=3,
            seed=0 if args.quick else None,
        ),
    )
    write_csv_artifact(
        args.output_dir / "q_learning" / "training_curve.csv",
        q_result.training_curve,
    )
    write_csv_artifact(
        args.output_dir / "q_learning" / "q_table.csv",
        q_table_rows(q_result.q_table),
    )
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
    reinforce_result = train_reinforce(episodes=reinforce_episodes)
    write_csv_artifact(
        args.output_dir / "policy_gradient" / "training_curve.csv",
        reinforce_result.training_curve,
    )
    # Offline RL evidence: the batch FQI convergence curve and the log's state-space coverage.
    write_csv_artifact(
        args.output_dir / "offline_rl" / "training_curve.csv",
        offline_result.training_curve,
    )
    offline_decision_states = {t.state.as_tuple() for t in offline_dataset.transitions}
    reachable = reachable_acting_states()
    write_csv_artifact(
        args.output_dir / "offline_rl" / "dataset_summary.csv",
        [
            {
                "num_transitions": len(offline_dataset),
                "num_decision_states": len(offline_decision_states),
                "num_reachable_states": len(reachable),
                "coverage_fraction": round(len(offline_decision_states) / len(reachable), 4),
                "behavior_policy": offline_dataset.behavior_policy_name,
                "epsilon": offline_dataset.epsilon,
            }
        ],
    )
    # Off-policy evaluation: estimate three targets' values from a behaviour log, graded vs truth.
    ope_dataset = collect_logged_dataset(episodes=ope_episodes, epsilon=0.3, seed=7)
    ope_targets: list[tuple[str, Policy]] = [
        ("heuristic_router", HeuristicRouterPolicy()),
        ("dp_optimal", QTablePolicy(q_table=optimal_values, name="dp_optimal")),
        ("random", RandomPolicy(seed=1)),
    ]
    write_csv_artifact(
        args.output_dir / "ope" / "estimator_comparison.csv",
        ope_report_rows(ope_dataset, ope_targets, episodes_per_scenario=ope_truth_episodes),
    )
    # Cost-aware cascade: the cost/quality (money + latency vs reward) operating frontier.
    write_csv_artifact(
        args.output_dir / "cost_cascade" / "cost_quality_curve.csv",
        cost_cascade_curve(effort_levels=(0, 1, 2, 3, 4), episodes_per_scenario=eval_episodes),
    )
    write_csv_artifact(args.output_dir / "eval" / "policy_comparison.csv", evaluation_summary)
    write_csv_artifact(args.output_dir / "eval" / "scenario_results.csv", evaluation_rows)
    write_text_artifact(
        args.output_dir / "reward" / "reward_hacking_report.md",
        reward_hacking_report(comparison_rows),
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
    # Lane A bridge: the learned policy driving an agent loop, each step mapped to an SDK construct.
    sdk_trace: list[dict[str, int | float | str]] = []
    for scenario_id in (0, 1, 2, 3, 4):
        sdk_trace.extend(
            run_bridged_episode(
                policy=learned,
                scenario_id=scenario_id,
                seed=0 if args.quick else None,
            )
        )
    write_csv_artifact(args.output_dir / "sdk_bridge" / "orchestration_trace.csv", sdk_trace)
    write_text_artifact(
        args.output_dir / "sdk_bridge" / "bridge_report.md",
        bridge_report_markdown(sdk_present=sdk_available()),
    )
    # Lane B: tune a toy LM's weights from preferences (RLHF/DPO/GRPO/RLVR) and compare them.
    preference_epochs = 120 if args.quick else 200
    comparison_rows, curve_rows = compare_preference_methods(epochs=preference_epochs)
    write_csv_artifact(args.output_dir / "preference" / "method_comparison.csv", comparison_rows)
    write_csv_artifact(args.output_dir / "preference" / "training_curves.csv", curve_rows)
    # Lane C: multi-agent coordination -- independent vs centralised (joint-action) learning.
    marl_seeds = 8 if args.quick else 30
    marl_episodes = 1500 if args.quick else 4000
    write_csv_artifact(
        args.output_dir / "marl" / "coordination_comparison.csv",
        marl_comparison_rows(CLIMBING_GAME, seeds=marl_seeds, episodes=marl_episodes),
    )
    marl_curve_rows: list[dict[str, int | float | str]] = []
    independent_run = train_independent_q_learning(CLIMBING_GAME, episodes=marl_episodes, seed=0)
    joint_run = train_joint_action_learner(CLIMBING_GAME, episodes=marl_episodes, seed=0)
    for marl_method, marl_result in (("independent", independent_run), ("joint", joint_run)):
        for marl_row in marl_result.training_curve:
            marl_curve_rows.append(
                {
                    "method": marl_method,
                    "episode": marl_row["episode"],
                    "greedy_team_reward": marl_row["greedy_team_reward"],
                }
            )
    write_csv_artifact(args.output_dir / "marl" / "training_curves.csv", marl_curve_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
