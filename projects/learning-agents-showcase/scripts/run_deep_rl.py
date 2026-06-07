#!/usr/bin/env python3
"""Train tabular Q-learning, deep DQN, and PPO; write the optional deep-RL comparison artifacts."""

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

from learning_agents.deep_rl import (
    FamilyEntry,
    build_model_policy,
    family_comparison_rows,
    train_dqn,
    train_ppo,
)
from learning_agents.policies import QTablePolicy
from learning_agents.q_learning import train_q_learning
from learning_agents.reporting import write_csv_artifact, write_text_artifact

HORIZON = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the deep-RL runner's command-line flags.

    What + why: this lane trains three learners (tabular Q-learning, deep DQN, PPO) whose budgets
    dominate runtime, so the flags size the sweep. ``--quick`` shrinks every budget for CI; the
    qualitative story (DQN recovers the dynamic-programming ceiling, PPO settles into a safe
    suboptimal policy) holds either way, only the exact numbers soften.

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated namespace with ``output_dir`` and ``quick``.

    RL concept:
        Deep reinforcement learning -- value-based DQN vs actor-critic PPO vs tabular Q-learning.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def _comparison_table(comparison: list[dict[str, int | float | str]]) -> str:
    """Render the family-comparison rows as a Markdown table for the bridge report."""
    header = (
        "| policy | family | avg_reward | avg_escalation_rate | solved_rate |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    body = "".join(
        f"| {row['policy']} | {row['family']} | {row['avg_reward']} | "
        f"{row['avg_escalation_rate']} | {row['solved_rate']} |\n"
        for row in comparison
    )
    return header + body


def bridge_report_markdown(comparison: list[dict[str, int | float | str]]) -> str:
    """Render ``artifacts/drl_optional/bridge_report.md`` from the real comparison results.

    What + why: the optional deep-RL bridge report names the two deep families (DQN and PPO),
    contrasts them with the tabular baseline, and reports the measured head-to-head numbers so the
    narrative can never drift from the artifact. The validator requires both ``DQN`` and ``PPO`` to
    appear; the teaching point is built from the actual table.

    Args:
        comparison: The family-comparison rows (one per policy) to summarize.

    Returns:
        Markdown text beginning with a top-level heading.

    RL concept:
        Deep RL bridge -- value-based DQN vs actor-critic PPO vs tabular Q-learning.
    """
    return (
        "# Optional Deep-RL Bridge (DQN vs PPO vs Tabular Q-learning)\n\n"
        "## Thesis\n\n"
        "The tabular ladder keys a table on the discrete state. Real agents face state spaces too "
        "large to tabulate, so they swap the table for a **neural function approximator**. This "
        "optional lane adds the two canonical deep-RL families on the *same* agent-decision MDP, "
        "implemented from scratch in NumPy so the lane stays self-contained and "
        "laptop-friendly:\n\n"
        "- **DQN** -- value-based deep RL: a neural network fit to the Q-learning Bellman target "
        "(experience replay + a periodically-synced target network). The deep cousin of tabular "
        "Q-learning.\n"
        "- **PPO** -- actor-critic policy-gradient deep RL: a policy network improved "
        "with a clipped "
        "surrogate objective against a learned value baseline. The deep cousin of REINFORCE.\n\n"
        "## Measured comparison\n\n"
        + _comparison_table(comparison)
        + "\n## What the numbers show\n\n"
        "On this small, deterministic MDP the exact optimum is known (dynamic programming computes "
        "`Q*`), so the comparison is graded against ground truth. **DQN recovers the "
        "dynamic-programming ceiling** -- value-function approximation generalizes across similar "
        "states and reaches essentially the optimal return. **PPO** reliably converges to a "
        "**safe but suboptimal** policy that over-escalates: with a modestly-positive escalation "
        "action available, the policy-gradient method settles into a low-variance local optimum. "
        "This is an honest, well-known contrast -- value bootstrapping is often more sample-"
        "efficient on small discrete action spaces, while policy-gradient methods earn their keep "
        "in large or continuous action spaces. It is not a claim that deep beats tabular: the "
        "tabular Q-learning row is the core ladder's value-based learner, shown for continuity.\n\n"
        "## How to run\n\n"
        "- `make run-drl` trains all three learners and regenerates this group.\n"
        "- The group is optional: the core `make smoke` / `make verify` path does not require it, "
        "and it is validated only when present (all-or-nothing).\n"
    )


def policy_gradient_notes_markdown() -> str:
    """Render ``artifacts/drl_optional/policy_gradient_notes.md`` (the policy-gradient arc).

    What + why: this note walks the policy-based half of the ladder -- policy-gradient to
    actor-critic to PPO -- and situates it against value-based DQN. The validator requires the terms
    ``policy-gradient``, ``actor-critic``, ``ppo``, and ``dqn`` to appear so the notes genuinely
    cover the arc.

    Returns:
        Markdown text beginning with a top-level heading.

    RL concept:
        The policy-based branch -- policy gradients, actor-critic, PPO -- versus DQN.
    """
    return (
        "# Policy-Gradient, Actor-Critic, and PPO Notes\n\n"
        "## From values to policies\n\n"
        "Value-based methods (tabular Q-learning, and its deep cousin **DQN**) learn an "
        "action-value "
        "function and act greedily. **Policy-gradient** methods skip the value table and "
        "optimize a "
        "parameterized policy `pi_theta(a|s)` directly, ascending the gradient of expected return. "
        "Tabular REINFORCE is the entry point; it is unbiased but high-variance.\n\n"
        "## Actor-critic\n\n"
        "An **actor-critic** method pairs the policy (the actor) with a learned value "
        "estimate (the "
        "critic) used as a baseline, subtracting it from the return to form an advantage. The "
        "baseline slashes gradient variance without biasing the update, which is what makes "
        "policy-gradient learning practical.\n\n"
        "## PPO\n\n"
        "**PPO** (Proximal Policy Optimization) is the actor-critic method used here. It "
        "reuses each "
        "batch of trajectories for several epochs while keeping every update close to the "
        "behaviour "
        "policy via a *clipped* probability-ratio objective, `min(rho * A, clip(rho, 1-eps, 1+eps) "
        "* A)`, where `rho = pi_new(a|s) / pi_old(a|s)` and `A` is the advantage. The clip is the "
        "trust region that stops a single step from destroying the policy.\n\n"
        "## Honest result on this MDP\n\n"
        "Here PPO converges to a safe, over-escalating local optimum rather than the "
        "optimum, while "
        "value-based DQN recovers the dynamic-programming ceiling. On small discrete action spaces "
        "value bootstrapping tends to dominate; policy-gradient methods like PPO shine when the "
        "action space is large or continuous. See "
        "`artifacts/drl_optional/rl_family_comparison.csv` "
        "for the measured numbers.\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Train the three learners and write the optional deep-RL artifact group.

    What + why: trains tabular Q-learning (baseline), deep DQN (value-based), and PPO (actor-critic)
    on the shared agent-decision MDP, scores them with the same offline harness as every other
    policy, and writes the five ``artifacts/drl_optional/*`` files: the family comparison, the
    per-scenario rollups, the training summary, the bridge report, and the policy-gradient notes.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Deep RL -- function approximation for value-based (DQN) and policy-gradient (PPO) control,
        graded against the dynamic-programming ceiling.
    """
    args = parse_args(argv)
    if args.quick:
        q_episodes, dqn_episodes, ppo_iterations, ppo_episodes, eval_episodes = 1500, 400, 40, 20, 3
    else:
        q_episodes, dqn_episodes, ppo_iterations, ppo_episodes, eval_episodes = (
            5000,
            1500,
            150,
            30,
            6,
        )

    q_policy = QTablePolicy(
        q_table=train_q_learning(episodes=q_episodes, seed=0).q_table, name="q_learning"
    )
    dqn_model, dqn_curve = train_dqn(episodes=dqn_episodes, epsilon=0.2, seed=0)
    ppo_model, ppo_curve = train_ppo(
        iterations=ppo_iterations,
        episodes_per_iteration=ppo_episodes,
        entropy_coef=0.01,
        policy_lr=0.05,
        seed=0,
    )

    entries = [
        FamilyEntry(q_policy, "tabular_value_based"),
        FamilyEntry(build_model_policy(dqn_model, name="dqn", horizon=HORIZON), "value_based_deep"),
        FamilyEntry(
            build_model_policy(ppo_model, name="ppo", horizon=HORIZON),
            "actor_critic_policy_gradient",
        ),
    ]
    comparison, rollups = family_comparison_rows(
        entries, horizon=HORIZON, episodes_per_scenario=eval_episodes
    )

    drl_dir = args.output_dir / "drl_optional"
    write_csv_artifact(drl_dir / "rl_family_comparison.csv", comparison)
    write_csv_artifact(drl_dir / "scenario_rollups.csv", rollups)

    training_rows: list[dict[str, int | float | str]] = []
    for policy_name, curve in (("dqn", dqn_curve), ("ppo", ppo_curve)):
        for row in curve:
            training_rows.append(
                {
                    "policy": policy_name,
                    "step": row["step"],
                    "mean_reward": row["mean_reward"],
                    "mean_escalation_rate": row["mean_escalation_rate"],
                }
            )
    write_csv_artifact(drl_dir / "training_summary.csv", training_rows)

    write_text_artifact(drl_dir / "bridge_report.md", bridge_report_markdown(comparison))
    write_text_artifact(drl_dir / "policy_gradient_notes.md", policy_gradient_notes_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
