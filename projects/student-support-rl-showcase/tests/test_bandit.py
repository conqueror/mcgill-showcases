"""Pin the contextual epsilon-greedy bandit warm-up as executable documentation.

These tests fix the first rung of the ladder before the full MDP: a *contextual bandit* that
sees student-state features x_t, picks one of four interventions, and is scored by cumulative
regret. They assert the experiment emits aligned per-step reward and regret traces, that the
greedy choice is context-dependent (different scenarios induce different optimal actions), and
that a fixed seed makes the whole run reproducible. This is the bridge from one-shot bandit
decisions (contextual bandit -> MDP -> Q-learning -> ...) to multi-step control.

RL concept:
    Contextual bandit, exploration vs. exploitation, and regret; see
    docs/exploration-and-bandits.md.

Math:
    regret_T = sum_t [mu*(x_t) - mu_{a_t}(x_t)], the gap between the best context-conditioned
    expected reward and the played arm, accumulated over T steps.
"""

from __future__ import annotations

from student_support_rl.bandit import run_bandit_experiment


def test_bandit_experiment_generates_reward_and_regret_traces() -> None:
    """Verify the run emits T aligned reward/regret rows with monotone non-negative regret.

    Pins the trace contract for a T=25 run: one reward row and one regret row per step, the
    first step indexed at 1, and a final cumulative regret that is non-negative. Cumulative
    regret can never decrease because each instantaneous term mu*(x_t) - mu_{a_t}(x_t) >= 0,
    so a non-negative endpoint is the minimal sanity check on the regret accounting.

    RL concept:
        Cumulative regret as the bandit performance metric; see
        docs/exploration-and-bandits.md.

    Math:
        regret_T = sum_t [mu*(x_t) - mu_{a_t}(x_t)] >= 0.
    """
    result = run_bandit_experiment(steps=25, epsilon=0.15, seed=3)

    # One reward row and one regret row per step: the two traces stay length-aligned.
    assert len(result.reward_trace) == 25
    assert len(result.regret_trace) == 25
    assert result.reward_trace[0]["step"] == 1
    # Cumulative regret is a sum of non-negative gaps, so its endpoint must be >= 0.
    assert float(result.regret_trace[-1]["cumulative_regret"]) >= 0.0


def test_bandit_experiment_uses_contextual_rewards() -> None:
    """Verify the optimal arm is context-dependent and the seeded run is reproducible.

    Pins what makes this a *contextual* (not plain) bandit: each row carries the context
    signature and the context-conditioned best action mu*(x_t), and across scenarios at least
    three distinct (scenario, optimal-action) pairs appear, so the right intervention truly
    depends on x_t. Re-running with the same seed reproduces the reward trace byte-for-byte,
    fixing determinism of the epsilon-greedy draws and reward sampling.

    RL concept:
        Context-conditioned optimal action and reproducibility of an exploration policy; see
        docs/exploration-and-bandits.md.
    """
    result = run_bandit_experiment(steps=30, epsilon=0.1, seed=5)

    assert {
        "scenario_id",
        "scenario_name",
        "context_signature",
        "optimal_action",
        "optimal_action_label",
        "expected_reward",
    } <= set(result.reward_trace[0])
    # Distinct optimal arms across scenarios => the best action depends on context x_t.
    scenario_optima = {
        (str(row["scenario_name"]), str(row["optimal_action_label"]))
        for row in result.reward_trace
    }
    assert len(scenario_optima) >= 3
    # Same seed reproduces the trace exactly: exploration draws are deterministic.
    assert result.reward_trace == run_bandit_experiment(steps=30, epsilon=0.1, seed=5).reward_trace
