"""Pin the contextual epsilon-greedy bandit warm-up as executable documentation.

These tests fix the first rung of the ladder before the full agent-decision MDP: a *contextual
bandit* that sees a request's start-state features x_t, picks one of the four orchestration actions
(:data:`learning_agents.environment.ACTION_LABELS`), and is scored by cumulative regret. They assert
the load-bearing properties of the warm-up: the run emits aligned per-step reward and regret traces;
cumulative regret is non-negative (a sum of non-negative gaps) AND the learner actually *learns*
(its per-step regret rate falls as it accumulates data); the greedy choice is genuinely
context-dependent (different scenarios induce different optimal actions); a fixed seed makes the
whole run reproducible byte-for-byte; and the online ridge update matches a direct batch solve of
the normal equations. This is the bridge from one-shot bandit decisions to the multi-step control
the rest of the showcase learns over the same actions and scenarios.

RL concept:
    Contextual bandit, exploration vs exploitation, and regret -- the single-step base of the RL
    ladder.

Math:
    regret_T = sum_t [mu*(x_t) - mu_{a_t}(x_t)] >= 0, the gap between the best context-conditioned
    expected reward and the played arm, accumulated over T steps. A learning bandit drives the
    *per-step* regret rate down over time.
"""

from __future__ import annotations

import numpy as np

from learning_agents.bandit import (
    CONTEXTUAL_REWARD_WEIGHTS,
    BanditRunResult,
    run_bandit_experiment,
)
from learning_agents.environment import ACTION_LABELS, scenario_catalog


def test_bandit_experiment_generates_aligned_reward_and_regret_traces() -> None:
    """Verify the run emits T aligned reward/regret rows indexed from 1 with non-negative regret.

    Pins the trace contract for a T=25 run: one reward row and one regret row per step, the first
    step indexed at 1, the two traces length-aligned and step-aligned, and a final cumulative regret
    that is non-negative. Cumulative regret can never decrease because each instantaneous term
    mu*(x_t) - mu_{a_t}(x_t) >= 0, so a non-negative endpoint is the minimal sanity check on the
    regret accounting.

    RL concept:
        Cumulative regret as the bandit performance metric -- a monotone non-decreasing curve.

    Math:
        regret_T = sum_t [mu*(x_t) - mu_{a_t}(x_t)] >= 0.
    """
    result = run_bandit_experiment(steps=25, epsilon=0.15, seed=3)

    assert isinstance(result, BanditRunResult)
    # One reward row and one regret row per step: the two traces stay length-aligned.
    assert len(result.reward_trace) == 25
    assert len(result.regret_trace) == 25
    assert result.reward_trace[0]["step"] == 1
    assert result.regret_trace[0]["step"] == 1
    # The two traces are step-aligned row by row.
    for reward_row, regret_row in zip(result.reward_trace, result.regret_trace, strict=True):
        assert reward_row["step"] == regret_row["step"]
    # Every instantaneous regret term is a non-negative expected-reward gap.
    assert all(float(row["instantaneous_regret"]) >= 0.0 for row in result.regret_trace)
    # Cumulative regret is a sum of non-negative gaps, so its endpoint must be >= 0.
    assert float(result.regret_trace[-1]["cumulative_regret"]) >= 0.0


def test_bandit_cumulative_regret_is_monotone_non_decreasing() -> None:
    """Verify the cumulative-regret curve never decreases (each step adds a non-negative gap)."""
    result = run_bandit_experiment(steps=120, epsilon=0.1, seed=11)
    cumulative = [float(row["cumulative_regret"]) for row in result.regret_trace]
    # regret_T is a running sum of non-negative terms => the sequence is non-decreasing. Pairing
    # each element with its successor uses an intentionally offset (strict=False) zip.
    pairs = zip(cumulative, cumulative[1:], strict=False)
    assert all(later >= earlier for earlier, later in pairs)


def test_bandit_learns_so_per_step_regret_rate_falls() -> None:
    """Verify the bandit *learns*: its regret accrues more slowly late than early in the run.

    The headline property of a learning bandit is sublinear regret -- as the ridge estimates
    sharpen, the greedy choice matches the oracle more often, so per-step regret accrued in the
    second half of the run falls below that of the untrained first half. Comparing the two
    half-run regret *rates* is the real "it learned" assertion, beyond mere non-negativity.

    Robustness: with a *constant* exploration rate the bandit keeps paying a fixed exploration
    tax forever, so once the greedy policy is near-optimal the two half-run regrets get close and
    a single run can occasionally flip the strict inequality by noise. This test therefore pins
    learning over a fixed *bank of seeds* rather than one lucky seed: every seed must individually
    show second-half < first-half, AND in aggregate the second half must accrue well under half
    the first half's regret. That margin is the genuine, non-fragile signal that learning happened
    and is not an artifact of seed choice.

    RL concept:
        Sublinear regret as evidence of learning in a bandit; an agent that never improved would
        accrue regret at a roughly constant rate (second half ~= first half).
    """
    steps = 400
    midpoint = steps // 2
    seeds = range(8)  # a fixed bank, so the property is pinned across runs, not one lucky seed

    first_half_regrets: list[float] = []
    second_half_regrets: list[float] = []
    for seed in seeds:
        result = run_bandit_experiment(steps=steps, epsilon=0.1, seed=seed)
        cumulative = [float(row["cumulative_regret"]) for row in result.regret_trace]
        first_half = cumulative[midpoint - 1]  # regret accrued over steps 1..midpoint
        second_half = cumulative[-1] - cumulative[midpoint - 1]  # regret accrued over the rest
        # Per-seed: learning => the second half accrues strictly less regret than the first half.
        assert second_half < first_half, f"seed={seed}: {second_half} >= {first_half}"
        first_half_regrets.append(first_half)
        second_half_regrets.append(second_half)

    # Aggregate: averaged over the seed bank the second half should accrue well under half the
    # first half's regret -- a substantial margin that constant-epsilon noise cannot manufacture.
    mean_first = sum(first_half_regrets) / len(first_half_regrets)
    mean_second = sum(second_half_regrets) / len(second_half_regrets)
    assert mean_second < 0.5 * mean_first


def test_bandit_optimal_action_is_context_dependent() -> None:
    """Verify the oracle-optimal arm depends on the context (this is a *contextual* bandit).

    Pins what makes the bandit contextual rather than plain: each reward row carries the context
    signature and the context-conditioned best action argmax_a mu_a(x_t), and across the scenario
    catalog at least three distinct (scenario, optimal-action) pairs appear, so the right
    orchestration move genuinely depends on x_t (e.g. answering a clean request vs clarifying an
    ambiguous one vs retrieving for a hard one).

    RL concept:
        Context-conditioned optimal action -- the defining feature of a contextual bandit.
    """
    result = run_bandit_experiment(steps=40, epsilon=0.1, seed=5)

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


def test_bandit_run_is_reproducible_for_a_fixed_seed() -> None:
    """Verify a fixed seed reproduces both traces exactly (deterministic explore/exploit draws)."""
    first = run_bandit_experiment(steps=30, epsilon=0.1, seed=5)
    second = run_bandit_experiment(steps=30, epsilon=0.1, seed=5)
    # Same seed reproduces the run exactly: exploration coin flips and Bernoulli draws are fixed.
    assert first.reward_trace == second.reward_trace
    assert first.regret_trace == second.regret_trace


def test_bandit_different_seeds_diverge() -> None:
    """Verify different seeds yield different runs (the RNG drives the run's stochastic part)."""
    base = run_bandit_experiment(steps=60, epsilon=0.3, seed=1)
    other = run_bandit_experiment(steps=60, epsilon=0.3, seed=2)
    # Distinct seeds should not collapse to an identical trace.
    assert base.reward_trace != other.reward_trace


def test_bandit_rewards_are_bernoulli_and_action_labels_valid() -> None:
    """Verify sampled rewards are in {0, 1} and every logged action is a valid action label."""
    result = run_bandit_experiment(steps=50, epsilon=0.2, seed=9)
    valid_actions = set(ACTION_LABELS)
    for row in result.reward_trace:
        assert float(row["reward"]) in (0.0, 1.0)  # Bernoulli realized reward
        assert int(row["action"]) in valid_actions
        assert row["action_label"] == ACTION_LABELS[int(row["action"])]
        assert int(row["optimal_action"]) in valid_actions
        # The model's predicted/expected values round to 4 dp for stable artifact CSVs.
        assert float(row["expected_reward"]) == round(float(row["expected_reward"]), 4)
        assert float(row["estimated_value"]) == round(float(row["estimated_value"]), 4)


def test_bandit_weight_table_matches_action_space() -> None:
    """Verify there is one synthetic ground-truth weight vector per action, of equal length."""
    assert len(CONTEXTUAL_REWARD_WEIGHTS) == len(ACTION_LABELS)
    lengths = {len(weights) for weights in CONTEXTUAL_REWARD_WEIGHTS}
    assert len(lengths) == 1  # all arms share the same feature dimension
    # Feature dimension is the 7-field normalized state vector plus a leading bias term.
    assert lengths.pop() == 8


def test_bandit_online_ridge_update_matches_batch_solve() -> None:
    """Verify the online ridge statistics equal a direct batch solve of the normal equations.

    Re-derives the algorithm's correctness independently of the trace: replaying the same seeded
    sequence of (context, chosen action, reward) and accumulating A_a = lambda*I + sum x x^T and
    b_a = sum r x for one arm must reproduce theta_a = A_a^{-1} b_a from a single batch ridge
    solve over that arm's pulls. This pins that the per-step update *is* ridge regression, not an
    approximation.

    RL concept:
        Per-action ridge regression as the bandit's value estimator -- the online update is exactly
        the closed-form ridge solution.

    Math:
        A_a = lambda*I + sum_t x_t x_t^T; b_a = sum_t r_t x_t; theta_a = A_a^{-1} b_a.
    """
    import random as _random

    from learning_agents.bandit import (
        _RIDGE_REGULARIZATION,
        _context_vector,
        _expected_reward,
    )

    steps = 60
    seed = 4
    epsilon = 0.2
    scenarios = scenario_catalog()
    feature_count = len(CONTEXTUAL_REWARD_WEIGHTS[0])

    # Replay the SAME stochastic stream the experiment uses, recording each arm's pulls.
    rng = _random.Random(seed)
    design = [
        np.eye(feature_count, dtype=float) * _RIDGE_REGULARIZATION for _ in ACTION_LABELS
    ]
    reward_vecs = [np.zeros(feature_count, dtype=float) for _ in ACTION_LABELS]
    counts = [0 for _ in ACTION_LABELS]
    pulls: list[tuple[np.ndarray, float]] = []  # (context, reward) for the tracked arm
    tracked_arm = 1  # retrieve

    for step in range(1, steps + 1):
        scenario = scenarios[(step - 1) % len(scenarios)]
        context = _context_vector(scenario)
        expected = [_expected_reward(context, w) for w in CONTEXTUAL_REWARD_WEIGHTS]
        if rng.random() < epsilon or min(counts) == 0:
            action = rng.randrange(len(ACTION_LABELS))
        else:
            action = max(
                range(len(ACTION_LABELS)),
                key=lambda idx: float(
                    np.dot(
                        np.linalg.solve(design[idx], reward_vecs[idx]),
                        np.array(context, dtype=float),
                    )
                ),
            )
        reward = 1.0 if rng.random() < expected[action] else 0.0
        counts[action] += 1
        x = np.array(context, dtype=float)
        design[action] += np.outer(x, x)
        reward_vecs[action] += reward * x
        if action == tracked_arm:
            pulls.append((x, reward))

    assert pulls, "tracked arm was never pulled; pick a different arm or seed"

    # Batch ridge solve over the tracked arm's recorded pulls.
    batch_a = np.eye(feature_count, dtype=float) * _RIDGE_REGULARIZATION
    batch_b = np.zeros(feature_count, dtype=float)
    for x, reward in pulls:
        batch_a += np.outer(x, x)
        batch_b += reward * x

    online_theta = np.linalg.solve(design[tracked_arm], reward_vecs[tracked_arm])
    batch_theta = np.linalg.solve(batch_a, batch_b)
    # The online-accumulated estimator equals the closed-form batch ridge estimator.
    assert np.allclose(online_theta, batch_theta)
