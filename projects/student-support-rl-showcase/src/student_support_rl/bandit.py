"""Contextual-bandit warm-up: epsilon-greedy ridge regression on student-support actions.

This module is the first rung on the RL ladder (contextual bandit -> MDP -> Q-learning ->
DQN -> policy gradient -> actor-critic -> PPO). A *contextual bandit* makes a single one-shot
decision per round: it sees a context vector ``x`` (the student's situation), picks one of four
support actions, and collects an immediate Bernoulli reward. Crucially there are NO transitions
and NO discounting -- the next context does not depend on the chosen action (here the contexts
simply cycle through a fixed catalog), so this is the degenerate single-step special case of an
MDP. It isolates the exploration-vs-exploitation trade-off before transitions and credit
assignment enter the picture in later modules.

The learner estimates each action's payoff with an independent per-action ridge regression
(a linear model), then acts epsilon-greedily on those estimates. HONESTY -- this is NOT LinUCB:
there is no upper-confidence-bound / optimism bonus added to the prediction; exploration comes
solely from the epsilon coin flip (plus a one-pull-per-arm warm-up), not from the model's
uncertainty. The synthetic ground-truth reward model lets us compute exact regret against the
known-optimal action, which is only knowable because the payoffs are simulated.

RL concept:
    Contextual bandit, exploration vs exploitation, and regret. See
    docs/exploration-and-bandits.md and docs/math-notes.md; the single-step framing connects
    forward to docs/mdp-and-environment.md.

Math:
    Per-action ridge least squares with regularizer lambda: A_a = lambda*I + sum_t x_t x_t^T,
    b_a = sum_t r_t x_t, theta_a = A_a^{-1} b_a, predicted payoff = theta_a . x. True reward is
    Bernoulli with mean sigma(w_a . x) where sigma(z) = 1/(1+exp(-z)). Cumulative regret
    regret_T = sum_t [mu*(x_t) - mu_{a_t}(x_t)] with mu the expected (not sampled) reward.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from student_support_rl.environment import (
    ACTION_LABELS,
    ScenarioDefinition,
    risk_from_metrics,
    scenario_catalog,
)

# Synthetic ground-truth weight vector w_a for each of the four actions, indexed by action id.
# The expected reward of action a in context x is sigma(w_a . x); these weights ARE the unknown
# the bandit is trying to estimate via ridge regression, and they let us compute exact regret.
CONTEXTUAL_REWARD_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (0.55, 0.6, 0.45, -0.8, -0.35, -1.0, -0.4),
    (0.2, -0.15, 0.3, -0.05, -0.1, 0.2, 0.0),
    (0.1, 0.05, -0.55, 0.55, -0.2, 0.85, 0.2),
    (-0.25, -0.2, -0.1, 0.95, 0.7, 1.1, 0.45),
)
# Ridge regularizer lambda: seeds each per-action design matrix as A_a = lambda*I, keeping
# A_a invertible before any data arrives and shrinking theta_a toward 0 (ridge regression).
_RIDGE_REGULARIZATION = 1.0


@dataclass(frozen=True)
class BanditRunResult:
    """Immutable record of one contextual-bandit run, split into reward and regret logs.

    Bundles the two per-step diagnostic traces produced by :func:`run_bandit_experiment` so
    callers (notebooks, plots, evaluation scripts) can chart learning and the explore/exploit
    trade-off. The reward trace tracks what the learner earned and predicted; the regret trace
    tracks the gap to the known-optimal action, the canonical bandit performance metric.

    Attributes:
        reward_trace: One dict per step with the chosen action, sampled reward, the model's
            predicted (estimated) value, the cumulative reward, and the oracle optimal action.
        regret_trace: One dict per step with the instantaneous regret mu*(x) - mu_{a}(x) and its
            running sum (cumulative regret), alongside the chosen and optimal action labels.

    RL concept:
        Reward and regret bookkeeping for bandits; see docs/exploration-and-bandits.md and
        docs/evaluation-and-governance.md.
    """

    reward_trace: list[dict[str, int | float | str]]
    regret_trace: list[dict[str, int | float | str]]


def run_bandit_experiment(
    *,
    steps: int = 600,
    epsilon: float = 0.2,
    seed: int = 7,
) -> BanditRunResult:
    """Run an epsilon-greedy ridge-regression contextual bandit and log reward and regret.

    Simulates ``steps`` one-shot decisions. Each round the learner observes a context vector for
    the next student scenario, estimates every action's payoff with its own ridge regression, and
    acts epsilon-greedily; with probability ``epsilon`` (and always until each arm has been pulled
    once) it explores uniformly at random, otherwise it exploits the highest predicted payoff. A
    Bernoulli reward is drawn from the synthetic ground-truth model, the chosen arm's ridge
    statistics are updated online, and both the reward and the regret-to-optimal are recorded.
    This is the bandit (single-step) base of the RL ladder; no transitions or discounting appear.

    Args:
        steps: Number of bandit rounds to simulate; must be positive.
        epsilon: Exploration rate in [0, 1] for the epsilon-greedy rule (probability of a uniform
            random action instead of the greedy/exploit choice).
        seed: Seed for the Python RNG governing exploration coin flips and Bernoulli rewards,
            making the whole run deterministic and reproducible.

    Returns:
        A :class:`BanditRunResult` holding the per-step reward and regret traces.

    Raises:
        ValueError: If ``steps`` is not positive.

    RL concept:
        Exploration vs exploitation and regret for contextual bandits; see
        docs/exploration-and-bandits.md and docs/math-notes.md.

    Math:
        Greedy choice a_t = argmax_a theta_a . x_t with theta_a = A_a^{-1} b_a; reward
        R_{t+1} ~ Bernoulli(sigma(w_{a_t} . x_t)); cumulative regret accumulates
        regret_T = sum_t [mu*(x_t) - mu_{a_t}(x_t)].
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = random.Random(seed)
    feature_count = len(CONTEXTUAL_REWARD_WEIGHTS[0])
    # One ridge design matrix per action, each initialized to A_a = lambda*I (the regularizer).
    design_matrices = [
        np.eye(feature_count, dtype=float) * _RIDGE_REGULARIZATION
        for _ in ACTION_LABELS
    ]
    # Per-action reward-feature accumulators b_a = sum r_t x_t, paired with the design matrices.
    reward_vectors = [np.zeros(feature_count, dtype=float) for _ in ACTION_LABELS]
    counts = [0 for _ in ACTION_LABELS]  # pulls per arm; drives the one-pull warm-up below
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    reward_trace: list[dict[str, int | float | str]] = []
    regret_trace: list[dict[str, int | float | str]] = []
    scenarios = scenario_catalog()

    for step in range(1, steps + 1):
        # Contexts cycle through a fixed catalog: bandit has NO transitions, the next x is
        # independent of the action just taken (single-step special case of an MDP).
        scenario = scenarios[(step - 1) % len(scenarios)]
        context = _context_vector(
            engagement=scenario.engagement,
            completion=scenario.completion,
            pressure=scenario.pressure,
            prior_interventions=scenario.prior_interventions,
        )
        # True expected reward mu_a(x) = sigma(w_a . x) for each arm, from the synthetic model.
        expected_rewards = [
            _expected_reward(context, weights) for weights in CONTEXTUAL_REWARD_WEIGHTS
        ]
        # Oracle optimal arm a* = argmax_a mu_a(x); knowable only because rewards are synthetic.
        optimal_action = max(range(len(expected_rewards)), key=lambda idx: expected_rewards[idx])

        # Epsilon-greedy action selection (plus forced warm-up: explore until every arm pulled).
        if rng.random() < epsilon or min(counts) == 0:
            action = rng.randrange(len(ACTION_LABELS))  # explore: uniform random arm
        else:
            # Exploit: pick the arm with the largest ridge prediction theta_a . x. NOTE: no UCB
            # optimism bonus is added here -- this is plain epsilon-greedy, not LinUCB.
            action = max(
                range(len(ACTION_LABELS)),
                key=lambda idx: _estimated_reward(
                    context=context,
                    design_matrix=design_matrices[idx],
                    reward_vector=reward_vectors[idx],
                ),
            )

        reward_probability = expected_rewards[action]
        # Sample the immediate reward R_{t+1} ~ Bernoulli(mu_{a_t}(x)).
        reward = 1.0 if rng.random() < reward_probability else 0.0
        counts[action] += 1
        context_vector = np.array(context, dtype=float)
        # Online ridge update for the chosen arm: A_a += x x^T and b_a += r x.
        design_matrices[action] += np.outer(context_vector, context_vector)
        reward_vectors[action] += reward * context_vector
        cumulative_reward += reward
        # Instantaneous regret = mu*(x) - mu_{a_t}(x), the expected-reward gap to the optimal arm.
        instantaneous_regret = expected_rewards[optimal_action] - reward_probability
        cumulative_regret += instantaneous_regret  # running regret_T = sum_t of the above

        reward_trace.append(
            {
                "step": step,
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "context_signature": _context_signature(scenario),
                "action": action,
                "action_label": ACTION_LABELS[action],
                "reward": reward,
                "expected_reward": round(reward_probability, 4),
                "cumulative_reward": round(cumulative_reward, 4),
                "estimated_value": round(
                    _estimated_reward(
                        context=context,
                        design_matrix=design_matrices[action],
                        reward_vector=reward_vectors[action],
                    ),
                    4,
                ),
                "optimal_action": optimal_action,
                "optimal_action_label": ACTION_LABELS[optimal_action],
            }
        )
        regret_trace.append(
            {
                "step": step,
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "action": action,
                "action_label": ACTION_LABELS[action],
                "optimal_action": optimal_action,
                "optimal_action_label": ACTION_LABELS[optimal_action],
                "instantaneous_regret": round(instantaneous_regret, 4),
                "cumulative_regret": round(cumulative_regret, 4),
            }
        )

    return BanditRunResult(reward_trace=reward_trace, regret_trace=regret_trace)


def _context_vector(
    *,
    engagement: int,
    completion: int,
    pressure: int,
    prior_interventions: int,
) -> tuple[float, ...]:
    """Build the feature vector x that conditions the bandit's per-action payoff estimates.

    Maps a student scenario's raw metrics into the normalized feature vector x on which both the
    synthetic reward model and the ridge regressions operate. The leading constant 1.0 is the
    intercept/bias feature, the middle entries are min-max-style normalizations of the raw
    metrics, and the trailing entry is a binary high-pressure indicator. In a contextual bandit
    this "context" is exactly what makes the optimal action depend on the situation.

    Args:
        engagement: Raw engagement metric (0-4 scale).
        completion: Raw completion metric (0-4 scale).
        pressure: Raw workload-pressure metric (0-4 scale).
        prior_interventions: Count of past interventions for this student.

    Returns:
        A 7-tuple x = (1, engagement/4, completion/4, pressure/4, prior_interventions/3,
        risk/3, [pressure>=3]); its length matches the per-action weight vectors.

    RL concept:
        Context features for a contextual bandit; see docs/exploration-and-bandits.md.
    """
    risk = risk_from_metrics(
        engagement=engagement,
        completion=completion,
        pressure=pressure,
        prior_interventions=prior_interventions,
    )
    # Feature vector x: index 0 is the bias term; remaining entries normalize the raw metrics.
    return (
        1.0,
        engagement / 4.0,
        completion / 4.0,
        pressure / 4.0,
        prior_interventions / 3.0,
        risk / 3.0,
        float(pressure >= 3),
    )


def _context_signature(scenario: ScenarioDefinition) -> str:
    """Render a scenario's raw metrics as a stable string key for trace logging.

    Produces a compact, human-readable signature of the context so each reward-trace row is
    self-describing and rows from the same scenario are easy to group. Purely a diagnostic
    formatting helper -- it plays no part in action selection or learning.

    Args:
        scenario: The scenario definition whose raw metrics are being summarized.

    Returns:
        A pipe-delimited string of the four raw context metrics.
    """
    return (
        f"engagement={scenario.engagement}|completion={scenario.completion}"
        f"|pressure={scenario.pressure}|prior_interventions={scenario.prior_interventions}"
    )


def _expected_reward(context: tuple[float, ...], weights: tuple[float, ...]) -> float:
    """Compute an action's true Bernoulli success probability under the synthetic reward model.

    Evaluates the ground-truth mean reward mu_a(x) = sigma(w_a . x) for one action: the dot
    product of context and that action's true weights is squashed through the logistic sigmoid
    into a probability in (0, 1). This is the oracle the learner never sees directly; it both
    samples the realized reward and defines the optimal action used for regret.

    Args:
        context: The feature vector x for the current scenario.
        weights: The action's synthetic ground-truth weight vector w_a (same length as x).

    Returns:
        The success probability sigma(w_a . x) in the open interval (0, 1).

    RL concept:
        Synthetic ground-truth reward model enabling exact regret; see docs/math-notes.md and
        docs/reward-design-and-hacking.md.

    Math:
        score = w_a . x; mu_a(x) = sigma(score) = 1 / (1 + exp(-score)).
    """
    # Linear score w_a . x, then logistic sigmoid -> Bernoulli success probability mu_a(x).
    score = sum(feature * weight for feature, weight in zip(context, weights, strict=True))
    return 1.0 / (1.0 + math.exp(-score))


def _estimated_reward(
    *,
    context: tuple[float, ...],
    design_matrix: np.ndarray,
    reward_vector: np.ndarray,
) -> float:
    """Predict an action's payoff from its ridge-regression statistics for the given context.

    Solves the per-action ridge normal equations for the weight estimate theta_a = A_a^{-1} b_a
    and returns the linear prediction theta_a . x. This is the learner's *current best guess* of
    the action's value; the greedy branch of the epsilon-greedy rule maximizes exactly this
    quantity. There is deliberately NO upper-confidence-bound bonus added -- exploration is
    handled by epsilon, so this is plain epsilon-greedy rather than LinUCB.

    Args:
        context: The feature vector x for the current scenario.
        design_matrix: The action's ridge design matrix A_a = lambda*I + sum x x^T.
        reward_vector: The action's reward-feature accumulator b_a = sum r x.

    Returns:
        The scalar predicted payoff theta_a . x.

    RL concept:
        Linear value estimation driving the exploit step; see docs/exploration-and-bandits.md and
        docs/value-based-learning.md.

    Math:
        theta_a = A_a^{-1} b_a (solved directly); predicted payoff = theta_a . x.
    """
    # Solve ridge normal equations A_a theta_a = b_a for the estimated weights theta_a.
    theta = np.linalg.solve(design_matrix, reward_vector)
    return float(np.dot(theta, np.array(context, dtype=float)))  # linear prediction theta_a . x
