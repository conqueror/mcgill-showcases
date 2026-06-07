"""Tests for tabular softmax REINFORCE: the softmax policy, the gradient, and that it learns.

REINFORCE optimizes the policy pi_theta *directly* by gradient ascent on expected return, rather
than learning action values and acting greedily as Q-learning does -- the *policy gradient* rung of
the ladder (contextual bandit -> MDP -> Q-learning -> DQN -> *policy gradient* -> actor-critic ->
PPO). These tests build the algorithm from its parts. First the softmax policy parameterization: a
valid distribution, correct values (not merely summing to 1), order-preserving, numerically stable
under large shifts, and rejecting empty logits. Then the score-function update itself, by hand: a
positive advantage must raise the chosen action's probability and a negative one must lower it
(pinning the gradient's sign, so a flipped baseline would fail). Finally the end-to-end claims:
training is deterministic and alpha=0 is a no-op control, the on-policy episode return rises
*because of the gradient* (the learned run's late window beats the alpha=0 control on the same
seed), and the trained stochastic policy collects more held-out return than the untrained uniform
fallback -- the value REINFORCE actually optimizes.

Why the held-out comparison uses the *stochastic* policy (not the greedy readout): in this MDP both
``answer_direct`` (0) and ``escalate`` (3) are one-step terminal commits, so from a fresh start they
return the same single reward and receive identical gradient updates; the deterministic tie-break
then pins the greedy readout to action 0 at the start state. REINFORCE's objective is the expected
return of the *sampled* policy pi_theta, so the honest learning claim is asserted on that sampled
value -- the quantity the gradient is actually climbing.

RL concept:
    Monte-Carlo policy gradient (REINFORCE) and softmax policies; the policy is optimized directly,
    and its objective is the expected return of the stochastic policy.

Math:
    softmax policy: pi(a|s) = exp(theta_{s,a}) / sum_{a'} exp(theta_{s,a'})
    return G_t = sum_k gamma^k R_{t+k+1};  baseline b reduces variance
    policy gradient: grad J = E[grad log pi(A|s) (G_t - b)],
    with d/dtheta_{s,a'} log pi(A|s) = 1[a'=A] - pi(a'|s).
"""

from __future__ import annotations

import math
import random

import pytest

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    AgentState,
    default_reward,
)
from learning_agents.policy_gradient import (
    ReinforcePolicy,
    ReinforceResult,
    _sample_action,
    softmax,
    train_reinforce,
)

ACTION_COUNT = len(ACTION_LABELS)


def test_softmax_is_a_valid_probability_distribution() -> None:
    """softmax(logits) is a proper distribution: right length, sums to 1, entries in [0, 1].

    Includes an extreme logit (1000.0) to confirm the max-subtraction shift keeps the result valid
    where a naive ``exp`` would overflow. This is the precondition for sampling actions
    A ~ pi(.|s) during on-policy rollout.

    RL concept:
        Softmax policy parameterization.

    Math:
        pi(a|s) = exp(theta_{s,a}) / sum_{a'} exp(theta_{s,a'}).
    """
    for logits in ([0.0, 0.0, 0.0, 0.0], [5.0, -3.0, 1000.0, -2.0], [-1.5, 2.7, 0.0, 9.9]):
        probabilities = softmax(logits)
        assert len(probabilities) == len(logits)
        assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_softmax_values_are_correct_not_just_a_distribution() -> None:
    """softmax returns the right numbers and preserves logit order, not merely a distribution.

    Stronger than the previous test: equal logits give exactly uniform probabilities, the full
    probability ranking matches the logit ranking (so ``argmax(pi) == argmax(logits)``, the
    property ``select_action`` relies on), and a closed-form two-logit case pins the actual values
    via the logistic ``1 / (1 + exp(-2))``.

    RL concept:
        Softmax monotonicity underpinning the greedy readout.
    """
    # Equal logits -> exactly uniform.
    assert softmax([2.0, 2.0, 2.0, 2.0]) == pytest.approx([0.25, 0.25, 0.25, 0.25])

    # Larger logit -> larger probability; the logit ordering is preserved exactly, so
    # argmax(logits) is argmax(pi). This is the property select_action relies on.
    logits = [0.1, 3.0, -1.0, 0.5]
    probabilities = softmax(logits)
    assert probabilities.index(max(probabilities)) == logits.index(max(logits))
    assert sorted(range(4), key=lambda i: probabilities[i]) == sorted(
        range(4), key=lambda i: logits[i]
    )

    # A closed-form two-logit case pins the actual numbers (not just "sums to 1").
    expected = 1.0 / (1.0 + math.exp(-2.0))
    assert softmax([2.0, 0.0]) == pytest.approx([expected, 1.0 - expected])


def test_softmax_subtract_max_leaves_distribution_unchanged() -> None:
    """softmax is shift-invariant: adding any constant to every logit leaves pi unchanged.

    The max-subtraction used for numerical stability must be a no-op on the distribution. Shifting
    all logits by +/-500 (which would overflow a naive ``exp``) must still reproduce the small-logit
    answer, confirming the stabilization is mathematically inert.

    RL concept:
        Numerical stability of the softmax policy.

    Math:
        softmax(theta + c) = softmax(theta) for any constant c.
    """
    base = softmax([0.3, -0.7, 1.1, 0.0])
    for offset in (-500.0, 0.0, 500.0):
        shifted = softmax([0.3 + offset, -0.7 + offset, 1.1 + offset, 0.0 + offset])
        assert shifted == pytest.approx(base, abs=1e-9)


def test_softmax_rejects_empty_logits() -> None:
    """Guard clause: ``softmax([])`` raises ``ValueError`` (an empty action set is undefined).

    RL concept:
        Input validation for the policy distribution.
    """
    with pytest.raises(ValueError, match="non-empty"):
        softmax([])


def test_sample_action_is_in_range_and_deterministic_under_seed() -> None:
    """``_sample_action`` returns a valid action index and is reproducible for a fixed RNG seed.

    On-policy exploration draws A ~ pi_theta(.|s); the draw must stay within the action space and
    replay identically when the RNG is reseeded, so a whole training run is deterministic.

    RL concept:
        On-policy exploration by sampling the stochastic policy (vs epsilon-greedy).
    """
    logits = [0.2, 1.0, -0.5, 0.3]
    # Determinism: two equally-seeded RNGs must yield the *same whole stream*, not just one draw
    # (a single-sample check would miss a sequencing bug). One shared RNG drawn twice must instead
    # advance and (here) produce a non-constant stream, proving the draws actually consume the RNG.
    rng_a, rng_b = random.Random(0), random.Random(0)
    stream_a = [_sample_action(logits, rng_a) for _ in range(64)]
    stream_b = [_sample_action(logits, rng_b) for _ in range(64)]
    assert stream_a == stream_b
    # In-range and non-degenerate: drawing many actions from a *single* RNG stream (the real
    # on-policy usage) must stay in the action space AND explore more than one action under these
    # non-uniform logits -- guarding against a sampler stuck on a fixed index (e.g. always 0 or
    # the last action from a broken inverse-CDF). One shared RNG is used deliberately: re-seeding a
    # fresh Random(k) per draw correlates the first draw and is not how rollout consumes the RNG.
    sampler_rng = random.Random(5)
    stream = [_sample_action(logits, sampler_rng) for _ in range(200)]
    assert all(0 <= action < ACTION_COUNT for action in stream)
    assert len(set(stream)) > 1


def test_train_reinforce_rejects_non_positive_episodes() -> None:
    """Guard clause: ``episodes <= 0`` raises ``ValueError`` (matches the documented Raises).

    RL concept:
        Input validation for the training loop.
    """
    with pytest.raises(ValueError, match="positive"):
        train_reinforce(episodes=0)


def test_training_returns_curve_with_expected_schema() -> None:
    """Training returns a per-episode curve and learned logits matching the documented schema.

    Shape contract: the curve has one row per episode with keys
    ``episode, scenario_id, total_reward, baseline, steps``; episodes are numbered 1..N; scenario
    ids cycle through the catalog (0,1,2,3,4,0,...); each rollout's length is within the finite
    horizon (1..H+1 steps, since commit actions can terminate early); every learned logit row has
    one entry per action; and ``greedy_policy()`` yields a ``ReinforcePolicy``.

    RL concept:
        REINFORCE training output and greedy readout.
    """
    horizon = 5
    result = train_reinforce(episodes=10, seed=7, horizon=horizon)

    assert isinstance(result, ReinforceResult)
    assert len(result.training_curve) == 10
    assert result.theta
    first = result.training_curve[0]
    assert set(first) == {"episode", "scenario_id", "total_reward", "baseline", "steps"}
    # Episodes are numbered 1..N.
    assert [int(row["episode"]) for row in result.training_curve] == list(range(1, 11))
    # Scenario ids cycle through the five-scenario catalog, one per episode.
    assert [int(row["scenario_id"]) for row in result.training_curve[:7]] == [0, 1, 2, 3, 4, 0, 1]
    # Each rollout terminates within the finite horizon (commit actions can end it early, so the
    # length ranges over 1..H+1 rather than always running the full horizon).
    assert all(1 <= int(row["steps"]) <= horizon + 1 for row in result.training_curve)
    # Every learned logit row has one entry per action.
    assert all(len(logits) == ACTION_COUNT for logits in result.theta.values())
    assert isinstance(result.greedy_policy(), ReinforcePolicy)


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    """Two runs with identical arguments produce identical logits and curves (seeded determinism).

    Determinism is required by the teaching repo: the only randomness is the seeded action-sampling
    RNG and the seeded start-state jitter, so a fixed seed must reproduce the run exactly.

    RL concept:
        Reproducibility of an on-policy stochastic learner under a fixed seed.
    """
    first = train_reinforce(episodes=40, seed=7)
    second = train_reinforce(episodes=40, seed=7)
    assert first.theta == second.theta
    assert first.training_curve == second.training_curve


def test_greedy_policy_falls_back_to_action_zero_for_unseen_state() -> None:
    """An unseen state returns action 0 (``answer_direct``), the conservative do-nothing fallback.

    With empty ``theta`` every state is unseen, so the greedy readout returns action 0. This keeps
    the deployed policy total over the whole state space even though training only visited a subset.

    RL concept:
        Greedy-readout coverage of a tabular policy.
    """
    policy = ReinforcePolicy(theta={})
    unseen = AgentState(
        step=0, intent=4, difficulty=2, ambiguity=2, evidence=0, attempts=0, budget=30
    )
    assert policy.name == "reinforce"
    assert policy.select_action(unseen) == 0


def test_greedy_policy_returns_argmax_of_learned_logits() -> None:
    """For a seen state, the greedy readout picks the highest-logit action, not the fallback.

    With logits whose maximum is at action 2 (``clarify``), ``select_action`` must return 2 --
    confirming the action-0 fallback fires only for *unseen* states. This is the deterministic mode
    of pi_theta used at evaluation time.

    RL concept:
        Deterministic deployment (the mode) of a softmax policy.
    """
    state = AgentState(
        step=1, intent=2, difficulty=1, ambiguity=2, evidence=0, attempts=0, budget=27
    )
    policy = ReinforcePolicy(theta={state.as_tuple(): [0.0, 0.5, 2.0, 0.1]})
    assert policy.select_action(state) == 2


def test_positive_advantage_update_increases_chosen_action_probability() -> None:
    """A positive advantage raises the chosen action's probability and lowers every other.

    Applies one REINFORCE step by hand with advantage = +1: the chosen action's logit becomes
    positive and all others negative, so ``pi(chosen)`` rises and every other probability falls.
    This pins both the form of the score-function gradient and its sign for a rewarding action.

    RL concept:
        REINFORCE / score-function update direction.

    Math:
        theta_{s,a'} += alpha * (G_t - b) * (1[a'=A] - pi(a'|s)),  here (G_t - b) > 0.
    """
    theta = [0.0, 0.0, 0.0, 0.0]
    chosen, alpha, advantage = 2, 0.1, 1.0
    before = softmax(theta)
    probabilities = softmax(theta)
    for action in range(4):
        indicator = 1.0 if action == chosen else 0.0
        theta[action] += alpha * advantage * (indicator - probabilities[action])
    after = softmax(theta)

    assert theta[chosen] > 0.0
    assert all(theta[action] < 0.0 for action in range(4) if action != chosen)
    assert after[chosen] > before[chosen]
    assert all(after[action] < before[action] for action in range(4) if action != chosen)


def test_negative_advantage_update_decreases_chosen_action_probability() -> None:
    """A negative advantage suppresses the chosen action, fixing the sign of the baseline term.

    Mirror of the positive case with advantage = -1: ``pi(chosen)`` must drop below its uniform
    0.25 start. If the baseline were *added* instead of subtracted (a sign flip in ``G_t - b``),
    this would fail -- so the test guards that exact convention.

    RL concept:
        Sign of the advantage / baseline term.

    Math:
        theta_{s,a'} += alpha * (G_t - b) * (1[a'=A] - pi(a'|s)),  here (G_t - b) < 0.
    """
    theta = [0.0, 0.0, 0.0, 0.0]
    chosen, alpha, advantage = 2, 0.1, -1.0
    probabilities = softmax(theta)
    for action in range(4):
        indicator = 1.0 if action == chosen else 0.0
        theta[action] += alpha * advantage * (indicator - probabilities[action])

    assert softmax(theta)[chosen] < 0.25


def test_disabling_learning_leaves_theta_untouched() -> None:
    """With alpha=0 the gradient ascent is a no-op: every logit stays at its 0.0 init.

    States are still visited and registered in ``theta``, but with zero step size no update moves
    them off 0.0. This is the no-learning control that the improvement test contrasts against,
    isolating learning from environment drift.

    RL concept:
        Learning-rate ablation control.
    """
    result = train_reinforce(episodes=50, seed=7, alpha=0.0)
    assert result.theta  # states were still visited and registered
    assert all(all(logit == 0.0 for logit in logits) for logits in result.theta.values())


def _window_means(rewards: list[float]) -> tuple[float, float]:
    """Return the mean episode return over the first and last 20% of a training run.

    Smooths the noisy per-episode return into early and late averages, so an improvement claim
    compares stable windows rather than individual stochastic episodes.

    Args:
        rewards: Per-episode total rewards in training order.

    Returns:
        A ``(first_window_mean, last_window_mean)`` pair over the first/last 20% of episodes.
    """
    window = max(1, len(rewards) // 5)  # first/last 20% of episodes
    return sum(rewards[:window]) / window, sum(rewards[-window:]) / window


def test_policy_improves_over_training_and_beats_no_learning_control() -> None:
    """REINFORCE raises the episode return, and the rise is caused by the gradient.

    The headline learning claim, asserted causally on the on-policy training return. (a) The learned
    run's late-window mean return exceeds its early window. (b) That gain is attributable to the
    gradient rather than environment/seed drift: the learned run's final window beats an alpha=0
    control on the SAME seed, isolating the effect of learning. Margins are modest here because both
    ``answer_direct`` and ``escalate`` are one-step commits that return immediately, so we demand a
    clear-but-conservative positive margin at the canonical seed.

    RL concept:
        Policy improvement via REINFORCE, isolated with a learning-rate ablation control.
    """
    learned = train_reinforce(episodes=600, seed=7, alpha=0.2)
    control = train_reinforce(episodes=600, seed=7, alpha=0.0)

    learned_first, learned_last = _window_means(
        [float(row["total_reward"]) for row in learned.training_curve]
    )
    _, control_last = _window_means(
        [float(row["total_reward"]) for row in control.training_curve]
    )

    # (a) Improvement is real: the late window beats the early window (seed=7 yields ~+0.27).
    assert learned_last > learned_first
    # (b) It is attributable to learning: the trained final window beats the no-learning control on
    # the SAME seed (seed=7 yields ~+0.36), so the gain is the gradient, not drift.
    assert learned_last > control_last + 0.1


def _sampled_mean_return(
    theta: dict[tuple[int, int, int, int, int, int, int], list[float]],
    *,
    horizon: int = 5,
    eval_seeds: range = range(50),
    sample_seed: int = 12345,
) -> float:
    """Mean on-policy return of the *stochastic* softmax policy over held-out start states.

    What + why: REINFORCE optimizes the expected return of the sampled policy pi_theta, so the
    honest deployment metric samples A ~ pi_theta(.|s) (with the action-0 uniform fallback for
    unseen states) rather than reading off the greedy mode. The evaluation seeds jitter the start
    state and are scored across all scenarios; a fixed ``sample_seed`` keeps the rollout
    reproducible.

    Args:
        theta: Learned per-state logits (empty dict -> the untrained uniform policy everywhere).
        horizon: Episode length H for the evaluation environment.
        eval_seeds: Start-state jitter seeds to average over, per scenario.
        sample_seed: Seed for the action-sampling RNG, fixed for reproducibility.

    Returns:
        Mean total episode return over ``len(eval_seeds) * 5`` rollouts.

    RL concept:
        Held-out evaluation of the *stochastic* policy -- the objective REINFORCE actually climbs.
    """
    rng = random.Random(sample_seed)
    total, episodes = 0.0, 0
    for scenario_id in range(5):
        for eval_seed in eval_seeds:
            environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=default_reward)
            state = environment.reset(seed=10_000 + eval_seed, scenario_id=scenario_id)
            episode_return = 0.0
            while not environment.is_done():
                logits = theta.get(state.as_tuple(), [0.0] * ACTION_COUNT)
                transition = environment.step(_sample_action(logits, rng))
                episode_return += transition.reward
                state = transition.state
            total += episode_return
            episodes += 1
    return total / episodes


def test_trained_stochastic_policy_outperforms_untrained_on_held_out_seeds() -> None:
    """The trained stochastic policy collects more held-out return than the untrained uniform one.

    Deployment value, asserted on the quantity REINFORCE optimizes: over scenarios and start-state
    seeds never used in training, the *sampled* trained policy must collect more mean return than
    the untrained uniform policy (empty ``theta`` -> uniform pi everywhere, with the action-0
    fallback).
    The greedy readout is deliberately not used here: both ``answer_direct`` and ``escalate`` are
    one-step terminal commits with equal updates from a fresh start, so the deterministic tie-break
    pins the greedy start action to 0; the learning shows up in the *distribution* the gradient
    shapes, which the sampled return measures.

    RL concept:
        Held-out evaluation of a learned stochastic policy against the uniform baseline.
    """
    trained = train_reinforce(episodes=600, seed=7, alpha=0.2).theta
    trained_value = _sampled_mean_return(trained)
    untrained_value = _sampled_mean_return({})  # empty -> uniform policy everywhere

    # seed=7 yields trained ~ -0.69 vs untrained ~ -0.80 (a clear ~+0.10 gain); demand >0.04.
    assert trained_value > untrained_value + 0.04
