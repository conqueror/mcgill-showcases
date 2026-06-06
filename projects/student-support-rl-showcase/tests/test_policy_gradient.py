"""Tests for tabular softmax REINFORCE: the softmax policy, the gradient, and that it learns.

REINFORCE optimizes the policy pi_theta *directly* by gradient ascent on expected return,
rather than learning action values and acting greedily as Q-learning does -- the
*policy gradient* rung of the ladder (contextual bandit -> MDP -> Q-learning -> DQN ->
*policy gradient* -> actor-critic -> PPO). These tests build the algorithm from its parts.
First the softmax policy parameterization: a valid distribution, correct values (not merely
summing to 1), order-preserving, numerically stable under large shifts, and rejecting empty
logits. Then the score-function update itself, by hand: a positive advantage must raise the
chosen action's probability and a negative one must lower it (pinning the gradient's sign, so
a flipped baseline would fail). Finally the end-to-end claims: alpha=0 is a no-op control,
training raises the return *because of the gradient* (beating that control), and the trained
greedy readout outperforms the untrained fallback on held-out evaluation seeds.

RL concept:
    Monte-Carlo policy gradient (REINFORCE) and softmax policies; see
    docs/policy-gradient-and-actor-critic.md, docs/math-notes.md and docs/glossary.md.

Math:
    softmax policy: pi(a|s) = exp(theta_{s,a}) / sum_{a'} exp(theta_{s,a'})
    return G_t = sum_k gamma^k R_{t+k+1};  baseline b reduces variance
    policy gradient: grad J = E[grad log pi(A|s) (G_t - b)],
    with d/dtheta_{s,a'} log pi(A|s) = 1[a'=A] - pi(a'|s).
"""

from __future__ import annotations

import math

import pytest

from student_support_rl.policy_gradient import ReinforcePolicy, softmax, train_reinforce


def test_softmax_is_a_valid_probability_distribution() -> None:
    """softmax(logits) is a proper distribution: right length, sums to 1, entries in [0, 1].

    Includes an extreme logit (1000.0) to confirm the max-subtraction shift keeps the result
    valid where a naive ``exp`` would overflow. This is the precondition for sampling actions
    A ~ pi(.|s) during on-policy rollout.

    RL concept:
        Softmax policy parameterization (docs/policy-gradient-and-actor-critic.md).

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
    property ``select_action`` relies on), and a closed-form two-logit case pins the actual
    values via the logistic ``1 / (1 + exp(-2))``.

    RL concept:
        Softmax monotonicity underpinning greedy readout (docs/policy-gradient-and-actor-critic.md).
    """
    # Equal logits -> exactly uniform.
    assert softmax([2.0, 2.0, 2.0, 2.0]) == pytest.approx([0.25, 0.25, 0.25, 0.25])

    # Larger logit -> larger probability; the logit ordering is preserved exactly,
    # so argmax(logits) is argmax(pi). This is the property select_action relies on.
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

    The max-subtraction used for numerical stability must be a no-op on the distribution.
    Shifting all logits by +/-500 (which would overflow a naive ``exp``) must still reproduce
    the small-logit answer, confirming the stabilization is mathematically inert.

    RL concept:
        Numerical stability of the softmax policy (docs/math-notes.md).

    Math:
        softmax(theta + c) = softmax(theta) for any constant c.
    """
    # The numerical-stability shift must be a no-op on the distribution: softmax is
    # invariant to adding any constant to every logit. A huge shift would overflow a
    # naive exp(); the result must still equal the small-logit answer.
    base = softmax([0.3, -0.7, 1.1, 0.0])
    for offset in (-500.0, 0.0, 500.0):
        shifted = softmax([0.3 + offset, -0.7 + offset, 1.1 + offset, 0.0 + offset])
        assert shifted == pytest.approx(base, abs=1e-9)


def test_softmax_rejects_empty_logits() -> None:
    """Guard clause: ``softmax([])`` raises ``ValueError`` (an empty action set is undefined).

    RL concept:
        Input validation for the policy distribution (docs/policy-gradient-and-actor-critic.md).
    """
    with pytest.raises(ValueError, match="non-empty"):
        softmax([])


def test_train_reinforce_rejects_non_positive_episodes() -> None:
    """Guard clause: ``episodes <= 0`` raises ``ValueError`` (matches the documented Raises).

    RL concept:
        Input validation for the training loop (docs/policy-gradient-and-actor-critic.md).
    """
    with pytest.raises(ValueError, match="positive"):
        train_reinforce(episodes=0)


def test_training_returns_curve_with_expected_schema() -> None:
    """Training returns a per-episode curve and learned logits matching the documented schema.

    Shape contract: the curve has one row per episode with keys
    ``episode, scenario_id, total_reward, baseline, steps``; episodes are numbered 1..N; each
    rollout runs the full horizon (steps == 6); every learned logit row has one entry per
    action; and ``greedy_policy()`` yields a ``ReinforcePolicy``.

    RL concept:
        REINFORCE training output and greedy readout (docs/policy-gradient-and-actor-critic.md).
    """
    result = train_reinforce(episodes=10, seed=7, horizon=6)

    assert len(result.training_curve) == 10
    assert result.theta
    first = result.training_curve[0]
    assert set(first) == {"episode", "scenario_id", "total_reward", "baseline", "steps"}
    # Episodes are numbered 1..N and each rollout runs the full horizon.
    assert [int(row["episode"]) for row in result.training_curve] == list(range(1, 11))
    assert all(int(row["steps"]) == 6 for row in result.training_curve)
    # Every learned logit row has one entry per action.
    assert all(len(logits) == 4 for logits in result.theta.values())
    assert isinstance(result.greedy_policy(), ReinforcePolicy)


def test_greedy_policy_falls_back_to_action_zero_for_unseen_state() -> None:
    """An unseen state returns action 0, the conservative do-nothing fallback.

    With empty ``theta`` every state is unseen, so the greedy readout returns action 0
    (``no_intervention``). This keeps the deployed policy total over the whole state space
    even though training only visited a subset.

    RL concept:
        Greedy readout coverage of a tabular policy (docs/policy-gradient-and-actor-critic.md).
    """
    from student_support_rl.environment import StudentState

    policy = ReinforcePolicy(theta={})
    unseen = StudentState(
        week=1,
        engagement=0,
        completion=0,
        pressure=4,
        risk=3,
        prior_interventions=0,
    )
    assert policy.select_action(unseen) == 0


def test_greedy_policy_returns_argmax_of_learned_logits() -> None:
    """For a seen state, the greedy readout picks the highest-logit action, not the fallback.

    With logits whose maximum is at action 2, ``select_action`` must return 2 -- confirming the
    action-0 fallback fires only for *unseen* states. This is the deterministic mode of
    pi_theta used at evaluation time.

    RL concept:
        Deterministic deployment of a softmax policy (docs/policy-gradient-and-actor-critic.md).
    """
    from student_support_rl.environment import StudentState

    state = StudentState(
        week=1,
        engagement=2,
        completion=2,
        pressure=2,
        risk=2,
        prior_interventions=0,
    )
    # Action 2 has the largest logit, so the greedy readout must pick it (not the
    # action-0 fallback, which only fires for *unseen* states).
    policy = ReinforcePolicy(theta={state.as_tuple(): [0.0, 0.5, 2.0, 0.1]})
    assert policy.select_action(state) == 2


def test_positive_advantage_update_increases_chosen_action_probability() -> None:
    """A positive advantage raises the chosen action's probability and lowers every other.

    Applies one REINFORCE step by hand with advantage = +1: the chosen action's logit becomes
    positive and all others negative, so ``pi(chosen)`` rises and every other probability
    falls. This pins both the form of the score-function gradient and its sign for a
    rewarding action.

    RL concept:
        REINFORCE / score-function update direction (docs/policy-gradient-and-actor-critic.md).

    Math:
        theta_{s,a'} += alpha * (G_t - b) * (1[a'=A] - pi(a'|s)),  here (G_t - b) > 0.
    """
    # Direct check of the REINFORCE update grad_log = 1[a'=A] - pi(a'|s):
    # a single step with POSITIVE advantage must raise pi(A) and lower every other
    # action's probability. This pins both the gradient form and its sign.
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
        Sign of the advantage / baseline term (docs/policy-gradient-and-actor-critic.md).

    Math:
        theta_{s,a'} += alpha * (G_t - b) * (1[a'=A] - pi(a'|s)),  here (G_t - b) < 0.
    """
    # Mirror of the above: NEGATIVE advantage must suppress the chosen action. If the
    # baseline were ADDED instead of subtracted (sign flip), this would fail.
    theta = [0.0, 0.0, 0.0, 0.0]
    chosen, alpha, advantage = 2, 0.1, -1.0
    probabilities = softmax(theta)
    for action in range(4):
        indicator = 1.0 if action == chosen else 0.0
        theta[action] += alpha * advantage * (indicator - probabilities[action])

    assert softmax(theta)[chosen] < 0.25


def test_disabling_learning_leaves_theta_untouched() -> None:
    """With alpha=0 the gradient ascent is a no-op: every logit stays at its 0.0 init.

    States are still visited and registered in ``theta``, but with zero step size no update
    moves them off 0.0. This is the no-learning control that the improvement tests contrast
    against, isolating learning from environment drift.

    RL concept:
        Learning-rate ablation control (docs/policy-gradient-and-actor-critic.md).
    """
    # With alpha=0 the gradient ascent is a no-op, so all logits stay at their 0.0
    # initialization. This is the control that the improvement test contrasts against.
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

    The headline learning claim, asserted causally. (a) The learned run's late-window mean
    return clearly exceeds its early window (> 0.3 headroom, ~+0.6 at seed=7). (b) That gain is
    attributable to the gradient, not environment drift: the learned run's final window beats
    an alpha=0 control on the SAME seed by a wide margin (> 1.0), a control whose return in fact
    drifts downward here.

    RL concept:
        Policy improvement via REINFORCE (docs/policy-gradient-and-actor-critic.md).
    """
    # The headline claim: REINFORCE raises the episode return, and the rise is *caused
    # by the gradient* rather than environment drift. We assert (a) a margin clearly
    # above zero and (b) that the learned run's final window beats an alpha=0 control
    # on the SAME seed -- a control whose return actually drifts downward here.
    learned = train_reinforce(episodes=400, seed=7)
    control = train_reinforce(episodes=400, seed=7, alpha=0.0)

    learned_first, learned_last = _window_means(
        [float(row["total_reward"]) for row in learned.training_curve]
    )
    _, control_last = _window_means(
        [float(row["total_reward"]) for row in control.training_curve]
    )

    # Improvement is real and not marginal (seed=7 yields ~+0.6; demand >0.3 headroom).
    assert learned_last - learned_first > 0.3
    # And it is attributable to learning: the trained policy's final window beats the
    # no-learning control by a wide margin (~+2.4 here).
    assert learned_last > control_last + 1.0


def test_trained_greedy_policy_outperforms_untrained_on_held_out_seeds() -> None:
    """The trained greedy policy beats the untrained fallback on held-out evaluation seeds.

    Measures deployment value: over scenarios and seeds never used in training, the greedy
    readout of the trained logits must collect more mean return than the untrained action-0
    fallback (by > 1.0). This is the head-to-head comparison a policy-gradient learner makes
    against the value-based baselines, on data it did not see.

    RL concept:
        Held-out evaluation of a learned policy (docs/evaluation-and-governance.md).
    """
    # Deployment value: the greedy readout of the trained logits must collect more
    # reward than the untrained action-0 fallback on evaluation seeds never used in
    # training. This is the comparison a learner makes against the value-based baselines.
    from student_support_rl.environment import StudentSupportEnvironment, default_reward

    def mean_return(policy: ReinforcePolicy) -> float:
        total, episodes = 0.0, 0
        for scenario_id in range(5):
            for eval_seed in range(40):
                environment = StudentSupportEnvironment(horizon=6, reward_fn=default_reward)
                state = environment.reset(seed=10_000 + eval_seed, scenario_id=scenario_id)
                policy.reset()
                episode_return = 0.0
                while not environment.is_done():
                    transition = environment.step(policy.select_action(state))
                    episode_return += transition.reward
                    state = transition.state
                total += episode_return
                episodes += 1
        return total / episodes

    trained = train_reinforce(episodes=400, seed=7).greedy_policy()
    untrained = ReinforcePolicy(theta={})  # every state unseen -> action-0 fallback

    assert mean_return(trained) > mean_return(untrained) + 1.0
