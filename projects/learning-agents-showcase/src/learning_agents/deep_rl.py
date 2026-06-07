"""Vendored deep-RL lane: neural DQN and actor-critic PPO on the agent-decision MDP.

What + why: the tabular ladder (Q-learning, SARSA, dynamic programming, REINFORCE) keys a table on
the discrete state. Real agents face state spaces too large to tabulate, so they replace the table
with a *function approximator* -- a neural network that maps state features to values or action
probabilities. This module is the showcase's deep-RL rung, implemented from scratch in NumPy (no
torch) to stay self-contained, deterministic, and laptop-friendly, exactly like the rest of the
vendored RL library. It provides two families:

* ``train_dqn`` -- a value-based deep method (a small multilayer perceptron trained with the
  Q-learning Bellman target, an experience-replay buffer, and a periodically-synced target network).
  This is the neural counterpart of the tabular ``q_learning`` rung.
* ``train_ppo`` -- a policy-gradient / actor-critic method (a policy network and a value-baseline
  critic, optimized with PPO's clipped surrogate objective). This is the neural counterpart of the
  tabular ``policy_gradient`` (REINFORCE) rung, plus a critic.

The teaching point is the comparison: on this *small, deterministic* MDP the exact optimum is known
(dynamic programming gives ``Q*``), so we can check that function approximation **recovers the
tabular ceiling** -- validating the method on a problem where the right answer is known before
trusting it where it is not. Both models expose the SB3-style ``predict`` surface so they slot into
:class:`~learning_agents.policies.ModelPolicy` and are scored by the same
:func:`~learning_agents.evaluation.evaluate_policies` harness as every other policy.

RL concept: deep reinforcement learning -- value-function and policy approximation with neural
networks; DQN (value-based) vs PPO (actor-critic, policy-gradient).

Math:
    DQN fits Q_theta(s, a) toward the Bellman target y = r + gamma * max_a' Q_target(s', a').
    PPO maximizes the clipped surrogate L = E[min(rho * A, clip(rho, 1-eps, 1+eps) * A)] where
    rho = pi_new(a|s) / pi_old(a|s) and A is the advantage (return minus the critic baseline).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    AgentState,
    RewardFunction,
    default_reward,
    scenario_catalog,
)
from learning_agents.evaluation import evaluate_policies
from learning_agents.policies import ModelPolicy, Policy

Array = NDArray[np.float64]

# Action/feature dimensions are derived from the live environment so the network shape follows the
# MDP rather than hard-coded constants drifting out of sync.
NUM_ACTIONS = len(ACTION_LABELS)
FEATURE_DIM = 7  # length of AgentState.as_normalized_vector
DEFAULT_GAMMA = 0.9  # matches the tabular q_learning / offline_rl discount


def state_features(state: AgentState, horizon: int) -> Array:
    """Encode a state as the network input feature vector.

    What + why: deep methods consume features, not table keys. This wraps
    :meth:`AgentState.as_normalized_vector` (which scales each field to ``[0, 1]``) as a float array
    so it can be fed to the MLP. Keeping the encoding in one place means DQN, PPO, and the
    ``ModelPolicy`` wrapper all see identical inputs.

    Args:
        state: The current MDP state.
        horizon: Episode length H, used to scale the time-like fields.

    Returns:
        A ``(FEATURE_DIM,)`` float array in ``[0, 1]``.

    RL concept: feature encoding for value/policy-function approximation.
    """
    features: Array = np.asarray(state.as_normalized_vector(horizon=horizon), dtype=np.float64)
    return features


def _relu(x: Array) -> Array:
    """Apply the elementwise ReLU activation ``max(0, x)``."""
    return np.maximum(0.0, x)


def _softmax(logits: Array) -> Array:
    """Convert a row of logits to a probability distribution (numerically stable softmax).

    Args:
        logits: A ``(num_actions,)`` array of unnormalized log-probabilities.

    Returns:
        A ``(num_actions,)`` array of probabilities summing to 1.

    RL concept: the policy head pi(a|s) = softmax(logits)_a of a policy-gradient agent.
    """
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    distribution: Array = exp / np.sum(exp)
    return distribution


@dataclass
class MLP:
    """A one-hidden-layer multilayer perceptron with manual forward/backward (NumPy only).

    What + why: this is the vendored function approximator shared by DQN (value head) and PPO
    (policy and critic heads). One hidden layer with ReLU is more than enough capacity for this
    small MDP and keeps the gradients easy to read -- the point is to *show* function approximation,
    not to scale it. Forward caches its activations so :meth:`backward` can compute exact analytic
    gradients for an arbitrary upstream loss-gradient ``d_output``.

    Attributes:
        w1, b1: First layer weights ``(in_dim, hidden)`` and bias ``(hidden,)``.
        w2, b2: Output layer weights ``(hidden, out_dim)`` and bias ``(out_dim,)``.

    RL concept: parametric function approximation Q_theta / pi_theta / V_theta.
    """

    w1: Array
    b1: Array
    w2: Array
    b2: Array

    @classmethod
    def initialize(cls, rng: np.random.Generator, in_dim: int, hidden: int, out_dim: int) -> MLP:
        """Create an MLP with He-initialized weights and zero biases.

        Args:
            rng: Seeded NumPy generator so initialization is reproducible.
            in_dim: Input feature dimension.
            hidden: Hidden-layer width.
            out_dim: Output dimension (action count for value/policy heads, 1 for the critic).

        Returns:
            A freshly initialized :class:`MLP`.

        RL concept: weight initialization for stable gradient-based training.
        """
        w1 = rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)
        w2 = rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden)
        return cls(w1=w1, b1=np.zeros(hidden), w2=w2, b2=np.zeros(out_dim))

    def copy(self) -> MLP:
        """Return a deep copy of this network (used to snapshot a DQN target network)."""
        return MLP(w1=self.w1.copy(), b1=self.b1.copy(), w2=self.w2.copy(), b2=self.b2.copy())

    def forward(self, x: Array) -> tuple[Array, tuple[Array, Array, Array]]:
        """Compute the output for a batch of inputs and cache activations for backprop.

        Args:
            x: A ``(batch, in_dim)`` input batch.

        Returns:
            ``(output, cache)`` where ``output`` is ``(batch, out_dim)`` and ``cache`` holds the
            tensors :meth:`backward` needs.

        Math:
            ``h = relu(x W1 + b1)``; ``y = h W2 + b2`` (linear output head).
        """
        pre = x @ self.w1 + self.b1
        hidden = _relu(pre)
        output: Array = hidden @ self.w2 + self.b2
        return output, (x, pre, hidden)

    def backward(
        self,
        cache: tuple[Array, Array, Array],
        d_output: Array,
        learning_rate: float,
        max_grad_norm: float = 5.0,
    ) -> None:
        """Backpropagate ``d_output`` and apply one clipped SGD update in place.

        What + why: given the gradient of the loss with respect to the network output, this computes
        the parameter gradients by the chain rule and takes a gradient-descent step. Global-norm
        clipping keeps the update stable when advantages or TD errors are momentarily large.

        Args:
            cache: The activations returned by :meth:`forward`.
            d_output: ``(batch, out_dim)`` gradient of the loss w.r.t. the network output.
            learning_rate: SGD step size.
            max_grad_norm: Global gradient-norm clip threshold.

        RL concept: gradient-based optimization of a function approximator.
        """
        x, pre, hidden = cache
        d_w2 = hidden.T @ d_output
        d_b2 = d_output.sum(axis=0)
        d_hidden = d_output @ self.w2.T
        d_pre = d_hidden * (pre > 0.0)
        d_w1 = x.T @ d_pre
        d_b1 = d_pre.sum(axis=0)

        grads = [d_w1, d_b1, d_w2, d_b2]
        total_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if total_norm > max_grad_norm and total_norm > 0.0:
            scale = max_grad_norm / total_norm
        else:
            scale = 1.0
        self.w1 -= learning_rate * scale * d_w1
        self.b1 -= learning_rate * scale * d_b1
        self.w2 -= learning_rate * scale * d_w2
        self.b2 -= learning_rate * scale * d_b2


# --------------------------------------------------------------------------------------------------
# DQN: value-based deep RL (neural Q-learning with replay + target network)
# --------------------------------------------------------------------------------------------------


@dataclass
class DQNModel:
    """A trained deep Q-network exposing the SB3-style ``predict`` surface.

    What + why: wraps the value network so it satisfies
    :class:`~learning_agents.policies.PredictModel` and can be adapted by
    :class:`~learning_agents.policies.ModelPolicy`. Greedy action selection is ``argmax_a Q(s, a)``.

    Attributes:
        network: The trained value MLP mapping features to per-action Q-values.
        horizon: Episode length used to encode states (kept so ``predict`` is self-contained).

    RL concept: a value-based deep policy pi(s) = argmax_a Q_theta(s, a).
    """

    network: MLP
    horizon: int

    def action_values(self, state: AgentState) -> Array:
        """Return the per-action Q-values Q(state, .) for one state."""
        features = state_features(state, self.horizon)[None, :]
        values, _ = self.network.forward(features)
        row_values: Array = values[0]
        return row_values

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """Map a feature vector to a greedy action (SB3 ``model.predict`` convention).

        Args:
            observation: The state feature vector (network input).
            deterministic: Ignored; a value-based greedy policy is already deterministic.

        Returns:
            ``(action, None)`` -- the greedy action index and an unused recurrent-state slot.
        """
        del deterministic
        features = np.asarray(observation, dtype=np.float64)[None, :]
        values, _ = self.network.forward(features)
        return int(np.argmax(values[0])), None


@dataclass
class _Transition:
    """One stored experience-replay tuple ``(s, a, r, s', done)`` in feature space."""

    features: Array
    action: int
    reward: float
    next_features: Array
    done: bool


def train_dqn(
    *,
    episodes: int = 400,
    hidden: int = 32,
    learning_rate: float = 0.05,
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = 0.2,
    batch_size: int = 64,
    target_sync: int = 25,
    horizon: int = 5,
    seed: int = 0,
    reward_fn: RewardFunction = default_reward,
    eval_every: int = 25,
) -> tuple[DQNModel, list[dict[str, int | float | str]]]:
    """Train a deep Q-network on the agent-decision MDP and return it with a training curve.

    What + why: the neural counterpart of tabular Q-learning. The agent collects experience with an
    epsilon-greedy behaviour policy, stores transitions in a replay buffer, and repeatedly fits the
    value network toward the Bellman target ``r + gamma * max_a' Q_target(s', a')`` computed from a
    periodically-synced *target* network (both standard DQN stabilizers). On this small MDP it
    converges to roughly the same greedy policy the tabular methods and dynamic programming find.

    Args:
        episodes: Number of training episodes (rollouts) to collect and learn from.
        hidden: Hidden-layer width of the value network.
        learning_rate: SGD step size for value-network updates.
        gamma: Discount factor.
        epsilon: Exploration rate of the epsilon-greedy behaviour policy.
        batch_size: Replay minibatch size per gradient step.
        target_sync: Sync the target network to the online network every this many episodes.
        horizon: Episode length H.
        seed: RNG seed for reproducibility (initialization, exploration, replay sampling).
        reward_fn: Reward function injected into the environment.
        eval_every: Record a training-curve point (greedy mean reward / escalation) this often.

    Returns:
        ``(model, curve)`` -- the trained :class:`DQNModel` and a list of
        ``{step, mean_reward, mean_escalation_rate}`` rows tracking greedy performance.

    RL concept: deep Q-learning -- value-function approximation with experience replay and a target
    network.

    Math:
        Minimize ``(Q_theta(s, a) - y)^2`` with
        ``y = r + gamma * (1 - done) * max_a' Q_bar(s', a')``.
    """
    rng = np.random.default_rng(seed)
    online = MLP.initialize(rng, FEATURE_DIM, hidden, NUM_ACTIONS)
    target = online.copy()
    scenarios = scenario_catalog()
    buffer: list[_Transition] = []
    curve: list[dict[str, int | float | str]] = []

    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=reward_fn)
        state = environment.reset(seed=seed + episode, scenario_id=scenario.scenario_id)
        done = False
        while not done:
            features = state_features(state, horizon)
            if rng.random() < epsilon:
                action = int(rng.integers(NUM_ACTIONS))
            else:
                values, _ = online.forward(features[None, :])
                action = int(np.argmax(values[0]))
            result = environment.step(action)
            buffer.append(
                _Transition(
                    features=features,
                    action=action,
                    reward=result.reward,
                    next_features=state_features(result.state, horizon),
                    done=result.done,
                )
            )
            state = result.state
            done = result.done

            if len(buffer) >= batch_size:
                _dqn_gradient_step(online, target, buffer, rng, batch_size, gamma, learning_rate)

        if episode % target_sync == 0:
            target = online.copy()
        if episode % eval_every == 0 or episode == episodes - 1:
            mean_reward, mean_esc = _greedy_rollout_stats(
                DQNModel(network=online, horizon=horizon), horizon=horizon, reward_fn=reward_fn
            )
            curve.append(
                {
                    "step": episode,
                    "mean_reward": round(mean_reward, 4),
                    "mean_escalation_rate": round(mean_esc, 4),
                }
            )

    return DQNModel(network=online, horizon=horizon), curve


def _dqn_gradient_step(
    online: MLP,
    target: MLP,
    buffer: list[_Transition],
    rng: np.random.Generator,
    batch_size: int,
    gamma: float,
    learning_rate: float,
) -> None:
    """Take one replay-minibatch DQN gradient step on the online network (in place)."""
    indices = rng.integers(0, len(buffer), size=batch_size)
    batch = [buffer[i] for i in indices]
    features = np.stack([t.features for t in batch])
    next_features = np.stack([t.next_features for t in batch])
    actions = np.array([t.action for t in batch])
    rewards = np.array([t.reward for t in batch], dtype=np.float64)
    not_done = np.array([0.0 if t.done else 1.0 for t in batch], dtype=np.float64)

    next_values, _ = target.forward(next_features)
    targets = rewards + gamma * not_done * np.max(next_values, axis=1)

    predicted, cache = online.forward(features)
    # MSE only on the taken action: d/dQ (Q[a]-y)^2 = 2(Q[a]-y), zero for other actions.
    d_output = np.zeros_like(predicted)
    row = np.arange(batch_size)
    d_output[row, actions] = (2.0 / batch_size) * (predicted[row, actions] - targets)
    online.backward(cache, d_output, learning_rate)


# --------------------------------------------------------------------------------------------------
# PPO: actor-critic policy-gradient with a clipped surrogate objective
# --------------------------------------------------------------------------------------------------


@dataclass
class PPOModel:
    """A trained PPO actor (policy network) exposing the SB3-style ``predict`` surface.

    What + why: holds the policy network whose softmax gives pi(a|s). ``predict`` takes the mode
    (argmax) action when deterministic and samples otherwise, matching the ``PredictModel`` contract
    that :class:`~learning_agents.policies.ModelPolicy` consumes. The critic is only needed during
    training (as the advantage baseline), so it is not stored on the deployed policy.

    Attributes:
        policy: The trained policy MLP mapping features to action logits.
        horizon: Episode length used to encode states.
        seed: Seed for the sampling RNG used when ``deterministic=False``.

    RL concept: a policy-gradient deep policy pi(a|s) = softmax(f_theta(s))_a.
    """

    policy: MLP
    horizon: int
    seed: int = 0

    def __post_init__(self) -> None:
        """Initialize the (only-used-when-sampling) RNG for stochastic action selection."""
        self._rng = np.random.default_rng(self.seed)

    def action_probabilities(self, state: AgentState) -> Array:
        """Return the policy distribution pi(.|state) for one state."""
        features = state_features(state, self.horizon)[None, :]
        logits, _ = self.policy.forward(features)
        return _softmax(logits[0])

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """Map a feature vector to an action (mode if deterministic, else a sample).

        Args:
            observation: The state feature vector (network input).
            deterministic: If True, return ``argmax_a pi(a|s)``; if False, sample from pi.

        Returns:
            ``(action, None)`` per the SB3 ``predict`` convention.
        """
        features = np.asarray(observation, dtype=np.float64)[None, :]
        logits, _ = self.policy.forward(features)
        probabilities = _softmax(logits[0])
        if deterministic:
            return int(np.argmax(probabilities)), None
        return int(self._rng.choice(NUM_ACTIONS, p=probabilities)), None


def train_ppo(
    *,
    iterations: int = 60,
    episodes_per_iteration: int = 20,
    hidden: int = 32,
    policy_lr: float = 0.05,
    value_lr: float = 0.05,
    gamma: float = DEFAULT_GAMMA,
    clip_epsilon: float = 0.2,
    epochs: int = 4,
    entropy_coef: float = 0.01,
    horizon: int = 5,
    seed: int = 0,
    reward_fn: RewardFunction = default_reward,
    eval_every: int = 5,
) -> tuple[PPOModel, list[dict[str, int | float | str]]]:
    """Train an actor-critic PPO agent on the agent-decision MDP and return it with a curve.

    What + why: the policy-gradient deep method. Each iteration collects on-policy trajectories by
    sampling actions from the current policy, computes discounted returns and advantages against a
    learned value baseline (the critic), then improves the policy with PPO's clipped surrogate over
    several epochs so each update stays close to the behaviour policy. The critic is regressed
    toward the returns. This is the clipped, variance-reduced neural descendant of REINFORCE.

    Args:
        iterations: Number of collect-then-improve PPO iterations.
        episodes_per_iteration: On-policy episodes collected per iteration.
        hidden: Hidden width of both the policy and value networks.
        policy_lr: Step size for the policy (actor) network.
        value_lr: Step size for the value (critic) network.
        gamma: Discount factor.
        clip_epsilon: PPO clip range for the probability ratio.
        epochs: Optimization epochs over each batch of collected data.
        entropy_coef: Weight on the entropy bonus (encourages exploration).
        horizon: Episode length H.
        seed: RNG seed for reproducibility.
        reward_fn: Reward function injected into the environment.
        eval_every: Record a training-curve point this often (in iterations).

    Returns:
        ``(model, curve)`` -- the trained :class:`PPOModel` and a list of
        ``{step, mean_reward, mean_escalation_rate}`` rows over training.

    RL concept: PPO -- a clipped-surrogate actor-critic policy-gradient method.

    Math:
        rho = pi_new(a|s) / pi_old(a|s); maximize E[min(rho A, clip(rho, 1-eps, 1+eps) A)]
        + c_ent * H(pi); critic minimizes E[(V_theta(s) - G)^2]; A = G - V(s) (normalized).
    """
    rng = np.random.default_rng(seed)
    policy = MLP.initialize(rng, FEATURE_DIM, hidden, NUM_ACTIONS)
    critic = MLP.initialize(rng, FEATURE_DIM, hidden, 1)
    scenarios = scenario_catalog()
    curve: list[dict[str, int | float | str]] = []

    for iteration in range(iterations):
        feats, actions, returns = _collect_ppo_batch(
            policy,
            scenarios,
            rng,
            episodes_per_iteration,
            gamma,
            horizon,
            reward_fn,
            seed,
            iteration,
        )
        values, _ = critic.forward(feats)
        baseline = values[:, 0]
        advantages = returns - baseline
        std = float(np.std(advantages))
        advantages = advantages / (std + 1e-8) if std > 1e-8 else advantages - np.mean(advantages)

        old_logits, _ = policy.forward(feats)
        old_logp = _log_prob(old_logits, actions)

        for _ in range(epochs):
            _ppo_policy_update(
                policy, feats, actions, advantages, old_logp, clip_epsilon, entropy_coef, policy_lr
            )
            _critic_update(critic, feats, returns, value_lr)

        if iteration % eval_every == 0 or iteration == iterations - 1:
            mean_reward, mean_esc = _greedy_rollout_stats(
                PPOModel(policy=policy, horizon=horizon, seed=seed),
                horizon=horizon,
                reward_fn=reward_fn,
            )
            curve.append(
                {
                    "step": iteration,
                    "mean_reward": round(mean_reward, 4),
                    "mean_escalation_rate": round(mean_esc, 4),
                }
            )

    return PPOModel(policy=policy, horizon=horizon, seed=seed), curve


def _collect_ppo_batch(
    policy: MLP,
    scenarios: Sequence[object],
    rng: np.random.Generator,
    episodes: int,
    gamma: float,
    horizon: int,
    reward_fn: RewardFunction,
    seed: int,
    iteration: int,
) -> tuple[Array, NDArray[np.int_], Array]:
    """Roll out on-policy episodes and return (features, actions, discounted returns).

    What + why: PPO is on-policy, so each iteration regenerates fresh trajectories by sampling
    actions from the current policy. We accumulate per-step features and actions and compute the
    Monte-Carlo discounted return-to-go for each step (the critic turns these into advantages).

    Returns:
        ``(features, actions, returns)`` stacked over every step of every collected episode.
    """
    all_features: list[Array] = []
    all_actions: list[int] = []
    all_returns: list[float] = []
    num_scenarios = len(scenarios)

    for episode in range(episodes):
        scenario_id = episode % num_scenarios
        environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=reward_fn)
        # Vary the start jitter per iteration+episode while staying reproducible (no Math.random).
        reset_seed = seed + iteration * episodes + episode
        state = environment.reset(seed=reset_seed, scenario_id=scenario_id)
        episode_features: list[Array] = []
        episode_actions: list[int] = []
        episode_rewards: list[float] = []
        done = False
        while not done:
            features = state_features(state, horizon)
            logits, _ = policy.forward(features[None, :])
            probabilities = _softmax(logits[0])
            action = int(rng.choice(NUM_ACTIONS, p=probabilities))
            result = environment.step(action)
            episode_features.append(features)
            episode_actions.append(action)
            episode_rewards.append(result.reward)
            state = result.state
            done = result.done

        running = 0.0
        returns_to_go = [0.0] * len(episode_rewards)
        for t in range(len(episode_rewards) - 1, -1, -1):
            running = episode_rewards[t] + gamma * running
            returns_to_go[t] = running
        all_features.extend(episode_features)
        all_actions.extend(episode_actions)
        all_returns.extend(returns_to_go)

    return (
        np.stack(all_features),
        np.asarray(all_actions, dtype=np.int_),
        np.asarray(all_returns, dtype=np.float64),
    )


def _log_prob(logits: Array, actions: NDArray[np.int_]) -> Array:
    """Return log pi(a|s) for each row's taken action under softmax logits."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum = np.log(np.sum(np.exp(shifted), axis=1))
    row = np.arange(logits.shape[0])
    log_probs: Array = shifted[row, actions] - log_sum
    return log_probs


def _ppo_policy_update(
    policy: MLP,
    feats: Array,
    actions: NDArray[np.int_],
    advantages: Array,
    old_logp: Array,
    clip_epsilon: float,
    entropy_coef: float,
    learning_rate: float,
) -> None:
    """Take one PPO clipped-surrogate gradient step on the policy network (in place).

    Math:
        For the clipped surrogate, the policy gradient flows through ``rho * A`` only when
        that branch is the binding minimum; otherwise (clipped) it is zero. The entropy bonus adds
        ``c_ent * dH/dlogits`` with ``H = -sum_a p_a log p_a``.
    """
    logits, cache = policy.forward(feats)
    batch = logits.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

    new_logp = _log_prob(logits, actions)
    ratio = np.exp(new_logp - old_logp)
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surr1 = ratio * advantages
    surr2 = clipped * advantages
    # Gradient of -min(surr1, surr2) flows through surr1 only where it is the binding branch.
    use_unclipped = (surr1 <= surr2).astype(np.float64)

    # d(new_logp)/d(logits) = onehot(action) - probabilities.
    onehot = np.zeros_like(logits)
    row = np.arange(batch)
    onehot[row, actions] = 1.0
    d_logp = onehot - probabilities

    # d(-objective_policy)/d(logits): chain through ratio = exp(new_logp - old_logp).
    coef = -(use_unclipped * ratio * advantages) / batch
    d_logits = coef[:, None] * d_logp

    # Entropy bonus: maximize H -> minimize -c_ent * H. dH/dlogits = -p * (log p + H).
    log_probabilities = np.log(probabilities + 1e-12)
    entropy = -np.sum(probabilities * log_probabilities, axis=1, keepdims=True)
    d_entropy = -probabilities * (log_probabilities + entropy)
    d_logits += (-entropy_coef * d_entropy) / batch

    policy.backward(cache, d_logits, learning_rate)


def _critic_update(critic: MLP, feats: Array, returns: Array, learning_rate: float) -> None:
    """Take one MSE gradient step regressing the critic toward the observed returns (in place)."""
    values, cache = critic.forward(feats)
    batch = values.shape[0]
    d_output = (2.0 / batch) * (values - returns[:, None])
    critic.backward(cache, d_output, learning_rate)


# --------------------------------------------------------------------------------------------------
# Shared evaluation helpers
# --------------------------------------------------------------------------------------------------


def _greedy_rollout_stats(
    model: DQNModel | PPOModel,
    *,
    horizon: int,
    reward_fn: RewardFunction,
) -> tuple[float, float]:
    """Score a model's greedy policy across all scenarios; return (mean_reward, mean_escalation).

    Used to build training curves without duplicating the evaluation harness.
    """
    policy = build_model_policy(model, name="_curve", horizon=horizon)
    # Average over a few jittered starts so the curve tracks the same robust signal as the final
    # comparison (a single un-jittered start can over-report a policy that is brittle to jitter).
    summary, _ = evaluate_policies(
        policies=[policy],
        scenario_ids=[scenario.scenario_id for scenario in scenario_catalog()],
        episodes_per_scenario=3,
        horizon=horizon,
        reward_fn=reward_fn,
    )
    row = summary[0]
    return float(row["avg_reward"]), float(row["avg_escalation_rate"])


def build_model_policy(
    model: DQNModel | PPOModel,
    *,
    name: str,
    horizon: int,
    deterministic: bool = True,
) -> Policy:
    """Wrap a trained DQN/PPO model as a :class:`~learning_agents.policies.Policy`.

    What + why: adapts a deep model to the same interface the tabular baselines use, so the standard
    evaluation harness can score it. The observation encoder is bound here so callers never have to
    re-derive the feature mapping.

    Args:
        model: A trained :class:`DQNModel` or :class:`PPOModel`.
        name: Policy label shown in evaluation tables (e.g. ``"dqn"`` or ``"ppo"``).
        horizon: Episode length used to encode states.
        deterministic: Whether to act greedily (True) or sample (False, PPO only).

    Returns:
        A :class:`~learning_agents.policies.ModelPolicy` satisfying the ``Policy`` protocol.

    RL concept: bridging a deep model into the shared policy-evaluation harness.
    """
    def encode(state: AgentState) -> list[float]:
        return [float(value) for value in state_features(state, horizon)]

    return ModelPolicy(
        model=model,
        observation_fn=encode,
        name=name,
        deterministic=deterministic,
    )


@dataclass(frozen=True)
class FamilyEntry:
    """One labelled policy in the deep-RL family comparison (policy + its RL family name)."""

    policy: Policy
    family: str


def family_comparison_rows(
    entries: Sequence[FamilyEntry],
    *,
    horizon: int = 5,
    episodes_per_scenario: int = 1,
    reward_fn: RewardFunction = default_reward,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    """Score each labelled policy and return (family-comparison rows, per-scenario rollup rows).

    What + why: produces the data for ``artifacts/drl_optional/rl_family_comparison.csv`` (one row
    per policy with its RL family and headline metrics) and
    ``artifacts/drl_optional/scenario_rollups.csv`` (per-policy, per-scenario return), both from
    the shared :func:`~learning_agents.evaluation.evaluate_policies` harness so the deep methods are
    judged exactly like the tabular ones.

    Args:
        entries: The labelled policies to compare (e.g. q_learning, dqn, ppo).
        horizon: Episode length H.
        episodes_per_scenario: Seeded rollouts per (policy, scenario) pair.
        reward_fn: Reward function injected into the environment.

    Returns:
        ``(comparison_rows, rollup_rows)`` matching the optional DRL artifact schemas.

    RL concept: head-to-head comparison of value-based vs policy-gradient deep RL vs the tabular
    baseline.
    """
    family_by_policy = {entry.policy.name: entry.family for entry in entries}
    summary, scenario_rows = evaluate_policies(
        policies=[entry.policy for entry in entries],
        scenario_ids=[scenario.scenario_id for scenario in scenario_catalog()],
        episodes_per_scenario=episodes_per_scenario,
        horizon=horizon,
        reward_fn=reward_fn,
    )
    comparison_rows: list[dict[str, int | float | str]] = [
        {
            "policy": str(row["policy"]),
            "family": family_by_policy[str(row["policy"])],
            "avg_reward": row["avg_reward"],
            "avg_escalation_rate": row["avg_escalation_rate"],
            "solved_rate": row["solved_rate"],
        }
        for row in summary
    ]
    rollup_rows: list[dict[str, int | float | str]] = [
        {
            "policy": str(row["policy"]),
            "scenario_id": row["scenario_id"],
            "scenario_name": row["scenario_name"],
            "total_reward": row["total_reward"],
        }
        for row in scenario_rows
    ]
    return comparison_rows, rollup_rows
