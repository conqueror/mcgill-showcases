"""Multi-agent RL on a cooperative coordination game (locus of learning C).

What + why: the third place learning can live in an agent system is the **coordination between
multiple agents** -- a researcher and a responder, a planner and a solver, a team of specialists. RL
that learns *how agents coordinate* is multi-agent RL (MARL), and its defining difficulty is that
when every agent learns at once the environment each one faces is **non-stationary**: the others are
changing too. This module makes that concrete and measurable on the smallest faithful testbed.

The game is the classic cooperative **Climbing game** (Claus & Boutilier, 1998), re-skinned to the
agent domain: a *researcher* picks an effort level and a *responder* picks an answer depth, and they
share a single team reward that depends on the *joint* action. The best outcome (thorough research +
detailed answer) pays the most, but the actions next to it are catastrophically penalised -- so an
agent that cannot count on its partner choosing the matching action is punished for reaching toward
the optimum. That payoff shape is exactly what trips up naive multi-agent learning.

Two learners are contrasted:

* **Independent Q-learning (IQL)**: each agent keeps its own action-value table and treats the other
  as part of the environment. It is simple and decentralised, but it suffers *relative
  overgeneralisation* -- each agent averages a risky action's value over the partner's exploration,
  so the high-value-but-risky optimum looks bad and both retreat to a safe, suboptimal equilibrium.
* **Joint-action learning (JAL)**: a single learner over the *joint* action space -- the simplest
  form of **centralised training**. With the full joint view it reliably finds the optimum, at the
  cost of an action space that grows multiplicatively with the number of agents (which is why real
  systems use CTDE methods like QMIX/MAPPO instead of a literal joint table).

The contrast is the lesson: on the same game IQL almost never coordinates to the optimum while JAL
almost always does -- a crisp, reproducible demonstration of why naive independence fails and why
centralised information helps.

RL concept:
    Locus of learning C -- multi-agent coordination. Non-stationarity and relative
    overgeneralisation in independent learners versus centralised (joint-action) training; the
    foundation under CTDE methods (QMIX, MAPPO, COMA).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = [
    "CLIMBING_GAME",
    "CooperativeMatrixGame",
    "MarlResult",
    "coordination_success_rate",
    "marl_comparison_rows",
    "optimal_joint_action",
    "optimal_team_reward",
    "team_reward",
    "train_independent_q_learning",
    "train_joint_action_learner",
]


@dataclass(frozen=True)
class CooperativeMatrixGame:
    """A two-agent, single-shot cooperative game: a shared payoff for each joint action.

    What + why: the minimal MARL environment -- two agents act simultaneously and receive the *same*
    team reward, read from a payoff matrix indexed by their joint action. Cooperative (shared
    reward) isolates the *coordination* problem from competition: the only difficulty is agreeing on
    a joint
    action, which is precisely what independent learners struggle with.

    Attributes:
        name: Human-readable game name.
        payoffs: Row-major team-reward matrix; ``payoffs[a0][a1]`` is the shared reward when agent 0
            plays ``a0`` and agent 1 plays ``a1``.
        agent0_actions: Action labels for agent 0 (the researcher).
        agent1_actions: Action labels for agent 1 (the responder).

    RL concept: a cooperative matrix (stateless) game -- the canonical testbed for coordination.
    """

    name: str
    payoffs: tuple[tuple[int, ...], ...]
    agent0_actions: tuple[str, ...]
    agent1_actions: tuple[str, ...]

    @property
    def num_agent0_actions(self) -> int:
        """Number of actions available to agent 0."""
        return len(self.payoffs)

    @property
    def num_agent1_actions(self) -> int:
        """Number of actions available to agent 1."""
        return len(self.payoffs[0])


# The Climbing game (Claus & Boutilier, 1998), re-skinned: agent 0 = researcher effort
# (deep_research > search > skim), agent 1 = responder depth (detailed > standard > brief). The
# optimum (deep_research, detailed) = 11 pays best, but its neighbours (deep+standard,
# search+detailed) = -30 are catastrophic, so independent learners retreat to (skim, brief) = 5.
CLIMBING_GAME = CooperativeMatrixGame(
    name="researcher_responder_climbing",
    payoffs=(
        (11, -30, 0),
        (-30, 7, 6),
        (0, 0, 5),
    ),
    agent0_actions=("deep_research", "search", "skim"),
    agent1_actions=("detailed", "standard", "brief"),
)


@dataclass(frozen=True)
class MarlResult:
    """The outcome of a MARL training run: the learned joint action and its convergence curve.

    Attributes:
        method: Which learner produced this (``"independent"`` or ``"joint"``).
        final_joint_action: The greedy joint action ``(a0, a1)`` after training.
        final_team_reward: The team reward of that greedy joint action.
        reached_optimum: Whether the greedy joint action is a team-reward maximiser.
        training_curve: Checkpoints ``[{"episode", "greedy_team_reward"}]`` -- the team reward of
            the current greedy joint action sampled across training (does it climb to the optimum?).

    RL concept: a MARL run's learned coordination and how it converged (or failed to).
    """

    method: str
    final_joint_action: tuple[int, int]
    final_team_reward: float
    reached_optimum: bool
    training_curve: list[dict[str, int | float]]


def team_reward(game: CooperativeMatrixGame, agent0_action: int, agent1_action: int) -> float:
    """Return the shared team reward for a joint action.

    Args:
        game: The cooperative game.
        agent0_action: Agent 0's action index.
        agent1_action: Agent 1's action index.

    Returns:
        The shared reward ``payoffs[agent0_action][agent1_action]`` as a float.

    RL concept: the common (cooperative) reward both agents jointly maximise.
    """
    return float(game.payoffs[agent0_action][agent1_action])


def optimal_joint_action(game: CooperativeMatrixGame) -> tuple[int, int]:
    """Return the team-reward-maximising joint action (ties broken by lowest index).

    Args:
        game: The cooperative game.

    Returns:
        The ``(a0, a1)`` joint action with the highest team reward.

    RL concept: the cooperative optimum -- the coordination target both learners aim for.
    """
    joint_actions = (
        (a0, a1)
        for a0 in range(game.num_agent0_actions)
        for a1 in range(game.num_agent1_actions)
    )
    return max(joint_actions, key=lambda joint: game.payoffs[joint[0]][joint[1]])


def optimal_team_reward(game: CooperativeMatrixGame) -> float:
    """Return the maximum achievable team reward (the value of the optimal joint action)."""
    a0, a1 = optimal_joint_action(game)
    return team_reward(game, a0, a1)


def _argmax(values: list[float]) -> int:
    """Return the index of the maximum value, breaking ties by lowest index."""
    best = max(values)
    return next(index for index, value in enumerate(values) if value == best)


def train_independent_q_learning(
    game: CooperativeMatrixGame,
    *,
    episodes: int = 4000,
    alpha: float = 0.1,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.999,
    epsilon_min: float = 0.02,
    seed: int = 7,
    checkpoints: int = 50,
) -> MarlResult:
    """Train two independent Q-learners that each treat the partner as part of the environment.

    What + why: the decentralised baseline. Each agent keeps its own action-value vector over its
    own actions, both act epsilon-greedily, both receive the shared reward, and each updates only
    its own table. Because each agent's estimate of an action averages over whatever the *partner*
    happened to do (including exploration), the risky-but-optimal action's value is dragged down --
    relative overgeneralisation -- and the pair typically converges to a safe, suboptimal joint
    action. This is the concrete face of non-stationarity in MARL.

    Args:
        game: The cooperative game to learn.
        episodes: Number of simultaneous-action rounds.
        alpha: TD learning rate for each agent's update.
        epsilon: Initial exploration rate (shared schedule for both agents).
        epsilon_decay: Multiplicative decay applied to epsilon each round.
        epsilon_min: Floor for epsilon.
        seed: RNG seed for reproducibility.
        checkpoints: Number of evenly spaced training-curve samples.

    Returns:
        A :class:`MarlResult` with the greedy joint action both agents settle on and the convergence
        curve.

    RL concept: independent Q-learning -- decentralised MARL beset by non-stationarity and relative
    overgeneralisation.
    """
    rng = random.Random(seed)
    num0, num1 = game.num_agent0_actions, game.num_agent1_actions
    q0 = [0.0] * num0
    q1 = [0.0] * num1
    current_epsilon = epsilon
    curve: list[dict[str, int | float]] = []
    checkpoint_every = max(1, episodes // checkpoints)
    for episode in range(1, episodes + 1):
        a0 = rng.randrange(num0) if rng.random() < current_epsilon else _argmax(q0)
        a1 = rng.randrange(num1) if rng.random() < current_epsilon else _argmax(q1)
        reward = team_reward(game, a0, a1)
        # Each agent updates ONLY its own table from the shared reward (independent learning).
        q0[a0] += alpha * (reward - q0[a0])
        q1[a1] += alpha * (reward - q1[a1])
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
        if episode % checkpoint_every == 0:
            greedy = team_reward(game, _argmax(q0), _argmax(q1))
            curve.append({"episode": episode, "greedy_team_reward": greedy})
    final_joint = (_argmax(q0), _argmax(q1))
    final_reward = team_reward(game, *final_joint)
    return MarlResult(
        method="independent",
        final_joint_action=final_joint,
        final_team_reward=final_reward,
        reached_optimum=final_reward == optimal_team_reward(game),
        training_curve=curve,
    )


def train_joint_action_learner(
    game: CooperativeMatrixGame,
    *,
    episodes: int = 4000,
    alpha: float = 0.1,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.999,
    epsilon_min: float = 0.02,
    seed: int = 7,
    checkpoints: int = 50,
) -> MarlResult:
    """Train a single Q-learner over the *joint* action space (centralised training).

    What + why: the centralised contrast. One learner holds an action-value for every *joint* action
    ``(a0, a1)`` and acts epsilon-greedily over the whole grid, so it sees the true value of the
    risky optimum directly instead of averaging it away. With the full joint view it reliably
    converges to the cooperative optimum -- the upside of centralised training. The catch (not a bug
    here, the point) is that the joint table grows multiplicatively with the number of agents, which
    is why scalable methods use centralised training with decentralised execution (CTDE) rather than
    a literal joint policy.

    Args:
        game: The cooperative game to learn.
        episodes: Number of rounds.
        alpha: TD learning rate.
        epsilon: Initial exploration rate over joint actions.
        epsilon_decay: Multiplicative decay applied to epsilon each round.
        epsilon_min: Floor for epsilon.
        seed: RNG seed for reproducibility.
        checkpoints: Number of evenly spaced training-curve samples.

    Returns:
        A :class:`MarlResult` with the greedy joint action and the convergence curve.

    RL concept: joint-action learning -- the simplest centralised MARL, optimal but exponential in
    the number of agents (the motivation for CTDE).
    """
    rng = random.Random(seed)
    num0, num1 = game.num_agent0_actions, game.num_agent1_actions
    joint_q = [[0.0] * num1 for _ in range(num0)]
    current_epsilon = epsilon
    curve: list[dict[str, int | float]] = []
    checkpoint_every = max(1, episodes // checkpoints)

    def greedy_joint() -> tuple[int, int]:
        """Return the current greedy joint action over the joint Q-table."""
        return max(
            ((a0, a1) for a0 in range(num0) for a1 in range(num1)),
            key=lambda joint: joint_q[joint[0]][joint[1]],
        )

    for episode in range(1, episodes + 1):
        if rng.random() < current_epsilon:
            a0, a1 = rng.randrange(num0), rng.randrange(num1)
        else:
            a0, a1 = greedy_joint()
        reward = team_reward(game, a0, a1)
        joint_q[a0][a1] += alpha * (reward - joint_q[a0][a1])
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
        if episode % checkpoint_every == 0:
            g0, g1 = greedy_joint()
            curve.append({"episode": episode, "greedy_team_reward": team_reward(game, g0, g1)})
    final_joint = greedy_joint()
    final_reward = team_reward(game, *final_joint)
    return MarlResult(
        method="joint",
        final_joint_action=final_joint,
        final_team_reward=final_reward,
        reached_optimum=final_reward == optimal_team_reward(game),
        training_curve=curve,
    )


def coordination_success_rate(
    game: CooperativeMatrixGame,
    *,
    method: str,
    seeds: int = 30,
    episodes: int = 4000,
) -> float:
    """Estimate how often a learner reaches the cooperative optimum across seeds.

    What + why: a single run is noisy, so the headline metric is the *fraction* of seeds on which a
    learner converges to the optimal joint action. This quantifies the coordination gap directly:
    independent learning scores low (it usually miscoordinates) while joint-action learning scores
    near 1.0.

    Args:
        game: The cooperative game.
        method: ``"independent"`` or ``"joint"``.
        seeds: Number of seeds (0..seeds-1) to average over.
        episodes: Training length per run.

    Returns:
        The fraction of seeds whose trained greedy joint action is a team-reward maximiser.

    Raises:
        ValueError: If ``method`` is not ``"independent"`` or ``"joint"``.

    RL concept: the coordination success rate -- a robust measure of the independent-vs-centralised
    gap.
    """
    if method == "independent":
        trainer = train_independent_q_learning
    elif method == "joint":
        trainer = train_joint_action_learner
    else:
        raise ValueError("method must be 'independent' or 'joint'")
    reached = sum(
        1 for seed in range(seeds) if trainer(game, episodes=episodes, seed=seed).reached_optimum
    )
    return round(reached / seeds, 4)


def marl_comparison_rows(
    game: CooperativeMatrixGame = CLIMBING_GAME,
    *,
    seeds: int = 30,
    episodes: int = 4000,
) -> list[dict[str, int | float | str]]:
    """Build the independent-vs-joint coordination comparison rows (the headline MARL artifact).

    What + why: runs both learners across many seeds on the cooperative game and reports, per
    method, the coordination success rate, a representative final joint action, and its team reward
    versus the optimum. The gap between the two rows is the lesson: independence miscoordinates,
    centralised training does not.

    Args:
        game: The cooperative game (defaults to :data:`CLIMBING_GAME`).
        seeds: Seeds to average the success rate over.
        episodes: Training length per run.

    Returns:
        One row per method with ``method``, ``coordination_success_rate``, ``final_joint_action``,
        ``final_team_reward``, and ``optimal_team_reward``.

    RL concept: a like-for-like comparison of decentralised vs centralised multi-agent learning.
    """
    optimum = optimal_team_reward(game)
    rows: list[dict[str, int | float | str]] = []
    for method, trainer in (
        ("independent", train_independent_q_learning),
        ("joint", train_joint_action_learner),
    ):
        result = trainer(game, episodes=episodes, seed=0)
        a0, a1 = result.final_joint_action
        rows.append(
            {
                "method": method,
                "coordination_success_rate": coordination_success_rate(
                    game, method=method, seeds=seeds, episodes=episodes
                ),
                "final_joint_action": f"{game.agent0_actions[a0]}+{game.agent1_actions[a1]}",
                "final_team_reward": result.final_team_reward,
                "optimal_team_reward": optimum,
            }
        )
    return rows
