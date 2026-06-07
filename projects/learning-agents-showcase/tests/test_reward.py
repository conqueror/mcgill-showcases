"""Tests for the judge-rubric reward and its hackable twin.

These assert the load-bearing reward-design properties of the showcase: that ``judge_reward`` ranks
a well-grounded answer above an under-grounded one and above a needless escalation, that escalation
pays off only for genuinely hard/ambiguous requests, that the rubric breakdown sums to the scalar
reward, and -- crucially -- that ``hackable_reward`` is gameable: trivial always-escalate and
always-retrieve policies score higher under it than a genuinely good policy, the reverse of the
aligned reward.
"""

from __future__ import annotations

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    AgentState,
    RewardFunction,
)
from learning_agents.policies import (
    AlwaysEscalatePolicy,
    HeuristicRouterPolicy,
    Policy,
    RandomPolicy,
)
from learning_agents.reward import (
    GOOD_REWARD,
    HACKABLE_REWARD,
    hackable_reward,
    judge_reward,
    rubric_breakdown,
)


def _state(
    *, difficulty: int, ambiguity: int, evidence: int, step: int = 0, budget: int = 30
) -> AgentState:
    """Build an AgentState fixing the fields the reward depends on (others held neutral)."""
    return AgentState(
        step=step,
        intent=0,
        difficulty=difficulty,
        ambiguity=ambiguity,
        evidence=evidence,
        attempts=0,
        budget=budget,
    )


def test_well_grounded_answer_beats_under_grounded_answer() -> None:
    """A correctly grounded answer scores strictly above an under-grounded one (hallucination)."""
    grounded = _state(difficulty=1, ambiguity=0, evidence=1)
    under_evidence = _state(difficulty=1, ambiguity=0, evidence=0)
    under_ambiguity = _state(difficulty=0, ambiguity=1, evidence=0)
    good = judge_reward(grounded, 0, grounded, True)
    bad_evidence = judge_reward(under_evidence, 0, under_evidence, True)
    bad_ambiguity = judge_reward(under_ambiguity, 0, under_ambiguity, True)
    assert good > bad_evidence
    assert good > bad_ambiguity
    # The under-grounded answer is actively penalized (negative), not merely smaller.
    assert bad_evidence < 0
    assert bad_ambiguity < 0


def test_well_grounded_answer_beats_needless_escalation() -> None:
    """Answering an easy, grounded request beats escalating it (which wastes a human)."""
    easy = _state(difficulty=0, ambiguity=0, evidence=0)
    answer_reward = judge_reward(easy, 0, easy, True)
    escalate_next = _state(difficulty=0, ambiguity=0, evidence=0, step=1, budget=15)
    escalate_reward = judge_reward(easy, 3, escalate_next, True)
    assert answer_reward > escalate_reward
    assert escalate_reward < 0  # escalating an easy request is a net loss


def test_escalation_pays_off_only_when_warranted() -> None:
    """Escalating a hard/ambiguous request scores higher than escalating an easy one."""
    easy = _state(difficulty=0, ambiguity=0, evidence=0)
    hard = _state(difficulty=2, ambiguity=2, evidence=0)
    easy_next = _state(difficulty=0, ambiguity=0, evidence=0, step=1, budget=15)
    hard_next = _state(difficulty=2, ambiguity=2, evidence=0, step=1, budget=15)
    escalate_easy = judge_reward(easy, 3, easy_next, True)
    escalate_hard = judge_reward(hard, 3, hard_next, True)
    assert escalate_hard > escalate_easy
    # On the genuinely hard/ambiguous request, escalation is worth it (positive net reward).
    assert escalate_hard > 0


def test_needless_retrieval_is_penalized_but_useful_retrieval_is_not() -> None:
    """Retrieving when already grounded costs extra; retrieving when needed is only charged cost."""
    already_grounded = _state(difficulty=0, ambiguity=0, evidence=1)
    needs_evidence = _state(difficulty=2, ambiguity=0, evidence=0)
    grounded_next = _state(difficulty=0, ambiguity=0, evidence=2, step=1, budget=25)
    needed_next = _state(difficulty=2, ambiguity=0, evidence=1, step=1, budget=25)
    needless = judge_reward(already_grounded, 1, grounded_next, False)
    useful = judge_reward(needs_evidence, 1, needed_next, False)
    # Both are non-positive (retrieval costs), but the needless one is strictly worse.
    assert needless < useful


def test_rubric_breakdown_sums_to_judge_reward() -> None:
    """The per-criterion breakdown sums (to 4 dp) to the scalar judge reward for every action."""
    cases: list[tuple[AgentState, int, AgentState]] = [
        (
            _state(difficulty=1, ambiguity=0, evidence=1),
            0,
            _state(difficulty=1, ambiguity=0, evidence=1),
        ),
        (
            _state(difficulty=1, ambiguity=0, evidence=0),
            0,
            _state(difficulty=1, ambiguity=0, evidence=0),
        ),
        (
            _state(difficulty=2, ambiguity=2, evidence=0),
            3,
            _state(difficulty=2, ambiguity=2, evidence=0, step=1, budget=15),
        ),
        (
            _state(difficulty=2, ambiguity=0, evidence=0),
            1,
            _state(difficulty=2, ambiguity=0, evidence=1, step=1, budget=25),
        ),
        (
            _state(difficulty=0, ambiguity=1, evidence=0),
            2,
            _state(difficulty=0, ambiguity=0, evidence=0, step=1, budget=27),
        ),
    ]
    for prev, action, nxt in cases:
        breakdown = rubric_breakdown(prev, action, nxt, False)
        assert round(sum(breakdown.values()), 4) == judge_reward(prev, action, nxt, False)


def test_rubric_breakdown_exposes_expected_criteria() -> None:
    """The breakdown names every criterion and fires the right one per action."""
    expected_keys = {
        "answer_quality",
        "hallucination_penalty",
        "escalation_value",
        "effort_penalty",
        "action_cost",
    }
    grounded = _state(difficulty=0, ambiguity=0, evidence=0)
    answer = rubric_breakdown(grounded, 0, grounded, True)
    assert set(answer) == expected_keys
    assert answer["answer_quality"] > 0
    assert answer["hallucination_penalty"] == 0.0

    under = _state(difficulty=2, ambiguity=0, evidence=0)
    under_answer = rubric_breakdown(under, 0, under, True)
    assert under_answer["answer_quality"] == 0.0
    assert under_answer["hallucination_penalty"] < 0


def _episode_return(policy: Policy, reward_fn: RewardFunction, scenario_id: int) -> float:
    """Roll one policy through one scenario and return its total (undiscounted) reward."""
    env = AgentDecisionEnvironment(reward_fn=reward_fn)
    policy.reset()
    state = env.reset(scenario_id=scenario_id)
    total = 0.0
    while not env.is_done():
        action = policy.select_action(state)
        result = env.step(action)
        total += result.reward
        state = result.state
    return round(total, 4)


class _GoodPolicy:
    """A genuinely good policy: clarify, then ground, then answer (used only in reward tests)."""

    name = "good"

    def reset(self) -> None:
        return None

    def select_action(self, state: AgentState) -> int:
        from learning_agents.reward import evidence_is_adequate

        if state.ambiguity > 0:
            return 2
        if not evidence_is_adequate(evidence=state.evidence, difficulty=state.difficulty):
            return 1
        return 0


def _total_over_scenarios(policy: Policy, reward_fn: RewardFunction) -> float:
    """Sum a policy's per-scenario return across all five scenarios."""
    return round(sum(_episode_return(policy, reward_fn, sc) for sc in range(5)), 4)


def test_good_policy_wins_under_judge_reward() -> None:
    """Under the aligned reward, the good policy beats always-escalate and uniform-random."""
    good = _total_over_scenarios(_GoodPolicy(), judge_reward)
    escalate = _total_over_scenarios(AlwaysEscalatePolicy(), judge_reward)
    random_policy = _total_over_scenarios(RandomPolicy(seed=0), judge_reward)
    assert good > escalate
    assert good > random_policy


def test_hackable_reward_is_gamed_by_always_escalate() -> None:
    """Always-escalate outscores the good policy under the hackable reward (rank reversal)."""
    good_proxy = _total_over_scenarios(_GoodPolicy(), hackable_reward)
    escalate_proxy = _total_over_scenarios(AlwaysEscalatePolicy(), hackable_reward)
    assert escalate_proxy > good_proxy


class _AlwaysRetrievePolicy:
    """Always retrieve -- a degenerate policy that farms the hackable reward's evidence bonus."""

    name = "always_retrieve"

    def reset(self) -> None:
        return None

    def select_action(self, state: AgentState) -> int:
        del state
        return 1


def test_hackable_reward_is_gamed_by_always_retrieve() -> None:
    """Always-retrieve also beats the good policy under the hackable reward."""
    good_proxy = _total_over_scenarios(_GoodPolicy(), hackable_reward)
    retrieve_proxy = _total_over_scenarios(_AlwaysRetrievePolicy(), hackable_reward)
    assert retrieve_proxy > good_proxy


def test_hacking_policies_are_actually_bad_under_judge_reward() -> None:
    """The policies that game the proxy are genuinely worse under the aligned reward (the point)."""
    good_true = _total_over_scenarios(_GoodPolicy(), judge_reward)
    escalate_true = _total_over_scenarios(AlwaysEscalatePolicy(), judge_reward)
    retrieve_true = _total_over_scenarios(_AlwaysRetrievePolicy(), judge_reward)
    # Both gaming strategies underperform the good policy on the true objective.
    assert escalate_true < good_true
    assert retrieve_true < good_true
    # Always-retrieve never commits, so it bleeds cost and scores deeply negative on true reward.
    assert retrieve_true < 0


def test_reward_aliases_point_at_the_right_functions() -> None:
    """GOOD_REWARD / HACKABLE_REWARD alias the aligned and proxy rewards respectively."""
    assert GOOD_REWARD is judge_reward
    assert HACKABLE_REWARD is hackable_reward


def test_rewards_are_rounded_to_four_decimals() -> None:
    """Both rewards round to 4 decimals so artifact CSVs are stable."""
    state = _state(difficulty=2, ambiguity=2, evidence=0)
    nxt = _state(difficulty=2, ambiguity=2, evidence=0, step=1, budget=15)
    for reward_value in (judge_reward(state, 3, nxt, True), hackable_reward(state, 3, nxt, True)):
        assert reward_value == round(reward_value, 4)


def test_heuristic_router_beats_baselines_under_judge_reward() -> None:
    """Sanity anchor: the hand-written router also clears random and always-escalate (aligned)."""
    router = _total_over_scenarios(HeuristicRouterPolicy(), judge_reward)
    escalate = _total_over_scenarios(AlwaysEscalatePolicy(), judge_reward)
    random_policy = _total_over_scenarios(RandomPolicy(seed=1), judge_reward)
    assert router > escalate
    assert router > random_policy
    # ACTION_LABELS sanity so the import is load-bearing and the action set is the expected one.
    assert ACTION_LABELS[3] == "escalate"
