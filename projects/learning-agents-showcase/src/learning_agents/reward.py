"""Contrast a well-aligned judge rubric with a hackable one to expose reward hacking.

What + why: reward design is where an MDP's *objective* lives -- the agent optimizes whatever
scalar R_{t+1} we hand it, not what we meant. This module is the reward *source* for the
agent-decision MDP. The agent never sees ground-truth answer text; instead a *judge rubric* scores
its committed action against multiple criteria, exactly as an LLM-as-judge or a learned reward
model would in practice. :func:`judge_reward` is the aligned rubric: it rewards a correctly
grounded answer, penalizes an under-grounded one (hallucination risk), charges for needless
retrieval/clarification, and gives escalation a modest safe payoff minus its high cost.
:func:`hackable_reward` is a deliberately misspecified twin that *overpays* for escalation and for
raw evidence regardless of need, so a trivial "always escalate" or "always retrieve" policy scores
high -- the reward-hacking lesson, made measurable.

This sits upstream of every method on the ladder (bandit -> MDP -> Q-learning -> SARSA -> REINFORCE
-> DQN): all of them inherit the reward, so a bad reward silently corrupts every learner. The
notation follows the project convention: reward after acting is R_{t+1}, the return is
G_t = sum_k gamma^k R_{t+k+1}.

RL concept: reward design, proxy/true-objective mismatch, and reward hacking. A judge-rubric reward
is a stand-in for a learned reward model; comparing the aligned and hackable rubrics shows how a
plausible-looking objective can be gamed by a degenerate policy.

Math:
    judge_reward(s, a, s', done) sums per-criterion terms minus c(a); see :func:`rubric_breakdown`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import only for type checking to avoid a runtime import cycle: environment.py imports
    # judge_reward from this module, so this module must not import environment.py at load time.
    from learning_agents.environment import AgentState

# Action cost c(a): the canonical price the reward charges per action. answer_direct is free to
# *attempt* (its quality is judged below); retrieve/clarify cost tool/turn budget; escalate is the
# most expensive because it consumes a scarce human. environment.py re-exports this constant, so
# this module is the single source of truth and there is no environment->reward->environment cycle.
ACTION_COSTS: dict[int, float] = {
    0: 0.0,
    1: 0.5,
    2: 0.3,
    3: 1.5,
}


def evidence_is_adequate(*, evidence: int, difficulty: int) -> bool:
    """Decide whether the gathered evidence is enough to ground an answer at this difficulty.

    What + why: "well-grounded" is the central quality criterion of the judge rubric, and the
    reward, the environment, and the heuristic router all need one shared definition of it. The rule
    is monotone: harder requests demand strictly more evidence. A difficulty-0 request needs no
    retrieval, difficulty 1 needs one unit, difficulty 2 needs two. Centralizing it here keeps every
    consumer in agreement on what "adequate grounding" means.

    Args:
        evidence: Units of grounding gathered so far (0 or more).
        difficulty: Request difficulty (0 or more).

    Returns:
        True iff ``evidence`` meets the required threshold for ``difficulty``.

    RL concept: a shared definition of answer quality used by both the reward and the baseline
    policy, so "grounded enough to answer" means the same thing everywhere.
    """
    return evidence >= difficulty


def _is_well_grounded(state: AgentState) -> bool:
    """Return whether a state would yield a high-quality direct answer.

    What + why: an ``answer_direct`` is "good" only when two conditions hold together -- the
    evidence is adequate for the difficulty AND the ambiguity is fully resolved (0). This helper
    expresses that conjunction once so :func:`judge_reward` and :func:`rubric_breakdown` agree.

    Args:
        state: The state at which an answer would be committed (the post-action ``next_state`` for
            ``answer_direct``, whose situational fields equal the pre-action state's).

    Returns:
        True iff evidence is adequate for the difficulty and ambiguity is 0.

    RL concept: the answer-quality predicate at the core of the judge rubric.
    """
    return evidence_is_adequate(evidence=state.evidence, difficulty=state.difficulty) and (
        state.ambiguity == 0
    )


def rubric_breakdown(
    previous_state: AgentState,
    action: int,
    next_state: AgentState,
    done: bool,
) -> dict[str, float]:
    """Return the aligned judge reward decomposed into named per-criterion terms.

    What + why: a single scalar reward hides *why* a transition scored as it did. This breakdown
    exposes each criterion the aligned rubric weighs -- answer quality, the hallucination penalty
    for an under-grounded answer, the safe-escalation payoff, the cost of needless tool use, and the
    raw action cost -- so artifacts and tests can inspect the objective term by term. The terms sum
    (after rounding) to exactly :func:`judge_reward`, which is asserted in the test-suite.

    The criteria (for the four actions):
        * ``answer_quality``: +2.0 for a well-grounded ``answer_direct``, else 0.
        * ``hallucination_penalty``: -1.5 for an under-grounded ``answer_direct`` (wrong/unsafe
          answer risk), else 0.
        * ``escalation_value``: a modest safe payoff for ``escalate`` that *scales with genuine
          need* (difficulty + unresolved ambiguity), so escalating a hard/ambiguous request earns
          more than escalating an easy one.
        * ``effort_penalty``: -0.2 for each retrieve/clarify that was *not* needed (evidence already
          adequate, or ambiguity already 0), discouraging busywork.
        * ``action_cost``: -c(a), the raw price of the action from :data:`ACTION_COSTS`.

    Args:
        previous_state: The state s before acting.
        action: The action a taken.
        next_state: The resulting state s'.
        done: Whether this transition terminated the episode (unused by the aligned rubric's terms
            but kept for signature parity with :data:`RewardFunction`).

    Returns:
        A mapping from criterion name to its (rounded, 4 dp) signed contribution.

    RL concept: per-criterion reward decomposition -- making a multi-objective reward auditable so
    its trade-offs are legible rather than buried in one number.
    """
    del done  # the aligned rubric judges the committed action directly; termination adds no term.
    answer_quality = 0.0
    hallucination_penalty = 0.0
    escalation_value = 0.0
    effort_penalty = 0.0

    if action == 0:
        # answer_direct: reward a well-grounded answer, penalize an under-grounded one.
        if _is_well_grounded(next_state):
            answer_quality = 2.0
        else:
            hallucination_penalty = -1.5
    elif action == 3:
        # escalate: a safe hand-off whose payoff scales with genuine need (hard/ambiguous), so it
        # is only worth its high cost for requests that truly warrant a human.
        need = previous_state.difficulty + previous_state.ambiguity
        escalation_value = round(0.6 + 0.45 * need, 4)
    elif action in (1, 2):
        # retrieve/clarify: charge a small penalty only when the move was needless.
        if action == 1 and evidence_is_adequate(
            evidence=previous_state.evidence, difficulty=previous_state.difficulty
        ):
            effort_penalty = -0.2  # retrieving when already adequately grounded
        if action == 2 and previous_state.ambiguity == 0:
            effort_penalty = -0.2  # clarifying when nothing is ambiguous

    action_cost = -ACTION_COSTS[action]
    return {
        "answer_quality": round(answer_quality, 4),
        "hallucination_penalty": round(hallucination_penalty, 4),
        "escalation_value": round(escalation_value, 4),
        "effort_penalty": round(effort_penalty, 4),
        "action_cost": round(action_cost, 4),
    }


def judge_reward(
    previous_state: AgentState,
    action: int,
    next_state: AgentState,
    done: bool,
) -> float:
    """Score one transition with the aligned judge rubric (the MDP's true objective).

    What + why: this is the reward the environment uses by default. It is a multi-criterion judge
    rubric, not a hand-shaped control reward: it rewards a *correct* answer (``answer_direct`` with
    evidence adequate for the difficulty AND ambiguity resolved), penalizes an *under-grounded*
    answer (the hallucination/wrong-answer risk), charges a small penalty for over-retrieving or
    needless clarifying, and gives ``escalate`` a modest safe payoff that scales with genuine need
    minus its high cost -- so escalation is only worth it for genuinely hard/ambiguous requests.
    Every action cost in :data:`ACTION_COSTS` is subtracted. The result is the sum of the terms in
    :func:`rubric_breakdown`.

    By construction this makes "answer when adequately grounded" the best behaviour: a well-grounded
    answer scores +2.0 at zero action cost, strictly above an under-grounded answer (-1.5) and above
    escalating a request that did not need a human (a net cost once the +0.6.. payoff is offset by
    the -1.5 cost on easy requests). The companion :func:`hackable_reward` breaks exactly this
    property.

    Args:
        previous_state: The state s before acting.
        action: The action a taken (indexes :data:`ACTION_COSTS`).
        next_state: The resulting state s'.
        done: Whether this transition terminated the episode.

    Returns:
        The scalar reward R_{t+1}, rounded to 4 decimals.

    RL concept: reward design as objective specification -- a judge-rubric reward (LLM-as-judge /
    reward-model stand-in) whose criteria are chosen so the aligned optimum is the genuinely good
    policy.

    Math:
        R_{t+1} = answer_quality + hallucination_penalty + escalation_value + effort_penalty
                  + action_cost (each as in :func:`rubric_breakdown`), rounded to 4 dp.
    """
    breakdown = rubric_breakdown(previous_state, action, next_state, done)
    return round(sum(breakdown.values()), 4)


def hackable_reward(
    previous_state: AgentState,
    action: int,
    next_state: AgentState,
    done: bool,
) -> float:
    """Score one transition with a deliberately misspecified (hackable) proxy reward.

    What + why: this is the teaching counter-example -- a proxy reward that *looks* sensible but is
    exploitable. It keeps the same shape as :func:`judge_reward` (a quality term, an escalation
    term, an evidence term, minus cost) but introduces two classic reward-design mistakes:

        1. It *overpays for escalation*: ``escalate`` earns a flat large bonus (3.0) regardless of
           whether the request actually needed a human, and that bonus dwarfs the action cost. A
           trivial "always escalate" policy therefore scores high on this proxy while never solving
           anything itself.
        2. It *pays for raw evidence regardless of need*: every retrieve earns a positive bonus
           (0.8) with no penalty for redundancy, so an "always retrieve" policy farms reward by
           piling up grounding it does not need.

    Crucially it *under-credits the aligned good behaviour*: a well-grounded direct answer earns
    only a small +1.0, less than what a couple of needless retrievals or a single escalation pull
    in. The net effect is a rank reversal -- degenerate policies beat the genuinely good one --
    the diagnostic signature of reward hacking.

    Args:
        previous_state: The state s before acting.
        action: The action a taken (indexes :data:`ACTION_COSTS`).
        next_state: The resulting state s'.
        done: Whether this transition terminated the episode. Intentionally ignored here.

    Returns:
        The proxy scalar reward R_{t+1}, rounded to 4 decimals -- upward-biased for escalation and
        for raw evidence, and under-paying the well-grounded answer.

    RL concept: reward hacking via a misspecified proxy reward -- contrast with the aligned
    :func:`judge_reward`; a learner that maximizes this proxy degrades the true objective.

    Math:
        answer_direct: +1.0 if well-grounded else -0.2; retrieve: +0.8 (always); clarify: +0.1;
        escalate: +3.0; then minus c(a). Rounded to 4 dp.
    """
    del done  # discarding need/termination is the bug: escalation is paid unconditionally.
    if action == 0:
        # Under-credits the genuinely good behaviour so it loses to gaming strategies.
        quality = 1.0 if _is_well_grounded(next_state) else -0.2
        return round(quality - ACTION_COSTS[action], 4)
    if action == 1:
        # Pays for raw evidence regardless of need -> "always retrieve" farms reward.
        return round(0.8 - ACTION_COSTS[action], 4)
    if action == 2:
        return round(0.1 - ACTION_COSTS[action], 4)
    # escalate: flat overpayment that dwarfs the cost -> "always escalate" scores high.
    return round(3.0 - ACTION_COSTS[action], 4)


# GOOD_REWARD is the aligned judge rubric (the environment's default objective); HACKABLE_REWARD is
# the misspecified proxy used to demonstrate reward hacking.
GOOD_REWARD = judge_reward
HACKABLE_REWARD = hackable_reward
