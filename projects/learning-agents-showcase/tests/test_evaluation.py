"""Pin that simulator-based policy evaluation reports governance metrics, not just reward.

These tests fix the evaluation-and-governance discipline that sits beside the algorithm ladder:
comparing agent policies on fixed request scenarios must expose more than average return. They
pin that both the per-policy summary and the per-scenario breakdown carry cost and safety columns
(action cost, escalation count, unsafe-or-questionable decisions, solved), that the aligned
judge rubric ranks the good heuristic router above the random and always-escalate baselines, and
-- crucially -- that swapping in the *hackable* reward reverses that ranking (a degenerate
always-escalate policy wins on the proxy while its poor ``solved_rate`` exposes the hack). They
also pin determinism, the summary ranking/aggregation contract, and the single-episode trace.

Note: :func:`evaluate_policies` re-simulates each policy inside the known
``AgentDecisionEnvironment``; it is NOT off-policy evaluation (OPE) from a fixed log of
trajectories collected by another behaviour policy.

RL concept:
    Simulator-based policy evaluation and multi-objective governance metrics -- task success and
    cost/safety reported alongside the scalar return so reward hacking stays visible.
"""

from __future__ import annotations

from learning_agents.environment import AgentDecisionEnvironment, scenario_catalog
from learning_agents.evaluation import evaluate_policies, simulate_episode
from learning_agents.policies import (
    AlwaysEscalatePolicy,
    HeuristicRouterPolicy,
    QTablePolicy,
    RandomPolicy,
)
from learning_agents.reward import hackable_reward, judge_reward

ALL_SCENARIOS = tuple(range(len(scenario_catalog())))


def _summary_by_name(summary_rows: list[dict[str, int | float | str]]) -> dict[str, float]:
    """Map each policy name to its avg_reward for convenient assertions."""
    return {str(row["policy"]): float(row["avg_reward"]) for row in summary_rows}


def test_evaluation_reports_multi_objective_rows() -> None:
    """Summary and per-scenario rows both expose cost and safety metrics, not just reward.

    Pins the evaluation contract: across the random, heuristic, and always-escalate policies, the
    summary rows must include average action cost, escalation count, unsafe-or-questionable
    decisions, and solved_rate, and the per-scenario rows must carry the un-averaged safety/cost
    fields plus the action trace. Pinning these columns guarantees reward hacking is observable --
    a high-reward policy that over-escalates or answers unsafely is flagged rather than hidden.
    """
    summary_rows, scenario_rows = evaluate_policies(
        policies=[RandomPolicy(seed=5), HeuristicRouterPolicy(), AlwaysEscalatePolicy()],
        scenario_ids=ALL_SCENARIOS,
        episodes_per_scenario=2,
    )

    # Summary granularity: per-policy averages must surface cost and safety, not just reward,
    # including the contract's avg_escalation_rate and avg_undergrounded_rate governance gates.
    assert {
        "avg_reward",
        "avg_action_cost",
        "avg_escalation_rate",
        "avg_undergrounded_rate",
        "avg_unsafe_or_questionable_decisions",
        "solved_rate",
    } <= set(summary_rows[0])
    # Per-scenario granularity: the raw safety/cost fields stay inspectable per episode.
    assert {
        "scenario_name",
        "unsafe_or_questionable_decisions",
        "action_cost",
        "escalation_count",
        "undergrounded_answer",
        "final_action_label",
        "solved",
        "actions",
    } <= set(scenario_rows[0])
    # One row per (policy, scenario, episode) and one summary row per policy.
    assert len(scenario_rows) == 3 * len(ALL_SCENARIOS) * 2
    assert len(summary_rows) == 3


def test_summary_is_sorted_by_avg_reward_descending() -> None:
    """The leaderboard orders policies by estimated value (highest mean return first)."""
    summary_rows, _ = evaluate_policies(
        policies=[RandomPolicy(seed=1), HeuristicRouterPolicy(), AlwaysEscalatePolicy()],
        scenario_ids=ALL_SCENARIOS,
    )
    rewards = [float(row["avg_reward"]) for row in summary_rows]
    assert rewards == sorted(rewards, reverse=True)


def test_heuristic_router_beats_random_under_aligned_reward() -> None:
    """Under the aligned judge rubric the hand-written router scores above uniform-random.

    This is the property the aligned reward is *designed* to have: a sensible policy that grounds,
    disambiguates, and answers must beat a policy that acts at random. It is the baseline a learned
    agent must also clear, validated here on the reference (non-learned) router.
    """
    summary_rows, _ = evaluate_policies(
        policies=[RandomPolicy(seed=11), HeuristicRouterPolicy()],
        scenario_ids=ALL_SCENARIOS,
        reward_fn=judge_reward,
    )
    rewards = _summary_by_name(summary_rows)
    assert rewards["heuristic_router"] > rewards["random"]
    # The leaderboard's top row is the heuristic router under the aligned objective.
    assert summary_rows[0]["policy"] == "heuristic_router"


def test_heuristic_router_solves_more_than_baselines() -> None:
    """The good router resolves more requests than the random and always-escalate baselines.

    ``solved`` is independent of the scalar reward, so a higher ``solved_rate`` for the router
    confirms it genuinely handles requests (well-grounded answers / needed escalations) rather
    than merely scoring well -- the outcome side of the governance report.
    """
    summary_rows, _ = evaluate_policies(
        policies=[RandomPolicy(seed=3), HeuristicRouterPolicy(), AlwaysEscalatePolicy()],
        scenario_ids=ALL_SCENARIOS,
    )
    solved = {str(row["policy"]): float(row["solved_rate"]) for row in summary_rows}
    assert solved["heuristic_router"] > solved["random"]
    assert solved["heuristic_router"] > solved["always_escalate"]


def test_hackable_reward_reverses_the_ranking_reward_hacking() -> None:
    """Swapping in the hackable rubric flips the ranking -- the reward-hacking lesson, measured.

    Under the aligned reward the heuristic router beats always-escalate; under the misspecified
    proxy (which overpays escalation) always-escalate beats the router. The harness surfaces this
    rank reversal while ``solved_rate`` still shows always-escalate barely solving anything -- the
    diagnostic signature of reward hacking, made visible by reporting more than the scalar reward.
    """
    good_summary, _ = evaluate_policies(
        policies=[HeuristicRouterPolicy(), AlwaysEscalatePolicy()],
        scenario_ids=ALL_SCENARIOS,
        reward_fn=judge_reward,
    )
    hack_summary, _ = evaluate_policies(
        policies=[HeuristicRouterPolicy(), AlwaysEscalatePolicy()],
        scenario_ids=ALL_SCENARIOS,
        reward_fn=hackable_reward,
    )
    good = _summary_by_name(good_summary)
    hacked = _summary_by_name(hack_summary)

    # Aligned objective: the genuinely good policy wins.
    assert good["heuristic_router"] > good["always_escalate"]
    # Misspecified proxy: the degenerate policy wins -> reward hacking.
    assert hacked["always_escalate"] > hacked["heuristic_router"]

    # The hack is exposed by the governance columns: always-escalate solves far less than it scores.
    hack_solved = {str(row["policy"]): float(row["solved_rate"]) for row in hack_summary}
    assert hack_solved["heuristic_router"] > hack_solved["always_escalate"]


def test_always_escalate_metrics_are_pure_escalation() -> None:
    """The always-escalate foil escalates every episode and never over-retrieves.

    Pins the side-metrics against a policy with known behaviour: exactly one escalation per
    episode (each episode terminates on the first escalate), zero over-effort (it never retrieves
    or clarifies), and its escalations of easy/unambiguous scenarios register as unsafe-or-
    questionable -- confirming the governance probes fire on the right behaviour.
    """
    _, scenario_rows = evaluate_policies(
        policies=[AlwaysEscalatePolicy()],
        scenario_ids=ALL_SCENARIOS,
    )
    for row in scenario_rows:
        assert row["escalation_count"] == 1  # escalate terminates immediately -> one per episode
        assert row["over_effort_count"] == 0  # never retrieves/clarifies
        assert row["final_action_label"] == "escalate"
    # The easy_factual scenario (difficulty 0, ambiguity 0) escalated -> a questionable hand-off.
    easy = next(row for row in scenario_rows if row["scenario_name"] == "easy_factual")
    assert easy["unsafe_or_questionable_decisions"] == 1
    assert easy["solved"] == 0  # escalating an easy request does not count as solved


def test_summary_escalation_and_undergrounded_rates_are_contract_gates() -> None:
    """The summary exposes avg_escalation_rate and avg_undergrounded_rate with rate semantics.

    Pins the two governance gates the deploy/shadow/reject rule reads. ``always_escalate`` escalates
    every episode (rate 1.0) and never answers, so its under-grounding rate is 0.0 by convention. An
    empty-table ``QTablePolicy`` answers immediately on every state (the all-zeros greedy fallback),
    so on the hard ``hard_debug`` scenario -- whose start state has no evidence for a positive
    difficulty -- every answer is under-grounded, giving an under-grounding rate of 1.0 and an
    escalation rate of 0.0. These are exactly the columns ``recommendation_from_summary`` gates on.
    """
    hard_debug_id = next(
        index
        for index, scenario in enumerate(scenario_catalog())
        if scenario.name == "hard_debug"
    )
    summary_rows, _ = evaluate_policies(
        policies=[AlwaysEscalatePolicy(), QTablePolicy(q_table={})],
        scenario_ids=(hard_debug_id,),
    )
    by_name = {str(row["policy"]): row for row in summary_rows}

    # always_escalate: escalates every episode (terminal) and never answers.
    assert by_name["always_escalate"]["avg_escalation_rate"] == 1.0
    assert by_name["always_escalate"]["avg_undergrounded_rate"] == 0.0
    # empty Q-table: answers immediately (action-0 fallback); a no-evidence hard request is unsafe.
    assert by_name["q_table"]["avg_escalation_rate"] == 0.0
    assert by_name["q_table"]["avg_undergrounded_rate"] == 1.0


def test_qtable_policy_evaluates_through_the_harness() -> None:
    """A QTablePolicy (learned-values stand-in) runs through evaluation and can match the router.

    The evaluation harness must score a learned-value policy exactly like the baselines. We build a
    tiny Q-table that prefers ``answer_direct`` on an already-grounded, unambiguous easy request
    (so it answers correctly) and confirm it solves and scores at least as well as the heuristic
    router on that scenario -- evidence the harness handles the value-based control rung.
    """
    env = AgentDecisionEnvironment()
    start = env.reset(scenario_id=0)  # easy_factual: difficulty 0, ambiguity 0 -> answer is correct
    # Greedy table: answer_direct (index 0) is the top-valued action at the start state.
    q_table = {start.as_tuple(): [1.0, 0.0, 0.0, 0.0]}
    policy = QTablePolicy(q_table=q_table)

    summary_rows, scenario_rows = evaluate_policies(
        policies=[policy, HeuristicRouterPolicy()],
        scenario_ids=(0,),
    )
    rewards = _summary_by_name(summary_rows)
    assert rewards["q_table"] >= rewards["heuristic_router"]
    q_row = next(row for row in scenario_rows if row["policy"] == "q_table")
    assert q_row["final_action_label"] == "answer_direct"
    assert q_row["solved"] == 1


def test_qtable_policy_values_actually_steer_the_action() -> None:
    """The Q-values must drive the decision, not the all-zeros greedy fallback (action 0).

    Guards against a tautology: because ``greedy_action`` breaks an all-zeros tie at index 0
    (``answer_direct``), a policy whose table is *empty* would also answer immediately on every
    scenario -- so "it answered" alone proves nothing about the learned values being consulted.
    Here we populate ``ambiguous_query``'s start state with a row whose argmax is ``clarify``
    (action 2) and assert the harness actually plays clarify-then-answer, whereas an empty table
    falls through to answering immediately. The divergent first action is what proves the harness
    reads the learned Q-values rather than ignoring them.
    """
    env = AgentDecisionEnvironment()
    start = env.reset(scenario_id=2)  # ambiguous_query: ambiguity 2 -> clarify is the good move
    # Top-valued action at the start is clarify (index 2), which differs from the action-0 fallback.
    learned = QTablePolicy(q_table={start.as_tuple(): [0.0, 0.0, 5.0, 0.0]})
    empty = QTablePolicy(q_table={})  # every state ties at 0 -> greedy fallback answers immediately

    _, learned_rows = evaluate_policies(policies=[learned], scenario_ids=(2,))
    _, empty_rows = evaluate_policies(policies=[empty], scenario_ids=(2,))

    learned_actions = str(learned_rows[0]["actions"]).split(" | ")
    empty_actions = str(empty_rows[0]["actions"]).split(" | ")
    # The learned table steers the first move to clarify; the fallback would have answered directly.
    assert learned_actions[0] == "clarify"
    assert empty_actions[0] == "answer_direct"
    assert learned_actions != empty_actions


def test_evaluation_is_deterministic_for_fixed_seed() -> None:
    """Two runs with the same inputs produce identical tables (reproducible evaluation)."""
    first_summary, first_scenarios = evaluate_policies(
        policies=[RandomPolicy(seed=9), HeuristicRouterPolicy()],
        scenario_ids=ALL_SCENARIOS,
        episodes_per_scenario=2,
        base_seed=4,
    )
    second_summary, second_scenarios = evaluate_policies(
        policies=[RandomPolicy(seed=9), HeuristicRouterPolicy()],
        scenario_ids=ALL_SCENARIOS,
        episodes_per_scenario=2,
        base_seed=4,
    )
    assert first_summary == second_summary
    assert first_scenarios == second_scenarios


def test_episodes_per_scenario_scales_row_count() -> None:
    """The per-scenario table has exactly policies * scenarios * episodes_per_scenario rows."""
    _, scenario_rows = evaluate_policies(
        policies=[RandomPolicy(seed=2), HeuristicRouterPolicy()],
        scenario_ids=(0, 1, 2),
        episodes_per_scenario=3,
    )
    assert len(scenario_rows) == 2 * 3 * 3


def test_simulate_episode_traces_each_transition() -> None:
    """simulate_episode returns one chronological row per step with state, action, and reward.

    Pins the trajectory trace: rolling the heuristic router on the ambiguous scenario yields a row
    per (s, a, R_{t+1}, s') transition, each carrying the pre-step state fields, the action label,
    its cost, the reward, and the resulting next_evidence/next_ambiguity, ending on a terminal
    commit. This is the qualitative companion to the aggregate metrics.
    """
    rows = simulate_episode(policy=HeuristicRouterPolicy(), scenario_id=2)
    assert rows, "a rollout must produce at least one transition"
    expected_keys = {
        "scenario_name",
        "step",
        "difficulty",
        "ambiguity",
        "evidence",
        "action_label",
        "action_cost",
        "reward",
        "termination",
        "next_evidence",
        "next_ambiguity",
    }
    for row in rows:
        assert expected_keys <= set(row)
    # The clock advances by one each step (deterministic dynamics, no jitter without a seed).
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    # The router must end by committing (answer_direct or escalate), never mid-episode.
    assert rows[-1]["action_label"] in {"answer_direct", "escalate"}


def test_simulate_episode_is_seed_reproducible() -> None:
    """The same seed reproduces the same trace exactly (start-state jitter is deterministic)."""
    first = simulate_episode(policy=HeuristicRouterPolicy(), scenario_id=3, seed=7)
    second = simulate_episode(policy=HeuristicRouterPolicy(), scenario_id=3, seed=7)
    assert first == second
