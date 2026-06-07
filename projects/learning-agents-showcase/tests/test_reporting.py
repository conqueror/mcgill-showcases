"""Pin the artifact contract and the deploy/shadow/reject rule for the learning-agents showcase.

These tests are not about an RL algorithm; they guard the *governance/reproducibility* boundary
around the whole pipeline. The showcase is only trustworthy if every required artifact (bandit
traces, Q-learning tables, DP ground-truth gaps, evaluation comparisons, reward and governance
memos) is present and well-formed before anyone reads results, and if the rollout decision refuses a
reward-hacking policy. The tests pin that:

* the validator fails on missing required outputs and on present-but-malformed required CSVs;
* it succeeds on a complete, well-formed set;
* the checked-in manifest stays in sync with the in-code contract;
* the optional deep-RL (DQN/PPO) comparison group is all-or-nothing;
* the Q-table schema is exactly the live :class:`AgentState` fields (no drift);
* episode/eval artifacts may only name actions the environment actually defines;
* the under-grounding metric is anchored to the reward's quality predicate;
* the deploy/shadow/reject rule rewards a genuinely-better learned policy but rejects one that
  wins reward only by over-escalating or by returning under-grounded answers (reward hacking).

RL concept:
    Evaluation governance and the reproducibility contract; the recommendation rule is the offline
    gate that connects reward design / reward hacking to a concrete rollout decision.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from learning_agents.environment import ACTION_LABELS, AgentState
from learning_agents.reporting import (
    OPTIONAL_DRL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    artifact_validation_errors,
    missing_required_artifacts,
    recommendation_from_summary,
    undergrounded_answer_rate,
    write_manifest,
    write_text_artifact,
)

# Minimal, schema-valid CSV bodies (header + one data row) for every required CSV artifact. The
# headers must match learning_agents.reporting._REQUIRED_CSV_COLUMNS exactly, so this table is also
# documentation of the contract a real run must satisfy.
_CSV_BODIES: dict[str, str] = {
    "artifacts/concepts/concept_map.csv": (
        "concept,showcase_component,artifact\n"
        "mdp_framing,mdp_spec_markdown,artifacts/concepts/mdp_spec.md\n"
    ),
    "artifacts/bandit/reward_trace.csv": (
        "step,scenario_name,context_signature,action_label,expected_reward,optimal_action_label\n"
        "1,ambiguous_query,intent=2|difficulty=1|ambiguity=2,clarify,0.5,clarify\n"
    ),
    "artifacts/bandit/regret_trace.csv": (
        "step,scenario_name,action_label,optimal_action_label,cumulative_regret\n"
        "1,ambiguous_query,clarify,clarify,0.0\n"
    ),
    "artifacts/mdp/sample_episodes.csv": (
        "scenario_name,step,action_label,reward,next_evidence\n"
        "hard_debug,1,retrieve,-0.5,1\n"
    ),
    "artifacts/q_learning/training_curve.csv": (
        "episode,scenario_id,total_reward,epsilon,steps\n1,3,1.5,0.4,3\n"
    ),
    "artifacts/q_learning/q_table.csv": (
        "step,intent,difficulty,ambiguity,evidence,attempts,budget,action,q_value\n"
        "0,3,2,1,0,0,30,1,1.2\n"
    ),
    "artifacts/dp/optimal_action_values.csv": ("step,action,optimal_q_value\n0,1,2.5\n"),
    "artifacts/dp/q_learning_gap.csv": (
        "step,action,learned_q_value,optimal_q_value,abs_gap\n0,1,2.0,2.5,0.5\n"
    ),
    "artifacts/sarsa/training_curve.csv": (
        "episode,scenario_id,total_reward,epsilon,steps\n1,3,1.5,0.4,3\n"
    ),
    "artifacts/sarsa/q_table.csv": (
        "step,intent,difficulty,ambiguity,evidence,attempts,budget,action,q_value\n"
        "0,3,2,1,0,0,30,1,1.1\n"
    ),
    "artifacts/policy_gradient/training_curve.csv": (
        "episode,scenario_id,total_reward,baseline,steps\n1,3,1.5,1.0,3\n"
    ),
    "artifacts/eval/policy_comparison.csv": (
        "policy,avg_reward,avg_escalation_rate,avg_undergrounded_rate,avg_action_cost,solved_rate\n"
        "q_learning,1.8,0.2,0.0,0.6,1.0\n"
    ),
    "artifacts/eval/scenario_results.csv": (
        "policy,scenario_id,scenario_name,total_reward,final_action_label,actions\n"
        "q_learning,3,hard_debug,1.8,answer_direct,retrieve | retrieve | answer_direct\n"
    ),
    "artifacts/offline_rl/training_curve.csv": (
        "sweep,bellman_residual,updated_state_action_pairs\n1,0.5,42\n"
    ),
    "artifacts/offline_rl/dataset_summary.csv": (
        "num_transitions,num_decision_states,num_reachable_states,coverage_fraction,"
        "behavior_policy,epsilon\n2164,196,371,0.5283,heuristic_router,0.3\n"
    ),
    "artifacts/ope/estimator_comparison.csv": (
        "target,estimator,estimate,true_value,abs_error\n"
        "heuristic_router,doubly_robust,1.151,1.179,0.028\n"
    ),
    "artifacts/cost_cascade/cost_quality_curve.csv": (
        "effort_budget,avg_action_cost,avg_steps,total_cost,avg_reward,"
        "avg_escalation_rate,solved_rate\n0,1.07,1.0,1.37,0.259,0.717,0.85\n"
    ),
    "artifacts/sdk_bridge/orchestration_trace.csv": (
        "step,scenario_name,action_label,sdk_role,sdk_target,reward,terminal\n"
        "0,hard_debug,retrieve,tool_call,retrieve_context,-0.5,0\n"
    ),
    "artifacts/preference/method_comparison.csv": (
        "method,expected_quality,win_rate_vs_reference,kl_to_reference\n"
        "dpo,0.9999,0.9,1.6074\n"
    ),
    "artifacts/preference/training_curves.csv": (
        "method,epoch,expected_quality,kl_to_reference\ndpo,1,0.5106,0.02\n"
    ),
    "artifacts/marl/coordination_comparison.csv": (
        "method,coordination_success_rate,final_joint_action,final_team_reward,optimal_team_reward\n"
        "joint,1.0,deep_research+detailed,11.0,11.0\n"
    ),
    "artifacts/marl/training_curves.csv": (
        "method,episode,greedy_team_reward\njoint,80,11.0\n"
    ),
}

# Minimal markdown bodies for every required markdown artifact. The ladder narrative must name each
# rung, which the markdown validator enforces.
_MARKDOWN_BODIES: dict[str, str] = {
    "artifacts/concepts/mdp_spec.md": "# Agent Decision MDP\n",
    "artifacts/concepts/algorithm_progression.md": (
        "# RL To DRL Progression\n\nQ-learning, DQN, policy gradients, actor-critic, and PPO.\n"
    ),
    "artifacts/reward/reward_hacking_report.md": "# Reward Hacking Report\n",
    "artifacts/reward/reward_spec_good.md": "# Good Reward\n",
    "artifacts/reward/reward_spec_bad.md": "# Bad Reward\n",
    "artifacts/governance/safety_controls.md": "# Safety Controls\n",
    "artifacts/governance/offline_eval_plan.md": "# Offline Eval Plan\n",
    "artifacts/business/deploy_shadow_reject_memo.md": "# Deploy Memo\n",
    "artifacts/sdk_bridge/bridge_report.md": "# OpenAI Agents SDK Bridge\n",
}


def _write_minimal_required_artifacts(root: Path) -> None:
    """Materialize one well-formed copy of every required artifact under ``root``.

    Shared setup for the "succeeds" and "invalid CSV" tests: it writes a schema-valid file per entry
    in :data:`REQUIRED_ARTIFACTS` (correct CSV headers with a single data row, or a heading-led
    Markdown stub) so validation passes unless a test deliberately corrupts one file afterward. Not
    an RL step -- it exists purely to exercise the artifact contract.

    Args:
        root: Temporary project root under which the ``artifacts/`` tree is created.
    """
    for relative_path in REQUIRED_ARTIFACTS:
        if relative_path == "artifacts/manifest.json":
            write_manifest(root / relative_path)
        elif relative_path in _CSV_BODIES:
            write_text_artifact(root / relative_path, _CSV_BODIES[relative_path])
        else:
            write_text_artifact(root / relative_path, _MARKDOWN_BODIES[relative_path])


def test_validation_fails_when_required_files_are_missing(tmp_path: Path) -> None:
    """An empty output tree reports every required artifact as missing.

    Pins the fail-closed half of the contract: with a manifest present but no artifact files,
    :func:`missing_required_artifacts` must list the whole required set, so a half-finished run can
    never be presented as complete.

    RL concept:
        Reproducibility gate in evaluation governance -- presence before results.
    """
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path, REQUIRED_ARTIFACTS)

    missing = missing_required_artifacts(output_dir=tmp_path, manifest_path=manifest_path)

    expected_missing = {
        path for path in REQUIRED_ARTIFACTS if path != "artifacts/manifest.json"
    }
    assert set(missing) == expected_missing


def test_validation_succeeds_for_complete_outputs(tmp_path: Path) -> None:
    """A complete, well-formed artifact set passes both presence and shape checks.

    Pins the fail-open half: once a minimal but valid copy of each required artifact is written,
    nothing is missing and no validation error is raised. Together with the missing-files test this
    brackets the verifier's accept/reject boundary on completeness.

    RL concept:
        Reproducibility gate in evaluation governance -- a complete run validates clean.
    """
    _write_minimal_required_artifacts(tmp_path)

    assert missing_required_artifacts(output_dir=tmp_path) == []
    assert artifact_validation_errors(output_dir=tmp_path) == []


def test_checked_in_manifest_matches_artifact_contract() -> None:
    """The committed manifest.json round-trips the in-code required/optional tuples.

    Writing the contract to a manifest and reading it back must reproduce
    :data:`REQUIRED_ARTIFACTS` / :data:`OPTIONAL_DRL_ARTIFACTS` exactly. This catches drift where a
    new artifact is added in code but the frozen contract is never updated, which would let an
    incomplete run pass verification.

    RL concept:
        Contract integrity for reproducible evaluation -- one source of truth.
    """
    repo_manifest = (
        Path(__file__).resolve().parent.parent / "artifacts" / "manifest.json"
    )
    payload = json.loads(repo_manifest.read_text(encoding="utf-8"))
    assert tuple(payload["required_files"]) == REQUIRED_ARTIFACTS
    assert tuple(payload["optional_files"]) == OPTIONAL_DRL_ARTIFACTS


def test_manifest_round_trip_reproduces_contract(tmp_path: Path) -> None:
    """Serializing then loading the manifest reproduces the exact contract tuples.

    Pins that :func:`write_manifest` is a faithful, version-pinned serialization of the live
    contract, so an audit run validated against a frozen manifest sees the same required/optional
    sets as the code.

    RL concept:
        Reproducibility -- a frozen, version-pinned evidence contract.
    """
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert tuple(payload["required_files"]) == REQUIRED_ARTIFACTS
    assert tuple(payload["optional_files"]) == OPTIONAL_DRL_ARTIFACTS


def test_validation_fails_for_invalid_required_csv_content(tmp_path: Path) -> None:
    """A present-but-column-deficient required CSV is rejected by shape validation.

    Pins that existence is not enough: after writing a complete set, overwriting the bandit reward
    trace with a CSV missing its required columns must surface a validation error naming the file.
    Schema validation, not just presence, protects the meaning of the reproducible outputs.

    RL concept:
        Schema-level reproducibility validation -- columns carry the meaning.
    """
    _write_minimal_required_artifacts(tmp_path)
    write_text_artifact(
        tmp_path / "artifacts" / "bandit" / "reward_trace.csv",
        "step,action_label\n1,clarify",
    )

    errors = artifact_validation_errors(output_dir=tmp_path)

    assert any("reward_trace.csv is missing required columns" in error for error in errors)


def test_validation_fails_for_empty_required_csv(tmp_path: Path) -> None:
    """A required CSV with a header but no data rows is rejected.

    Pins the "at least one data row" half of the columnar contract: an empty training curve is not
    valid evidence even if its header is correct.

    RL concept:
        Evaluation evidence must be non-empty to be meaningful.
    """
    _write_minimal_required_artifacts(tmp_path)
    write_text_artifact(
        tmp_path / "artifacts" / "q_learning" / "training_curve.csv",
        "episode,scenario_id,total_reward,epsilon,steps",
    )

    errors = artifact_validation_errors(output_dir=tmp_path)

    assert any("training_curve.csv must contain at least one data row" in e for e in errors)


def test_q_table_schema_matches_live_agent_state_fields() -> None:
    """The Q-table CSV's state columns equal AgentState's declared fields, in order.

    Pins that the reporting contract is *derived from the live MDP*, not re-typed: the columns the
    validator requires for ``artifacts/q_learning/q_table.csv`` (minus action/q_value) are exactly
    :meth:`AgentState.as_tuple`'s field order. If a state field is added or renamed, the artifact
    contract must follow automatically, and this test fails until the fixtures do.

    RL concept:
        State representation -- the tabular evidence keys are the MDP's state variables.
    """
    state_field_names = tuple(f.name for f in fields(AgentState))
    header = _CSV_BODIES["artifacts/q_learning/q_table.csv"].splitlines()[0].split(",")

    assert tuple(header[: len(state_field_names)]) == state_field_names
    assert header[len(state_field_names) :] == ["action", "q_value"]


def test_validation_rejects_unknown_action_label(tmp_path: Path) -> None:
    """An episode artifact that names a non-existent action is malformed.

    Pins that the evidence can only reference actions the environment actually defines: corrupting
    the sample-episodes action label to something outside :data:`ACTION_LABELS` must raise an error.
    This ties the governance layer back to the live action space.

    RL concept:
        Evidence integrity -- artifacts may only mention the MDP's real action space.
    """
    _write_minimal_required_artifacts(tmp_path)
    write_text_artifact(
        tmp_path / "artifacts" / "mdp" / "sample_episodes.csv",
        "scenario_name,step,action_label,reward,next_evidence\nhard_debug,1,teleport,0.0,0",
    )

    errors = artifact_validation_errors(output_dir=tmp_path)

    assert "teleport" not in set(ACTION_LABELS.values())  # guard: the corrupt label is truly bogus
    assert any("unknown action label 'teleport'" in error for error in errors)


def test_optional_drl_group_requires_complete_comparison_outputs(tmp_path: Path) -> None:
    """The optional deep-RL comparison group is enforced as all-or-nothing.

    Pins the group rule for the upper rungs of the ladder (DQN, PPO): once any deep-RL comparison
    artifact appears, the remaining ones become mandatory. Writing only the bridge report plus one
    comparison CSV must raise an "incomplete" error, so a partial DQN/PPO comparison can never
    masquerade as a finished one.

    RL concept:
        Completeness contract for the deep-RL comparison stage.
    """
    write_text_artifact(
        tmp_path / "artifacts" / "drl_optional" / "bridge_report.md",
        "# Optional DRL Bridge\n\nDQN and PPO.\n",
    )
    write_text_artifact(
        tmp_path / "artifacts" / "drl_optional" / "rl_family_comparison.csv",
        "policy,family,avg_reward,avg_escalation_rate,solved_rate\n"
        "q_learning,tabular_value_based,1.0,0.2,1.0\n",
    )

    errors = artifact_validation_errors(output_dir=tmp_path)

    assert any("Optional DRL comparison output is incomplete" in error for error in errors)


def test_optional_drl_family_comparison_requires_all_three_families(tmp_path: Path) -> None:
    """A complete deep-RL group whose comparison omits a family is rejected.

    Pins that the family-comparison CSV must contrast all three families head-to-head: a group that
    is otherwise complete but lists only ``q_learning`` (no ``dqn``/``ppo`` rows) must raise an
    error, so the deep-RL bridge cannot claim a comparison it did not run.

    RL concept:
        Deep-RL bridge -- value-based DQN vs actor-critic PPO vs tabular Q-learning.
    """
    base = tmp_path / "artifacts" / "drl_optional"
    write_text_artifact(base / "bridge_report.md", "# DRL Bridge\n\nDQN and PPO.\n")
    write_text_artifact(
        base / "rl_family_comparison.csv",
        "policy,family,avg_reward,avg_escalation_rate,solved_rate\n"
        "q_learning,tabular_value_based,1.0,0.2,1.0\n",
    )
    write_text_artifact(
        base / "scenario_rollups.csv",
        "policy,scenario_id,scenario_name,total_reward\nq_learning,3,hard_debug,1.0\n",
    )
    write_text_artifact(
        base / "training_summary.csv",
        "policy,step,mean_reward,mean_escalation_rate\nq_learning,1,1.0,0.2\n",
    )
    write_text_artifact(
        base / "policy_gradient_notes.md",
        "# Notes\n\npolicy-gradient, actor-critic, ppo, dqn.\n",
    )

    errors = artifact_validation_errors(output_dir=tmp_path)

    assert any("must include q_learning, dqn, and ppo rows" in error for error in errors)


def test_undergrounded_rate_anchored_to_quality_predicate() -> None:
    """The under-grounding metric matches the reward module's adequacy predicate.

    Pins that ``undergrounded_answer_rate`` counts an answer as under-grounded exactly when
    :func:`learning_agents.reward.evidence_is_adequate` is False -- a difficulty-2 answer with one
    unit of evidence is under-grounded, a difficulty-1 answer with one unit is not. This keeps the
    governance metric and the reward objective on one definition, so reward hacking via
    under-grounded answers is measured the same way it is penalized.

    RL concept:
        Evaluation safety metric anchored to the reward's quality predicate.
    """
    base = dict(step=1, intent=0, ambiguity=0, attempts=0, budget=30)
    undergrounded = AgentState(difficulty=2, evidence=1, **base)  # needs 2, has 1 -> bad
    well_grounded = AgentState(difficulty=1, evidence=1, **base)  # needs 1, has 1 -> ok

    assert undergrounded_answer_rate([]) == 0.0
    assert undergrounded_answer_rate([well_grounded]) == 0.0
    assert undergrounded_answer_rate([undergrounded]) == 1.0
    assert undergrounded_answer_rate([undergrounded, well_grounded]) == 0.5


def test_recommendation_shadows_a_genuinely_better_safe_policy() -> None:
    """A learned policy that safely beats the heuristic router earns a guarded rollout.

    Pins the positive branch of the deploy/shadow/reject rule: when ``q_learning`` has higher
    average reward than ``heuristic_router`` while staying within both safety bounds (low escalation
    rate, low under-grounding rate), the recommendation is ``shadow``.

    RL concept:
        Offline evaluation gate -- promote a learned policy only when it is both better and safe.
    """
    summary = [
        {
            "policy": "q_learning",
            "avg_reward": 1.8,
            "avg_escalation_rate": 0.2,
            "avg_undergrounded_rate": 0.0,
        },
        {
            "policy": "heuristic_router",
            "avg_reward": 1.2,
            "avg_escalation_rate": 0.2,
            "avg_undergrounded_rate": 0.0,
        },
    ]

    decision, _rationale = recommendation_from_summary(summary)

    assert decision == "shadow"


def test_recommendation_rejects_over_escalating_reward_hacker() -> None:
    """A high-reward policy that wins by over-escalating is rejected by the safety gate.

    Pins the reward-hacking guard: even though ``q_learning`` has the higher average reward, an
    escalation rate above 0.5 (the always-escalate failure mode the hackable reward induces) trips
    the safety gate *before* the reward comparison, so the recommendation is ``reject``.

    RL concept:
        Reward hacking -- a proxy-maximizing, over-escalating policy must not reach rollout.
    """
    summary = [
        {
            "policy": "q_learning",
            "avg_reward": 9.0,  # inflated by over-escalation under a hackable reward
            "avg_escalation_rate": 0.9,
            "avg_undergrounded_rate": 0.0,
        },
        {
            "policy": "heuristic_router",
            "avg_reward": 1.2,
            "avg_escalation_rate": 0.2,
            "avg_undergrounded_rate": 0.0,
        },
    ]

    decision, rationale = recommendation_from_summary(summary)

    assert decision == "reject"
    assert "escalate" in rationale.lower()


def test_recommendation_rejects_under_grounded_reward_hacker() -> None:
    """A high-reward policy that answers without grounding is rejected.

    Pins the second safety gate: an under-grounding rate above 0.5 (answering before retrieving
    enough evidence) trips the gate regardless of reward, so a hallucination-prone policy cannot be
    promoted.

    RL concept:
        Reward hacking -- under-grounded answers (hallucination risk) must not reach rollout.
    """
    summary = [
        {
            "policy": "q_learning",
            "avg_reward": 5.0,
            "avg_escalation_rate": 0.1,
            "avg_undergrounded_rate": 0.8,
        },
        {
            "policy": "heuristic_router",
            "avg_reward": 1.2,
            "avg_escalation_rate": 0.2,
            "avg_undergrounded_rate": 0.0,
        },
    ]

    decision, rationale = recommendation_from_summary(summary)

    assert decision == "reject"
    assert "under-grounded" in rationale.lower()


def test_recommendation_rejects_insufficient_margin() -> None:
    """A safe-but-not-better learned policy is rejected for an insufficient margin.

    Pins the middle branch: when ``q_learning`` is safe but does not beat the heuristic router on
    reward, training did not pay off and the recommendation is ``reject``.

    RL concept:
        Offline evaluation gate -- learning must clear the incumbent baseline to justify rollout.
    """
    summary = [
        {
            "policy": "q_learning",
            "avg_reward": 1.0,
            "avg_escalation_rate": 0.2,
            "avg_undergrounded_rate": 0.0,
        },
        {
            "policy": "heuristic_router",
            "avg_reward": 1.5,
            "avg_escalation_rate": 0.2,
            "avg_undergrounded_rate": 0.0,
        },
    ]

    decision, _rationale = recommendation_from_summary(summary)

    assert decision == "reject"


def test_recommendation_rejects_when_baseline_or_learned_row_missing() -> None:
    """Missing a comparable baseline or learned row forces a conservative reject.

    Pins the fail-closed default: with no ``heuristic_router`` row to compare against, the rule has
    no evidence to justify a rollout and must reject.

    RL concept:
        Governance default -- absent comparative evidence, do not deploy.
    """
    summary = [
        {
            "policy": "q_learning",
            "avg_reward": 2.0,
            "avg_escalation_rate": 0.1,
            "avg_undergrounded_rate": 0.0,
        },
    ]

    decision, _rationale = recommendation_from_summary(summary)

    assert decision == "reject"


def test_recommendation_accepts_csv_round_tripped_string_values() -> None:
    """The rule parses string-typed fields (as read back from CSV) the same as numbers.

    Pins that summaries round-tripped through CSV (all values are strings) yield the same decision
    as in-memory numeric summaries, so the gate works on artifacts read from disk.

    RL concept:
        Robust offline evaluation -- the gate consumes serialized evidence faithfully.
    """
    summary = [
        {
            "policy": "q_learning",
            "avg_reward": "1.8",
            "avg_escalation_rate": "0.2",
            "avg_undergrounded_rate": "0.0",
        },
        {
            "policy": "heuristic_router",
            "avg_reward": "1.2",
            "avg_escalation_rate": "0.2",
            "avg_undergrounded_rate": "0.0",
        },
    ]

    decision, _rationale = recommendation_from_summary(summary)

    assert decision == "shadow"
