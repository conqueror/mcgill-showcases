"""Define and enforce the showcase artifact contract, plus the deploy/shadow/reject rule.

What + why: this module is the *governance and evidence* layer of the learning-agents showcase. It
does no learning itself, but it pins down exactly which files every training/evaluation run must
emit, writes them to disk, validates their shape, and turns an offline-evaluation summary into a
rollout decision. Conceptually it sits at the far end of the RL ladder (contextual bandit -> MDP ->
Q-learning -> SARSA -> dynamic programming -> policy gradient -> optional deep RL): once an agent's
*orchestration policy* has been learned, RL practice demands a disciplined offline check *before*
any deployment, and this module is where that discipline is encoded as code rather than prose.

The artifact contract is the source of truth shared by three parties: the runner scripts that
*produce* artifacts, ``scripts/verify_artifacts.py`` that *checks* them in CI, and
``tests/test_reporting.py`` that *tests* the checker. ``REQUIRED_ARTIFACTS`` and
``OPTIONAL_DRL_ARTIFACTS`` are that contract's data; do not edit them casually because the
checked-in ``artifacts/manifest.json`` and the test suite assert byte-for-byte agreement.

This module imports ONLY from :mod:`learning_agents` so the showcase stays self-contained: it pulls
the action vocabulary (:data:`~learning_agents.environment.ACTION_LABELS`), the answer-quality
predicate (:func:`~learning_agents.reward.evidence_is_adequate`), and the state shape
(:class:`~learning_agents.environment.AgentState`) from this package's own environment/reward so the
CSV schemas and the recommendation rule stay aligned with the live MDP.

RL concept:
    Evaluation and governance -- offline evaluation gates and reproducible evidence. The
    recommendation rule connects to reward design and reward hacking: a policy that wins reward by
    over-escalating or returning under-grounded answers must be caught here, not in production.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

from learning_agents.environment import ACTION_LABELS, AgentState
from learning_agents.reward import evidence_is_adequate

# Public surface of this module: the artifact contract data, the typed writers/validators, the
# narrative builders, and the deploy/shadow/reject decision rule.
__all__ = [
    "OPTIONAL_DRL_ARTIFACTS",
    "REQUIRED_ARTIFACTS",
    "algorithm_progression_markdown",
    "artifact_validation_errors",
    "concept_map_rows",
    "governance_artifacts",
    "mdp_spec_markdown",
    "missing_required_artifacts",
    "optional_artifacts",
    "recommendation_from_summary",
    "required_artifacts",
    "undergrounded_answer_rate",
    "write_csv_artifact",
    "write_json_artifact",
    "write_manifest",
    "write_text_artifact",
]

# Artifact contract (data, not behavior): the exact set of relative paths a complete showcase run
# must emit. Mirrored verbatim in artifacts/manifest.json and asserted by the test suite. The set
# spans every rung of this showcase's ladder so a reader can inspect each method's evidence, and it
# includes the checked-in manifest itself so the frozen contract file is part of the verifier gate.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "artifacts/manifest.json",
    "artifacts/concepts/mdp_spec.md",
    "artifacts/concepts/concept_map.csv",
    "artifacts/concepts/algorithm_progression.md",
    "artifacts/bandit/reward_trace.csv",
    "artifacts/bandit/regret_trace.csv",
    "artifacts/mdp/sample_episodes.csv",
    "artifacts/q_learning/training_curve.csv",
    "artifacts/q_learning/q_table.csv",
    "artifacts/dp/optimal_action_values.csv",
    "artifacts/dp/q_learning_gap.csv",
    "artifacts/sarsa/training_curve.csv",
    "artifacts/sarsa/q_table.csv",
    "artifacts/policy_gradient/training_curve.csv",
    "artifacts/offline_rl/training_curve.csv",
    "artifacts/offline_rl/dataset_summary.csv",
    "artifacts/ope/estimator_comparison.csv",
    "artifacts/cost_cascade/cost_quality_curve.csv",
    "artifacts/eval/policy_comparison.csv",
    "artifacts/eval/scenario_results.csv",
    "artifacts/reward/reward_hacking_report.md",
    "artifacts/reward/reward_spec_good.md",
    "artifacts/reward/reward_spec_bad.md",
    "artifacts/governance/safety_controls.md",
    "artifacts/governance/offline_eval_plan.md",
    "artifacts/business/deploy_shadow_reject_memo.md",
    "artifacts/sdk_bridge/bridge_report.md",
    "artifacts/sdk_bridge/orchestration_trace.csv",
    "artifacts/preference/method_comparison.csv",
    "artifacts/preference/training_curves.csv",
    "artifacts/marl/coordination_comparison.csv",
    "artifacts/marl/training_curves.csv",
)

OPTIONAL_DRL_ARTIFACTS: tuple[str, ...] = (
    # Optional deep-RL bridge (DQN/PPO). Either none of these exist, or the whole comparison group
    # must exist together (see artifact_validation_errors for the all-or-nothing rule).
    "artifacts/drl_optional/bridge_report.md",
    "artifacts/drl_optional/rl_family_comparison.csv",
    "artifacts/drl_optional/scenario_rollups.csv",
    "artifacts/drl_optional/training_summary.csv",
    "artifacts/drl_optional/policy_gradient_notes.md",
)

# Derive the state-key column order *from the live MDP* rather than re-typing it, so the Q-table
# schema provably equals AgentState.as_tuple()'s field order; if a field is ever added/renamed the
# contract follows automatically. This is what keeps the artifacts aligned with the environment.
_STATE_KEY_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(AgentState))

# The set of legal action labels (e.g. "answer_direct", "escalate"); used to validate that the
# evaluation/episode artifacts only ever name actions the environment actually defines.
_ACTION_LABEL_SET: frozenset[str] = frozenset(ACTION_LABELS.values())

# Per-path required CSV columns: the columnar contract for every tabular artifact. Derived from the
# live MDP -- the Q-table columns are exactly the seven AgentState fields plus action/value, and the
# evaluation columns are the multi-objective rollup of the judge rubric (reward, residual risk via
# escalation, under-grounding, cost). ``scripts/verify_artifacts.py`` and the reporting test suite
# rely on exactly these rules.
_REQUIRED_CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "artifacts/concepts/concept_map.csv": (
        "concept",
        "showcase_component",
        "artifact",
    ),
    "artifacts/bandit/reward_trace.csv": (
        "step",
        "scenario_name",
        "context_signature",
        "action_label",
        "expected_reward",
        "optimal_action_label",
    ),
    "artifacts/bandit/regret_trace.csv": (
        "step",
        "scenario_name",
        "action_label",
        "optimal_action_label",
        "cumulative_regret",
    ),
    "artifacts/mdp/sample_episodes.csv": (
        "scenario_name",
        "step",
        "action_label",
        "reward",
        "next_evidence",
    ),
    "artifacts/q_learning/training_curve.csv": (
        "episode",
        "scenario_id",
        "total_reward",
        "epsilon",
        "steps",
    ),
    "artifacts/q_learning/q_table.csv": (*_STATE_KEY_COLUMNS, "action", "q_value"),
    "artifacts/dp/optimal_action_values.csv": (
        "step",
        "action",
        "optimal_q_value",
    ),
    "artifacts/dp/q_learning_gap.csv": (
        "step",
        "action",
        "learned_q_value",
        "optimal_q_value",
        "abs_gap",
    ),
    "artifacts/sarsa/training_curve.csv": (
        "episode",
        "scenario_id",
        "total_reward",
        "epsilon",
        "steps",
    ),
    "artifacts/sarsa/q_table.csv": (*_STATE_KEY_COLUMNS, "action", "q_value"),
    "artifacts/policy_gradient/training_curve.csv": (
        "episode",
        "scenario_id",
        "total_reward",
        "baseline",
        "steps",
    ),
    "artifacts/offline_rl/training_curve.csv": (
        "sweep",
        "bellman_residual",
        "updated_state_action_pairs",
    ),
    "artifacts/offline_rl/dataset_summary.csv": (
        "num_transitions",
        "num_decision_states",
        "num_reachable_states",
        "coverage_fraction",
        "behavior_policy",
        "epsilon",
    ),
    "artifacts/ope/estimator_comparison.csv": (
        "target",
        "estimator",
        "estimate",
        "true_value",
        "abs_error",
    ),
    "artifacts/cost_cascade/cost_quality_curve.csv": (
        "effort_budget",
        "avg_action_cost",
        "avg_steps",
        "total_cost",
        "avg_reward",
        "avg_escalation_rate",
        "solved_rate",
    ),
    "artifacts/sdk_bridge/orchestration_trace.csv": (
        "step",
        "scenario_name",
        "action_label",
        "sdk_role",
        "sdk_target",
        "reward",
        "terminal",
    ),
    "artifacts/preference/method_comparison.csv": (
        "method",
        "expected_quality",
        "win_rate_vs_reference",
        "kl_to_reference",
    ),
    "artifacts/preference/training_curves.csv": (
        "method",
        "epoch",
        "expected_quality",
        "kl_to_reference",
    ),
    "artifacts/marl/coordination_comparison.csv": (
        "method",
        "coordination_success_rate",
        "final_joint_action",
        "final_team_reward",
        "optimal_team_reward",
    ),
    "artifacts/marl/training_curves.csv": (
        "method",
        "episode",
        "greedy_team_reward",
    ),
    "artifacts/eval/policy_comparison.csv": (
        "policy",
        "avg_reward",
        "avg_escalation_rate",
        "avg_undergrounded_rate",
        "avg_action_cost",
        "solved_rate",
    ),
    "artifacts/eval/scenario_results.csv": (
        "policy",
        "scenario_id",
        "scenario_name",
        "total_reward",
        "final_action_label",
        "actions",
    ),
    "artifacts/drl_optional/rl_family_comparison.csv": (
        "policy",
        "family",
        "avg_reward",
        "avg_escalation_rate",
        "solved_rate",
    ),
    "artifacts/drl_optional/scenario_rollups.csv": (
        "policy",
        "scenario_id",
        "scenario_name",
        "total_reward",
    ),
    "artifacts/drl_optional/training_summary.csv": (
        "policy",
        "step",
        "mean_reward",
        "mean_escalation_rate",
    ),
}


def write_manifest(
    path: Path,
    required_files: Sequence[str] = REQUIRED_ARTIFACTS,
    optional_files: Sequence[str] = OPTIONAL_DRL_ARTIFACTS,
) -> None:
    """Serialize the artifact contract to a versioned JSON manifest.

    What + why: writes the required/optional artifact lists to ``path`` so downstream tooling can
    validate a run against a *frozen* contract instead of whatever code happens to be loaded. The
    checked-in ``artifacts/manifest.json`` is produced this way and is asserted to match the
    module-level tuples by ``tests/test_reporting.py``.

    Args:
        path: Destination of the manifest JSON file (parents are created as needed).
        required_files: Relative paths that must always be present. Defaults to
            :data:`REQUIRED_ARTIFACTS`.
        optional_files: Relative paths for the optional deep-RL comparison group. Defaults to
            :data:`OPTIONAL_DRL_ARTIFACTS`.

    RL concept:
        Evaluation and governance -- a reproducible, version-pinned evidence contract.
    """
    payload = {
        "version": 1,
        "required_files": list(required_files),
        "optional_files": list(optional_files),
    }
    write_json_artifact(path, payload)


def write_csv_artifact(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a list of uniform mappings as a CSV artifact.

    What + why: the header is taken from the keys of the first row, so every mapping is expected to
    share the same schema. This is the workhorse for tabular evidence (training curves, Q-tables,
    policy comparisons) whose required columns are enforced by :func:`_validate_csv_artifact`.

    Args:
        path: Destination CSV path (parent directories are created as needed).
        rows: Sequence of mappings sharing a common key set. An empty sequence writes an empty file
            (no header), which the validator later treats as a contract violation.

    RL concept:
        Evaluation and governance -- machine-checkable, columnar evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Write a mapping as a stably-ordered, indented JSON artifact.

    What + why: keys are sorted and the output is indented so diffs stay small and deterministic
    across runs -- important for the manifest, which is committed to version control and compared
    against the in-code contract in tests.

    Args:
        path: Destination JSON path (parent directories are created as needed).
        payload: Mapping to serialize; converted to a plain ``dict`` before dumping.

    RL concept:
        Evaluation and governance -- deterministic, reviewable run metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text_artifact(path: Path, content: str) -> None:
    """Write a text/markdown artifact with normalized leading/trailing whitespace.

    What + why: strips surrounding whitespace and guarantees exactly one trailing newline so the
    narrative artifacts (MDP spec, governance memos, reward-hacking report) are byte-stable. The
    validators only require a leading ``#`` heading and certain key terms, both preserved by this
    normalization.

    Args:
        path: Destination text path (parent directories are created as needed).
        content: Raw text body; outer whitespace is stripped and one newline appended.

    RL concept:
        Evaluation and governance -- reproducible narrative evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def missing_required_artifacts(
    *,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> list[str]:
    """Return the required artifact paths that do not exist under ``output_dir``.

    What + why: this is the first gate ``scripts/verify_artifacts.py`` applies -- presence before
    shape. A run is incomplete if any required relative path is missing on disk.

    Args:
        output_dir: Project root that should contain the ``artifacts/`` tree.
        manifest_path: Optional frozen contract; when given and present, its required list is used
            instead of :data:`REQUIRED_ARTIFACTS`.

    Returns:
        The subset of required relative paths absent from ``output_dir`` (empty if complete).

    RL concept:
        Evaluation and governance -- completeness check before validation.
    """
    required_files = required_artifacts(manifest_path)
    return [
        relative_path
        for relative_path in required_files
        if not (output_dir / relative_path).exists()
    ]


def required_artifacts(manifest_path: Path | None = None) -> list[str]:
    """Resolve the required-artifact list from a manifest, or fall back to the built-in tuple.

    What + why: lets the verifier validate against either a version-pinned manifest (reproducible
    audits) or the live in-code contract (developer convenience) using the same code path.

    Args:
        manifest_path: Optional manifest JSON. Used only if it both is not ``None`` and exists on
            disk; otherwise :data:`REQUIRED_ARTIFACTS` is returned.

    Returns:
        The list of required relative artifact paths.

    RL concept:
        Evaluation and governance -- a single source of truth for the evidence contract.
    """
    if manifest_path is not None and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(item) for item in payload["required_files"]]
    return list(REQUIRED_ARTIFACTS)


def optional_artifacts(manifest_path: Path | None = None) -> list[str]:
    """Resolve the optional deep-RL artifact list from a manifest, or use the built-in tuple.

    What + why: mirror of :func:`required_artifacts` for the optional DQN/PPO comparison group.
    Reads the ``optional_files`` key defensively (defaults to empty) so older manifests without that
    key still load.

    Args:
        manifest_path: Optional manifest JSON. Used only if it both is not ``None`` and exists on
            disk; otherwise :data:`OPTIONAL_DRL_ARTIFACTS` is returned.

    Returns:
        The list of optional relative artifact paths.

    RL concept:
        Evaluation and governance -- gating the optional deep-RL bridge.
    """
    if manifest_path is not None and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(item) for item in payload.get("optional_files", [])]
    return list(OPTIONAL_DRL_ARTIFACTS)


def artifact_validation_errors(
    *,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> list[str]:
    """Validate the *shape* of every present artifact and return human-readable errors.

    What + why: runs after the presence check. Each existing required artifact is validated for its
    contract (required CSV columns / required markdown headings and terms). The optional deep-RL
    group is all-or-nothing -- if *any* comparison output beyond the bridge report is present, then
    the entire :data:`OPTIONAL_DRL_ARTIFACTS` group is required and validated; otherwise a lone
    ``bridge_report.md`` is validated on its own. Missing *required* files are skipped here because
    :func:`missing_required_artifacts` already reports them.

    Args:
        output_dir: Project root containing the ``artifacts/`` tree.
        manifest_path: Optional frozen contract used to resolve the required/optional lists.

    Returns:
        A list of validation error messages; empty means every present artifact is well-formed.

    RL concept:
        Evaluation and governance -- structural validation of the evidence contract.
    """
    errors: list[str] = []
    required_files = required_artifacts(manifest_path)

    for relative_path in required_files:
        artifact_path = output_dir / relative_path
        if not artifact_path.exists():
            continue
        errors.extend(_validate_required_artifact(relative_path, artifact_path))

    optional_files = optional_artifacts(manifest_path)
    optional_paths = {relative_path: output_dir / relative_path for relative_path in optional_files}
    present_optional = {
        relative_path: artifact_path
        for relative_path, artifact_path in optional_paths.items()
        if artifact_path.exists()
    }
    # All-or-nothing rule: any optional file *other than* the standalone bridge report signals an
    # intent to ship the full DQN/PPO comparison, which then requires the whole group.
    has_drl_comparison = any(
        relative_path != "artifacts/drl_optional/bridge_report.md"
        for relative_path in present_optional
    )
    if has_drl_comparison:
        for relative_path, artifact_path in optional_paths.items():
            if not artifact_path.exists():
                errors.append(
                    "Optional DRL comparison output is incomplete: "
                    f"missing {relative_path}."
                )
                continue
            errors.extend(_validate_optional_drl_artifact(relative_path, artifact_path))
    elif "artifacts/drl_optional/bridge_report.md" in present_optional:
        errors.extend(
            _validate_optional_drl_artifact(
                "artifacts/drl_optional/bridge_report.md",
                present_optional["artifacts/drl_optional/bridge_report.md"],
            )
        )
    return errors


def mdp_spec_markdown() -> str:
    """Render the human-readable specification of the agent-decision MDP.

    What + why: produces the ``artifacts/concepts/mdp_spec.md`` body -- the (state, action,
    transition, horizon, reward) tuple that defines the finite-horizon MDP every learner in the
    showcase operates on, plus the bridge narrative from tabular Bellman control up to DQN and PPO.
    This is the "bottom of the ladder made explicit": naming the agent-orchestration MDP that the
    bandit warm-up, Q-learning, SARSA, REINFORCE, and the optional deep-RL methods all share.

    Returns:
        Markdown text beginning with a top-level heading (satisfies the markdown validator).

    RL concept:
        MDP framing -- states, actions, transitions, finite horizon, reward. The novelty is that
        the *agent's own decision loop* is the MDP.

    Math:
        The reward after acting at step t is R_{t+1}; the discounted return is
        G_t = sum_k gamma^k R_{t+k+1}, and control methods target
        Q*(s,a) = E[R_{t+1} + gamma * max_a' Q*(s',a')].
    """
    return (
        "# Agent Decision MDP\n\n"
        "## Core MDP Elements\n\n"
        "- State: step, intent, difficulty, ambiguity, evidence, attempts, and remaining budget.\n"
        "- Actions: answer directly, retrieve more evidence, ask a clarifying question, "
        "or escalate to a human.\n"
        "- Transition: a small deterministic teaching simulator where retrieve adds grounding, "
        "clarify resolves ambiguity, and answer/escalate are terminal commit actions.\n"
        "- Horizon: 5 orchestration decisions, with a hard cost budget.\n"
        "- Policy set: random, heuristic router, always-escalate, tabular Q-learning, SARSA, "
        "tabular REINFORCE, optional DQN, and optional PPO.\n"
        "- Reward: a judge rubric that scores answer quality, grounding, cost, and safety, "
        "compared against a hackable proxy reward.\n\n"
        "## Bellman And Evaluation Bridge\n\n"
        "- Q-learning updates action values with the Bellman target.\n"
        "- SARSA learns on-policy, bootstrapping from the action actually taken next.\n"
        "- Dynamic programming (backward induction) computes the exact optimal Q* for this "
        "finite-horizon MDP as a ground-truth baseline for Q-learning.\n"
        "- DQN reuses the same control framing with a neural value approximator.\n"
        "- PPO adds an actor-critic, policy-gradient baseline on the same environment family.\n"
        "- Offline evaluation checks reward, escalation rate, under-grounded answers, and "
        "action cost before any rollout recommendation.\n"
    )


def algorithm_progression_markdown() -> str:
    """Render the RL-to-DRL ladder narrative for the showcase.

    What + why: produces ``artifacts/concepts/algorithm_progression.md`` -- the ordered story from
    contextual bandit -> Q-learning -> dynamic programming -> SARSA -> DQN -> policy gradients ->
    actor-critic -> PPO, with pointers to the artifact that lets a reader inspect each rung. The
    validator requires the terms "q-learning", "dqn", "policy gradients", "actor-critic", and "ppo"
    to appear, so the rungs are named explicitly in the prose.

    Returns:
        Markdown text beginning with a top-level heading and naming each ladder rung.

    RL concept:
        The algorithm ladder -- value-based to policy-based to actor-critic.
    """
    return (
        "# RL To DRL Progression\n\n"
        "## The ladder in this showcase\n\n"
        "1. **Contextual bandit**: make one decision from the request features, learn which "
        "action works best for that context, and measure regret.\n"
        "2. **Tabular Q-learning**: move from one-step decisions to a multi-step MDP and learn "
        "action values with the Bellman update.\n"
        "3. **Dynamic programming**: solve the known finite-horizon MDP exactly by backward "
        "induction to get the optimal Q*, the ground truth Q-learning is compared against.\n"
        "4. **SARSA**: on-policy TD control that bootstraps from the action actually taken next "
        "instead of the greedy max.\n"
        "5. **DQN**: keep the value-learning idea from Q-learning, but replace the Q-table with "
        "a neural network so the agent can work from continuous observations.\n"
        "6. **Policy gradients**: optimize a policy directly instead of learning values first.\n"
        "7. **Actor-critic**: combine a policy learner (actor) with a value estimator (critic) "
        "to reduce variance and stabilize training.\n"
        "8. **PPO**: use an actor-critic policy-gradient method with clipped updates so the "
        "policy improves more steadily.\n\n"
        "## How to inspect each step\n\n"
        "- Contextual bandit: `artifacts/bandit/reward_trace.csv` and "
        "`artifacts/bandit/regret_trace.csv`\n"
        "- Q-learning: `artifacts/q_learning/training_curve.csv` and "
        "`artifacts/q_learning/q_table.csv`\n"
        "- Dynamic programming ground truth: `artifacts/dp/optimal_action_values.csv` and "
        "`artifacts/dp/q_learning_gap.csv`\n"
        "- SARSA (on-policy): `artifacts/sarsa/training_curve.csv`\n"
        "- Policy gradient (REINFORCE): `artifacts/policy_gradient/training_curve.csv`\n"
        "- DQN vs PPO vs Q-learning: `artifacts/drl_optional/rl_family_comparison.csv`, "
        "`artifacts/drl_optional/scenario_rollups.csv`, and "
        "`artifacts/drl_optional/training_summary.csv`\n"
        "- Policy-gradient and actor-critic notes: "
        "`artifacts/drl_optional/policy_gradient_notes.md`\n"
    )


def concept_map_rows() -> list[dict[str, str]]:
    """Build the concept-to-component-to-artifact crosswalk rows.

    What + why: each row links one RL concept (e.g. exploration/exploitation, the Bellman update,
    on-policy vs off-policy, policy gradients, governance) to the showcase component that implements
    it and the artifact that evidences it. Serialized to ``artifacts/concepts/concept_map.csv``,
    whose required columns are ``concept``, ``showcase_component``, and ``artifact`` -- exactly the
    keys produced here. This is the table-of-contents for the whole ladder.

    Returns:
        A list of dicts, each with ``concept``, ``showcase_component``, and ``artifact`` keys.

    RL concept:
        Curriculum map across the ladder -- bandit to MDP to value/policy methods to deep RL.
    """
    return [
        {
            "concept": "agent_environment_loop",
            "showcase_component": "AgentDecisionEnvironment.reset/step/observe/is_done",
            "artifact": "artifacts/mdp/sample_episodes.csv",
        },
        {
            "concept": "mdp_framing",
            "showcase_component": "mdp_spec_markdown",
            "artifact": "artifacts/concepts/mdp_spec.md",
        },
        {
            "concept": "exploration_vs_exploitation",
            "showcase_component": "contextual epsilon-greedy bandit warm-up",
            "artifact": "artifacts/bandit/regret_trace.csv",
        },
        {
            "concept": "bellman_and_q_learning",
            "showcase_component": "train_q_learning",
            "artifact": "artifacts/q_learning/training_curve.csv",
        },
        {
            "concept": "dynamic_programming_ground_truth",
            "showcase_component": "optimal_action_values (backward induction)",
            "artifact": "artifacts/dp/q_learning_gap.csv",
        },
        {
            "concept": "on_policy_vs_off_policy",
            "showcase_component": "train_sarsa versus train_q_learning",
            "artifact": "artifacts/sarsa/training_curve.csv",
        },
        {
            "concept": "policy_gradient_reinforce",
            "showcase_component": "train_reinforce (tabular softmax)",
            "artifact": "artifacts/policy_gradient/training_curve.csv",
        },
        {
            "concept": "deep_rl_comparison",
            "showcase_component": "run_drl_comparison",
            "artifact": "artifacts/drl_optional/rl_family_comparison.csv",
        },
        {
            "concept": "policy_gradient_actor_critic",
            "showcase_component": "PPO policy-gradient notes",
            "artifact": "artifacts/drl_optional/policy_gradient_notes.md",
        },
        {
            "concept": "reward_design",
            "showcase_component": "judge_reward versus hackable_reward",
            "artifact": "artifacts/reward/reward_hacking_report.md",
        },
        {
            "concept": "offline_evaluation",
            "showcase_component": "fixed scenario policy comparison",
            "artifact": "artifacts/eval/policy_comparison.csv",
        },
        {
            "concept": "governance",
            "showcase_component": "deploy-shadow-reject framing",
            "artifact": "artifacts/business/deploy_shadow_reject_memo.md",
        },
    ]


def undergrounded_answer_rate(
    answered_states: Sequence[AgentState],
) -> float:
    """Compute the fraction of committed direct answers that were *not* adequately grounded.

    What + why: this is the metric that fills the ``avg_undergrounded_rate`` column in
    ``artifacts/eval/policy_comparison.csv`` and feeds the under-grounding safety gate in
    :func:`recommendation_from_summary`. It is defined directly in terms of the reward module's
    quality predicate :func:`~learning_agents.reward.evidence_is_adequate`, so the *evidence* the
    governance layer reports cannot drift from the *objective* the agent was trained on: an answer
    counts as under-grounded exactly when the judge rubric would have penalized it for missing
    grounding. Centralizing it here keeps the evaluation runner, the CSV column, and the rollout
    gate on one shared definition.

    Args:
        answered_states: The states at which a policy committed an ``answer_direct`` action across
            the evaluation episodes (one entry per direct answer). An empty sequence means the
            policy never answered directly.

    Returns:
        The proportion of those answers that were under-grounded (evidence below the difficulty
        threshold), in [0, 1]; ``0.0`` when ``answered_states`` is empty.

    RL concept:
        Evaluation and governance -- a safety metric anchored to the reward's quality predicate, so
        reward hacking via under-grounded answers is measured the same way it is penalized.
    """
    if not answered_states:
        return 0.0
    undergrounded = sum(
        1
        for state in answered_states
        if not evidence_is_adequate(evidence=state.evidence, difficulty=state.difficulty)
    )
    return round(undergrounded / len(answered_states), 4)


def governance_artifacts() -> dict[str, str]:
    """Build the static governance narrative artifacts: safety controls and the offline-eval plan.

    What + why: returns the markdown bodies written to
    ``artifacts/governance/safety_controls.md`` and ``artifacts/governance/offline_eval_plan.md``.
    These encode the human-oversight and offline-gating discipline that wraps any RL deployment, and
    they articulate the same over-escalation / under-grounding concern that
    :func:`recommendation_from_summary` enforces numerically. The deploy/shadow/reject memo itself
    is deliberately NOT here: it is run-specific (its verdict depends on the offline-eval summary),
    so it is generated dynamically from :func:`recommendation_from_summary` by the runner scripts
    rather than hard-coded to a fixed recommendation that could contradict the live numbers.

    Returns:
        A mapping with keys ``safety_controls`` and ``offline_eval_plan``, each a markdown string
        beginning with a top-level heading.

    RL concept:
        Evaluation and governance -- safety controls and the offline-evaluation gate that precedes
        any deploy/shadow/reject decision.
    """
    return {
        "safety_controls": (
            "# Safety Controls\n\n"
            "- Use synthetic requests only.\n"
            "- Keep human escalation reviewed by a person in any real deployment.\n"
            "- Run offline evaluation before shadow or live rollout.\n"
        ),
        "offline_eval_plan": (
            "# Offline Evaluation Plan\n\n"
            "1. Hold scenarios fixed across policies.\n"
            "2. Compare reward, escalation rate, under-grounded-answer rate, and action cost.\n"
            "3. Reject any policy that improves reward only by over-escalating or by returning "
            "under-grounded answers.\n"
        ),
    }


def recommendation_from_summary(
    summary_rows: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    """Decide deploy/shadow/reject for the learned policy from offline-evaluation summaries.

    What + why: implements the showcase's rollout gate -- the practical payoff of putting evaluation
    and governance at the top of the ladder. Intuition: a learned policy earns a guarded rollout
    only if it is both *safe enough* and *better than the incumbent heuristic router*; anything else
    is rejected. The rule, applied to the ``q_learning`` row using ``heuristic_router`` as baseline:

    1. Reject if it is unsafe -- ``avg_escalation_rate > 0.5`` (over-escalation / reward hacking) or
       ``avg_undergrounded_rate > 0.5`` (too many hallucination-risk answers).
    2. Otherwise shadow if it beats the heuristic router -- ``avg_reward > baseline_reward`` while
       staying within both safety bounds.
    3. Otherwise reject for an insufficient margin over the heuristic router.

    The escalation/under-grounding gates are what prevent accepting a reward-hacking policy that
    inflates reward by always-escalating or by answering without adequate grounding (the failure
    modes the reward-design comparison demonstrates).

    Args:
        summary_rows: Per-policy offline-eval summary mappings. Must include rows whose ``policy``
            is ``"q_learning"`` and ``"heuristic_router"``, each carrying ``avg_reward``,
            ``avg_escalation_rate``, and ``avg_undergrounded_rate`` fields.

    Returns:
        A ``(decision, rationale)`` tuple where ``decision`` is one of ``"shadow"`` or ``"reject"``.
        Returns ``("reject", ...)`` if either required policy row is absent.

    RL concept:
        Evaluation and governance -- the deploy/shadow/reject decision rule, with a guard against
        reward hacking (over-escalation and under-grounded answers).
    """
    by_policy = {str(row["policy"]): row for row in summary_rows}
    q_learning = by_policy.get("q_learning")
    baseline = by_policy.get("heuristic_router")
    if q_learning is None or baseline is None:
        return ("reject", "Missing comparable heuristic-router or learned-policy evidence.")

    avg_reward = _coerce_float(q_learning["avg_reward"])
    baseline_reward = _coerce_float(baseline["avg_reward"])
    escalation_rate = _coerce_float(q_learning["avg_escalation_rate"])
    undergrounded_rate = _coerce_float(q_learning["avg_undergrounded_rate"])

    # Safety gate: reject reward-hacking policies (over-escalation or under-grounded answers) before
    # comparing reward, so a high-reward-but-gaming policy never reaches a rollout recommendation.
    if escalation_rate > 0.5 or undergrounded_rate > 0.5:
        return (
            "reject",
            "The learned policy still over-escalates or returns too many under-grounded answers.",
        )
    # Shadow only if the learned policy beats the heuristic-router baseline (within safety bounds).
    if avg_reward > baseline_reward:
        return (
            "shadow",
            "The learned policy beats the heuristic router offline, but still needs "
            "guarded rollout.",
        )
    return (
        "reject",
        "The learned policy does not clearly outperform the heuristic router with enough margin.",
    )


def _coerce_float(value: object) -> float:
    """Coerce a numeric-or-string artifact value to ``float``, rejecting other types.

    What + why: summary rows may arrive with values typed as ``int``/``float`` (in-memory) or
    ``str`` (round-tripped through CSV), so both are accepted; anything else is a programming error
    and raises rather than silently mis-parsing.

    Args:
        value: The value to coerce.

    Returns:
        The value as a ``float``.

    Raises:
        TypeError: If ``value`` is not an ``int``, ``float``, or ``str``.

    RL concept:
        Evaluation and governance -- robust parsing of round-tripped evaluation summaries.
    """
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected int, float, or str value, got {type(value)!r}")


def _validate_required_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Dispatch a required artifact to the CSV or markdown validator by file extension.

    Args:
        relative_path: The artifact's contract-relative path (also selects its rules).
        artifact_path: Resolved path to the file on disk.

    Returns:
        Validation error messages for this artifact (empty if it passes or is neither ``.csv`` nor
        ``.md``).

    RL concept:
        Evaluation and governance -- per-artifact contract enforcement.
    """
    if relative_path.endswith(".csv"):
        return _validate_csv_artifact(relative_path, artifact_path)
    if relative_path.endswith(".md"):
        return _validate_markdown_artifact(relative_path, artifact_path)
    return []


def _validate_optional_drl_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Validate an optional deep-RL artifact, adding DQN/PPO-specific content checks.

    What + why: beyond the generic CSV/markdown checks, the optional group asserts that the
    comparison actually contrasts the three families that motivate the deep-RL bridge. The family
    comparison CSV must contain ``q_learning``, ``dqn``, and ``ppo`` rows; the bridge report must
    name both DQN and PPO; and the policy-gradient notes must mention the defining terms
    ("policy-gradient", "actor-critic", "ppo", "dqn").

    Args:
        relative_path: The artifact's contract-relative path (also selects its rules).
        artifact_path: Resolved path to the file on disk.

    Returns:
        Validation error messages for this artifact (empty if it passes).

    RL concept:
        Deep RL bridge -- value-based DQN vs actor-critic PPO vs tabular Q-learning.
    """
    if relative_path.endswith(".csv"):
        errors = _validate_csv_artifact(relative_path, artifact_path)
        if relative_path == "artifacts/drl_optional/rl_family_comparison.csv":
            with artifact_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            policies = {row["policy"] for row in rows}
            # Contract: the comparison must contrast all three families head-to-head.
            if {"q_learning", "dqn", "ppo"} - policies:
                errors.append(
                    "Optional DRL comparison must include q_learning, dqn, and ppo rows."
                )
        return errors
    if relative_path.endswith(".md"):
        errors = _validate_markdown_artifact(relative_path, artifact_path)
        content = artifact_path.read_text(encoding="utf-8")
        if relative_path == "artifacts/drl_optional/bridge_report.md":
            if "DQN" not in content or "PPO" not in content:
                errors.append("Optional DRL bridge report must mention both DQN and PPO.")
        if relative_path == "artifacts/drl_optional/policy_gradient_notes.md":
            # Required terms ensure the notes actually cover the policy-gradient/actor-critic arc.
            required_terms = ("policy-gradient", "actor-critic", "ppo", "dqn")
            lowered_content = content.lower()
            missing_terms = [term for term in required_terms if term not in lowered_content]
            if missing_terms:
                errors.append(
                    "Policy-gradient notes are missing required terms: "
                    + ", ".join(missing_terms)
                    + "."
                )
        return errors
    return []


def _validate_csv_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Validate a CSV artifact: required header columns present and at least one data row.

    What + why: the per-path :data:`_REQUIRED_CSV_COLUMNS` table *is* the columnar contract for
    every tabular artifact (bandit traces, training curves, Q-tables, DP gap, policy comparisons,
    the deep-RL rollups). A file passes only if its header is non-empty, contains every required
    column for its path, and carries at least one data row. ``scripts/verify_artifacts.py`` and
    ``tests/test_reporting.py`` rely on exactly these rules.

    Args:
        relative_path: The artifact's contract-relative path; selects which columns are required.
        artifact_path: Path to the CSV file on disk.

    Returns:
        Validation error messages (empty if the CSV satisfies its contract).

    RL concept:
        Evaluation and governance -- machine-checkable columnar evidence.
    """
    errors: list[str] = []
    with artifact_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [f"{relative_path} is empty or missing a header row."]
        missing_columns = [
            column
            for column in _REQUIRED_CSV_COLUMNS.get(relative_path, ())
            if column not in reader.fieldnames
        ]
        if missing_columns:
            errors.append(
                f"{relative_path} is missing required columns: {', '.join(missing_columns)}."
            )
        rows = list(reader)
        if not rows:
            errors.append(f"{relative_path} must contain at least one data row.")
        errors.extend(_validate_action_labels(relative_path, rows))
    return errors


# Artifacts that carry a single action label in a known column: the value must be one the
# environment actually defines, so a report can never reference a non-existent action.
_ACTION_LABEL_COLUMN: dict[str, str] = {
    "artifacts/mdp/sample_episodes.csv": "action_label",
    "artifacts/eval/scenario_results.csv": "final_action_label",
    "artifacts/sdk_bridge/orchestration_trace.csv": "action_label",
}


def _validate_action_labels(
    relative_path: str,
    rows: Sequence[Mapping[str, str]],
) -> list[str]:
    """Check that action-label columns only ever name actions the environment defines.

    What + why: the episode and scenario-result artifacts record which orchestration action was
    taken. Because the legal set is owned by :data:`~learning_agents.environment.ACTION_LABELS`,
    validating against it here ties the *evidence* back to the *live action space* -- a report that
    mentions an action the MDP cannot take is malformed by definition. Paths without an action-label
    column are skipped.

    Args:
        relative_path: The artifact's contract-relative path; selects the action-label column (if
            any) to validate.
        rows: The already-parsed CSV data rows.

    Returns:
        One error per row whose action-label value is not in
        :data:`~learning_agents.environment.ACTION_LABELS` (empty if all are legal or the path has
        no action-label column).

    RL concept:
        Evaluation and governance -- evidence must reference only the MDP's real action space.
    """
    column = _ACTION_LABEL_COLUMN.get(relative_path)
    if column is None:
        return []
    errors: list[str] = []
    for row in rows:
        label = row.get(column)
        if label is not None and label not in _ACTION_LABEL_SET:
            errors.append(
                f"{relative_path} has an unknown action label {label!r} in column {column!r}."
            )
    return errors


def _validate_markdown_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Validate a markdown artifact: non-empty, heading-led, and (for the ladder) rung-complete.

    What + why: generic rule -- the file must be non-empty and start with a ``#`` heading.
    Path-specific rule -- ``artifacts/concepts/algorithm_progression.md`` must mention each ladder
    rung ("q-learning", "dqn", "policy gradients", "actor-critic", and "ppo") so the progression
    narrative cannot silently skip a method.

    Args:
        relative_path: The artifact's contract-relative path; selects path-specific rules.
        artifact_path: Path to the markdown file on disk.

    Returns:
        Validation error messages (empty if the markdown satisfies its contract).

    RL concept:
        Evaluation and governance -- narrative evidence that covers the full ladder.
    """
    content = artifact_path.read_text(encoding="utf-8").strip()
    if not content:
        return [f"{relative_path} is empty."]
    if relative_path == "artifacts/concepts/algorithm_progression.md":
        # Contract: the ladder narrative must name every rung (no silently-skipped method).
        required_terms = ("q-learning", "dqn", "policy gradients", "actor-critic", "ppo")
        lowered_content = content.lower()
        missing_terms = [term for term in required_terms if term not in lowered_content]
        if missing_terms:
            return [
                "artifacts/concepts/algorithm_progression.md is missing: "
                + ", ".join(missing_terms)
                + "."
            ]
    if not content.startswith("#"):
        return [f"{relative_path} should start with a Markdown heading."]
    return []
