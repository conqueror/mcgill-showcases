"""Artifact contract, text generation, and simple verification helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from adaptive_course_assistant_rl.agent_bridge import (
    action_mapping_markdown,
    policy_router_payload,
)
from adaptive_course_assistant_rl.environment import (
    ACTION_LABELS,
    BANDIT_ACTIONS,
    STATE_FIELD_NAMES,
)

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "artifacts/manifest.json",
    "artifacts/concepts/mdp_spec.md",
    "artifacts/concepts/algorithm_progression.md",
    "artifacts/concepts/state_action_reward_schema.csv",
    "artifacts/concepts/agentic_rl_bridge.md",
    "artifacts/concepts/interpretation_prompts.md",
    "artifacts/assistant/episode_trace.json",
    "artifacts/assistant/resource_matches.csv",
    "artifacts/bandit/contextual_policy_metrics.csv",
    "artifacts/bandit/regret_trace.csv",
    "artifacts/bandit/action_breakdown.csv",
    "artifacts/mdp/sample_episodes.csv",
    "artifacts/policy/rule_policy_summary.csv",
    "artifacts/policy/intervention_decisions.csv",
    "artifacts/q_learning/training_curve.csv",
    "artifacts/q_learning/q_table.csv",
    "artifacts/sarsa/training_curve.csv",
    "artifacts/policy_gradient/training_curve.csv",
    "artifacts/reward/reward_hacking_report.md",
    "artifacts/reward/reward_spec_bad.md",
    "artifacts/reward/reward_spec_good.md",
    "artifacts/eval/offline_policy_eval.csv",
    "artifacts/eval/scenario_rollups.csv",
    "artifacts/eval/safety_summary.md",
    "artifacts/bridge/policy_router.json",
    "artifacts/bridge/action_mapping.md",
    "artifacts/bridge/learning_agent_story.md",
    "artifacts/business/deployment_recommendation.md",
)

OPTIONAL_DRL_ARTIFACTS: tuple[str, ...] = (
    "artifacts/drl_optional/dqn_training_summary.csv",
    "artifacts/drl_optional/ppo_training_summary.csv",
    "artifacts/drl_optional/rl_family_comparison.csv",
    "artifacts/drl_optional/scenario_rollups.csv",
    "artifacts/drl_optional/policy_gradient_bridge_notes.md",
)


def write_manifest(
    path: Path,
    required_files: Sequence[str] = REQUIRED_ARTIFACTS,
    optional_files: Sequence[str] = OPTIONAL_DRL_ARTIFACTS,
) -> None:
    """Write the JSON artifact manifest."""
    write_json_artifact(
        path,
        {"version": 1, "required_files": list(required_files), "optional_files": list(optional_files)},
    )


def write_csv_artifact(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a list of dictionaries as a CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON artifact with deterministic key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text_artifact(path: Path, content: str) -> None:
    """Write normalized markdown or plain-text artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def required_artifacts(manifest_path: Path | None = None) -> list[str]:
    """Resolve the required artifact list from a manifest or the in-code contract."""
    if manifest_path is not None and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(item) for item in payload["required_files"]]
    return list(REQUIRED_ARTIFACTS)


def optional_artifacts(manifest_path: Path | None = None) -> list[str]:
    """Resolve the optional artifact list from a manifest or the in-code contract."""
    if manifest_path is not None and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(item) for item in payload.get("optional_files", [])]
    return list(OPTIONAL_DRL_ARTIFACTS)


def missing_required_artifacts(*, output_dir: Path, manifest_path: Path | None = None) -> list[str]:
    """Return required artifacts missing from ``output_dir``."""
    return [
        relative_path
        for relative_path in required_artifacts(manifest_path)
        if not (output_dir / relative_path).exists()
    ]


def missing_optional_artifacts(*, output_dir: Path, manifest_path: Path | None = None) -> list[str]:
    """Return optional DRL artifacts missing from ``output_dir``."""
    return [
        relative_path
        for relative_path in optional_artifacts(manifest_path)
        if not (output_dir / relative_path).exists()
    ]


def optional_drl_validation_errors(*, artifacts_dir: Path) -> list[str]:
    """Validate the optional DRL bundle relative to the artifact directory itself."""
    errors: list[str] = []
    for relative_path in OPTIONAL_DRL_ARTIFACTS:
        artifact_path = artifacts_dir / Path(relative_path).relative_to("artifacts")
        if not artifact_path.exists():
            errors.append(f"Required optional DRL artifact is missing: {relative_path}.")
            continue
        errors.extend(_validate_optional_artifact(relative_path, artifact_path))
    return errors


def artifact_validation_errors(
    *,
    output_dir: Path,
    manifest_path: Path | None = None,
    require_optional_drl: bool = False,
) -> list[str]:
    """Run light structural validation over present artifacts."""
    errors: list[str] = []
    for relative_path in required_artifacts(manifest_path):
        artifact_path = output_dir / relative_path
        if not artifact_path.exists():
            continue
        errors.extend(_validate_required_artifact(relative_path, artifact_path))

    optional_paths = {path: output_dir / path for path in optional_artifacts(manifest_path)}
    present_optional = {path: artifact for path, artifact in optional_paths.items() if artifact.exists()}
    if require_optional_drl:
        for path, artifact in optional_paths.items():
            if not artifact.exists():
                errors.append(f"Required optional DRL artifact is missing: {path}.")
                continue
            errors.extend(_validate_optional_artifact(path, artifact))
        return errors

    if present_optional:
        for path, artifact in optional_paths.items():
            if not artifact.exists():
                errors.append(f"Optional DRL artifact group is incomplete: missing {path}.")
                continue
            errors.extend(_validate_optional_artifact(path, artifact))
    return errors


def state_action_reward_rows() -> list[dict[str, object]]:
    """Render the state/action/reward schema as CSV rows."""
    state_notes = {
        "intent_type": "What the student is trying to do; helps choose concept, debug, study, or exam support.",
        "difficulty_level": "How hard the content is; harder states usually need grounding or escalation sooner.",
        "confidence_level": "How confident the student seems; low confidence favors hints, examples, or rephrasing.",
        "misconception_type": "What kind of misunderstanding is present; drives hint vs worked-example choices.",
        "retrieval_quality": "How strong the course evidence is; poor retrieval makes hints/examples risky.",
        "intent_uncertainty": "How unclear the request is; high uncertainty should favor clarification first.",
        "cognitive_load": "How overloaded the student is; high load favors slowing down and rephrasing.",
        "turn_index": "Where we are in the short tutoring episode; late risky states should escalate faster.",
        "attempt_count": "How many intervention attempts have already happened; repeated attempts raise risk.",
        "last_action": "The previous intervention; used to penalize churn between unrelated actions.",
        "safety_risk": "Current need for caution or escalation; high risk should discourage normal tutoring loops.",
        "resolved_flag": "Whether the issue is solved; terminal success requires enough grounding and certainty.",
    }
    action_notes = {
        0: "Use when intent is unclear before giving content help.",
        1: "Use when the next answer needs stronger course grounding.",
        2: "Use when evidence is adequate and the student needs a small nudge.",
        3: "Use when a deeper misconception needs a fuller demonstration.",
        4: "Use when the student is ready to practice a targeted next step.",
        5: "Use to verify that apparent progress is real.",
        6: "Use when cognitive load is high or the explanation needs simplification.",
        7: "Use when the state remains risky or outside the assistant's safe scope.",
    }
    rows: list[dict[str, object]] = [
        {
            "category": "state_field",
            "name": field,
            "notes": state_notes[field],
        }
        for field in STATE_FIELD_NAMES
    ]
    rows.extend(
        {
            "category": "action",
            "name": ACTION_LABELS[action],
            "notes": action_notes[action],
        }
        for action in ACTION_LABELS
    )
    rows.extend(
        [
            {"category": "reward", "name": "resolved_with_grounding", "notes": "Reward grounded resolution strongly."},
            {"category": "reward", "name": "extra_turn_penalty", "notes": "Charge for dragging the conversation out."},
            {"category": "reward", "name": "ungrounded_penalty", "notes": "Penalize hint/example actions with poor retrieval."},
            {"category": "reward", "name": "missed_escalation_penalty", "notes": "Penalize unsafe failure to escalate."},
            {
                "category": "reward_scope",
                "name": "bandit_first_turn_reward",
                "notes": "The contextual bandit uses a one-step Bernoulli reward model for first-intervention teaching.",
            },
            {
                "category": "reward_scope",
                "name": "mdp_multi_turn_reward",
                "notes": "Q-learning, SARSA, REINFORCE, DQN, and PPO use the multi-turn MDP reward.",
            },
        ]
    )
    return rows


def mdp_spec_markdown() -> str:
    """Describe the tutoring MDP in plain language."""
    return (
        "# Adaptive Tutoring MDP\n\n"
        "The deterministic assistant already knows what kind of question it is looking at. The RL layer decides the next intervention.\n\n"
        "## State\n\n"
        "- Intent, difficulty, confidence, misconception type, retrieval quality, uncertainty, cognitive load, turn index, attempt count, last action, safety risk, and whether the issue is already resolved.\n\n"
        "## Actions\n\n"
        "- Ask a clarifying question.\n"
        "- Retrieve a course note.\n"
        "- Give a hint.\n"
        "- Give a worked example.\n"
        "- Assign targeted practice.\n"
        "- Check understanding.\n"
        "- Slow down and rephrase.\n"
        "- Escalate to a human.\n\n"
        "## Horizon\n\n"
        "- Five decisions per episode.\n\n"
        "## Reward\n\n"
        "- Reward grounded resolution.\n"
        "- Reward safer, lower-risk states.\n"
        "- Penalize extra turns, ungrounded help, intervention churn, and missed escalation.\n"
        "- Keep the contextual-bandit reward separate: it teaches first-turn action selection with a one-step Bernoulli reward, while the MDP algorithms learn from the multi-turn reward above.\n"
    )


def algorithm_progression_markdown() -> str:
    """Explain the algorithm ladder in a student-friendly way."""
    return (
        "# Algorithm Progression\n\n"
        "| Method | What is learned | Representation | Update signal | Artifact to inspect | Common misread |\n"
        "|---|---|---|---|---|---|\n"
        "| Rule-based policy | Nothing; it is hand-written | If/then rules | Human design | `artifacts/policy/rule_policy_summary.csv` | Treating it as trained |\n"
        "| Contextual bandit | First intervention only | Linear reward estimates per action | Immediate Bernoulli reward | `artifacts/bandit/contextual_policy_metrics.csv` | Assuming it models future turns |\n"
        "| Q-learning | Action values | Tabular Q(s,a) | Off-policy Bellman backup | `artifacts/q_learning/q_table.csv` | Assuming the table writes responses |\n"
        "| SARSA | Action values under the behavior policy | Tabular Q(s,a) | On-policy TD backup | `artifacts/sarsa/training_curve.csv` | Missing the on-policy contrast |\n"
        "| REINFORCE | Action preferences | Tabular logits | Monte Carlo return | `artifacts/policy_gradient/training_curve.csv` | Expecting a critic to exist |\n"
        "| DQN | Action values | Neural approximation of Q(s,a) | Q-learning-style TD loss | `artifacts/drl_optional/dqn_training_summary.csv` | Treating it as a different objective |\n"
        "| PPO | Policy plus critic | Actor-critic neural policy | Clipped policy-gradient update | `artifacts/drl_optional/ppo_training_summary.csv` | Treating PPO as magic deployment proof |\n\n"
        "Actor-critic is represented here through PPO rather than a separate standalone training artifact. "
        "The table is the important learning path: tabular value learning -> value learning with function "
        "approximation -> direct policy optimization -> actor-critic policy optimization.\n"
    )


def agentic_rl_bridge_markdown() -> str:
    """Describe how the RL layer fits around the deterministic assistant."""
    return (
        "# Agentic RL Bridge\n\n"
        "This project does not train the assistant to write answers from scratch. It trains a smaller controller that chooses the next intervention after the deterministic assistant has already understood the request.\n\n"
        "That boundary matters. It keeps the offline simulation readable, and it makes the export story honest: a learned policy can plug back into a deterministic assistant without turning the whole assistant into a black box.\n\n"
        "The exported `policy_router.json` is therefore an assistant-side routing contract. It does not serialize learned weights or claim to be a deployable champion model by itself.\n"
    )


def interpretation_prompts_markdown() -> str:
    """Give students concrete prompts for reading the artifacts."""
    return (
        "# Interpretation Prompts\n\n"
        "- Which scenarios make the rule-based policy look strong, and which ones expose its limits?\n"
        "- Does the contextual bandit choose different first actions for different tutoring contexts?\n"
        "- Where does Q-learning improve solved rate without simply spending more intervention cost?\n"
        "- Does the optional DRL bridge actually beat the tabular baseline on the same scenarios?\n"
        "- Which bad reward behavior would you miss if you only looked at average reward?\n"
    )


def safety_summary_markdown(summary_rows: Sequence[dict[str, int | float | str]]) -> str:
    """Turn summary rows into a short safety/governance note."""
    lines = [
        (
            f"- `{row['policy']}`: reward={row['avg_reward']}, solved_rate={row['solved_rate']}, "
            f"final_safety_risk={row['avg_final_safety_risk']}, "
            f"ungrounded_actions={row['avg_ungrounded_action_count']}, "
            f"evidence_mode={row.get('evidence_mode', 'unknown')}, "
            f"unique_trajectories={row.get('unique_trajectory_count', 'unknown')}/"
            f"{row.get('episode_count', 'unknown')}"
        )
        for row in summary_rows
    ]
    return (
        "# Safety Summary\n\n"
        "These rows are local simulator rollouts, not formal off-policy evaluation over a logged "
        "behavior-policy dataset. Replayed deterministic trajectories are useful for teaching, but "
        "they are not independent deployment evidence.\n\n"
        + "\n".join(lines)
        + "\n"
    )


def recommendation_from_summary(summary_rows: Sequence[dict[str, int | float | str]]) -> tuple[str, str]:
    """Produce a simple deploy/shadow/reject recommendation."""
    best = max(summary_rows, key=lambda row: float(row["avg_reward"]))
    risky = float(best["avg_final_safety_risk"]) > 1.0 or float(best["avg_ungrounded_action_count"]) > 0.6
    if risky:
        return "reject", "The highest-reward policy still carries too much safety or grounding risk."
    if float(best["solved_rate"]) < 0.75:
        return "shadow", "The policy is promising, but it needs more evidence before it should act automatically."
    return "shadow", "Even the best policy stays in shadow mode in this teaching repo."


def deployment_recommendation_markdown(summary_rows: Sequence[dict[str, int | float | str]]) -> str:
    """Write the final deployment recommendation artifact."""
    recommendation, rationale = recommendation_from_summary(summary_rows)
    return (
        "# Deployment Recommendation\n\n"
        f"Recommendation: **{recommendation}**.\n\n"
        "## Why\n\n"
        f"{rationale}\n\n"
        "## Teaching takeaway\n\n"
        "A policy can look interesting in a simulator and still be nowhere near ready to run unsupervised around real students.\n"
    )


def resource_match_rows() -> list[dict[str, str]]:
    """Return a tiny deterministic set of assistant-side resource matches."""
    return [
        {"intent_type": "concept_help", "resource": "course-note-linear-algebra.md", "why": "Good fallback when the policy chooses retrieval."},
        {"intent_type": "debug_help", "resource": "debug-checklist.md", "why": "Useful when the policy wants grounding before a hint."},
        {"intent_type": "study_plan", "resource": "weekly-planning-template.md", "why": "Pairs well with targeted practice or pacing."},
        {"intent_type": "exam_review", "resource": "exam-review-guide.md", "why": "Helps the assistant ground late-stage support."},
    ]


def bridge_artifacts() -> tuple[dict[str, object], str]:
    """Return the router payload and action-mapping markdown."""
    return policy_router_payload(), action_mapping_markdown()


def _validate_required_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    if artifact_path.suffix == ".csv":
        return _validate_csv_artifact(relative_path, artifact_path)
    if artifact_path.suffix == ".json":
        return _validate_json_artifact(relative_path, artifact_path)
    if artifact_path.suffix == ".md":
        return _validate_markdown_artifact(relative_path, artifact_path)
    return []


def _validate_optional_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    return _validate_required_artifact(relative_path, artifact_path)


def _validate_csv_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    content = artifact_path.read_text(encoding="utf-8").strip()
    if not content:
        return [f"{relative_path} is empty."]
    rows = list(csv.DictReader(content.splitlines()))
    header = list(rows[0].keys()) if rows else content.splitlines()[0].split(",")
    required_columns = {
        "artifacts/bandit/contextual_policy_metrics.csv": ["step", "scenario_name", "action", "reward"],
        "artifacts/bandit/regret_trace.csv": ["step", "action", "cumulative_regret"],
        "artifacts/bandit/action_breakdown.csv": ["action", "pull_count", "pull_share"],
        "artifacts/q_learning/training_curve.csv": ["episode", "total_reward", "epsilon"],
        "artifacts/q_learning/q_table.csv": ["intent_type", "action", "q_value"],
        "artifacts/sarsa/training_curve.csv": ["episode", "total_reward", "epsilon"],
        "artifacts/policy_gradient/training_curve.csv": ["episode", "total_reward", "baseline"],
        "artifacts/eval/offline_policy_eval.csv": ["policy", "avg_reward", "solved_rate"],
        "artifacts/eval/scenario_rollups.csv": ["policy", "scenario_name", "total_reward"],
        "artifacts/policy/rule_policy_summary.csv": ["policy", "avg_reward", "solved_rate"],
        "artifacts/policy/intervention_decisions.csv": ["policy", "avg_reward", "interpretation"],
        "artifacts/drl_optional/dqn_training_summary.csv": [
            "policy",
            "timesteps",
            "avg_reward",
            "solved_rate",
            "avg_final_safety_risk",
        ],
        "artifacts/drl_optional/ppo_training_summary.csv": [
            "policy",
            "timesteps",
            "avg_reward",
            "solved_rate",
            "avg_final_safety_risk",
        ],
        "artifacts/drl_optional/rl_family_comparison.csv": [
            "policy",
            "family",
            "avg_reward",
            "solved_rate",
            "avg_final_safety_risk",
        ],
        "artifacts/drl_optional/scenario_rollups.csv": [
            "policy",
            "scenario_name",
            "total_reward",
            "actions",
        ],
    }.get(relative_path, [])
    missing = [column for column in required_columns if column not in header]
    errors = [f"{relative_path} is missing columns: {', '.join(missing)}."] if missing else []
    if missing:
        return errors
    if required_columns and not rows:
        return [f"{relative_path} has no data rows."]
    errors.extend(_validate_csv_semantics(relative_path, rows))
    return errors


def _validate_csv_semantics(relative_path: str, rows: Sequence[Mapping[str, str]]) -> list[str]:
    errors: list[str] = []
    if relative_path == "artifacts/drl_optional/dqn_training_summary.csv":
        errors.extend(_require_policy_values(relative_path, rows, {"dqn"}))
    elif relative_path == "artifacts/drl_optional/ppo_training_summary.csv":
        errors.extend(_require_policy_values(relative_path, rows, {"ppo"}))
    elif relative_path == "artifacts/drl_optional/rl_family_comparison.csv":
        expected_families = {
            "q_learning": "tabular_value_based",
            "dqn": "deep_value_based",
            "ppo": "actor_critic_policy_gradient",
        }
        seen_policies = {row["policy"] for row in rows}
        missing_policies = sorted(set(expected_families) - seen_policies)
        if missing_policies:
            errors.append(f"{relative_path} is missing policy rows: {', '.join(missing_policies)}.")
        for row in rows:
            policy = row["policy"]
            expected_family = expected_families.get(policy)
            if expected_family is None:
                errors.append(f"{relative_path} has unknown policy: {policy}.")
            elif row["family"] != expected_family:
                errors.append(f"{relative_path} has wrong family for {policy}: {row['family']}.")
    elif relative_path == "artifacts/drl_optional/scenario_rollups.csv":
        errors.extend(_require_policy_values(relative_path, rows, {"q_learning", "dqn", "ppo"}))
    return errors


def _require_policy_values(
    relative_path: str,
    rows: Sequence[Mapping[str, str]],
    expected_policies: set[str],
) -> list[str]:
    seen_policies = {row["policy"] for row in rows}
    errors: list[str] = []
    unknown_policies = sorted(seen_policies - expected_policies)
    if unknown_policies:
        errors.append(f"{relative_path} has unexpected policy rows: {', '.join(unknown_policies)}.")
    missing_policies = sorted(expected_policies - seen_policies)
    if missing_policies:
        errors.append(f"{relative_path} is missing policy rows: {', '.join(missing_policies)}.")
    return errors


def _validate_json_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{relative_path} is not valid JSON: {exc}"]
    if relative_path != "artifacts/bridge/policy_router.json":
        return []

    errors: list[str] = []
    required_keys = {
        "allowed_actions",
        "bandit_subset",
        "decision_boundary",
        "export_kind",
        "exports_champion_policy_parameters",
        "exports_learned_weights",
        "notes",
        "router_version",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        errors.append(f"{relative_path} is missing keys: {', '.join(missing_keys)}.")
    if payload.get("export_kind") != "assistant_side_action_contract":
        errors.append(f"{relative_path} must keep export_kind=assistant_side_action_contract.")
    if payload.get("router_version") != 1:
        errors.append(f"{relative_path} must keep router_version=1.")
    if payload.get("decision_boundary") != "pedagogical_intervention_only":
        errors.append(f"{relative_path} must keep decision_boundary=pedagogical_intervention_only.")
    if payload.get("exports_learned_weights") is not False:
        errors.append(f"{relative_path} must not claim to export learned weights.")
    if payload.get("exports_champion_policy_parameters") is not False:
        errors.append(f"{relative_path} must not claim to export champion policy parameters.")
    allowed_actions = payload.get("allowed_actions", [])
    bandit_subset = payload.get("bandit_subset", [])
    if not isinstance(allowed_actions, list) or not allowed_actions:
        errors.append(f"{relative_path} must list allowed_actions.")
    elif allowed_actions != list(ACTION_LABELS.values()):
        errors.append(f"{relative_path} allowed_actions must match the canonical action vocabulary.")
    if not isinstance(bandit_subset, list) or not bandit_subset:
        errors.append(f"{relative_path} must list a non-empty bandit_subset.")
    elif bandit_subset != [ACTION_LABELS[action] for action in BANDIT_ACTIONS]:
        errors.append(f"{relative_path} bandit_subset must match the canonical first-turn bandit actions.")
    return errors


def _validate_markdown_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    content = artifact_path.read_text(encoding="utf-8").strip()
    if not content.startswith("# "):
        return [f"{relative_path} must begin with a top-level heading."]
    return []
