"""Pin the artifact contract that makes the showcase reproducible and auditable.

These tests are not about an RL algorithm; they guard the *governance/reproducibility*
boundary around the whole pipeline. The showcase is only trustworthy if every required
artifact (bandit traces, Q-learning tables, DP ground-truth gaps, evaluation comparisons,
reward and governance memos) is present and well-formed before anyone reads results. The
tests pin that the verifier fails on missing or malformed required outputs, succeeds on a
complete set, keeps the checked-in manifest in sync with the code, and treats the optional
deep-RL (DQN/PPO) comparison group as all-or-nothing.

RL concept:
    Evaluation governance and the reproducibility contract; see
    docs/evaluation-and-governance.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_artifacts import main as verify_artifacts_main
from student_support_rl.reporting import (
    OPTIONAL_DRL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    artifact_validation_errors,
    write_manifest,
    write_text_artifact,
)


def test_verify_artifacts_fails_when_required_files_are_missing(tmp_path: Path) -> None:
    """Verify the verifier returns exit code 1 when required artifacts are absent.

    Pins the fail-closed half of the contract: a manifest exists but no artifact files do, so
    ``verify_artifacts`` must reject (exit 1) rather than silently pass. This is what stops a
    half-finished run from being presented as a complete, reproducible result.

    RL concept:
        Reproducibility gate in evaluation governance; see
        docs/evaluation-and-governance.md.
    """
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path, REQUIRED_ARTIFACTS)

    exit_code = verify_artifacts_main(
        [
            "--output-dir",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 1


def test_verify_artifacts_succeeds_for_complete_outputs(tmp_path: Path) -> None:
    """Verify the verifier returns exit code 0 when every required artifact is present.

    Pins the fail-open half: once a minimal but well-formed copy of each required artifact is
    written, verification passes (exit 0). Together with the missing-files test this brackets
    the verifier's accept/reject boundary on completeness.

    RL concept:
        Reproducibility gate in evaluation governance; see
        docs/evaluation-and-governance.md.
    """
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path, REQUIRED_ARTIFACTS)

    _write_minimal_required_artifacts(tmp_path)

    exit_code = verify_artifacts_main(
        [
            "--output-dir",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 0


def test_checked_in_manifest_matches_required_artifacts() -> None:
    """Verify the committed manifest.json stays in sync with the code's artifact lists.

    Pins that the on-disk manifest's required and optional file tuples exactly equal the
    ``REQUIRED_ARTIFACTS`` / ``OPTIONAL_DRL_ARTIFACTS`` constants. This catches drift where a
    new artifact is added in code but the checked-in contract is never updated, which would
    let an incomplete run pass verification.

    RL concept:
        Contract integrity for reproducible evaluation; see
        docs/evaluation-and-governance.md.
    """
    manifest_path = (
        Path(__file__).resolve().parent.parent / "artifacts" / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert tuple(payload["required_files"]) == REQUIRED_ARTIFACTS
    assert tuple(payload["optional_files"]) == OPTIONAL_DRL_ARTIFACTS


def test_verify_artifacts_fails_for_invalid_required_csv_content(tmp_path: Path) -> None:
    """Verify the verifier rejects a present-but-malformed required CSV (exit code 1).

    Pins that existence is not enough: after writing a complete artifact set, overwriting the
    bandit reward trace with a CSV missing its required columns (scenario_name,
    context_signature, expected_reward, optimal_action_label, ...) must fail verification.
    Schema validation, not just presence, protects the meaning of the reproducible outputs.

    RL concept:
        Schema-level reproducibility validation; see docs/evaluation-and-governance.md.
    """
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path, REQUIRED_ARTIFACTS, OPTIONAL_DRL_ARTIFACTS)

    _write_minimal_required_artifacts(tmp_path)
    # Overwrite a valid trace with a column-deficient one to trip schema validation.
    write_text_artifact(
        tmp_path / "artifacts" / "bandit" / "reward_trace.csv",
        "step,action\n1,resource_email",
    )

    exit_code = verify_artifacts_main(
        [
            "--output-dir",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 1


def test_optional_drl_group_requires_complete_comparison_outputs(tmp_path: Path) -> None:
    """Verify the optional deep-RL comparison group is enforced as all-or-nothing.

    Pins the group rule for the upper rungs of the ladder (DQN, PPO -- the deep-RL and
    actor-critic stages): once any deep-RL comparison artifact appears, the remaining ones in
    the group become mandatory. Writing only the bridge report plus one comparison CSV must
    raise an "incomplete" error, so a partial DQN/PPO comparison can never masquerade as a
    finished one.

    RL concept:
        Completeness contract for the deep-RL comparison stage; see docs/deep-rl.md and
        docs/policy-gradient-and-actor-critic.md.
    """
    bridge_report = tmp_path / "artifacts" / "drl_optional" / "bridge_report.md"
    write_text_artifact(bridge_report, "# Optional DRL Bridge\n\nDQN and PPO.\n")
    write_text_artifact(
        tmp_path / "artifacts" / "drl_optional" / "rl_family_comparison.csv",
        "policy,family,avg_reward,avg_final_risk,solved_rate\nq_learning,tabular_value_based,1.0,1.0,1.0\n",
    )

    errors = artifact_validation_errors(output_dir=tmp_path)

    # Partial deep-RL group => the validator flags the comparison as incomplete.
    assert any("Optional DRL comparison output is incomplete" in error for error in errors)


def _write_minimal_required_artifacts(root: Path) -> None:
    """Write a minimal, schema-valid copy of every required artifact under ``root``.

    Shared setup for the "succeeds" and "invalid CSV" tests: it materializes one well-formed
    file per entry in ``REQUIRED_ARTIFACTS`` (correct CSV headers with a single data row, or a
    heading-led Markdown stub) so that verification passes unless a test deliberately corrupts
    one file afterward. Not an RL step -- it exists purely to exercise the artifact contract.

    Args:
        root: Temporary project root under which the ``artifacts/`` tree is created.
    """
    rows_by_path = {
        "artifacts/concepts/concept_map.csv": (
            "concept,showcase_component,artifact\n"
            "mdp_framing,mdp_spec_markdown,artifacts/concepts/mdp_spec.md\n"
        ),
        "artifacts/bandit/reward_trace.csv": (
            "step,scenario_name,context_signature,action_label,expected_reward,optimal_action_label\n"
            "1,medium_risk_student,engagement=2|completion=2|pressure=2|prior_interventions=0,"
            "ta_session,0.7,ta_session\n"
        ),
        "artifacts/bandit/regret_trace.csv": (
            "step,scenario_name,action_label,optimal_action_label,cumulative_regret\n"
            "1,medium_risk_student,ta_session,ta_session,0.0\n"
        ),
        "artifacts/mdp/sample_episodes.csv": (
            "scenario_name,week,action,reward,next_risk\n"
            "high_risk_student,1,advisor_meeting,1.5,2\n"
        ),
        "artifacts/q_learning/training_curve.csv": (
            "episode,scenario_id,total_reward,epsilon,steps\n"
            "1,2,2.1,0.4,6\n"
        ),
        "artifacts/q_learning/q_table.csv": (
            "week,engagement,completion,pressure,risk,prior_interventions,action,q_value\n"
            "1,1,1,3,3,0,3,1.2\n"
        ),
        "artifacts/dp/optimal_action_values.csv": (
            "week,engagement,completion,pressure,risk,prior_interventions,action,optimal_q_value\n"
            "1,1,1,3,3,0,2,3.5\n"
        ),
        "artifacts/dp/q_learning_gap.csv": (
            "week,engagement,completion,pressure,risk,prior_interventions,action,"
            "learned_q_value,optimal_q_value,abs_gap\n"
            "1,1,1,3,3,0,2,3.0,3.5,0.5\n"
        ),
        "artifacts/sarsa/training_curve.csv": (
            "episode,scenario_id,total_reward,epsilon,steps\n"
            "1,2,2.1,0.4,6\n"
        ),
        "artifacts/sarsa/q_table.csv": (
            "week,engagement,completion,pressure,risk,prior_interventions,action,q_value\n"
            "1,1,1,3,3,0,3,1.2\n"
        ),
        "artifacts/policy_gradient/training_curve.csv": (
            "episode,scenario_id,total_reward,baseline,steps\n"
            "1,2,2.1,1.0,6\n"
        ),
        "artifacts/eval/policy_comparison.csv": (
            "policy,avg_reward,avg_final_risk,avg_intervention_cost,solved_rate\n"
            "q_learning,3.2,1.0,1.1,1.0\n"
        ),
        "artifacts/eval/scenario_results.csv": (
            "policy,scenario_id,scenario_name,total_reward,final_risk,actions\n"
            "q_learning,2,high_risk_student,3.2,1,ta_session | advisor_meeting\n"
        ),
    }
    markdown_by_path = {
        "artifacts/concepts/mdp_spec.md": "# MDP Spec\n",
        "artifacts/concepts/algorithm_progression.md": (
            "# RL To DRL Progression\n\n"
            "Q-learning, DQN, policy gradients, actor-critic, and PPO.\n"
        ),
        "artifacts/reward/reward_hacking_report.md": "# Reward Hacking Report\n",
        "artifacts/reward/reward_spec_good.md": "# Good Reward\n",
        "artifacts/reward/reward_spec_bad.md": "# Bad Reward\n",
        "artifacts/governance/safety_controls.md": "# Safety Controls\n",
        "artifacts/governance/offline_eval_plan.md": "# Offline Eval Plan\n",
        "artifacts/business/deploy_shadow_reject_memo.md": "# Deploy Memo\n",
    }

    for relative_path in REQUIRED_ARTIFACTS:
        if relative_path in rows_by_path:
            write_text_artifact(root / relative_path, rows_by_path[relative_path])
        else:
            write_text_artifact(root / relative_path, markdown_by_path[relative_path])
