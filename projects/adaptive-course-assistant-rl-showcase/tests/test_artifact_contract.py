import importlib.util
from pathlib import Path
from types import ModuleType

from adaptive_course_assistant_rl.reporting import (
    OPTIONAL_DRL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    artifact_validation_errors,
    missing_required_artifacts,
    optional_artifacts,
    required_artifacts,
    write_json_artifact,
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    script_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_matches_required_artifact_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path)

    assert manifest_path.exists()
    missing = missing_required_artifacts(output_dir=tmp_path, manifest_path=manifest_path)
    expected = [path for path in REQUIRED_ARTIFACTS if path != "artifacts/manifest.json"]
    assert missing == expected


def test_optional_drl_artifacts_can_be_explicitly_required(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path)

    errors = artifact_validation_errors(
        output_dir=tmp_path,
        manifest_path=manifest_path,
        require_optional_drl=True,
    )

    assert errors == [f"Required optional DRL artifact is missing: {path}." for path in OPTIONAL_DRL_ARTIFACTS]


def test_optional_drl_artifacts_require_comparison_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path)
    optional_dir = tmp_path / "artifacts" / "drl_optional"
    optional_dir.mkdir(parents=True)

    (optional_dir / "dqn_training_summary.csv").write_text(
        "policy,timesteps,avg_reward,solved_rate\n"
        "dqn,10,1.0,0.5\n",
        encoding="utf-8",
    )
    (optional_dir / "ppo_training_summary.csv").write_text(
        "policy,timesteps,avg_reward,solved_rate,avg_final_safety_risk\n"
        "ppo,10,1.0,0.5,0.2\n",
        encoding="utf-8",
    )
    (optional_dir / "rl_family_comparison.csv").write_text(
        "policy,family,avg_reward,solved_rate,avg_final_safety_risk\n"
        "q_learning,tabular_value_based,1.0,0.5,0.2\n"
        "dqn,deep_value_based,1.0,0.5,0.2\n"
        "ppo,actor_critic_policy_gradient,1.0,0.5,0.2\n",
        encoding="utf-8",
    )
    (optional_dir / "scenario_rollups.csv").write_text(
        "policy,scenario_name,total_reward,actions\n"
        "q_learning,demo,1.0,retrieve_course_note\n"
        "dqn,demo,1.0,retrieve_course_note\n"
        "ppo,demo,1.0,retrieve_course_note\n",
        encoding="utf-8",
    )
    (optional_dir / "policy_gradient_bridge_notes.md").write_text("# Notes\n\nDemo.\n", encoding="utf-8")

    errors = artifact_validation_errors(
        output_dir=tmp_path,
        manifest_path=manifest_path,
        require_optional_drl=True,
    )

    assert errors == [
        "artifacts/drl_optional/dqn_training_summary.csv is missing columns: avg_final_safety_risk."
    ]


def test_optional_drl_artifacts_require_semantic_policy_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    write_manifest(manifest_path)
    optional_dir = tmp_path / "artifacts" / "drl_optional"
    optional_dir.mkdir(parents=True)

    (optional_dir / "dqn_training_summary.csv").write_text(
        "policy,timesteps,avg_reward,solved_rate,avg_final_safety_risk\n"
        "ppo,10,1.0,0.5,0.2\n",
        encoding="utf-8",
    )
    (optional_dir / "ppo_training_summary.csv").write_text(
        "policy,timesteps,avg_reward,solved_rate,avg_final_safety_risk\n"
        "dqn,10,1.0,0.5,0.2\n",
        encoding="utf-8",
    )
    (optional_dir / "rl_family_comparison.csv").write_text(
        "policy,family,avg_reward,solved_rate,avg_final_safety_risk\n"
        "q_learning,tabular_value_based,1.0,0.5,0.2\n"
        "dqn,deep_value_based,1.0,0.5,0.2\n"
        "ppo,tabular_value_based,1.0,0.5,0.2\n",
        encoding="utf-8",
    )
    (optional_dir / "scenario_rollups.csv").write_text(
        "policy,scenario_name,total_reward,actions\n"
        "dqn,demo,1.0,retrieve_course_note\n",
        encoding="utf-8",
    )
    (optional_dir / "policy_gradient_bridge_notes.md").write_text("# Notes\n\nDemo.\n", encoding="utf-8")

    errors = artifact_validation_errors(
        output_dir=tmp_path,
        manifest_path=manifest_path,
        require_optional_drl=True,
    )

    assert "artifacts/drl_optional/dqn_training_summary.csv has unexpected policy rows: ppo." in errors
    assert "artifacts/drl_optional/ppo_training_summary.csv has unexpected policy rows: dqn." in errors
    assert "artifacts/drl_optional/rl_family_comparison.csv has wrong family for ppo: tabular_value_based." in errors
    assert "artifacts/drl_optional/scenario_rollups.csv is missing policy rows: ppo, q_learning." in errors


def test_policy_router_contract_is_semantically_validated(tmp_path: Path) -> None:
    router_path = tmp_path / "artifacts" / "bridge" / "policy_router.json"
    write_json_artifact(
        router_path,
        {
            "allowed_actions": ["ask_clarifying_question"],
            "bandit_subset": ["not_a_known_action"],
            "decision_boundary": "answer_generation",
            "export_kind": "wrong_contract",
            "exports_champion_policy_parameters": True,
            "exports_learned_weights": True,
            "notes": "Bad teaching contract.",
            "router_version": 999,
        },
    )

    errors = artifact_validation_errors(output_dir=tmp_path)

    assert "artifacts/bridge/policy_router.json must keep export_kind=assistant_side_action_contract." in errors
    assert "artifacts/bridge/policy_router.json must keep router_version=1." in errors
    assert "artifacts/bridge/policy_router.json must keep decision_boundary=pedagogical_intervention_only." in errors
    assert "artifacts/bridge/policy_router.json must not claim to export learned weights." in errors
    assert "artifacts/bridge/policy_router.json must not claim to export champion policy parameters." in errors
    assert "artifacts/bridge/policy_router.json allowed_actions must match the canonical action vocabulary." in errors
    assert "artifacts/bridge/policy_router.json bandit_subset must match the canonical first-turn bandit actions." in errors


def test_verify_script_accepts_artifact_directory_as_output_dir(tmp_path: Path) -> None:
    verify_artifacts = _load_script("verify_artifacts_test", "scripts/verify_artifacts.py")
    artifacts_dir = tmp_path / "custom-artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "manifest.json").write_text(
        '{"version": 1, "required_files": [], "optional_files": []}\n',
        encoding="utf-8",
    )

    assert verify_artifacts.main(["--output-dir", str(artifacts_dir)]) == 0


def test_verify_script_fails_closed_when_manifest_is_missing(tmp_path: Path) -> None:
    verify_artifacts = _load_script("verify_artifacts_missing_manifest_test", "scripts/verify_artifacts.py")

    assert verify_artifacts.main(["--output-dir", str(tmp_path)]) == 1
    assert verify_artifacts.main(["--output-dir", str(tmp_path), "--manifest", "/tmp/no-such-manifest.json"]) == 1


def test_checked_in_manifest_matches_artifact_contract() -> None:
    manifest_path = PROJECT_ROOT / "artifacts" / "manifest.json"

    assert tuple(required_artifacts(manifest_path)) == REQUIRED_ARTIFACTS
    assert tuple(optional_artifacts(manifest_path)) == OPTIONAL_DRL_ARTIFACTS
