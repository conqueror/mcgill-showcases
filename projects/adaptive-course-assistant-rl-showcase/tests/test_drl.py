import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from _pytest.monkeypatch import MonkeyPatch

from adaptive_course_assistant_rl.drl import DRLComparisonResult, OptionalDRLError, _policy_family
from adaptive_course_assistant_rl.environment import AssistantState
from adaptive_course_assistant_rl.policies import Policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    script_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_family_labels_are_stable() -> None:
    assert _policy_family("q_learning") == "tabular_value_based"
    assert _policy_family("dqn") == "deep_value_based"
    assert _policy_family("ppo") == "actor_critic_policy_gradient"


def test_optional_drl_error_is_a_runtime_error() -> None:
    assert issubclass(OptionalDRLError, RuntimeError)


def test_run_drl_optional_fails_when_extras_are_missing(monkeypatch: MonkeyPatch) -> None:
    run_drl_optional = _load_script("run_drl_optional_test", "scripts/run_drl_optional.py")

    def raise_missing_extras(*args: object, **kwargs: object) -> object:
        raise OptionalDRLError("gymnasium is not installed")

    monkeypatch.setattr(run_drl_optional, "run_drl_comparison", raise_missing_extras)

    assert run_drl_optional.main(["--output-dir", str(PROJECT_ROOT / "artifacts")]) == 1


def test_quick_rl_family_comparison_uses_quick_reinforce_budget(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_rl_family_comparison = _load_script(
        "run_rl_family_comparison_test",
        "scripts/run_rl_family_comparison.py",
    )
    captured: dict[str, int] = {}

    class DummyPolicy:
        name = "dummy"

        def reset(self) -> None:
            return None

        def select_action(self, state: AssistantState) -> int:
            del state
            return 0

    class DummyResult:
        def greedy_policy(self) -> Policy:
            return DummyPolicy()

    def fake_train_reinforce(*, episodes: int, **kwargs: object) -> DummyResult:
        captured["episodes"] = episodes
        return DummyResult()

    def fake_train_tabular(**kwargs: object) -> DummyResult:
        return DummyResult()

    def fake_evaluate_policies(**kwargs: object) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [
                {
                    "policy": "dummy",
                    "avg_reward": 1.0,
                    "solved_rate": 1.0,
                    "avg_final_safety_risk": 0.0,
                    "avg_ungrounded_action_count": 0.0,
                }
            ],
            [{"policy": "dummy", "scenario_name": "demo", "total_reward": 1.0}],
        )

    def fake_write_artifact(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(run_rl_family_comparison, "train_reinforce", fake_train_reinforce)
    monkeypatch.setattr(run_rl_family_comparison, "train_q_learning", fake_train_tabular)
    monkeypatch.setattr(run_rl_family_comparison, "train_sarsa", fake_train_tabular)
    monkeypatch.setattr(run_rl_family_comparison, "evaluate_policies", fake_evaluate_policies)
    monkeypatch.setattr(run_rl_family_comparison, "write_csv_artifact", fake_write_artifact)
    monkeypatch.setattr(run_rl_family_comparison, "write_text_artifact", fake_write_artifact)

    assert run_rl_family_comparison.main(["--quick", "--output-dir", str(tmp_path)]) == 0
    assert captured["episodes"] == 180


def test_rl_family_comparison_uses_shared_training_seed_and_distinct_eval_seed(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_rl_family_comparison = _load_script(
        "run_rl_family_comparison_seed_test",
        "scripts/run_rl_family_comparison.py",
    )
    captured: dict[str, object] = {}

    class DummyPolicy:
        name = "dummy"

        def reset(self) -> None:
            return None

        def select_action(self, state: AssistantState) -> int:
            del state
            return 0

    class DummyResult:
        def greedy_policy(self) -> Policy:
            return DummyPolicy()

    def fake_train_q_learning(*, seed: int, **kwargs: object) -> DummyResult:
        captured["q_seed"] = seed
        return DummyResult()

    def fake_train_sarsa(*, seed: int, **kwargs: object) -> DummyResult:
        captured["sarsa_seed"] = seed
        return DummyResult()

    def fake_train_reinforce(*, seed: int, **kwargs: object) -> DummyResult:
        captured["reinforce_seed"] = seed
        return DummyResult()

    def fake_evaluate_policies(**kwargs: object) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        captured["eval_base_seed"] = kwargs["base_seed"]
        return (
            [
                {
                    "policy": "dummy",
                    "avg_reward": 1.0,
                    "solved_rate": 1.0,
                    "avg_final_safety_risk": 0.0,
                    "avg_ungrounded_action_count": 0.0,
                }
            ],
            [{"policy": "dummy", "scenario_name": "demo", "total_reward": 1.0}],
        )

    def fake_write_artifact(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(run_rl_family_comparison, "train_q_learning", fake_train_q_learning)
    monkeypatch.setattr(run_rl_family_comparison, "train_sarsa", fake_train_sarsa)
    monkeypatch.setattr(run_rl_family_comparison, "train_reinforce", fake_train_reinforce)
    monkeypatch.setattr(run_rl_family_comparison, "evaluate_policies", fake_evaluate_policies)
    monkeypatch.setattr(run_rl_family_comparison, "write_csv_artifact", fake_write_artifact)
    monkeypatch.setattr(run_rl_family_comparison, "write_text_artifact", fake_write_artifact)

    assert run_rl_family_comparison.main(["--output-dir", str(tmp_path)]) == 0
    assert captured["q_seed"] == captured["sarsa_seed"] == captured["reinforce_seed"] == 23
    assert captured["eval_base_seed"] == 123


def test_run_drl_optional_supports_custom_output_directory(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_drl_optional = _load_script("run_drl_optional_custom_output_test", "scripts/run_drl_optional.py")

    monkeypatch.setattr(
        run_drl_optional,
        "run_drl_comparison",
        lambda **kwargs: DRLComparisonResult(
            comparison_rows=[
                {
                    "policy": "q_learning",
                    "family": "tabular_value_based",
                    "avg_reward": 1.0,
                    "solved_rate": 1.0,
                    "avg_final_safety_risk": 0.0,
                },
                {
                    "policy": "dqn",
                    "family": "deep_value_based",
                    "avg_reward": 1.0,
                    "solved_rate": 1.0,
                    "avg_final_safety_risk": 0.0,
                },
                {
                    "policy": "ppo",
                    "family": "actor_critic_policy_gradient",
                    "avg_reward": 1.0,
                    "solved_rate": 1.0,
                    "avg_final_safety_risk": 0.0,
                }
            ],
            scenario_rows=[
                {
                    "policy": "q_learning",
                    "scenario_name": "demo",
                    "total_reward": 1.0,
                    "actions": "retrieve_course_note",
                },
                {
                    "policy": "dqn",
                    "scenario_name": "demo",
                    "total_reward": 1.0,
                    "actions": "retrieve_course_note",
                },
                {
                    "policy": "ppo",
                    "scenario_name": "demo",
                    "total_reward": 1.0,
                    "actions": "retrieve_course_note",
                }
            ],
            dqn_training_rows=[
                {
                    "policy": "dqn",
                    "timesteps": 10,
                    "avg_reward": 1.0,
                    "solved_rate": 1.0,
                    "avg_final_safety_risk": 0.0,
                }
            ],
            ppo_training_rows=[
                {
                    "policy": "ppo",
                    "timesteps": 10,
                    "avg_reward": 1.0,
                    "solved_rate": 1.0,
                    "avg_final_safety_risk": 0.0,
                }
            ],
            policy_gradient_notes="# Policy-Gradient Bridge Notes\n\nDemo note.\n",
        ),
    )

    output_dir = tmp_path / "custom-artifacts"

    assert run_drl_optional.main(["--output-dir", str(output_dir)]) == 0
    assert (output_dir / "drl_optional" / "rl_family_comparison.csv").exists()
    assert (output_dir / "drl_optional" / "policy_gradient_bridge_notes.md").exists()


def test_run_policy_export_does_not_overwrite_business_recommendation(tmp_path: Path) -> None:
    run_policy_export = _load_script("run_policy_export_test", "scripts/run_policy_export.py")

    assert run_policy_export.main(["--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "bridge" / "policy_router.json").exists()
    assert (tmp_path / "bridge" / "action_mapping.md").exists()
    assert not (tmp_path / "business" / "deployment_recommendation.md").exists()


def test_policy_router_export_is_marked_as_contract_only() -> None:
    run_policy_export = _load_script("run_policy_export_contract_test", "scripts/run_policy_export.py")

    payload = run_policy_export.policy_router_payload()

    assert payload["export_kind"] == "assistant_side_action_contract"
    assert payload["exports_learned_weights"] is False
    assert payload["exports_champion_policy_parameters"] is False
