"""Tests for the optional deep-RL bridge CLI: graceful degradation and artifact wiring.

The optional bridge runs DQN and PPO (via Gymnasium + Stable-Baselines3) on the same
student-support environment family as tabular Q-learning, situating the deep-RL rungs of the
ladder next to the tabular ones (contextual bandit -> MDP -> Q-learning -> DQN -> policy
gradient -> actor-critic -> PPO). Because those deep-RL dependencies are heavy and optional,
these tests deliberately never invoke the real trainers: they ``monkeypatch`` the comparison
function so the CLI's *plumbing* is exercised deterministically and fast. One test forces the
missing-dependency path (``OptionalDRLError``) and asserts the CLI still exits 0 and writes a
human-readable bridge report; the other stubs a successful run and asserts the comparison
artifacts (CSV/Markdown) are written with the expected RL-family taxonomy
(``tabular_value_based`` / ``deep_value_based`` / ``actor_critic_policy_gradient``).

RL concept:
    Mapping algorithms onto the value-based -> policy-gradient -> actor-critic ladder; see
    docs/deep-rl.md, docs/policy-gradient-and-actor-critic.md and docs/evaluation-and-governance.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pytest import MonkeyPatch

from scripts.run_drl_optional import main as run_drl_optional_main


def test_drl_optional_report_mentions_dqn_and_ppo_when_deps_are_missing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """When optional deps are missing, the CLI still exits 0 and writes an explanatory report.

    The comparison function is patched to raise ``OptionalDRLError`` (the exception the real
    code raises when Gymnasium/SB3 are absent). The CLI must degrade gracefully: return 0 and
    emit ``bridge_report.md`` that still names DQN and PPO and surfaces the import error, so a
    learner without the heavy extras gets guidance rather than a crash.

    RL concept:
        Optional deep-RL bridge (DQN/PPO) on the same MDP (docs/deep-rl.md).
    """
    from student_support_rl.drl import OptionalDRLError

    def fake_run_drl_comparison(**_: object) -> None:
        raise OptionalDRLError("missing optional dependencies")

    monkeypatch.setattr("scripts.run_drl_optional.run_drl_comparison", fake_run_drl_comparison)

    exit_code = run_drl_optional_main(["--output-dir", str(tmp_path), "--quick"])

    report = (tmp_path / "drl_optional" / "bridge_report.md").read_text(encoding="utf-8")

    assert exit_code == 0
    assert "DQN" in report
    assert "PPO" in report
    assert "Import error:" in report


def test_drl_optional_writes_comparison_artifacts_from_stubbed_run(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A stubbed successful run writes the CSV/Markdown artifacts with the RL-family taxonomy.

    The comparison function is patched to return a fixed ``DRLComparisonResult`` (q_learning,
    dqn, ppo), so the test checks the CLI's serialization plumbing without training anything.
    It asserts the expected files exist and that the comparison CSV preserves both row order
    (q_learning, dqn, ppo) and the family label that locates each method on the ladder --
    PPO's row carries ``actor_critic_policy_gradient``.

    RL concept:
        Algorithm-family taxonomy across the ladder (docs/deep-rl.md, docs/glossary.md).
    """
    from student_support_rl import drl

    def fake_run_drl_comparison(
        *,
        timesteps: int,
        output_dir: Path,
        quick: bool,
    ) -> drl.DRLComparisonResult:
        del timesteps, quick
        return drl.DRLComparisonResult(
            comparison_rows=[
                {
                    "policy": "q_learning",
                    "family": "tabular_value_based",
                    "avg_reward": 4.2,
                    "avg_final_risk": 0.8,
                    "solved_rate": 1.0,
                },
                {
                    "policy": "dqn",
                    "family": "deep_value_based",
                    "avg_reward": 4.5,
                    "avg_final_risk": 0.7,
                    "solved_rate": 1.0,
                },
                {
                    "policy": "ppo",
                    "family": "actor_critic_policy_gradient",
                    "avg_reward": 4.4,
                    "avg_final_risk": 0.7,
                    "solved_rate": 1.0,
                },
            ],
            scenario_rows=[
                {
                    "policy": "dqn",
                    "scenario_id": 2,
                    "scenario_name": "high_risk_student",
                    "total_reward": 5.0,
                }
            ],
            training_rows=[
                {"policy": "dqn", "step": 1, "mean_reward": 1.5, "mean_final_risk": 0.8},
                {"policy": "ppo", "step": 1, "mean_reward": 1.4, "mean_final_risk": 0.7},
            ],
            policy_gradient_notes=(
                "# Policy Gradient Coverage\n\n"
                "PPO provides the actor-critic reference point in this showcase.\n\n"
                "This policy-gradient bridge compares PPO against DQN.\n"
            ),
            bridge_report="# Optional DRL Bridge\n\nRan DQN and PPO.\n",
        )

    monkeypatch.setattr("scripts.run_drl_optional.run_drl_comparison", fake_run_drl_comparison)

    exit_code = run_drl_optional_main(["--output-dir", str(tmp_path), "--quick"])

    comparison_path = tmp_path / "drl_optional" / "rl_family_comparison.csv"
    training_path = tmp_path / "drl_optional" / "training_summary.csv"
    notes_path = tmp_path / "drl_optional" / "policy_gradient_notes.md"

    assert exit_code == 0
    assert comparison_path.exists()
    assert training_path.exists()
    assert notes_path.exists()
    comparison_rows = list(csv.DictReader(comparison_path.open(encoding="utf-8")))
    assert [row["policy"] for row in comparison_rows] == ["q_learning", "dqn", "ppo"]
    assert comparison_rows[2]["family"] == "actor_critic_policy_gradient"
