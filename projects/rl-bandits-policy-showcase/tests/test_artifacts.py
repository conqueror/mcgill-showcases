from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = [
    "artifacts/sim/policy_comparison.csv",
    "artifacts/sim/policy_recommendation.md",
    "artifacts/sim/regret_trace.csv",
    "artifacts/sim/reward_trace.csv",
]


def _run_script(root: Path, name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = root / "scripts" / name
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / "scripts" / name, script)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _generate_bundle(root: Path) -> None:
    for script in ("run_simulation.py", "run_policy_comparison.py"):
        result = _run_script(root, script, "--quick")
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "scripts",
    [
        ("run_simulation.py", "run_policy_comparison.py"),
        ("run_policy_comparison.py", "run_simulation.py"),
    ],
)
def test_manifest_and_verification_do_not_depend_on_command_order(
    tmp_path: Path, scripts: tuple[str, str]
) -> None:
    for script in scripts:
        result = _run_script(tmp_path, script, "--quick")
        assert result.returncode == 0, result.stdout + result.stderr

    manifest = json.loads((tmp_path / "artifacts/manifest.json").read_text())
    assert sorted(manifest["required_files"]) == EXPECTED_FILES
    verified = _run_script(tmp_path, "verify_artifacts.py")
    assert verified.returncode == 0, verified.stdout + verified.stderr


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1},
        {"version": 1, "required_files": []},
        {"version": 1, "required_files": ["artifacts/sim/reward_trace.csv"]},
    ],
)
def test_verifier_rejects_manifests_that_omit_the_expected_bundle(
    tmp_path: Path, payload: object
) -> None:
    _generate_bundle(tmp_path)
    (tmp_path / "artifacts/manifest.json").write_text(json.dumps(payload))

    result = _run_script(tmp_path, "verify_artifacts.py")
    assert result.returncode != 0


@pytest.mark.parametrize(
    "path,content",
    [
        ("reward_trace.csv", ""),
        ("reward_trace.csv", "round,strategy,reward,cumulative_reward\n"),
        ("reward_trace.csv", "wrong,columns\n1,2\n"),
        ("reward_trace.csv", "round,strategy,reward,cumulative_reward\n1,epsilon_greedy,NaN,1\n"),
        ("reward_trace.csv", "round,strategy,reward,cumulative_reward\n1,epsilon_greedy,2,2\n"),
        ("regret_trace.csv", "round,strategy,instant_regret,cumulative_regret\n1,ucb1,-0.5,-0.5\n"),
        ("policy_comparison.csv", "strategy,cumulative_reward,cumulative_regret\nucb1,inf,1\n"),
        ("policy_recommendation.md", ""),
    ],
)
def test_verifier_rejects_empty_or_corrupt_artifacts(
    tmp_path: Path, path: str, content: str
) -> None:
    _generate_bundle(tmp_path)
    (tmp_path / "artifacts/sim" / path).write_text(content)

    result = _run_script(tmp_path, "verify_artifacts.py")
    assert result.returncode != 0


def test_verifier_rejects_inconsistent_cumulative_totals(tmp_path: Path) -> None:
    _generate_bundle(tmp_path)
    path = tmp_path / "artifacts/sim/reward_trace.csv"
    lines = path.read_text().splitlines()
    cells = lines[1].split(",")
    cells[-1] = "99"
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")

    result = _run_script(tmp_path, "verify_artifacts.py")
    assert result.returncode != 0


@pytest.mark.parametrize(
    "name,column,value",
    [
        ("reward_trace.csv", "reward", "not-a-number"),
        ("reward_trace.csv", "reward", "nan"),
        ("reward_trace.csv", "reward", "2"),
        ("regret_trace.csv", "instant_regret", "-0.5"),
        ("policy_comparison.csv", "cumulative_reward", "inf"),
        ("reward_trace.csv", "round", "1.5"),
        ("reward_trace.csv", "strategy", "unknown_policy"),
    ],
)
def test_verifier_rejects_invalid_values_in_an_otherwise_complete_bundle(
    tmp_path: Path, name: str, column: str, value: str
) -> None:
    _generate_bundle(tmp_path)
    path = tmp_path / "artifacts/sim" / name
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][column] = value
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = _run_script(tmp_path, "verify_artifacts.py")
    assert result.returncode != 0
