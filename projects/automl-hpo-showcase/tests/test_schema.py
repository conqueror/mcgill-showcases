from __future__ import annotations

from pathlib import Path

from automl_hpo_showcase.search_random_grid import run_random_search


def test_trials_have_required_columns() -> None:
    frame = run_random_search(budget=3, seed=11, random_state=11)
    required = {"strategy", "trial_id", "n_estimators", "max_depth", "min_samples_split", "score"}
    assert required.issubset(frame.columns)


def test_artifact_verifier_rejects_empty_contracts(tmp_path: Path) -> None:
    import json
    import shutil
    import subprocess
    import sys

    script = tmp_path / "scripts/verify_artifacts.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py", script)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "empty.csv").touch()
    for payload in [{}, {"required_files": []}, {"required_files": ["artifacts/empty.csv"]}]:
        (artifacts / "manifest.json").write_text(json.dumps(payload))
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert result.returncode != 0
