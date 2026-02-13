from __future__ import annotations

from eda_leakage_showcase.data import make_dataset


def test_dataset_contains_leakage_column() -> None:
    bundle = make_dataset(n_samples=100, random_state=7)
    assert "leak_target_copy" in bundle.frame.columns
    assert len(bundle.frame) == 100
    assert len(bundle.target) == 100
