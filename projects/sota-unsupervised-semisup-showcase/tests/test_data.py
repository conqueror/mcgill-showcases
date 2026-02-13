from __future__ import annotations

import numpy as np

from sota_showcase.data import load_digits_dataset, make_semisupervised_labels


def test_semisupervised_labels_keep_every_class() -> None:
    dataset = load_digits_dataset(scale=True)
    masked = make_semisupervised_labels(dataset.y, labeled_fraction=0.08, random_state=42)

    labeled = masked[masked != -1]
    assert labeled.size > 0
    assert set(np.unique(labeled).tolist()) == set(np.unique(dataset.y).tolist())


def test_semisupervised_labels_include_unlabeled_points() -> None:
    dataset = load_digits_dataset(scale=True)
    masked = make_semisupervised_labels(dataset.y, labeled_fraction=0.1, random_state=42)

    unlabeled_ratio = float(np.mean(masked == -1))
    assert 0.75 <= unlabeled_ratio <= 0.95
