"""Tests for evaluation helpers."""

from __future__ import annotations

import torch

from pytorch_training_regularization_showcase import evaluation


def test_compute_accuracy_and_error_analysis_table() -> None:
    """Evaluation helpers should expose readable metrics and example-level outputs."""

    logits = torch.tensor(
        [
            [3.0, 0.5, -1.0],
            [0.2, 2.0, 0.1],
            [1.5, 1.2, 0.2],
        ],
    )
    targets = torch.tensor([0, 1, 1])

    accuracy = evaluation.compute_accuracy(logits, targets)
    table = evaluation.build_error_analysis_table(
        logits,
        targets,
        class_names=["class_0", "class_1", "class_2"],
    )

    assert accuracy == 2 / 3
    assert list(table.columns) == [
        "example_index",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    ]
    assert len(table) == 3
