"""Evaluation helpers for the PyTorch showcase."""

from __future__ import annotations

import pandas as pd
import torch


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return categorical accuracy for a batch of logits."""

    predictions = torch.argmax(logits, dim=1)
    correct = int((predictions == targets).sum().item())
    return correct / len(targets)


def build_error_analysis_table(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
) -> pd.DataFrame:
    """Create an example-level table for model error analysis."""

    probabilities = torch.softmax(logits, dim=1)
    predicted_indices = torch.argmax(probabilities, dim=1)
    confidences = probabilities.max(dim=1).values

    rows = []
    for index in range(len(targets)):
        target_index = int(targets[index].item())
        predicted_index = int(predicted_indices[index].item())
        rows.append(
            {
                "example_index": index,
                "true_class": class_names[target_index],
                "predicted_class": class_names[predicted_index],
                "confidence": float(confidences[index].item()),
                "correct": target_index == predicted_index,
            },
        )
    return pd.DataFrame(rows)
