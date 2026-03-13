#!/usr/bin/env python3
"""Entry point for the neural network foundations showcase."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neural_network_foundations_showcase import (
    activations,
    backprop,
    config,
    data,
    losses,
    networks,
    plots,
    reporting,
    training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the showcase runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help="Directory where artifacts should be written.",
    )
    return parser.parse_args(argv)


def _build_decision_boundary_outputs(
    random_state: int = 7,
) -> tuple[pd.DataFrame, list[plots.DecisionBoundaryExperiment]]:
    """Train the models used in the decision-boundary figure and summary."""

    linear_dataset = data.make_toy_dataset(
        "linearly_separable",
        samples_per_class=36,
        random_state=random_state,
    )
    xor_dataset = data.make_toy_dataset(
        "xor",
        samples_per_class=36,
        random_state=random_state + 1,
    )

    linear_result = training.train_network(
        linear_dataset,
        training.TrainingConfig(
            layer_sizes=(2, 1),
            epochs=90,
            learning_rate=0.25,
            random_state=random_state,
        ),
    )
    xor_linear_result = training.train_network(
        xor_dataset,
        training.TrainingConfig(
            layer_sizes=(2, 1),
            epochs=90,
            learning_rate=0.25,
            random_state=random_state + 2,
        ),
    )
    xor_mlp_result = training.train_network(
        xor_dataset,
        training.TrainingConfig(
            layer_sizes=(2, 8, 1),
            epochs=220,
            learning_rate=0.25,
            random_state=random_state + 3,
        ),
    )

    summary_rows = []
    experiments: list[plots.DecisionBoundaryExperiment] = []
    for title, dataset, result, teaching_point in (
        (
            "Perceptron on linearly separable data",
            linear_dataset,
            linear_result,
            "A single weighted sum can place one separating line.",
        ),
        (
            "Perceptron on XOR",
            xor_dataset,
            xor_linear_result,
            "Linear models fail when the classes need multiple regions.",
        ),
        (
            "Hidden-layer network on XOR",
            xor_dataset,
            xor_mlp_result,
            "A hidden layer bends the boundary into a nonlinear shape.",
        ),
    ):
        final = result.history.iloc[-1]
        summary_rows.append(
            {
                "model_title": title,
                "dataset": dataset.name,
                "train_accuracy": float(final["train_accuracy"]),
                "validation_accuracy": float(final["validation_accuracy"]),
                "teaching_point": teaching_point,
            },
        )
        experiments.append(
            plots.DecisionBoundaryExperiment(
                title=title,
                dataset=dataset,
                network=result.network,
            ),
        )

    return pd.DataFrame(summary_rows), experiments


def main(argv: list[str] | None = None) -> int:
    """Generate the core showcase artifacts."""

    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = reporting.expected_artifact_paths(output_dir)
    activation_table = activations.build_activation_comparison_table()
    loss_table = losses.build_loss_comparison_table()
    gradient_dataset = data.make_toy_dataset(
        "xor",
        samples_per_class=12,
        random_state=9,
    )
    gradient_network = networks.build_network(
        layer_sizes=(2, 6, 1),
        init_strategy="xavier",
        random_state=9,
    )
    gradient_trace = backprop.build_backprop_gradient_trace(
        gradient_network,
        gradient_dataset.features[:16],
        gradient_dataset.labels[:16],
    )
    initialization_table = networks.build_initialization_comparison_table()
    regime_table = training.build_underfit_overfit_table()
    training_curves = training.build_training_curve_table()
    decision_boundary_summary, decision_boundary_experiments = (
        _build_decision_boundary_outputs()
    )

    reporting.write_csv_artifact(
        activation_table,
        artifact_paths["activation_comparison"],
    )
    reporting.write_csv_artifact(
        loss_table,
        artifact_paths["loss_function_comparison"],
    )
    reporting.write_csv_artifact(
        gradient_trace,
        artifact_paths["backprop_gradient_trace"],
    )
    reporting.write_csv_artifact(
        initialization_table,
        artifact_paths["initialization_comparison"],
    )
    reporting.write_csv_artifact(
        regime_table,
        artifact_paths["underfit_overfit_examples"],
    )
    reporting.write_csv_artifact(
        training_curves,
        artifact_paths["training_curves"],
    )
    reporting.write_csv_artifact(
        decision_boundary_summary,
        artifact_paths["decision_boundary_summary"],
    )
    plots.plot_decision_boundaries(
        decision_boundary_experiments,
        artifact_paths["decision_boundaries"],
    )

    well_fit_gap = regime_table.loc[
        regime_table["regime"] == "well_fit",
        "generalization_gap",
    ].iloc[0]

    summary = reporting.build_summary_markdown(
        project_title="Neural Network Foundations Showcase",
        highlights=[
            reporting.to_highlight(
                "Best XOR validation accuracy",
                f"{decision_boundary_summary.iloc[-1]['validation_accuracy']:.3f}",
            ),
            reporting.to_highlight(
                "Well-fit regime gap",
                f"{well_fit_gap:.3f}",
            ),
            reporting.to_highlight(
                "Largest backprop gradient norm",
                f"{gradient_trace['weight_grad_norm'].max():.3f}",
            ),
        ],
        next_steps=[
            "Read docs/learning-flow.md before changing any hyperparameters.",
            "Inspect artifacts/decision_boundary_summary.csv next to the PNG figure.",
            (
                "Compare underfit, well-fit, and overfit curves in "
                "artifacts/training_curves.csv."
            ),
        ],
        extra_sections={
            "Generated Artifacts": [path.name for path in artifact_paths.values()],
        },
    )
    reporting.write_markdown_artifact(summary, artifact_paths["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
