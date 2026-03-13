#!/usr/bin/env python3
"""Entry point for the PyTorch training regularization showcase."""

from __future__ import annotations

import argparse
from pathlib import Path

from pytorch_training_regularization_showcase import (
    config,
    data,
    defaults,
    experiments,
    models,
    reporting,
    training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the showcase runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="digits",
        choices=["synthetic", "digits", "fashion_mnist"],
        help="Dataset bundle to train on.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller dataset slice and fewer epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help="Directory where artifacts should be written.",
    )
    return parser.parse_args(argv)


def build_baseline_metrics(
    bundle: data.DatasetBundle,
    baseline_run: training.TrainingResult,
) -> dict[str, float | int | str]:
    """Create the baseline metrics JSON artifact."""

    return {
        "dataset_name": bundle.dataset_name,
        "input_dim": bundle.input_dim,
        "num_classes": bundle.num_classes,
        "best_epoch": baseline_run.best_epoch,
        "best_validation_accuracy": baseline_run.best_validation_accuracy,
        "test_accuracy": baseline_run.test_metrics["test_accuracy"],
        "test_loss": baseline_run.test_metrics["test_loss"],
        "trainable_parameters": models.count_trainable_parameters(baseline_run.model),
    }


def main(argv: list[str] | None = None) -> int:
    """Generate the core showcase artifacts."""

    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = data.build_dataset_bundle(
        dataset_name=args.dataset,
        batch_size=32 if args.quick else 64,
        random_state=7,
        quick=args.quick,
    )
    base_config = defaults.default_training_config(
        args.dataset,
        args.quick,
        random_state=7,
    )
    artifact_paths = reporting.expected_artifact_paths(output_dir)

    baseline_run = training.train_classifier(bundle, base_config)
    optimizer_table = experiments.run_optimizer_comparison(bundle, base_config)
    scheduler_table = experiments.run_scheduler_comparison(bundle, base_config)
    regularization_table = experiments.run_regularization_ablation(bundle, base_config)
    error_analysis = experiments.build_error_analysis_table(baseline_run, bundle)
    gradient_table = training.measure_gradient_health(
        baseline_run.model,
        bundle.train_loader,
    )

    reporting.write_json_artifact(
        build_baseline_metrics(bundle, baseline_run),
        artifact_paths["baseline_metrics"],
    )
    reporting.write_csv_artifact(
        baseline_run.history,
        artifact_paths["training_history"],
    )
    reporting.write_csv_artifact(
        optimizer_table,
        artifact_paths["optimizer_comparison"],
    )
    reporting.write_csv_artifact(
        scheduler_table,
        artifact_paths["learning_rate_schedule_comparison"],
    )
    reporting.write_csv_artifact(
        regularization_table,
        artifact_paths["regularization_ablation"],
    )
    reporting.write_csv_artifact(
        error_analysis,
        artifact_paths["error_analysis"],
    )
    reporting.write_markdown_artifact(
        reporting.build_gradient_health_report_markdown(
            gradient_table,
            bundle.dataset_name,
        ),
        artifact_paths["gradient_health_report"],
    )

    best_optimizer = optimizer_table.sort_values(
        "best_validation_accuracy",
        ascending=False,
    ).iloc[0]
    best_regularization = regularization_table.sort_values(
        "best_validation_accuracy",
        ascending=False,
    ).iloc[0]

    summary = reporting.build_summary_markdown(
        project_title="PyTorch Training Regularization Showcase",
        highlights=[
            reporting.to_highlight(
                "Baseline test accuracy",
                f"{baseline_run.test_metrics['test_accuracy']:.3f}",
            ),
            reporting.to_highlight(
                "Best optimizer",
                best_optimizer["optimizer"],
            ),
            reporting.to_highlight(
                "Best regularization setup",
                best_regularization["experiment"],
            ),
        ],
        next_steps=[
            (
                "Inspect artifacts/training_history.csv before changing any "
                "hyperparameters."
            ),
            (
                "Compare optimizer_comparison.csv against "
                "learning_rate_schedule_comparison.csv."
            ),
            "Use gradient_health_report.md to reason about stability and signal flow.",
        ],
        extra_sections={
            "Generated Artifacts": [path.name for path in artifact_paths.values()],
        },
    )
    reporting.write_markdown_artifact(summary, artifact_paths["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
