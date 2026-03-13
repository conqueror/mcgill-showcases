#!/usr/bin/env python3
"""Entry point for the deep learning math foundations showcase."""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_learning_math_foundations_showcase import (
    calculus,
    config,
    information_theory,
    linear_algebra,
    optimization,
    probability,
    reporting,
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


def main(argv: list[str] | None = None) -> int:
    """Generate the core showcase artifacts."""

    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = reporting.expected_artifact_paths(output_dir)
    vector_table = linear_algebra.build_vector_operations_table()
    matrix_table = linear_algebra.build_matrix_transformations_table()
    derivative_table = calculus.build_derivative_examples_table()
    probability_table = probability.build_probability_simulations_table()
    gradient_trace = optimization.run_gradient_descent_trace()
    info_summary = information_theory.build_information_theory_summary_markdown()

    reporting.write_csv_artifact(vector_table, artifact_paths["vector_operations"])
    reporting.write_csv_artifact(
        matrix_table,
        artifact_paths["matrix_transformations"],
    )
    reporting.write_csv_artifact(
        derivative_table,
        artifact_paths["derivative_examples"],
    )
    reporting.write_csv_artifact(
        gradient_trace,
        artifact_paths["gradient_descent_trace"],
    )
    reporting.write_csv_artifact(
        probability_table,
        artifact_paths["probability_simulations"],
    )
    reporting.write_markdown_artifact(
        info_summary,
        artifact_paths["information_theory_summary"],
    )

    summary = reporting.build_summary_markdown(
        project_title="Deep Learning Math Foundations Showcase",
        highlights=[
            reporting.to_highlight(
                "Vector addition result",
                vector_table.iloc[0]["result"],
            ),
            reporting.to_highlight(
                "Gradient descent final x",
                f"{gradient_trace.iloc[-1]['x']:.6f}",
            ),
            reporting.to_highlight(
                "Binary entropy",
                f"{information_theory.entropy([0.5, 0.5]):.6f} bits",
            ),
        ],
        next_steps=[
            "Read docs/learning-flow.md to connect each artifact to the study path.",
            (
                "Inspect artifacts/gradient_descent_trace.csv to see iterative "
                "optimization."
            ),
            (
                "Compare entropy and cross-entropy in "
                "artifacts/information_theory_summary.md."
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
