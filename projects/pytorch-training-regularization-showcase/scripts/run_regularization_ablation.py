#!/usr/bin/env python3
"""Regularization ablation entry point for the PyTorch showcase."""

from __future__ import annotations

import argparse
from pathlib import Path

from pytorch_training_regularization_showcase import (
    config,
    data,
    defaults,
    experiments,
    reporting,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the regularization script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="digits",
        choices=["synthetic", "digits", "fashion_mnist"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the regularization ablation and write its CSV artifact."""

    args = parse_args(argv)
    bundle = data.build_dataset_bundle(
        dataset_name=args.dataset,
        batch_size=32 if args.quick else 64,
        random_state=7,
        quick=args.quick,
    )
    table = experiments.run_regularization_ablation(
        bundle,
        defaults.default_training_config(
            args.dataset,
            args.quick,
            random_state=7,
        ),
    )
    artifact_path = reporting.expected_artifact_paths(args.output_dir)[
        "regularization_ablation"
    ]
    reporting.write_csv_artifact(table, artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
