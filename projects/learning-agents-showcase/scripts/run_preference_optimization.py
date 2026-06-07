#!/usr/bin/env python3
"""Preference-tune a toy LM with RLHF/DPO/GRPO/RLVR and write the Lane B comparison artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from learning_agents.preference_optimization import compare_preference_methods
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the preference-optimization runner's command-line flags.

    What + why: this lane tunes the toy LM's *weights* from preferences, so the only knob is how
    long to train each method. Exposes ``--output-dir``, ``--epochs`` (training length applied
    equally to every method; ``None`` defers to the default in ``main``), and ``--quick`` (for CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir``, ``epochs``, and ``quick``.

    RL concept:
        Locus of learning B -- preference optimisation of the policy weights.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train RLHF, DPO, GRPO, and RLVR on the toy LM and write the Lane B artifacts.

    What + why: runs all four preference-optimization methods from the same uniform reference and
    writes ``artifacts/preference/method_comparison.csv`` (final expected quality, win rate, and KL
    per method) and ``artifacts/preference/training_curves.csv`` (each method's per-epoch quality
    and KL). Together they show the "learning the LLM weights" locus: every method -- reward-model
    RLHF, reward-model-free DPO, critic-free GRPO, and verifier-driven RLVR -- lifts the toy policy
    from the ~0.49 reference toward the optimum, by different mechanisms.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Preference optimisation family (RLHF / DPO / GRPO / RLVR) compared like-for-like.
    """
    args = parse_args(argv)
    epochs = args.epochs if args.epochs is not None else (120 if args.quick else 200)
    comparison_rows, curve_rows = compare_preference_methods(epochs=epochs)
    write_csv_artifact(args.output_dir / "preference" / "method_comparison.csv", comparison_rows)
    write_csv_artifact(args.output_dir / "preference" / "training_curves.csv", curve_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
