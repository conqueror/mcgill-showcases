#!/usr/bin/env python
"""Train and persist the demand API demo model bundle under artifacts/."""

from __future__ import annotations

from pathlib import Path

from demand_api_observability_showcase.model.demo_training import train_demo_model


def main() -> None:
    """CLI entrypoint for demo training used by Makefile targets."""

    train_demo_model(Path("artifacts"))


if __name__ == "__main__":
    main()
