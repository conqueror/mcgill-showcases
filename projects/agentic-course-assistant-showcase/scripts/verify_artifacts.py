#!/usr/bin/env python
"""Validate required agentic course assistant artifacts."""

from pathlib import Path

from agentic_course_assistant.artifact_contract import verify


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors = verify(root)
    if errors:
        raise SystemExit("; ".join(errors))

    print("Artifact contract verified.")


if __name__ == "__main__":
    main()
