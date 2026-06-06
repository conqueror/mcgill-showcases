"""Bootstrap ``sys.path`` so the test suite imports the showcase without installation.

This conftest pins one thing only: that ``student_support_rl`` (under ``src/``) and the
top-level ``scripts`` package resolve during collection, even when the project is run from a
fresh checkout that has not been ``pip``/``uv`` installed. It adds the project root and its
``src/`` directory to ``sys.path`` so the value-based, bandit, and policy-gradient modules
on the RL ladder are importable by every test. No fixtures and no RL behavior live here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root = the directory above tests/; src/ holds the importable package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
