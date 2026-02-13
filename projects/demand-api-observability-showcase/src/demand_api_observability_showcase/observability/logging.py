from __future__ import annotations

import logging
import sys


def init_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
