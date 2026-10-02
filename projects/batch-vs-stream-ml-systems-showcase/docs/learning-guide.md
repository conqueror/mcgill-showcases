# Learning Guide

1. Run `make run-compare` to generate events and both pipelines.
2. Inspect parity differences by window.
3. Change allowed lateness with `uv run python scripts/run_stream.py --allowed-lateness 1 --seed 42`.
   Each run regenerates events from the seed; use the same seed and quick setting to compare inputs.
   The watermark starts at -1 and advances to `max(previous watermark, arrival_time - allowed_lateness)`.
   Events with an earlier event time are dropped.
4. Discuss tradeoffs between latency and completeness.

