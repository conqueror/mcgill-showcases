# Batch vs Stream ML Systems Showcase

Compare offline batch and near-real-time stream KPI processing on the same event data.

## Learning outcomes
- Understand event-time vs arrival-time behavior.
- Measure parity gaps between batch and stream outputs.
- Inspect the impact of late-event handling.

## Quickstart
```bash
cd projects/batch-vs-stream-ml-systems-showcase
make sync
make run-compare
make verify
```

## Key outputs
- `artifacts/batch/kpi_output.csv`
- `artifacts/stream/kpi_output.csv`
- `artifacts/compare/parity_report.csv`
- `artifacts/compare/latency_throughput_summary.md`

