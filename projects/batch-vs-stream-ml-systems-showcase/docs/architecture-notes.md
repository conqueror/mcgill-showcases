# Architecture Notes

- Batch mode aggregates by event time on full data.
- Stream mode aggregates by arrival order with watermark and lateness limits.
- Reconciliation compares KPI parity and quantifies divergence.

