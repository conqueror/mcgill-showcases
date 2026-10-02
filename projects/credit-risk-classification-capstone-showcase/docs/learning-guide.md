# Learning Guide

1. Run `make run` to generate full diagnostics and model artifacts.
2. Read `artifacts/leakage/leakage_report.csv` before trusting metrics.
3. Inspect `artifacts/models/strategy_comparison.csv` for imbalance strategy effects.
4. Use the validation-only `artifacts/eval/threshold_analysis.csv` to choose an F1 threshold; inspect the locked threshold's test metrics before interpreting the result. This table does not measure calibration or decision costs.
5. Open notebook `notebooks/02_modeling_thresholds.ipynb` to narrate your decision.
