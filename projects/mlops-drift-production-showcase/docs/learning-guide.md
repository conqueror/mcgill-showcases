# Learning Guide

1. Run `make run` and inspect model quality in `artifacts/metrics/train_eval_summary.csv`.
2. Run `make run-drift` and inspect per-feature drift signals.
3. Open `artifacts/policy/retrain_recommendation.json` and discuss why the policy chose monitor/retrain.
4. Start the API with `make serve` and test with a sample prediction payload.

Key reflection questions:
- What level of drift should trigger retraining in a business context?
- What happens if drift is high but model quality appears stable?

