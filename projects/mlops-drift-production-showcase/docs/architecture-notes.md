# Architecture Notes

Pipeline stages:
1. Data generation/load.
2. Baseline model train + evaluate.
3. Tracking writeback.
4. Drift monitor against reference data.
5. Policy recommendation.
6. Serving layer via FastAPI.

