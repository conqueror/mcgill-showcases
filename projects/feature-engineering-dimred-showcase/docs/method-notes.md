# Method Notes

- One-hot encoding avoids fake ordinality but can increase dimensionality.
- L1 scores take the largest absolute coefficient across classes; adding mutual information is a heuristic because the two scores are not normalized.
- t-SNE helps visualization, but not direct downstream modeling by default.
- The advanced tsfresh example treats four different Iris measurements as artificial positions 0–3. It demonstrates the API, not a chronological feature window or an as-of cutoff. Advanced autofeat output is fitted on the whole demonstration frame and is not a held-out evaluation.
