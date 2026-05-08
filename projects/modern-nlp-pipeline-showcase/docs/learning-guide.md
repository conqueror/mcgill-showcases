# Learning Guide

## Suggested order

1. Run `make smoke`.
2. Read `artifacts/summary.md`.
3. Inspect `artifacts/classification/metrics_summary.csv`.
4. Inspect `artifacts/retrieval/retrieval_metrics.csv`.
5. Open `artifacts/retrieval/retrieval_examples.json`.
6. Open `artifacts/generation/qa_outputs.csv`.
7. Open `artifacts/generation/query_summaries.json`.

## Questions to ask yourself

- Which queries reward exact terminology and help lexical retrieval?
- Which queries require semantic matching and help dense retrieval?
- Does the better retriever also support better grounded answers?
- Which classification baseline is easier to explain?
- Which artifact gives the clearest evidence that a model is grounded?

## Interpretation prompts

- Explain the best classifier in one sentence.
- Explain the best retriever in one sentence.
- Pick one query where lexical retrieval wins and explain why.
- Pick one query where dense retrieval wins and explain why.
- Compare a generated answer to its supporting passages and judge whether it is grounded.
