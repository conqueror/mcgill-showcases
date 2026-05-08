# Architecture Notes

This showcase intentionally uses one shared corpus and one script-first execution path.

## Pipeline Shape

1. Load the local corpus of research abstracts and summaries.
2. Build chunk-level passages for retrieval.
3. Fit a lexical baseline with `TF-IDF`.
4. Build dense representations with a compact sentence encoder when available.
5. Compare lexical and dense topic classification.
6. Compare lexical and dense retrieval on the same query set.
7. Run retrieval-grounded QA and summarization.
8. Write stable artifacts and verify them.

## Why this design works for teaching

- It avoids three disconnected NLP demos.
- It makes sparse-vs-dense tradeoffs visible on the same documents.
- It keeps retrieval quality tied to downstream generation quality.
- It remains runnable on a laptop with a fast fallback mode.
