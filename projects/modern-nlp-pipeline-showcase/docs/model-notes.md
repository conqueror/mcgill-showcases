# Model Notes

## Lexical baseline

The lexical path uses `TF-IDF` with unigrams and bigrams.

Why it matters:

- strong baseline for exact terminology,
- fast and transparent,
- easy to inspect and explain.

## Dense encoder

The default dense encoder target is `BAAI/bge-small-en-v1.5`.

Why this is a good fit here:

- modern embedding model,
- compact enough for local experimentation,
- suitable for sentence and passage retrieval.

## Fallback behavior

The project includes deterministic fallback backends for:

- dense sentence encoding,
- question answering,
- summarization.

Why that matters:

- tests remain offline and reproducible,
- `make smoke` stays fast,
- the project still teaches the pipeline even when model downloads are unavailable.

## Generation caveat

Retrieval-grounded QA and summarization are only as good as the retrieved evidence. If the retriever surfaces weak passages, the generation step will usually degrade too.
