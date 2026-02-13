from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class EmbeddingResult:
    method: str
    embedding: npt.NDArray[np.float64]


def run_embeddings(
    x_values: npt.NDArray[np.float64],
    *,
    random_state: int = 42,
    quick: bool = False,
) -> list[EmbeddingResult]:
    pca = PCA(n_components=2, random_state=random_state)
    tsne_perplexity = 20 if quick else 30
    tsne = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=tsne_perplexity)

    outputs = [
        EmbeddingResult(method="pca", embedding=pca.fit_transform(x_values)),
        EmbeddingResult(method="tsne", embedding=tsne.fit_transform(x_values)),
    ]

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        outputs.append(EmbeddingResult(method="umap", embedding=reducer.fit_transform(x_values)))
    except Exception:
        pass

    return outputs


def embedding_quality(
    embeddings: list[EmbeddingResult],
    labels: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    y = labels.to_numpy()
    for item in embeddings:
        score = silhouette_score(item.embedding, y)
        rows.append(
            {
                "method": item.method,
                "silhouette_vs_label": float(score),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(by="silhouette_vs_label", ascending=False)
        .reset_index(drop=True)
    )
