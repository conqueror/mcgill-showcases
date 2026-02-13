from __future__ import annotations

import numpy as np
import pandas as pd

from feature_dimred_showcase.dimensionality_reduction import embedding_quality, run_embeddings


def test_dimred_contains_core_methods() -> None:
    rng = np.random.default_rng(42)
    x_values = rng.normal(size=(120, 8))
    y = pd.Series(rng.integers(0, 3, size=120))

    embeddings = run_embeddings(x_values, random_state=42, quick=True)
    methods = {item.method for item in embeddings}
    assert {"pca", "tsne"}.issubset(methods)

    quality = embedding_quality(embeddings, y)
    assert "silhouette_vs_label" in quality.columns
