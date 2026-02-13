"""Semi-supervised learning methods mapped to the lecture concepts."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import split_two_views

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


def _build_lgbm(random_state: int, n_classes: int) -> LGBMClassifier:
    if n_classes > 2:
        return LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=-1,
            num_leaves=31,
            n_jobs=4,
            random_state=random_state,
            verbosity=-1,
            objective="multiclass",
            num_class=n_classes,
        )
    return LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=-1,
        num_leaves=31,
        n_jobs=4,
        random_state=random_state,
        verbosity=-1,
        objective="binary",
    )


def _evaluate(name: str, y_true: ArrayI, y_pred: ArrayI) -> dict[str, float | str]:
    return {
        "method": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


class Autoencoder(nn.Module):
    """Small MLP autoencoder for tabular representation learning."""

    def __init__(self, input_dim: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def _train_autoencoder(
    X_train: ArrayF,
    random_state: int,
    epochs: int = 25,
    batch_size: int = 128,
    embedding_dim: int = 32,
) -> Autoencoder:
    torch.manual_seed(random_state)
    model = Autoencoder(input_dim=X_train.shape[1], embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

    return model


def _encode(model: Autoencoder, X: ArrayF) -> ArrayF:
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(torch.from_numpy(X).float()).numpy()
    return encoded.astype(np.float64)


def _cluster_pseudolabel(
    X_train: ArrayF,
    y_train_masked: ArrayI,
    n_clusters: int,
    random_state: int,
) -> ArrayI:
    """Assign pseudo-labels per cluster using majority vote from known labels."""

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    cluster_ids = kmeans.fit_predict(X_train)

    pseudo = y_train_masked.copy()
    for cluster_id in np.unique(cluster_ids):
        in_cluster = cluster_ids == cluster_id
        known_labels = y_train_masked[in_cluster]
        known_labels = known_labels[known_labels != -1]
        if known_labels.size == 0:
            continue
        majority_label = int(np.bincount(known_labels).argmax())
        pseudo[(in_cluster) & (pseudo == -1)] = majority_label

    return pseudo


def _co_training_consensus(
    X_train: ArrayF,
    y_train_masked: ArrayI,
    random_state: int,
    n_classes: int,
    confidence_threshold: float = 0.9,
    max_rounds: int = 3,
    max_new_per_round: int = 80,
) -> ArrayI:
    """A practical co-training variant using two feature views and consensus pseudo-labels."""

    X_left, X_right = split_two_views(X_train)
    pseudo_labels = y_train_masked.copy()

    for _ in range(max_rounds):
        labeled_mask = pseudo_labels != -1
        unlabeled_idx = np.flatnonzero(~labeled_mask)
        if unlabeled_idx.size == 0:
            break

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, "
                    "but LGBMClassifier was fitted with feature names"
                ),
                category=UserWarning,
            )
            model_left = _build_lgbm(random_state=random_state, n_classes=n_classes)
            model_right = _build_lgbm(random_state=random_state + 1, n_classes=n_classes)
            model_left.fit(X_left[labeled_mask], pseudo_labels[labeled_mask])
            model_right.fit(X_right[labeled_mask], pseudo_labels[labeled_mask])

            left_proba = model_left.predict_proba(X_left[unlabeled_idx])
            right_proba = model_right.predict_proba(X_right[unlabeled_idx])

        left_pred = left_proba.argmax(axis=1)
        right_pred = right_proba.argmax(axis=1)
        left_conf = left_proba.max(axis=1)
        right_conf = right_proba.max(axis=1)

        consensus = (left_pred == right_pred) & (left_conf >= confidence_threshold) & (
            right_conf >= confidence_threshold
        )

        candidate_idx = unlabeled_idx[consensus]
        candidate_labels = left_pred[consensus]

        if candidate_idx.size == 0:
            break

        # Keep the most confident consensus points first for safer pseudo-labeling.
        scores = np.minimum(left_conf[consensus], right_conf[consensus])
        order = np.argsort(-scores)
        selected = order[:max_new_per_round]

        pseudo_labels[candidate_idx[selected]] = candidate_labels[selected].astype(np.int64)

    return pseudo_labels


def run_semi_supervised_benchmark(
    X_train: ArrayF,
    X_test: ArrayF,
    y_train: ArrayI,
    y_test: ArrayI,
    y_train_masked: ArrayI,
    random_state: int,
    semisup_kmeans_clusters: int,
    self_training_threshold: float,
) -> pd.DataFrame:
    """Run multiple semi-supervised strategies used in this project."""

    rows: list[dict[str, float | str]] = []
    labeled_mask = y_train_masked != -1
    n_classes = len(np.unique(y_train))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, "
                "but LGBMClassifier was fitted with feature names"
            ),
            category=UserWarning,
        )
        baseline = _build_lgbm(random_state=random_state, n_classes=n_classes)
        baseline.fit(X_train[labeled_mask], y_train_masked[labeled_mask])
        baseline_pred = baseline.predict(X_test)
    rows.append(
        _evaluate(
            "Supervised_Only_Labeled",
            y_true=y_test,
            y_pred=baseline_pred.astype(np.int64),
        ),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, "
                "but LGBMClassifier was fitted with feature names"
            ),
            category=UserWarning,
        )
        self_training = SelfTrainingClassifier(
            estimator=_build_lgbm(random_state=random_state, n_classes=n_classes),
            threshold=self_training_threshold,
            max_iter=6,
            verbose=False,
        )
        self_training.fit(X_train, y_train_masked)
        st_pred = self_training.predict(X_test)
    rows.append(_evaluate("SelfTraining_LightGBM", y_true=y_test, y_pred=st_pred.astype(np.int64)))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        label_spreading = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=50)
        label_spreading.fit(X_train, y_train_masked)
        lp_pred = label_spreading.predict(X_test)
    rows.append(_evaluate("LabelSpreading", y_true=y_test, y_pred=lp_pred.astype(np.int64)))

    cluster_pseudo = _cluster_pseudolabel(
        X_train=X_train,
        y_train_masked=y_train_masked,
        n_clusters=semisup_kmeans_clusters,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, "
                "but LGBMClassifier was fitted with feature names"
            ),
            category=UserWarning,
        )
        cluster_model = _build_lgbm(random_state=random_state, n_classes=n_classes)
        cluster_model.fit(X_train[cluster_pseudo != -1], cluster_pseudo[cluster_pseudo != -1])
        cluster_pred = cluster_model.predict(X_test)
    rows.append(
        _evaluate(
            "ClusterPseudoLabel_LightGBM",
            y_true=y_test,
            y_pred=cluster_pred.astype(np.int64),
        ),
    )

    co_train_labels = _co_training_consensus(
        X_train=X_train,
        y_train_masked=y_train_masked,
        random_state=random_state,
        n_classes=n_classes,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, "
                "but LGBMClassifier was fitted with feature names"
            ),
            category=UserWarning,
        )
        co_train_model = _build_lgbm(random_state=random_state, n_classes=n_classes)
        co_train_model.fit(X_train[co_train_labels != -1], co_train_labels[co_train_labels != -1])
        co_train_pred = co_train_model.predict(X_test)
    rows.append(
        _evaluate(
            "CoTraining_Consensus",
            y_true=y_test,
            y_pred=co_train_pred.astype(np.int64),
        ),
    )

    autoencoder = _train_autoencoder(X_train=X_train, random_state=random_state)
    X_train_latent = _encode(autoencoder, X_train)
    X_test_latent = _encode(autoencoder, X_test)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, "
                "but LGBMClassifier was fitted with feature names"
            ),
            category=UserWarning,
        )
        ae_model = _build_lgbm(random_state=random_state, n_classes=n_classes)
        ae_model.fit(X_train_latent[labeled_mask], y_train_masked[labeled_mask])
        ae_pred = ae_model.predict(X_test_latent)
    rows.append(
        _evaluate(
            "AutoencoderFeatures_LightGBM",
            y_true=y_test,
            y_pred=ae_pred.astype(np.int64),
        ),
    )

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
