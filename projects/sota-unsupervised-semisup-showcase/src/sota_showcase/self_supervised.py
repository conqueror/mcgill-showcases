"""Self-supervised representation learning on tabularized digit images."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


@dataclass(frozen=True)
class SelfSupervisedArtifacts:
    """Outputs required by downstream plotting and reporting."""

    metrics: pd.DataFrame
    train_embeddings: ArrayF
    test_embeddings: ArrayF


class ContrastiveMLP(nn.Module):
    """Encoder + projection head for SimCLR-style learning."""

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        projection = self.projector(embedding)
        return embedding, projection


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Create an augmented tabular view with noise + random feature masking."""

    noise = torch.randn_like(x) * 0.08
    mask = (torch.rand_like(x) > 0.1).float()
    return (x + noise) * mask


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.mm(z, z.t()) / temperature

    diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity.device)
    similarity = similarity.masked_fill(diag_mask, -1e9)

    positives = torch.cat([
        torch.diag(similarity, batch_size),
        torch.diag(similarity, -batch_size),
    ])
    denominator = torch.logsumexp(similarity, dim=1)

    return -(positives - denominator).mean()


def _train_contrastive_encoder(
    X_train: ArrayF,
    random_state: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
) -> ContrastiveMLP:
    torch.manual_seed(random_state)
    model = ContrastiveMLP(input_dim=X_train.shape[1], embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            view_1 = _augment(batch)
            view_2 = _augment(batch)
            _, proj_1 = model(view_1)
            _, proj_2 = model(view_2)
            loss = _nt_xent_loss(proj_1, proj_2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def _encode(model: ContrastiveMLP, X: ArrayF) -> ArrayF:
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(torch.from_numpy(X).float())
    return embeddings.numpy().astype(np.float64)


def _evaluate_predictions(name: str, y_true: ArrayI, y_pred: ArrayI) -> dict[str, float | str]:
    return {
        "method": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def _build_lgbm(random_state: int, n_classes: int) -> LGBMClassifier:
    if n_classes > 2:
        return LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            n_jobs=4,
            random_state=random_state,
            verbosity=-1,
            objective="multiclass",
            num_class=n_classes,
        )
    return LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        n_jobs=4,
        random_state=random_state,
        verbosity=-1,
        objective="binary",
    )


def run_self_supervised_benchmark(
    X_train: ArrayF,
    X_test: ArrayF,
    y_train: ArrayI,
    y_test: ArrayI,
    y_train_masked: ArrayI,
    random_state: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
) -> SelfSupervisedArtifacts:
    """Train a contrastive encoder and evaluate downstream classifiers."""

    model = _train_contrastive_encoder(
        X_train=X_train,
        random_state=random_state,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
    )

    X_train_embed = _encode(model, X_train)
    X_test_embed = _encode(model, X_test)

    labeled_mask = y_train_masked != -1
    n_classes = len(np.unique(y_train))
    rows: list[dict[str, float | str]] = []

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
        baseline.fit(X_train[labeled_mask], y_train[labeled_mask])
        baseline_pred = baseline.predict(X_test)
    rows.append(
        _evaluate_predictions(
            "RawFeatures_LightGBM",
            y_test,
            baseline_pred.astype(np.int64),
        ),
    )

    linear_probe = LogisticRegression(max_iter=2000)
    linear_probe.fit(X_train_embed[labeled_mask], y_train[labeled_mask])
    linear_pred = linear_probe.predict(X_test_embed)
    rows.append(
        _evaluate_predictions(
            "ContrastiveEmbedding_LinearProbe",
            y_test,
            linear_pred.astype(np.int64),
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
        lgbm_embed = _build_lgbm(random_state=random_state, n_classes=n_classes)
        lgbm_embed.fit(X_train_embed[labeled_mask], y_train[labeled_mask])
        lgbm_embed_pred = lgbm_embed.predict(X_test_embed)
    rows.append(
        _evaluate_predictions(
            "ContrastiveEmbedding_LightGBM",
            y_test,
            lgbm_embed_pred.astype(np.int64),
        ),
    )

    metrics = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return SelfSupervisedArtifacts(
        metrics=metrics,
        train_embeddings=X_train_embed,
        test_embeddings=X_test_embed,
    )
