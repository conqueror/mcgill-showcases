"""Educational Deep Embedded Clustering (DEC) implementation in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


@dataclass(frozen=True)
class DECArtifacts:
    """DEC metrics and labels for reporting/visualization."""

    metrics: pd.DataFrame
    labels: ArrayI


class DECAutoencoder(nn.Module):
    """Autoencoder backbone used by DEC."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction


class DECModel(nn.Module):
    """DEC objective on top of pretrained encoder."""

    def __init__(self, autoencoder: DECAutoencoder, n_clusters: int, latent_dim: int) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Student-t kernel from DEC paper."""

        distances = torch.cdist(z, self.cluster_centers) ** 2
        q = 1.0 / (1.0 + distances)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = q**2 / torch.sum(q, dim=0, keepdim=True)
        return weight / torch.sum(weight, dim=1, keepdim=True)


def run_dec_benchmark(
    X: ArrayF,
    y: ArrayI,
    random_state: int,
    pretrain_epochs: int,
    finetune_epochs: int,
    n_clusters: int = 10,
    latent_dim: int = 16,
    batch_size: int = 256,
) -> DECArtifacts:
    """Run an intentionally compact DEC example for tutorial demonstration."""

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    autoencoder = DECAutoencoder(input_dim=X.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    reconstruction_loss = nn.MSELoss()

    tensor_data = torch.from_numpy(X).float()
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoencoder.train()
    for _ in range(pretrain_epochs):
        for (batch,) in loader:
            _, recon = autoencoder(batch)
            loss = reconstruction_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    autoencoder.eval()
    with torch.no_grad():
        latent = autoencoder.encoder(tensor_data).numpy()

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    kmeans_labels = kmeans.fit_predict(latent)

    dec_model = DECModel(autoencoder=autoencoder, n_clusters=n_clusters, latent_dim=latent_dim)
    dec_model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    dec_optimizer = torch.optim.Adam(dec_model.parameters(), lr=1e-3)

    dec_model.train()
    for _ in range(finetune_epochs):
        for (batch,) in loader:
            z, recon = dec_model.autoencoder(batch)
            q = dec_model.soft_assign(z)
            p = dec_model.target_distribution(q).detach()

            kl = F.kl_div(torch.log(q + 1e-9), p, reduction="batchmean")
            rec = reconstruction_loss(recon, batch)
            loss = kl + 0.1 * rec

            dec_optimizer.zero_grad()
            loss.backward()
            dec_optimizer.step()

    dec_model.eval()
    with torch.no_grad():
        z_final = dec_model.autoencoder.encoder(tensor_data)
        q_final = dec_model.soft_assign(z_final)
        dec_labels = q_final.argmax(dim=1).numpy().astype(np.int64)

    metrics = pd.DataFrame(
        [
            {
                "algorithm": "DEC",
                "ari": float(adjusted_rand_score(y, dec_labels)),
                "nmi": float(normalized_mutual_info_score(y, dec_labels)),
                "silhouette": float(silhouette_score(X, dec_labels)),
                "pretrain_epochs": pretrain_epochs,
                "finetune_epochs": finetune_epochs,
                "n_clusters": n_clusters,
            },
            {
                "algorithm": "KMeans_on_Pretrained_Latent",
                "ari": float(adjusted_rand_score(y, kmeans_labels)),
                "nmi": float(normalized_mutual_info_score(y, kmeans_labels)),
                "silhouette": float(silhouette_score(X, kmeans_labels)),
                "pretrain_epochs": pretrain_epochs,
                "finetune_epochs": 0,
                "n_clusters": n_clusters,
            },
        ],
    )

    return DECArtifacts(metrics=metrics, labels=dec_labels)
