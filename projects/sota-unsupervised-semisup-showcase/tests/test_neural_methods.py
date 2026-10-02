import numpy as np
import pytest
import torch

from sota_showcase.dec import DECModel, run_dec_benchmark
from sota_showcase.self_supervised import ContrastiveMLP, _nt_xent_loss, _train_contrastive_encoder
from sota_showcase.semi_supervised import _co_training_consensus


def test_co_training_keeps_noncontiguous_class_ids() -> None:
    X = np.array(
        [[0.0, 0.0, 0.0, 0.0]] * 30
        + [[10.0, 10.0, 10.0, 10.0]] * 30
        + [[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]]
    )
    masked = np.array([2] * 30 + [4] * 30 + [-1, -1], dtype=np.int64)
    labels = _co_training_consensus(X, masked, random_state=7, n_classes=2, max_rounds=1)
    np.testing.assert_array_equal(labels[-2:], [2, 4])
    np.testing.assert_array_equal(labels[:-2], masked[:-2])


def test_dec_collapsed_solution_reports_nan_silhouette() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    output = run_dec_benchmark(
        X,
        np.array([0, 0, 1, 1]),
        random_state=7,
        pretrain_epochs=1,
        finetune_epochs=1,
        n_clusters=1,
        latent_dim=2,
        batch_size=2,
    )
    assert output.metrics["silhouette"].isna().all()
    assert output.metrics.iloc[0]["algorithm"] == "DEC_minibatch_reconstruction"


def test_small_contrastive_dataset_actually_updates_encoder() -> None:
    torch.manual_seed(7)
    initial = ContrastiveMLP(input_dim=3, embedding_dim=4)
    X = np.array([[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    trained = _train_contrastive_encoder(
        X, random_state=7, embedding_dim=4, epochs=1, batch_size=128
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            initial.encoder.parameters(), trained.encoder.parameters(), strict=True
        )
    )


@pytest.mark.parametrize("n_rows", [0, 1])
def test_contrastive_training_rejects_too_few_rows(n_rows: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        _train_contrastive_encoder(np.ones((n_rows, 3)), 7, 2, 1, 128)


def test_dec_target_distribution_matches_hand_calculation() -> None:
    # Frequencies are 6/5 and 4/5. q^2/f then row normalization gives
    # (8/15, 1/20)/(7/12) and (2/15, 9/20)/(7/12).
    q = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float64)
    expected = torch.tensor([[32 / 35, 3 / 35], [8 / 35, 27 / 35]], dtype=torch.float64)
    torch.testing.assert_close(DECModel.target_distribution(q), expected)


def test_nt_xent_matches_hand_calculation() -> None:
    # Two orthogonal pairs: each anchor has a positive exp(1) and two negatives exp(0).
    # Loss per anchor is log(e + 2) - 1; all four anchors have that same loss.
    views = torch.eye(2, dtype=torch.float64)
    loss = _nt_xent_loss(views, views, temperature=1.0)
    assert loss.item() == pytest.approx(0.5514447139320511)
