import numpy as np

from sota_supervised_showcase.data import (
    build_binary_target,
    build_multilabel_targets,
    load_digits_split,
    load_regression_split,
    rebalance_binary_training_data,
)


def test_load_digits_split_shapes() -> None:
    split = load_digits_split()
    assert split.x_train.shape[1] == 64
    assert split.x_test.shape[1] == 64
    assert len(split.y_train) == split.x_train.shape[0]
    assert len(split.y_test) == split.x_test.shape[0]


def test_binary_and_multilabel_targets() -> None:
    labels = np.array([0, 1, 5, 8, 9])
    binary = build_binary_target(labels, positive_digit=0)
    multilabel = build_multilabel_targets(labels)
    assert binary.tolist() == [1, 0, 0, 0, 0]
    assert multilabel.shape == (5, 2)
    assert multilabel[2].tolist() == [1, 1]  # 5 is large and odd


def test_rebalance_binary_training_data_changes_class_balance() -> None:
    split = load_digits_split()
    y_train_binary = build_binary_target(split.y_train, positive_digit=0)

    x_up, y_up = rebalance_binary_training_data(
        split.x_train, y_train_binary, strategy="upsample_minority"
    )
    x_down, y_down = rebalance_binary_training_data(
        split.x_train, y_train_binary, strategy="downsample_majority"
    )

    up_counts = np.bincount(y_up)
    down_counts = np.bincount(y_down)

    assert x_up.shape[0] == len(y_up)
    assert x_down.shape[0] == len(y_down)
    assert up_counts[0] == up_counts[1]
    assert down_counts[0] == down_counts[1]


def test_load_regression_split_shapes() -> None:
    split = load_regression_split()
    assert split.x_train.shape[1] == 10
    assert split.x_test.shape[1] == 10
    assert split.target_name == "disease_progression"
