"""Package import smoke test for the PyTorch showcase."""

from pytorch_training_regularization_showcase import PROJECT_NAME, __version__


def test_package_metadata_is_exposed() -> None:
    """The top-level package should expose stable metadata."""

    assert PROJECT_NAME == "pytorch-training-regularization-showcase"
    assert __version__ == "0.1.0"
