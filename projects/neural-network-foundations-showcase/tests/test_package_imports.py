"""Package import smoke test for the neural network foundations showcase."""

from neural_network_foundations_showcase import PROJECT_NAME, __version__


def test_package_metadata_is_exposed() -> None:
    """The top-level package should expose stable metadata."""

    assert PROJECT_NAME == "neural-network-foundations-showcase"
    assert __version__ == "0.1.0"
