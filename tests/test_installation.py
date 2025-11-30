"""Tests to check that installation is correct."""


def test_installation():
    """Tests that dependencies are installed correctly."""
    import langchain  # noqa: F401, PLC0415
    import torch  # noqa: F401, PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: F401, PLC0415
