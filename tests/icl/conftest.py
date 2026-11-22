"""Fixtures for the ICL tests. Nothing here touches the network, a GPU or vllm."""

import pytest

from icl_helpers import ToyEmbedder


@pytest.fixture
def embedder():
    return ToyEmbedder()
