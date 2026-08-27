import os

import numpy as np
import pytest

from app.embeddings.base import EmbeddingProvider
from tests.fakes import FakeEmbedder

# The real model needs a ~90MB download on first use. Opt in explicitly so the
# suite stays offline and deterministic by default.
RUN_MODEL_TESTS = os.environ.get("RUN_MODEL_TESTS") == "1"


def test_fake_embedder_satisfies_the_interface():
    assert isinstance(FakeEmbedder(), EmbeddingProvider)


def test_vectors_are_unit_length():
    vectors = FakeEmbedder().embed(["kafka pipeline", "lemon tart"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedding_is_deterministic():
    a = FakeEmbedder().embed(["same text"])
    b = FakeEmbedder().embed(["same text"])
    assert np.array_equal(a, b)


def test_embed_empty_list():
    result = FakeEmbedder().embed([])
    assert result.shape == (0, 32)


def test_embed_query_returns_one_dimension():
    assert FakeEmbedder().embed_query("hello").shape == (32,)


def test_blank_text_still_yields_a_unit_vector():
    vector = FakeEmbedder().embed_query("   ")
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_shared_vocabulary_scores_higher_than_unrelated():
    e = FakeEmbedder()
    m = e.embed([
        "kafka streaming ingestion pipeline",
        "kafka streaming ingestion service",
        "lemon tart dessert recipe",
    ])
    assert float(m[0] @ m[1]) > float(m[0] @ m[2])


@pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set RUN_MODEL_TESTS=1 to exercise the real ONNX model")
def test_real_onnx_embedder():
    from app.embeddings.onnx_embedder import OnnxEmbedder

    embedder = OnnxEmbedder()
    m = embedder.embed([
        "I built a Kafka ingestion pipeline.",
        "I designed a streaming ingestion system with Kafka.",
        "My favourite dessert is lemon tart.",
    ])
    assert m.shape == (3, embedder.dimension)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert float(m[0] @ m[1]) > float(m[0] @ m[2]) + 0.2
