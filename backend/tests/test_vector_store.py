import numpy as np
import pytest

from app.vector_store.base import VectorStoreError
from app.vector_store.faiss_store import FaissVectorStore


def unit(values: list[float]) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_add_and_search(vector_store, embedder):
    vectors = embedder.embed(["kafka streaming pipeline", "lemon tart recipe"])
    vector_store.add([10, 20], vectors)

    results = vector_store.search(embedder.embed_query("kafka streaming pipeline"), top_k=2)
    assert results[0][0] == 10
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_on_empty_index_returns_nothing(vector_store, embedder):
    assert vector_store.search(embedder.embed_query("anything"), top_k=5) == []


def test_search_caps_at_index_size(vector_store, embedder):
    vector_store.add([1], embedder.embed(["only one"]))
    assert len(vector_store.search(embedder.embed_query("only one"), top_k=10)) == 1


def test_remove_ids(vector_store, embedder):
    vector_store.add([1, 2, 3], embedder.embed(["alpha", "beta", "gamma"]))
    assert vector_store.size == 3

    assert vector_store.remove([2]) == 1
    assert vector_store.size == 2

    ids = [i for i, _ in vector_store.search(embedder.embed_query("beta"), top_k=5)]
    assert 2 not in ids


def test_remove_missing_id_is_harmless(vector_store, embedder):
    vector_store.add([1], embedder.embed(["alpha"]))
    assert vector_store.remove([999]) == 0
    assert vector_store.size == 1


def test_persist_and_reload(tmp_path, embedder):
    path = tmp_path / "faiss" / "index.faiss"
    store = FaissVectorStore(embedder.dimension, path)
    store.add([7], embedder.embed(["persisted vector"]))
    store.persist()
    assert path.exists()

    reloaded = FaissVectorStore(embedder.dimension, path)
    assert reloaded.size == 1
    assert reloaded.search(embedder.embed_query("persisted vector"), top_k=1)[0][0] == 7


def test_dimension_mismatch_on_add_is_rejected(vector_store):
    with pytest.raises(VectorStoreError, match="dim vectors"):
        vector_store.add([1], np.zeros((1, 999), dtype=np.float32))


def test_mismatched_ids_and_vectors_rejected(vector_store, embedder):
    with pytest.raises(VectorStoreError, match="same length"):
        vector_store.add([1, 2], embedder.embed(["only one"]))


def test_reload_with_wrong_dimension_is_rejected(tmp_path, embedder):
    path = tmp_path / "faiss" / "index.faiss"
    store = FaissVectorStore(embedder.dimension, path)
    store.add([1], embedder.embed(["text"]))
    store.persist()

    with pytest.raises(VectorStoreError, match="expected"):
        FaissVectorStore(embedder.dimension + 1, path).size


def test_adding_nothing_is_a_noop(vector_store):
    vector_store.add([], np.zeros((0, 32), dtype=np.float32))
    assert vector_store.size == 0
