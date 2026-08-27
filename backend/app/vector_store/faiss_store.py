import threading
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.vector_store.base import VectorStore, VectorStoreError

logger = get_logger(__name__)


class FaissVectorStore(VectorStore):
    """IndexIDMap2 over IndexFlatIP.

    Flat + inner product on unit vectors == exact cosine similarity, so there is
    no training step, no recall loss, and no tuning. IndexIDMap2 lets us own the
    ids (allocated from SQLite) and supports remove_ids, which the IVF/HNSW
    families do not do cleanly.
    """

    def __init__(self, dimension: int, index_path: Path) -> None:
        self._dimension = dimension
        self._index_path = index_path
        self._lock = threading.Lock()
        self._index = None

    def _faiss(self):
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError(f"faiss is not available: {exc}") from exc
        return faiss

    def _ensure_index(self):
        if self._index is not None:
            return self._index
        faiss = self._faiss()
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            logger.info("faiss_index_loaded vectors=%d", self._index.ntotal)
            if self._index.d != self._dimension:
                raise VectorStoreError(
                    f"Index at {self._index_path} has dimension {self._index.d}, "
                    f"expected {self._dimension}. Delete it to re-ingest with the "
                    "current embedding model."
                )
        else:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
            logger.info("faiss_index_created dim=%d", self._dimension)
        return self._index

    @property
    def size(self) -> int:
        with self._lock:
            return self._ensure_index().ntotal

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise VectorStoreError("ids and vectors must be the same length")
        if len(ids) == 0:
            return
        if vectors.shape[1] != self._dimension:
            raise VectorStoreError(
                f"Expected {self._dimension}-dim vectors, got {vectors.shape[1]}"
            )
        with self._lock:
            index = self._ensure_index()
            index.add_with_ids(
                np.ascontiguousarray(vectors, dtype=np.float32),
                np.asarray(ids, dtype=np.int64),
            )

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        with self._lock:
            index = self._ensure_index()
            if index.ntotal == 0 or top_k <= 0:
                return []
            vector = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
            scores, ids = index.search(vector, min(top_k, index.ntotal))
        # faiss pads with -1 when fewer than top_k results exist.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        faiss = self._faiss()
        with self._lock:
            index = self._ensure_index()
            selector = faiss.IDSelectorArray(np.asarray(ids, dtype=np.int64))
            return int(index.remove_ids(selector))

    def persist(self) -> None:
        faiss = self._faiss()
        with self._lock:
            index = self._ensure_index()
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file then replace: a crash mid-write must not
            # leave a truncated index that fails to load on next start.
            tmp = self._index_path.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(tmp))
            tmp.replace(self._index_path)
