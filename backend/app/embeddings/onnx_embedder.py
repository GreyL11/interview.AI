import threading

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingError, EmbeddingProvider
from app.model_status import tracker

logger = get_logger(__name__)

_BATCH_SIZE = 32


class OnnxEmbedder(EmbeddingProvider):
    """all-MiniLM-L6-v2 via onnxruntime. Deliberately torch-free: the whole
    packaged app avoids a ~1.5GB dependency this way.

    Model files are downloaded once into the local HF cache under DATA_DIR and
    reused offline afterwards.
    """

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._cache_dir = cache_dir or str(settings.data_dir / "models" / "hf")
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()

    @property
    def dimension(self) -> int:
        return settings.embedding_dimension

    def _ensure_loaded(self) -> None:
        # Lazy: constructing the provider must not download 90MB or block startup.
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise EmbeddingError(f"Embedding backend unavailable: {exc}") from exc

            # A previous failure must not colour this attempt's reporting.
            tracker.reset("embedding")
            tracker.downloading("embedding")
            try:
                # Resolved from the local cache without network when the files
                # are already there, so this is the download step only on a
                # first run.
                model_path = hf_hub_download(
                    self._model_name, "onnx/model.onnx", cache_dir=self._cache_dir
                )
                tokenizer_path = hf_hub_download(
                    self._model_name, "tokenizer.json", cache_dir=self._cache_dir
                )
            except Exception as exc:
                message = (
                    f"Could not download the document search model "
                    f"'{self._model_name}'. The first run needs internet access. "
                    f"Cause: {exc}"
                )
                tracker.failed("embedding", message)
                raise EmbeddingError(message) from exc

            tracker.loading("embedding")
            try:
                tokenizer = Tokenizer.from_file(tokenizer_path)
                tokenizer.enable_truncation(max_length=settings.embedding_max_tokens)
                tokenizer.enable_padding()

                options = ort.SessionOptions()
                options.intra_op_num_threads = 0  # let ORT pick based on the CPU
                session = ort.InferenceSession(
                    model_path, options, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                # Files present but unreadable: a truncated or corrupted cache.
                # Say which directory to delete, because that is the fix.
                message = (
                    f"The document search model files are unreadable and are "
                    f"probably incomplete. Delete '{self._cache_dir}' and try "
                    f"again to re-download them. Cause: {exc}"
                )
                tracker.failed("embedding", message)
                raise EmbeddingError(message) from exc

            self._tokenizer = tokenizer
            self._session = session
            self._input_names = {i.name for i in session.get_inputs()}
            tracker.ready("embedding")
            logger.info("embedding_model_loaded model=%s", self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        self._ensure_loaded()
        out = [self._embed_batch(texts[i : i + _BATCH_SIZE]) for i in range(0, len(texts), _BATCH_SIZE)]
        return np.vstack(out)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )

        token_embeddings = self._session.run(None, inputs)[0]
        return _normalize(_mean_pool(token_embeddings, attention_mask))


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool over real tokens only — padding must not drag vectors toward zero."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, a_min=1e-12, a_max=None)).astype(np.float32)
