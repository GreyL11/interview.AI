from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_timeout_seconds: float = 30.0

    # Local storage
    data_dir: Path = Path("./data")

    # Embeddings (ONNX backend — no torch)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_max_tokens: int = 256

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    rag_top_k: int = 5
    rag_min_similarity: float = 0.25
    rag_overfetch: int = 4  # search top_k * this, since SQLite filtering drops hits

    # Session memory
    memory_max_tokens: int = 1200  # verbatim window; older turns get summarised

    # Question detection (only final transcripts ever reach this)
    question_min_words: int = 3
    question_min_confidence: float = 0.5
    question_coalesce_ms: int = 1200  # a follow-on clause corrects, not re-asks

    # Realtime transport
    ws_replay_buffer: int = 200
    api_token: str = ""  # set by the desktop shell at spawn; empty disables the check

    log_level: str = "INFO"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def faiss_path(self) -> Path:
        return self.data_dir / "faiss" / "index.faiss"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "metadata" / "app.db"


settings = Settings()
