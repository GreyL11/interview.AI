from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_timeout_seconds: float = 30.0
    gemini_fallback_models: str = ""
    gemini_fallback_model: str = ""
    gemini_retry_max_attempts: int = 3
    gemini_retry_initial_delay_seconds: float = 0.5
    gemini_retry_max_delay_seconds: float = 4.0

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
    # 2, not 3: real prompts get this short ("Explain decorators"). One-word
    # noise is still filtered, and acknowledgements are rejected by pattern
    # rather than by length.
    question_min_words: int = 2
    question_min_confidence: float = 0.5
    question_coalesce_ms: int = 1200  # a follow-on clause corrects, not re-asks
    # How far back a rejected interviewer utterance ("just write a character
    # count program") can still be prepended as context once the very next
    # utterance turns out to be the actual question. Deliberately short: this
    # bridges two halves of one thought, not a running transcript history.
    question_context_window_ms: int = 4000
    # How long after the last accepted question a short fragment ("Why?", "How?")
    # may still count as a follow-up rather than noise. Long enough to cover
    # "candidate read the streamed answer, then asked a quick follow-up".
    question_followup_window_ms: int = 45_000
    # How long to hold a question that looks mid-clause ("...what happens
    # when") before asking it, giving a quick continuation a chance to
    # supersede it first. A complete question is never delayed -- see
    # question_detector._looks_incomplete.
    question_stabilization_ms: int = 400
    # Off by default: logs every detector decision (accepted or rejected,
    # with category/confidence/reason) via the existing structured metrics
    # logger, for inspecting real-world detector behavior during manual
    # testing. Never covers MIC audio -- that never reaches the detector.
    question_detector_diagnostics: bool = False

    # Realtime transport
    ws_replay_buffer: int = 200
    api_token: str = ""  # set by the desktop shell at spawn; empty disables the check

    # Audio capture
    audio_queue_frames: int = 200  # ~6s at 32ms/frame before dropping oldest
    audio_capture_mic: bool = True        # candidate, recorded for review
    audio_capture_loopback: bool = True   # interviewer, drives question detection

    # Voice activity detection
    vad_threshold: float = 0.5
    vad_start_frames: int = 2        # ~64ms of speech to open an utterance
    vad_silence_ms: int = 700        # sustained silence to close one
    vad_min_utterance_ms: int = 300  # below this it is a click, not speech
    vad_max_utterance_ms: int = 30_000

    # Speech to text
    stt_model: str = "distil-small.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 5
    stt_language: str = "en"
    stt_partial_interval_ms: int = 1000

    # STT execution. One shared CTranslate2 model serves every channel, so
    # concurrency here is what the model is actually told to support: raising
    # it without raising num_workers just moves the queue into C++.
    stt_inference_concurrency: int = 1
    stt_cpu_threads: int = 0  # 0 = let CTranslate2 size its own intra-op pool

    # Partial transcripts are display-only best effort; finals are the product.
    stt_enable_partials: bool = True
    stt_partial_min_audio_ms: int = 500      # below this a partial says nothing useful
    stt_max_partials_per_utterance: int = 4  # bounds re-transcription of one utterance

    # Inference queue bands, lowest number first out. Every final outranks
    # every partial regardless of channel -- a candidate-final must never wait
    # behind an interviewer-partial -- and loopback still wins within each
    # band, since only the interviewer's final transcript can produce an answer.
    stt_priority_loopback_final: int = 0
    stt_priority_mic_final: int = 1
    stt_priority_loopback_partial: int = 2
    stt_priority_mic_partial: int = 3

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
