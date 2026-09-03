import logging
import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Folder name under %LOCALAPPDATA% used when the desktop shell did not pass
#: --data-dir. Matches the Tauri bundle identifier so both processes agree on
#: one location even if they disagree about who chose it.
APP_IDENTIFIER = "com.callassistant.desktop"

#: Identifiers this app shipped under before it was renamed. The desktop
#: shell derives the per-user data directory from the bundle identifier, so
#: changing the name points the app at an empty folder and every existing
#: user silently loses their documents, session history and ~350MB of
#: downloaded models. `migrate_legacy_data_dir` moves the old one across.
LEGACY_APP_IDENTIFIERS = ("com.interviewcoach.desktop",)


def default_data_dir() -> Path:
    """Where mutable state goes when nothing overrode it.

    A frozen build must never write beside its executable: a per-user NSIS
    install can land somewhere read-only, and `./data` would then resolve
    against whatever working directory Explorer happened to hand the process --
    in practice `C:\\Windows\\System32`. The desktop shell normally passes
    --data-dir; this is the fallback for when resolving that path failed, and
    for anyone running the frozen exe directly.

    Source checkouts keep the old `./data` so a developer's tree stays
    self-contained.
    """
    if not getattr(sys, "frozen", False):
        return Path("./data")

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / APP_IDENTIFIER
    return Path.home() / ".call-assistant"


class Settings(BaseSettings):
    # `.env` is a *development* convenience and is resolved relative to the
    # working directory. A frozen build inherits whatever directory it was
    # launched from -- the install folder, the user's desktop, System32 -- so
    # honouring `.env` there means the app's configuration depends on where its
    # icon happened to be double-clicked from. Worse, it lets an unrelated
    # `.env` sitting in that directory silently supply an API key.
    #
    # The packaged app is configured by its command line (from the desktop
    # shell) and the OS credential store, both of which are unambiguous.
    model_config = SettingsConfigDict(
        env_file=None if getattr(sys, "frozen", False) else ".env",
        extra="ignore",
    )

    # --- LLM (Groq is the only cloud provider) ---
    groq_api_key: str = ""
    #: The single place a model is named. Everything else reads
    #: `settings.groq_model`, so changing it here (or via PUT /settings) is the
    #: whole change. Validated by app.llm.groq_client.validate_model.
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 30.0
    #: Passed to the Groq SDK, which retries only connection errors, 408, 429
    #: and 5xx. Deterministic failures (bad key, unknown model) are never
    #: retried by the SDK and must never be retried here either.
    groq_max_retries: int = 2

    # Local storage
    data_dir: Path = default_data_dir()

    # Embeddings (ONNX backend — no torch)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_max_tokens: int = 256

    # Document OCR (scanned PDFs). Runs only on pages native extraction could
    # not read -- see app.documents.parsers.pdf.
    #: Below this many characters, a page is treated as a picture of text rather
    #: than text. Not zero: scanners stamp headers and page numbers onto scanned
    #: pages, so an emptiness test would pass a page that has no real content.
    ocr_min_chars_per_page: int = 60
    #: Rasterisation resolution. Below ~200 recognition accuracy on body text
    #: drops sharply; above ~300 the extra pixels cost time without helping.
    ocr_render_dpi: int = 220
    #: Ceiling on pages OCR'd per document. At roughly a second a page, this
    #: bounds one upload's hold on the ingest lock to a few minutes.
    ocr_max_pages: int = 60
    #: Cores OCR may use. 0 picks a quarter of the machine, capped at four.
    #: Bounded on purpose: ingestion can overlap a live interview, and OCR
    #: saturating every core would push Whisper inference -- which is on the
    #: critical path to an answer -- behind background document work.
    ocr_threads: int = 0

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
    question_context_window_ms: int = 8000
    # How long after the last accepted question a short fragment ("Why?", "How?")
    # may still count as a follow-up rather than noise. Long enough to cover
    # "candidate read the streamed answer, then asked a quick follow-up".
    question_followup_window_ms: int = 45_000
    # --- turn accumulation budgets -------------------------------------
    # All three are windows on the SPEECH clock, and all three are spent only
    # by a turn that is not yet answerable. A complete one-shot question and
    # an obvious follow-up are held for 0ms and are unaffected by every value
    # below.
    #
    # The sizing is not arbitrary. Two fragments of one spoken turn have their
    # speech_end timestamps separated by `pause + duration(next fragment)` --
    # each fragment stamps speech_end vad_silence_ms after its own audio ends,
    # so those two closes cancel out -- and the next fragment is *delivered*
    # one STT pass after that. Accumulation therefore only catches a
    # continuation when:
    #
    #     pause + duration(next fragment) + stt_inference  <=  hold
    #
    # For natural speech that right-hand side is 1.6-2.8s (a 300ms pause plus
    # a 1.5s clause plus ~150ms inference is 1950ms). Anything under ~2s
    # cannot assemble a fragmented question at all -- it fires on the first
    # fragment and then pays a second provider call to correct itself.
    #
    # How long to hold a request that is grammatically whole but worded the
    # way interviewers word one with a constraint still coming ("find two
    # numbers", "tell me about a time"). It may already be finished, so this
    # is the smallest of the three.
    question_stabilization_ms: int = 2500
    # How long to hold a request that is provably *not* whole yet -- a
    # dangling conjunction, a bare "Can you explain", a premise with nothing
    # asked for. Longer than the above because sending it can only ever be
    # wrong, so the only cost of waiting is on the rare utterance that is
    # abandoned mid-sentence. Still bounded, and still never applied to a
    # complete question, which is held for 0ms.
    #
    # A continuation arrives as its own VAD-bounded utterance, so it cannot
    # land sooner than vad_silence_ms plus its own speech and inference time.
    # This is the knob that decides whether a split question costs one
    # provider call or two -- see Finality.ACCUMULATING. Tune on real audio.
    question_hold_incomplete_ms: int = 3000
    #: After a fragment has merged into the turn but the result still carries
    #: no terminal "?", the interviewer has demonstrated that this turn
    #: arrives in pieces -- so the next piece gets this long to land. Restarted
    #: by each new fragment, which is what turns N fragments into one provider
    #: call instead of N. Never applied to a turn that has not fragmented, so
    #: it costs a one-shot question nothing.
    question_continuation_ms: int = 2000
    #: Hard ceiling on one turn's accumulation, and the merge window used
    #: while a turn is still accumulating. Bounds the worst case for an
    #: interviewer who never stops qualifying, and stops a held fragment from
    #: outliving the window that would let its continuation merge -- if the
    #: hold outlived the merge window, the held words would be discarded
    #: rather than assembled.
    question_max_accumulation_ms: int = 10_000
    # Off by default: logs every detector decision (accepted or rejected,
    # with category/confidence/reason) via the existing structured metrics
    # logger, for inspecting real-world detector behavior during manual
    # testing. Never covers MIC audio -- that never reaches the detector.
    question_detector_diagnostics: bool = False

    # --- question understanding ----------------------------------------
    #: One classification call per *completed* interviewer turn, between
    #: finality and the answer. Never runs on partials or fragments. Off
    #: degrades to the deterministic reading the detector already produced --
    #: answers still happen, context selection is just less precise.
    question_understanding_enabled: bool = True
    #: Hard budget. This sits on the realtime path, so it is far tighter than
    #: `groq_timeout_seconds`: past this the fallback is faster than the
    #: classifier and the interviewer is waiting either way.
    question_understanding_timeout_ms: int = 1500
    #: Empty means "use groq_model". Split only if a smaller/faster model is
    #: worth the extra configuration surface -- a decision for real audio.
    question_understanding_model: str = ""

    # --- interviewer-pasted material -----------------------------------
    #: How close to a question a paste has to be to belong to it. Short on
    #: purpose: this binds "here is the table" to "what is wrong with it",
    #: not a paste from earlier in the interview to whatever is asked next.
    #: Applies in both directions -- paste-then-ask and ask-then-paste are
    #: the same event in different orders.
    context_attachment_window_ms: int = 10_000
    #: Per-attachment ceiling. An attachment over this is rejected with a
    #: reason rather than truncated: a partial table or query looks complete
    #: to the model and is worse than none at all.
    context_attachment_max_chars: int = 20_000
    #: How many pastes one turn may carry (a schema, a query, an error, ...).
    context_attachment_max_items: int = 6

    # Realtime transport
    ws_replay_buffer: int = 200
    api_token: str = ""  # set by the desktop shell at spawn; empty disables the check

    # Audio capture
    audio_queue_frames: int = 200  # ~6s at 32ms/frame before dropping oldest
    #: Low-pass the WASAPI loopback stream before decimating it to 16kHz.
    #: Linear interpolation alone does not reject anything above the output
    #: Nyquist, so without this a 15kHz tone lands at ~1kHz and broadband
    #: content above 8kHz sits only ~9dB below the speech itself -- both
    #: measured on this resampler. Exposed as a switch purely so the two
    #: arms can be A/B'd on real interview audio; there is no reason to
    #: prefer the aliased path. Mic capture is unaffected (PortAudio
    #: resamples that one itself).
    audio_loopback_antialias: bool = True
    #: Off by default. The microphone is transcribed for review only -- it never
    #: drives question detection -- but it was measured at 70% of all STT work
    #: (548s of 783s), which is what pushed transcription past real time on a
    #: laptop CPU. Turn it on when the review transcript is worth that cost.
    audio_capture_mic: bool = False       # candidate, recorded for review
    audio_capture_loopback: bool = True   # interviewer, drives question detection

    # Voice activity detection
    vad_threshold: float = 0.5
    vad_start_frames: int = 2        # ~64ms of speech to open an utterance
    vad_silence_ms: int = 700        # sustained silence to close one
    vad_min_utterance_ms: int = 300  # below this it is a click, not speech
    vad_max_utterance_ms: int = 30_000

    # Speech to text
    stt_model: str = "distil-small.en"
    #: "auto" detects an NVIDIA GPU and uses it, falling back to the CPU when
    #: there is not one -- so the same build runs on NVIDIA, AMD, Intel and in
    #: a VM with no configuration. "cuda" demands the GPU and fails loudly if
    #: it is missing; "cpu" never probes.
    stt_device: str = "auto"
    stt_compute_type: str = "int8"          # CPU
    stt_cuda_compute_type: str = "float16"  # NVIDIA GPU
    stt_beam_size: int = 5
    stt_language: str = "en"
    stt_partial_interval_ms: int = 1000

    # STT execution. One shared CTranslate2 model serves every channel, so
    # concurrency here is what the model is actually told to support: raising
    # it without raising num_workers just moves the queue into C++.
    stt_inference_concurrency: int = 1
    #: 0 lets CTranslate2 size its own pool. Measured here, auto-detection was
    #: roughly 2.5x slower than an explicit count on identical audio, so this
    #: defaults to a real number -- see `resolved_stt_cpu_threads`.
    stt_cpu_threads: int = 0

    # Load shedding. Transcription is a real-time product: once the queue is
    # this far behind, the words arrive after the moment they were useful, and
    # running them anyway costs the CPU the *next* utterance needs. Without a
    # deadline the backlog grows without bound -- observed reaching 100s.
    #: Interviewer audio, which drives question detection. Generous, because
    #: dropping this loses a question outright.
    stt_max_job_age_seconds: float = 25.0
    #: The microphone is transcribed for review only and nothing waits on it,
    #: so it yields first when the machine cannot keep up.
    stt_max_mic_job_age_seconds: float = 8.0

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

    @property
    def logs_dir(self) -> Path:
        """Where the rotating log lives. Under data_dir so the packaged app
        writes to LOCALAPPDATA rather than Program Files, and so the desktop
        shell's "Open Logs Folder" has one place to point at."""
        return self.data_dir / "logs"


settings = Settings()


#: Created as a side effect of the first log line, before anything the user
#: cares about exists. Treating it as "in use" would block the migration on
#: every install, which is exactly what it did the first time.
_INCIDENTAL_ENTRIES = {"logs"}


def _significant(directory: Path):
    """Entries that mean this directory is genuinely in use."""
    return [
        entry for entry in directory.iterdir()
        if entry.name not in _INCIDENTAL_ENTRIES or any(entry.iterdir())
    ]


def migrate_legacy_data_dir(target: Path | None = None) -> Path | None:
    """Adopt the data directory this app used under its previous name.

    The desktop shell derives `--data-dir` from the Tauri bundle identifier, so
    renaming the product renames the folder -- and an upgraded install would
    start against an empty one. The user would see no documents, no history, and
    a re-download of both models, with the originals still on disk under a name
    they have no reason to look for.

    Deliberately conservative. The move happens only when the new location does
    not exist or is empty, so it can never overwrite data the renamed app has
    already written, and it is a no-op on every start after the first. Returns
    the directory moved from, or None.
    """
    import shutil

    target = target or settings.data_dir
    if target.exists() and any(_significant(target)):
        return None  # already in use; never merge into live data

    for legacy_name in LEGACY_APP_IDENTIFIERS:
        legacy = target.parent / legacy_name
        if legacy == target or not legacy.is_dir():
            continue
        if not any(legacy.iterdir()):
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(legacy), str(target))
            else:
                # An empty `logs/` is already here, so the directory itself
                # cannot be moved onto it. Move the contents instead.
                for entry in list(legacy.iterdir()):
                    destination = target / entry.name
                    if destination.exists():
                        continue  # never overwrite what is already there
                    shutil.move(str(entry), str(destination))
                if not any(legacy.iterdir()):
                    legacy.rmdir()
        except OSError:
            # A locked file or a cross-volume failure. Losing the migration is
            # recoverable (the old folder is untouched); crashing startup over
            # it is not.
            logging.getLogger(__name__).warning(
                "data_dir_migration_failed from=%s to=%s", legacy, target
            )
            return None
        logging.getLogger(__name__).info(
            "data_dir_migrated from=%s to=%s", legacy, target
        )
        return legacy
    return None


def resolved_stt_cpu_threads() -> int:
    """How many cores CTranslate2 should use for transcription.

    `STT_CPU_THREADS=0` asks CTranslate2 to decide, which is what shipped and
    what was measured running ~2.5x slower than an explicit count on identical
    audio. Auto-detection also behaves badly inside a frozen build, where the
    process sees the machine's full core count but is competing with the
    embedding model, OCR and the WebView for it.

    Half the machine, floored at 2 and capped at 8: past roughly eight threads
    the per-utterance gain flattens while contention with everything else does
    not.
    """
    if settings.stt_cpu_threads > 0:
        return settings.stt_cpu_threads
    return max(2, min(8, (os.cpu_count() or 4) // 2))
