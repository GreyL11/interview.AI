"""Download and load lifecycle for the two local models.

The speech model and the embedding model are both fetched from Hugging Face on
first use and then reused offline. Before this module the UI could only ask "are
there files on disk", which cannot tell "downloading right now" from "failed
halfway" from "never started" -- so a first run looked identical to a broken one
for as long as the download took.

The states are the ones the UI actually has to distinguish:

    not_downloaded -> downloading -> downloaded -> loading -> ready
                             \\                        \\
                              +-------> failed <--------+

`downloaded` means the bytes are on disk; `ready` means the model is loaded and
has served a request. `failed` carries a sentence the user can act on.

Deliberately in-process and not persisted: a state that survives a restart would
be a claim about a previous process this one cannot verify. On startup the
tracker re-derives `not_downloaded` / `downloaded` from disk, which is a fact it
can check.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

NOT_DOWNLOADED = "not_downloaded"
DOWNLOADING = "downloading"
DOWNLOADED = "downloaded"
LOADING = "loading"
READY = "ready"
FAILED = "failed"

#: States that mean the bytes are present, whatever is happening on top of them.
_ON_DISK = frozenset({DOWNLOADED, LOADING, READY})

#: What a finished download of each kind leaves behind. Checked by extension
#: rather than by exact filename because the ONNX and CTranslate2 layouts differ
#: and both are versioned by Hugging Face, not by this app.
_ARTIFACT_SUFFIXES = ("*.onnx", "*.bin")


@dataclass
class _ModelState:
    kind: str
    state: str = NOT_DOWNLOADED
    detail: str | None = None
    #: What the model actually ended up running on ("cpu" / "cuda"), once it
    #: has loaded. Reported so the UI can state the active accelerator as a
    #: fact rather than repeating what was merely configured.
    device: str | None = None


class ModelTracker:
    """Lifecycle for every local model, shared across threads.

    The loaders run on worker threads (STT inference, document ingestion) while
    `/models/status` is answered on the event loop, so every transition takes
    the lock. Transitions are cheap and rare; contention is not a concern.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, _ModelState] = {
            "stt": _ModelState(kind="stt"),
            "embedding": _ModelState(kind="embedding"),
        }

    # ---------------------------------------------------------- transitions

    def _set(
        self,
        kind: str,
        state: str,
        detail: str | None = None,
        device: str | None = None,
    ) -> None:
        with self._lock:
            entry = self._states.get(kind)
            if entry is None:
                return
            entry.state = state
            entry.detail = detail
            if device is not None:
                entry.device = device
        logger.info("model_state kind=%s state=%s device=%s", kind, state, device or "-")

    def downloading(self, kind: str) -> None:
        self._set(kind, DOWNLOADING)

    def loading(self, kind: str) -> None:
        self._set(kind, LOADING)

    def ready(self, kind: str, device: str | None = None) -> None:
        self._set(kind, READY, device=device)

    def failed(self, kind: str, detail: str) -> None:
        # The detail is shown to the user, so it is the loader's own sentence
        # ("First run needs network access...") rather than a raw traceback.
        self._set(kind, FAILED, detail)

    def reset(self, kind: str) -> None:
        """Forget a failure so the next attempt is judged on its own.

        Called when a load is retried: leaving `failed` in place would make a
        successful retry look like it had not happened.
        """
        self._set(kind, self._state_from_disk(kind))

    # -------------------------------------------------------------- reading

    def _state_from_disk(self, kind: str) -> str:
        return DOWNLOADED if _has_artifacts(model_dir(kind)) else NOT_DOWNLOADED

    def snapshot(self) -> list[dict]:
        """Current state of every model, for the API.

        In-flight states (`downloading`, `loading`, `failed`) are reported as
        recorded. Otherwise the answer is re-derived from disk, so a model that
        was downloaded by a previous run -- or deleted behind the app's back --
        is reported correctly without this process having watched it happen.
        """
        with self._lock:
            entries = [
                (name, entry.state, entry.detail, entry.device)
                for name, entry in self._states.items()
            ]

        out: list[dict] = []
        for kind, state, detail, device in entries:
            if state in (NOT_DOWNLOADED, DOWNLOADED):
                state = self._state_from_disk(kind)
            directory = model_dir(kind)
            out.append(
                {
                    "name": _model_name(kind),
                    "kind": kind,
                    "state": state,
                    # Derived, never stored separately, so the boolean an older
                    # client reads can never contradict `state`.
                    "downloaded": state in _ON_DISK,
                    "path": str(directory),
                    "detail": detail,
                    # None until the model has loaded -- claiming an
                    # accelerator before anything ran on it would be a guess.
                    "device": device,
                }
            )
        return out


def model_dir(kind: str) -> Path:
    """Where a model's files live.

    Always under DATA_DIR, which the desktop shell points at the per-user
    application data directory. Never the installation directory: a per-user
    NSIS install can land somewhere read-only, and never the PyInstaller
    extraction directory, which is discarded on exit.
    """
    if kind == "stt":
        return settings.data_dir / "models" / "whisper"
    return settings.data_dir / "models" / "hf"


def _model_name(kind: str) -> str:
    return settings.stt_model if kind == "stt" else settings.embedding_model


def _has_artifacts(directory: Path) -> bool:
    """Does this directory hold a finished model?

    Only completed files count. Hugging Face downloads into `*.incomplete` in
    the blob store and renames on success, so a partial or interrupted download
    never matches these patterns -- which is what makes a half-finished download
    report `not_downloaded` (retryable) instead of `downloaded` (broken).
    """
    if not directory.exists():
        return False
    return any(
        any(directory.rglob(pattern)) for pattern in _ARTIFACT_SUFFIXES
    )


#: One tracker per process, shared by the loaders and the API.
tracker = ModelTracker()


def model_states() -> list[dict]:
    return tracker.snapshot()
