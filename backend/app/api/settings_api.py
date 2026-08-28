from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.audio.base import AudioChannel, DeviceInfo
from app.audio.devices import audio_available, list_devices
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["settings"])


class SettingsView(BaseModel):
    """Everything the Setup screen needs.

    The Gemini key is never returned — only whether one is configured. It is
    write-only across this API by design.
    """

    gemini_key_configured: bool
    gemini_model: str
    embedding_model: str
    stt_model: str
    stt_device: str
    stt_compute_type: str
    chunk_size: int
    chunk_overlap: int
    rag_top_k: int
    rag_min_similarity: float
    data_dir: str
    audio_capture_mic: bool
    audio_capture_loopback: bool
    audio_available: bool


class SettingsUpdate(BaseModel):
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    stt_model: str | None = None
    stt_device: str | None = None
    rag_top_k: int | None = None
    rag_min_similarity: float | None = None
    audio_capture_mic: bool | None = None
    audio_capture_loopback: bool | None = None


class ModelStatus(BaseModel):
    name: str
    kind: str
    downloaded: bool
    path: str


def _view() -> SettingsView:
    return SettingsView(
        gemini_key_configured=bool(settings.gemini_api_key),
        gemini_model=settings.gemini_model,
        embedding_model=settings.embedding_model,
        stt_model=settings.stt_model,
        stt_device=settings.stt_device,
        stt_compute_type=settings.stt_compute_type,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        rag_top_k=settings.rag_top_k,
        rag_min_similarity=settings.rag_min_similarity,
        data_dir=str(settings.data_dir),
        audio_capture_mic=settings.audio_capture_mic,
        audio_capture_loopback=settings.audio_capture_loopback,
        audio_available=audio_available(),
    )


@router.get("/settings", response_model=SettingsView)
async def get_settings() -> SettingsView:
    return _view()


@router.put("/settings", response_model=SettingsView)
async def update_settings(update: SettingsUpdate) -> SettingsView:
    changed = update.model_dump(exclude_none=True)
    for key, value in changed.items():
        setattr(settings, key, value)

    # Never log the key itself, only that it was set.
    logger.info(
        "settings_updated fields=%s",
        sorted("gemini_api_key" if k == "gemini_api_key" else k for k in changed),
    )

    if "gemini_api_key" in changed or "gemini_model" in changed:
        from app.core.deps import get_llm_client, get_summarizer

        get_llm_client.cache_clear()
        get_summarizer.cache_clear()
    if "stt_model" in changed or "stt_device" in changed:
        from app.core.deps import get_stt_engine
        from app.stt.scheduler import reset_shared_scheduler

        get_stt_engine.cache_clear()
        # The scheduler's worker count is derived from STT settings, so it has
        # to be rebuilt alongside the engine rather than outliving it.
        reset_shared_scheduler()

    return _view()


@router.get("/audio/devices", response_model=list[DeviceInfo])
async def get_audio_devices() -> list[DeviceInfo]:
    if not audio_available():
        return []
    try:
        return list_devices()
    except Exception as exc:
        logger.warning("device_enumeration_failed error=%s", exc)
        return []


@router.get("/models/status", response_model=list[ModelStatus])
async def models_status() -> list[ModelStatus]:
    """Which local models are already on disk. Drives the first-run download UI."""
    embedding_dir = settings.data_dir / "models" / "hf"
    whisper_dir = settings.data_dir / "models" / "whisper"
    return [
        ModelStatus(
            name=settings.embedding_model,
            kind="embedding",
            downloaded=_has_files(embedding_dir),
            path=str(embedding_dir),
        ),
        ModelStatus(
            name=settings.stt_model,
            kind="stt",
            downloaded=_has_files(whisper_dir),
            path=str(whisper_dir),
        ),
    ]


def _has_files(directory: Path) -> bool:
    return directory.exists() and any(directory.rglob("*.onnx")) or (
        directory.exists() and any(directory.rglob("*.bin"))
    )


@router.get("/audio/channels")
async def audio_channels() -> dict:
    return {
        "mic": settings.audio_capture_mic,
        "loopback": settings.audio_capture_loopback,
        "note": (
            "Question detection runs on the loopback channel (the interviewer). "
            "The microphone is transcribed for review only."
        ),
        "channels": [c.value for c in AudioChannel],
    }
