from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.audio.base import AudioChannel, DeviceInfo
from app.audio.devices import audio_available, list_devices
from app.core.config import settings
from app.core.logging import get_logger
from app.core.secret_config import forget_secret, persist_secret
from app.core.secrets import SecretStoreUnavailable, secret_store

logger = get_logger(__name__)

router = APIRouter(tags=["settings"])


class ProviderStatus(BaseModel):
    """One LLM provider as the Settings screen should see it.

    No key material, not even a masked prefix: the API is write-only for keys,
    and this stays that way so a screenshot of Settings can never leak one.
    """

    name: str
    model: str
    configured: bool
    enabled: bool
    #: Wired into the router this launch. A provider can be configured and
    #: enabled but absent here if the key arrived after the router was built.
    active: bool
    available: bool
    cooling_down: bool
    cooldown_remaining_seconds: float | None = None
    role: str | None = None


class SettingsView(BaseModel):
    """Everything the Settings screen needs.

    API keys are never returned — only whether one is configured. They are
    write-only across this API by design.
    """

    providers: list[ProviderStatus]
    provider_priority: str
    #: True when the OS credential store is usable, so a saved key survives a
    #: restart. False in environments without one -- the UI says "this session
    #: only" rather than implying it was saved.
    secure_storage_available: bool
    #: Mutations here apply to the running process only; nothing is written
    #: back to .env. The UI says so rather than implying they persist.
    settings_persist: bool = False
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
    """Non-secret settings only.

    API keys are deliberately absent: they go through
    PUT/DELETE /providers/{name}/key, which is the single path that also
    persists them. Two ways to set a key, one persisting and one not, is
    exactly the confusion this split exists to prevent.
    """

    gemini_model: str | None = None
    groq_model: str | None = None
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


def _provider_statuses() -> list[ProviderStatus]:
    """Static config joined with the router's live health.

    The router is the only thing that knows about cooldowns, so its view wins
    where the two overlap. It is read defensively: a Settings screen must not
    fail to render because the LLM layer is unhappy.
    """
    configured = {
        "groq": (bool(settings.groq_api_key), settings.groq_enabled, settings.groq_model),
        "gemini": (
            bool(settings.gemini_api_key),
            settings.gemini_enabled,
            settings.gemini_model,
        ),
    }

    live: dict[str, dict] = {}
    try:
        from app.core.deps import get_llm_client

        client = get_llm_client()
        if hasattr(client, "provider_status"):
            live = {entry["name"]: entry for entry in client.provider_status()}
    except Exception as exc:
        logger.warning("provider_status_unavailable error=%s", exc)

    order = [name.strip().lower() for name in settings.llm_provider_priority.split(",")]
    ordered = [name for name in order if name in configured]
    ordered += [name for name in configured if name not in ordered]

    return [
        ProviderStatus(
            name=name,
            model=configured[name][2],
            configured=configured[name][0],
            enabled=configured[name][1],
            active=name in live,
            # build_router() deliberately wires Gemini even with no key, so its
            # existing "not configured" message is what the user sees on the
            # first question. That must not surface here as "available" -- a
            # provider with no key is never usable, whatever the router says.
            available=(
                configured[name][0]
                and configured[name][1]
                and live.get(name, {}).get("available", False)
            ),
            cooling_down=live.get(name, {}).get("cooling_down", False),
            cooldown_remaining_seconds=live.get(name, {}).get("cooldown_remaining_seconds"),
            role=live.get(name, {}).get("role"),
        )
        for name in ordered
    ]


def _view() -> SettingsView:
    return SettingsView(
        providers=_provider_statuses(),
        provider_priority=settings.llm_provider_priority,
        secure_storage_available=secret_store().available,
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
        sorted(changed),
    )

    if changed.keys() & {"gemini_model", "groq_model"}:
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


# ------------------------------------------------------------- provider keys


class ProviderKeyUpdate(BaseModel):
    api_key: str


class ProviderKeyResult(BaseModel):
    """What the UI is told after a key changes.

    Carries no key material and nothing derived from it -- no prefix, no
    length, no hash. `persisted` is the honest answer to "will this survive a
    restart", which is False whenever there is no OS credential store.
    """

    provider: str
    configured: bool
    persisted: bool
    detail: str


_KEY_FIELDS = {"groq": "groq_api_key", "gemini": "gemini_api_key"}


def _key_field(provider: str) -> str:
    field = _KEY_FIELDS.get(provider.lower())
    if field is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    return field


def _refresh_router() -> None:
    """Rebuild the router so the change takes effect without a restart.

    `get_llm_client` is lru_cached, so clearing it swaps the router for
    *future* callers only: a request already holding a reference finishes on
    the router it started with. That is the atomic swap -- no lock needed,
    because nothing mutates the old instance.
    """
    from app.core.deps import get_llm_client, get_summarizer

    get_llm_client.cache_clear()
    get_summarizer.cache_clear()


@router.put("/providers/{provider}/key", response_model=ProviderKeyResult)
async def set_provider_key(provider: str, update: ProviderKeyUpdate) -> ProviderKeyResult:
    field = _key_field(provider)
    value = update.api_key.strip()
    if not value:
        raise HTTPException(status_code=400, detail="The key cannot be empty.")

    try:
        persist_secret(field, value)
        persisted = True
        detail = "Saved. It will still be here after you restart Interview Coach."
    except SecretStoreUnavailable:
        # Honest fallback: apply it now, but do not claim it was saved and do
        # not write it to a file to make the claim true.
        setattr(settings, field, value)
        persisted = False
        detail = "Applied for this session. It cannot be saved on this machine."

    _refresh_router()
    logger.info("provider_key_set provider=%s persisted=%s", provider, persisted)
    return ProviderKeyResult(
        provider=provider, configured=True, persisted=persisted, detail=detail
    )


@router.delete("/providers/{provider}/key", response_model=ProviderKeyResult)
async def delete_provider_key(provider: str) -> ProviderKeyResult:
    field = _key_field(provider)

    # Asked *before* clearing: forget_secret empties the in-memory value, which
    # would make this look like nothing had ever supplied it.
    from app.core.secret_config import env_supplied

    from_environment = env_supplied(field)
    forget_secret(field)
    _refresh_router()

    # An environment-supplied key returns on the next start; this app cannot
    # unset a variable its parent process set, and saying otherwise would be a
    # lie the user discovers later.
    detail = (
        "Removed for this session. It is set in this machine's environment, so it "
        "will come back when Interview Coach restarts."
        if from_environment
        else "Removed."
    )
    logger.info("provider_key_removed provider=%s", provider)
    return ProviderKeyResult(
        provider=provider, configured=False, persisted=False, detail=detail
    )
