import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.audio.base import AudioChannel, DeviceInfo
from app.audio.devices import audio_available, list_devices
from app.core.config import settings
from app.core.logging import get_logger
from app.core.secret_config import forget_secret, persist_secret
from app.core.secrets import SecretStoreUnavailable, secret_store
from app.model_status import model_states

logger = get_logger(__name__)

router = APIRouter(tags=["settings"])

#: A provider path segment must map onto a `<provider>_api_key` secret name.
#: The same shape the credential store enforces, checked here so a bad path
#: produces a 404 rather than a 500 from deeper down.
_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,54}$")


class ProviderStatus(BaseModel):
    """The cloud LLM provider as the Settings screen should see it.

    No key material, not even a masked prefix: the API is write-only for keys,
    and this stays that way so a screenshot of Settings can never leak one.
    """

    name: str
    model: str
    configured: bool
    #: Wired into the running engine. False only if the engine could not be
    #: consulted at all, which the UI reports as needing a restart.
    active: bool
    #: How the last request to this provider failed, as a classification --
    #: never the provider's own error text. None when the last request
    #: succeeded or nothing has been asked yet.
    last_error_kind: str | None = None


class SettingsView(BaseModel):
    """Everything the Settings screen needs.

    API keys are never returned — only whether one is configured. They are
    write-only across this API by design.
    """

    providers: list[ProviderStatus]
    #: True when the OS credential store is usable, so a saved key survives a
    #: restart. False in environments without one -- the UI says "this session
    #: only" rather than implying it was saved.
    secure_storage_available: bool
    #: Mutations here apply to the running process only; nothing is written
    #: back to .env. The UI says so rather than implying they persist.
    settings_persist: bool = False
    groq_key_configured: bool
    groq_model: str
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

    groq_model: str | None = None
    stt_model: str | None = None
    stt_device: str | None = None
    rag_top_k: int | None = None
    rag_min_similarity: float | None = None
    audio_capture_mic: bool | None = None
    audio_capture_loopback: bool | None = None


class ModelStatus(BaseModel):
    """One local model's download lifecycle.

    `state` is the field to render. `downloaded` is kept because it is the one
    thing an older client understood, and it is derived from `state` so the two
    can never disagree.
    """

    name: str
    kind: str
    state: str
    downloaded: bool
    path: str
    detail: str | None = None


def _provider_statuses() -> list[ProviderStatus]:
    """Static configuration joined with the running client's own view.

    Read defensively: a Settings screen must not fail to render because the LLM
    layer is unhappy.
    """
    active = False
    last_error: str | None = None
    try:
        from app.core.deps import get_llm_client

        client = get_llm_client()
        active = True
        kind = getattr(client, "last_error_kind", None)
        last_error = kind.value if kind is not None else None
    except Exception as exc:
        logger.warning("provider_status_unavailable error=%s", type(exc).__name__)

    return [
        ProviderStatus(
            name="groq",
            model=settings.groq_model,
            configured=bool(settings.groq_api_key),
            active=active,
            last_error_kind=last_error,
        )
    ]


def _view() -> SettingsView:
    return SettingsView(
        providers=_provider_statuses(),
        secure_storage_available=secret_store().available,
        groq_key_configured=bool(settings.groq_api_key),
        groq_model=settings.groq_model,
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

    if "groq_model" in changed:
        # Rejected here rather than discovered on the next question: a blank or
        # malformed model would otherwise take the provider down silently.
        from app.llm.groq_client import GroqConfigError, validate_model

        try:
            changed["groq_model"] = validate_model(changed["groq_model"])
        except GroqConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    for key, value in changed.items():
        setattr(settings, key, value)

    # Never log a value, only which fields moved.
    logger.info("settings_updated fields=%s", sorted(changed))

    if "groq_model" in changed:
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
    """Where each local model is in its download lifecycle. Drives the
    first-run download UI."""
    return [ModelStatus(**state) for state in model_states()]


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


def _key_field(provider: str) -> str:
    """Map a provider path segment onto its secret name.

    Generic rather than a fixed table: the credential store accepts any
    `<name>_api_key`, so a provider does not need a code change to be storable.
    """
    name = provider.strip().lower()
    if not _PROVIDER_PATTERN.match(name):
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    return f"{name}_api_key"


def _refresh_engine() -> None:
    """Rebuild the LLM client so the change takes effect without a restart.

    `get_llm_client` is lru_cached, so clearing it swaps the client for
    *future* callers only: a request already holding a reference finishes on
    the client it started with. That is the atomic swap -- no lock needed,
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
        detail = "Saved. It will still be here after you restart Call Assistant."
    except SecretStoreUnavailable:
        # Honest fallback: apply it now, but do not claim it was saved and do
        # not write it to a file to make the claim true.
        if field in type(settings).model_fields:
            setattr(settings, field, value)
        persisted = False
        detail = (
            "Applied for this session. This machine has no credential store, so "
            "it cannot be saved and will be gone after a restart."
        )

    _refresh_engine()
    # provider and outcome only -- never the key, its length, or a prefix.
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
    _refresh_engine()

    # An environment-supplied key returns on the next start; this app cannot
    # unset a variable its parent process set, and saying otherwise would be a
    # lie the user discovers later.
    detail = (
        "Removed for this session. It is set in this machine's environment, so it "
        "will come back when Call Assistant restarts."
        if from_environment
        else "Removed."
    )
    logger.info("provider_key_removed provider=%s", provider)
    return ProviderKeyResult(
        provider=provider, configured=False, persisted=False, detail=detail
    )
