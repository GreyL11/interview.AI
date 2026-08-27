from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.post("/shutdown")
async def shutdown(request: Request) -> dict[str, str]:
    """Graceful stop, called by the desktop shell when its window closes.

    First of three shutdown layers: the shell force-kills if this does not
    land, and the child watchdog plus the Windows Job Object cover the case
    where the shell dies without asking.
    """
    from app.core.lifecycle import request_exit
    from app.realtime.manager import session_manager

    logger.info("shutdown_requested active_sessions=%d", session_manager.active_count)
    try:
        await session_manager.close_all()
    except Exception:
        logger.exception("session_cleanup_failed during shutdown")

    request_exit()
    return {"status": "stopping"}


@router.get("/ready")
async def ready() -> dict[str, object]:
    """Startup detail for the shell's first screen: what is configured and what
    still needs the user's attention."""
    from app.audio.devices import audio_available

    return {
        "status": "ready",
        "gemini_configured": bool(settings.gemini_api_key),
        "audio_available": audio_available(),
        "data_dir": str(settings.data_dir),
    }
