import asyncio
from collections.abc import Callable

from app.audio.base import AudioChannel, AudioError, AudioSource
from app.audio.device_source import DeviceAudioSource
from app.core.config import settings
from app.core.logging import get_logger
from app.realtime.session import LiveSession
from app.stt.base import SttEngine
from app.stt.pipeline import AudioPipeline, TranscriptionWorker
from app.stt.vad import SpeechDetector, build_speech_detector

logger = get_logger(__name__)


def build_pipeline(
    live: LiveSession,
    engine: SttEngine,
    sources: list[AudioSource] | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    detector_factory: Callable[[], SpeechDetector] = build_speech_detector,
) -> AudioPipeline:
    """Attach capture to a live session.

    Each channel gets its own VAD state — mic and loopback speak at different
    times and sharing a segmenter between them would splice the candidate's
    words onto the interviewer's question.
    """
    loop = loop or asyncio.get_running_loop()
    sources = sources if sources is not None else default_sources()

    workers = [
        TranscriptionWorker(
            source=source,
            detector=detector_factory(),
            engine=engine,
            loop=loop,
            on_transcript=live.on_transcript,
        )
        for source in sources
    ]
    return AudioPipeline(workers)


def default_sources() -> list[AudioSource]:
    """Open the configured channels, tolerating a missing loopback device.

    Loopback is the channel that hears the interviewer, so losing it matters —
    but a machine with no loopback should degrade to solo practice rather than
    refuse to start a session.
    """
    sources: list[AudioSource] = []

    if settings.audio_capture_loopback:
        try:
            source = DeviceAudioSource(AudioChannel.LOOPBACK)
            source.describe()
            sources.append(source)
        except AudioError as exc:
            logger.warning("loopback_unavailable degrading_to_mic_only error=%s", exc)

    if settings.audio_capture_mic:
        try:
            source = DeviceAudioSource(AudioChannel.MIC)
            source.describe()
            sources.append(source)
        except AudioError as exc:
            logger.warning("mic_unavailable error=%s", exc)

    if not sources:
        raise AudioError(
            "No capture device is available. Check the audio settings, or use "
            "typed questions instead of live audio."
        )
    return sources
