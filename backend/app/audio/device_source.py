import queue
from collections.abc import Iterator

import numpy as np

from app.audio.base import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioChannel,
    AudioError,
    AudioSource,
    DeviceInfo,
)
from app.audio.devices import _sounddevice, default_device
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DeviceAudioSource(AudioSource):
    """Live capture from one device via PortAudio.

    PortAudio delivers audio on its own real-time thread. That callback must
    never block or allocate unpredictably, so it does exactly one thing: drop a
    copy into a bounded queue. When the consumer falls behind, the oldest frame
    is discarded and counted — stale audio is worthless for live transcription,
    and blocking the callback would glitch the capture itself.
    """

    def __init__(
        self,
        channel: AudioChannel,
        device: DeviceInfo | None = None,
        queue_frames: int | None = None,
    ) -> None:
        self._channel = channel
        self._device = device
        self._queue: queue.Queue = queue.Queue(
            maxsize=queue_frames or settings.audio_queue_frames
        )
        self._stream = None
        self._running = False
        self.dropped_frames = 0

    @property
    def channel(self) -> AudioChannel:
        return self._channel

    def describe(self) -> DeviceInfo:
        if self._device is None:
            resolved = default_device(self._channel)
            if resolved is None:
                raise AudioError(f"No {self._channel.value} capture device is available")
            self._device = resolved
        return self._device

    def start(self) -> None:
        if self._running:
            return
        sd = _sounddevice()
        device = self.describe()

        extra = None
        if self._channel == AudioChannel.LOOPBACK:
            # WASAPI loopback opens an output device for input. Without this the
            # interviewer's audio simply cannot be captured on Windows.
            try:
                extra = sd.WasapiSettings(loopback=True)
            except Exception as exc:
                raise AudioError(
                    f"WASAPI loopback is unavailable on this device: {exc}"
                ) from exc

        def callback(indata, frames, time_info, status):
            if status:
                logger.debug("audio_status channel=%s status=%s", self._channel.value, status)
            mono = indata[:, 0].astype(np.float32, copy=True)
            try:
                self._queue.put_nowait(mono)
            except queue.Full:
                self.dropped_frames += 1
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(mono)
                except queue.Empty:
                    pass

        try:
            self._stream = sd.InputStream(
                device=device.index,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_SAMPLES,
                dtype="float32",
                callback=callback,
                extra_settings=extra,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioError(f"Could not open {self._channel.value} device: {exc}") from exc

        self._running = True
        logger.info("audio_started channel=%s device=%s", self._channel.value, device.name)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.debug("audio_stop_failed channel=%s", self._channel.value)
            self._stream = None
        if self.dropped_frames:
            logger.warning(
                "audio_frames_dropped channel=%s count=%d", self._channel.value, self.dropped_frames
            )

    def frames(self) -> Iterator[np.ndarray]:
        while self._running:
            try:
                yield self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
