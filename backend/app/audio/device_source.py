import queue
import sys
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

_LOOPBACK_DIAGNOSTIC_INTERVAL = 500


def _pyaudiowpatch():
    """Load the Windows-only WASAPI loopback backend when it is needed."""
    if sys.platform != "win32":
        raise AudioError("System-audio loopback capture is available only on Windows.")
    try:
        import pyaudiowpatch as pyaudio
    except (ImportError, OSError) as exc:
        raise AudioError(
            "Windows system-audio capture requires PyAudioWPatch. Install the "
            "Windows audio dependency and ensure WASAPI is available."
        ) from exc
    return pyaudio


def _default_loopback_device(pyaudio, audio) -> dict:
    """Find the WASAPI loopback input paired with the default output device."""
    try:
        wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
        output = audio.get_device_info_by_index(wasapi["defaultOutputDevice"])
    except (KeyError, OSError, ValueError) as exc:
        raise AudioError("WASAPI or its default output device is unavailable.") from exc

    if output.get("isLoopbackDevice"):
        return output

    output_name = output.get("name", "")
    for loopback in audio.get_loopback_device_info_generator():
        if output_name and output_name in loopback.get("name", ""):
            return loopback

    raise AudioError(
        "No WASAPI loopback device was found for the default Windows output. "
        "Run `python -m pyaudiowpatch` to inspect available devices."
    )


class DeviceAudioSource(AudioSource):
    """Live microphone or Windows WASAPI loopback capture.

    Microphone capture stays on sounddevice. Windows system audio uses
    PyAudioWPatch because its PortAudio build exposes WASAPI loopback devices
    as input streams.
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
        self._loopback_audio = None
        self._loopback_sample_rate = SAMPLE_RATE
        self._resample_input = np.empty(0, dtype=np.float32)
        self._resample_position = 0.0
        self._resampled_output = np.empty(0, dtype=np.float32)
        self._loopback_callback_count = 0
        self._loopback_queue_count = 0
        self._loopback_yield_count = 0
        self._running = False
        self.dropped_frames = 0

    @property
    def channel(self) -> AudioChannel:
        return self._channel

    def describe(self) -> DeviceInfo:
        if self._device is not None:
            return self._device

        if self._channel == AudioChannel.MIC:
            resolved = default_device(self._channel)
            if resolved is None:
                raise AudioError("No MIC capture device is available")
            self._device = resolved
            return resolved

        pyaudio = _pyaudiowpatch()
        audio = pyaudio.PyAudio()
        try:
            loopback = _default_loopback_device(pyaudio, audio)
            self._device = DeviceInfo(
                index=int(loopback["index"]),
                name=loopback.get("name", "WASAPI loopback"),
                channel=AudioChannel.LOOPBACK,
                sample_rate=int(loopback.get("defaultSampleRate", SAMPLE_RATE)),
                is_default=True,
            )
            return self._device
        finally:
            try:
                audio.terminate()
            except Exception:
                logger.debug("loopback_terminate_failed")

    def start(self) -> None:
        if self._running:
            return
        if self._channel == AudioChannel.MIC:
            self._start_microphone()
        else:
            self._start_loopback()
        self._running = True
        logger.info("audio_started channel=%s device=%s", self._channel.value, self._device.name)

    def _start_microphone(self) -> None:
        sd = _sounddevice()
        device = self.describe()
        stream = None
        try:
            stream = sd.InputStream(
                device=device.index,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_SAMPLES,
                dtype="float32",
                callback=self._sounddevice_callback,
            )
            stream.start()
            self._stream = stream
        except Exception as exc:
            if stream is not None:
                self._close_sounddevice_stream(stream)
            raise AudioError(f"Could not open MIC device: {exc}") from exc

    def _start_loopback(self) -> None:
        pyaudio = _pyaudiowpatch()
        audio = pyaudio.PyAudio()
        stream = None
        try:
            loopback = _default_loopback_device(pyaudio, audio)
            channels = int(loopback.get("maxInputChannels", 0))
            if channels < 1:
                raise AudioError("The default WASAPI loopback device has no input channels.")

            self._device = DeviceInfo(
                index=int(loopback["index"]),
                name=loopback.get("name", "WASAPI loopback"),
                channel=AudioChannel.LOOPBACK,
                sample_rate=int(loopback.get("defaultSampleRate", SAMPLE_RATE)),
                is_default=True,
            )
            native_rate = self._device.sample_rate
            if native_rate <= 0:
                raise AudioError("The default WASAPI loopback device has no valid sample rate.")
            self._loopback_sample_rate = native_rate
            self._reset_loopback_buffers()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=native_rate,
                input=True,
                input_device_index=self._device.index,
                frames_per_buffer=FRAME_SAMPLES,
                stream_callback=lambda data, count, time_info, status: self._loopback_callback(
                    pyaudio, data, count, time_info, status, channels
                ),
                start=False,
            )
            stream.start_stream()
            self._stream = stream
            self._loopback_audio = audio
        except Exception as exc:
            if stream is not None:
                self._close_loopback_stream(stream)
            audio.terminate()
            if isinstance(exc, AudioError):
                raise
            raise AudioError(f"Could not open LOOPBACK device: {exc}") from exc

    def _sounddevice_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("audio_status channel=%s status=%s", self._channel.value, status)
        self._enqueue(indata[:, 0].astype(np.float32, copy=True))

    def _loopback_callback(self, pyaudio, in_data, frame_count, time_info, status, channels):
        self._loopback_callback_count += 1
        if status:
            logger.debug("audio_status channel=%s status=%s", self._channel.value, status)
        samples = np.frombuffer(in_data, dtype=np.int16)
        expected = frame_count * channels
        if samples.size != expected:
            logger.warning(
                "loopback_frame_size_mismatch expected=%d received=%d", expected, samples.size
            )
        else:
            mono = samples.reshape(frame_count, channels).mean(axis=1, dtype=np.float32)
            normalized = mono / np.float32(32768.0)
            produced, queued, leftover = self._queue_resampled_loopback(normalized)
            if self._loopback_callback_count % _LOOPBACK_DIAGNOSTIC_INTERVAL == 0:
                logger.debug(
                    "loopback_callback count=%d bytes=%d frame_count=%d channels=%d "
                    "native_rate=%d decoded_samples=%d rms=%.6f peak=%.6f "
                    "resampled=%d queued_frames=%d leftover=%d",
                    self._loopback_callback_count,
                    len(in_data),
                    frame_count,
                    channels,
                    self._loopback_sample_rate,
                    samples.size,
                    float(np.sqrt(np.mean(np.square(normalized)))),
                    float(np.max(np.abs(normalized))),
                    produced,
                    queued,
                    leftover,
                )
        return (None, pyaudio.paContinue)

    def _reset_loopback_buffers(self) -> None:
        self._resample_input = np.empty(0, dtype=np.float32)
        self._resample_position = 0.0
        self._resampled_output = np.empty(0, dtype=np.float32)

    def _queue_resampled_loopback(self, mono: np.ndarray) -> tuple[int, int, int]:
        """Resample native-rate mono audio and emit only complete VAD frames."""
        resampled = self._resample_loopback(mono)
        if resampled.size:
            self._resampled_output = np.concatenate((self._resampled_output, resampled))

        queued = 0
        while self._resampled_output.size >= FRAME_SAMPLES:
            frame = self._resampled_output[:FRAME_SAMPLES].copy()
            self._loopback_queue_count += 1
            queue_before = self._queue.qsize()
            self._enqueue(frame)
            if self._loopback_queue_count % _LOOPBACK_DIAGNOSTIC_INTERVAL == 0:
                logger.debug(
                    "loopback_queue_frame count=%d queue_before=%d queue_after=%d "
                    "shape=%s dtype=%s rms=%.6f",
                    self._loopback_queue_count,
                    queue_before,
                    self._queue.qsize(),
                    frame.shape,
                    frame.dtype,
                    float(np.sqrt(np.mean(np.square(frame)))),
                )
            self._resampled_output = self._resampled_output[FRAME_SAMPLES:]
            queued += 1
        return resampled.size, queued, self._resampled_output.size

    def _resample_loopback(self, mono: np.ndarray) -> np.ndarray:
        """Streaming linear resampler that retains unconsumed source samples.

        WASAPI delivers native-rate chunks of arbitrary size. Keeping the next
        source position relative to the retained tail preserves continuity over
        callback boundaries without adding a heavy DSP dependency.
        """
        if self._loopback_sample_rate == SAMPLE_RATE:
            return mono.astype(np.float32, copy=False)

        samples = np.concatenate((self._resample_input, mono))
        if samples.size < 2:
            self._resample_input = samples
            return np.empty(0, dtype=np.float32)

        step = self._loopback_sample_rate / SAMPLE_RATE
        positions = np.arange(self._resample_position, samples.size - 1, step)
        if positions.size == 0:
            self._resample_input = samples
            return np.empty(0, dtype=np.float32)

        left = positions.astype(np.intp)
        fraction = (positions - left).astype(np.float32)
        output = (
            samples[left] * (np.float32(1.0) - fraction) + samples[left + 1] * fraction
        ).astype(np.float32, copy=False)

        next_position = positions[-1] + step
        consumed = int(next_position)
        self._resample_input = samples[consumed:]
        self._resample_position = next_position - consumed
        return output

    def _enqueue(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except queue.Empty:
                pass

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            if self._channel == AudioChannel.MIC:
                self._close_sounddevice_stream(self._stream)
            else:
                self._close_loopback_stream(self._stream)
            self._stream = None
        if self._loopback_audio is not None:
            try:
                self._loopback_audio.terminate()
            except Exception:
                logger.debug("loopback_terminate_failed")
            self._loopback_audio = None
        if self.dropped_frames:
            logger.warning(
                "audio_frames_dropped channel=%s count=%d", self._channel.value, self.dropped_frames
            )

    def _close_sounddevice_stream(self, stream) -> None:
        try:
            stream.stop()
        except Exception:
            logger.debug("audio_stop_failed channel=%s", self._channel.value)
        try:
            stream.close()
        except Exception:
            logger.debug("audio_close_failed channel=%s", self._channel.value)

    def _close_loopback_stream(self, stream) -> None:
        try:
            stream.stop_stream()
        except Exception:
            logger.debug("loopback_stop_failed")
        try:
            stream.close()
        except Exception:
            logger.debug("loopback_close_failed")

    def frames(self) -> Iterator[np.ndarray]:
        while self._running:
            try:
                frame = self._queue.get(timeout=0.25)
                if self._channel == AudioChannel.LOOPBACK:
                    self._loopback_yield_count += 1
                    if self._loopback_yield_count % _LOOPBACK_DIAGNOSTIC_INTERVAL == 0:
                        logger.debug(
                            "loopback_frame_yielded count=%d queue_after_get=%d channel=%s "
                            "samples=%d dtype=%s rms=%.6f",
                            self._loopback_yield_count,
                            self._queue.qsize(),
                            self._channel.value,
                            frame.size,
                            frame.dtype,
                            float(np.sqrt(np.mean(np.square(frame)))),
                        )
                yield frame
            except queue.Empty:
                continue
