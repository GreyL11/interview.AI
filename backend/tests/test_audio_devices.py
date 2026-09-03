"""Device enumeration and capture-source tests.

sounddevice/PortAudio is not installed in this environment, so these tests
exercise the abstraction and the failure paths. Real device capture is verified
manually — see README, "Pending hardware verification".
"""

import numpy as np
import pytest

from app.audio.base import FRAME_SAMPLES, AudioChannel, AudioError
from app.audio.device_source import DeviceAudioSource
from app.audio.devices import audio_available, list_devices
from tests.fakes import FakeAudioSource, silence_frames, speech_frames


class FakeSounddeviceStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSounddevice:
    def __init__(self):
        self.stream = None

    def InputStream(self, **kwargs):
        self.stream = FakeSounddeviceStream(**kwargs)
        return self.stream


class FakeLoopbackStream:
    def __init__(self, start_error=None, **kwargs):
        self.kwargs = kwargs
        self.start_error = start_error
        self.started = False
        self.stopped = False
        self.closed = False

    def start_stream(self):
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakePyAudioManager:
    def __init__(self, loopbacks, open_error=None, start_error=None):
        self.loopbacks = loopbacks
        self.open_error = open_error
        self.start_error = start_error
        self.terminated = False
        self.open_calls = []
        self.stream = None

    def get_host_api_info_by_type(self, host_api):
        return {"defaultOutputDevice": 4}

    def get_device_info_by_index(self, index):
        assert index == 4
        return {"index": 4, "name": "Speakers", "isLoopbackDevice": False}

    def get_loopback_device_info_generator(self):
        yield from self.loopbacks

    def open(self, **kwargs):
        self.open_calls.append(kwargs)
        if self.open_error is not None:
            raise self.open_error
        self.stream = FakeLoopbackStream(start_error=self.start_error, **kwargs)
        return self.stream

    def terminate(self):
        self.terminated = True


class FakePyAudioModule:
    paWASAPI = 13
    paInt16 = 8
    paContinue = 0

    def __init__(self, managers):
        self._managers = iter(managers)

    def PyAudio(self):
        return next(self._managers)


def loopback_device():
    return {
        "index": 9,
        "name": "Speakers (loopback)",
        "isLoopbackDevice": True,
        "maxInputChannels": 2,
        "defaultSampleRate": 48_000,
    }


def test_audio_available_reports_without_raising():
    assert isinstance(audio_available(), bool)


@pytest.mark.skipif(audio_available(), reason="sounddevice is installed here")
def test_missing_sounddevice_raises_actionable_error():
    with pytest.raises(AudioError, match="Audio capture is unavailable"):
        list_devices()


@pytest.mark.skipif(not audio_available(), reason="requires sounddevice")
def test_enumeration_tags_channels():
    for device in list_devices():
        assert device.channel in (AudioChannel.MIC, AudioChannel.LOOPBACK)
        assert device.name


def test_fake_source_yields_fixed_frames():
    source = FakeAudioSource(speech_frames(3))
    source.start()
    frames = list(source.frames())
    assert len(frames) == 3
    assert all(f.shape == (FRAME_SAMPLES,) for f in frames)
    assert all(f.dtype == np.float32 for f in frames)


def test_fake_source_stops_early():
    source = FakeAudioSource(silence_frames(100))
    source.start()
    produced = []
    for frame in source.frames():
        produced.append(frame)
        if len(produced) == 5:
            source.stop()
    assert len(produced) == 5


def test_fake_source_describes_its_channel():
    source = FakeAudioSource([], channel=AudioChannel.LOOPBACK)
    assert source.describe().channel == AudioChannel.LOOPBACK
    assert source.channel == AudioChannel.LOOPBACK


def test_microphone_source_uses_sounddevice_and_closes_stream(monkeypatch):
    from app.audio import device_source

    fake_sd = FakeSounddevice()
    device = device_source.DeviceInfo(index=2, name="Microphone", channel=AudioChannel.MIC)
    monkeypatch.setattr(device_source, "_sounddevice", lambda: fake_sd)
    monkeypatch.setattr(device_source, "default_device", lambda channel: device)

    source = DeviceAudioSource(AudioChannel.MIC)
    source.start()

    assert fake_sd.stream.started
    assert fake_sd.stream.kwargs["samplerate"] == 16_000
    assert fake_sd.stream.kwargs["blocksize"] == FRAME_SAMPLES
    assert fake_sd.stream.kwargs["dtype"] == "float32"

    source.stop()
    assert fake_sd.stream.stopped
    assert fake_sd.stream.closed


def test_loopback_uses_matching_default_output_device_and_downmixes(monkeypatch):
    from app.audio import device_source

    # Decimation arithmetic in isolation. The anti-alias prefilter is off here
    # deliberately: it swallows its first `taps-1` input samples while priming
    # and smooths edges by design, so leaving it on would test the filter
    # rather than the downmix. Its own behaviour is pinned separately below.
    monkeypatch.setattr(device_source.settings, "audio_loopback_antialias", False)

    describe_manager = FakePyAudioManager([loopback_device()])
    capture_manager = FakePyAudioManager([loopback_device()])
    module = FakePyAudioModule([describe_manager, capture_manager])
    monkeypatch.setattr(device_source, "_pyaudiowpatch", lambda: module)

    source = DeviceAudioSource(AudioChannel.LOOPBACK)
    described = source.describe()
    assert described.index == 9
    assert described.is_default
    assert describe_manager.terminated

    source.start()
    assert capture_manager.open_calls[0]["input_device_index"] == 9
    assert capture_manager.open_calls[0]["rate"] == 48_000
    assert capture_manager.open_calls[0]["frames_per_buffer"] == FRAME_SAMPLES

    callback = capture_manager.stream.kwargs["stream_callback"]
    native_frames = 1_536
    data = np.tile(np.array([32767, -32768], dtype=np.int16), native_frames).tobytes()
    assert callback(data, native_frames, None, None) == (None, module.paContinue)
    frame = source._queue.get_nowait()
    assert frame.shape == (FRAME_SAMPLES,)
    assert frame.dtype == np.float32
    assert np.allclose(frame, -1 / 65536)

    source.stop()
    assert capture_manager.stream.stopped
    assert capture_manager.stream.closed
    assert capture_manager.terminated


def test_loopback_preserves_resampled_leftovers_across_callbacks(monkeypatch):
    from app.audio import device_source

    # As above: the step edge this asserts is exactly what a low-pass is
    # supposed to round off, so the prefilter is off to keep this test about
    # leftover retention across callbacks.
    monkeypatch.setattr(device_source.settings, "audio_loopback_antialias", False)

    manager = FakePyAudioManager([loopback_device()])
    module = FakePyAudioModule([manager])
    monkeypatch.setattr(device_source, "_pyaudiowpatch", lambda: module)

    source = DeviceAudioSource(AudioChannel.LOOPBACK)
    source.start()
    callback = manager.stream.kwargs["stream_callback"]
    first_half = np.ones((768, 2), dtype=np.int16).tobytes()
    second_half = np.full((768, 2), 2, dtype=np.int16).tobytes()

    callback(first_half, 768, None, None)
    assert source._queue.empty()

    callback(second_half, 768, None, None)
    frame = source._queue.get_nowait()
    assert frame.shape == (FRAME_SAMPLES,)
    assert frame.dtype == np.float32
    # 768 native samples produce 256 samples at 16 kHz; both callbacks are
    # needed to make one complete 512-sample AudioSource frame.
    assert np.allclose(frame[:256], 1 / 32768)
    assert np.allclose(frame[256:], 2 / 32768)
    source.stop()


# ------------------------------------------------- loopback anti-aliasing
# Linear interpolation is not a decimation filter. Without a prefilter,
# content above the 16kHz output Nyquist folds down into the band Whisper
# reads: measured on this resampler, a 15kHz tone reappeared at ~1kHz at
# full amplitude. These pin the fix without needing a device or a model.


def _resample_tone(freq_hz: int, antialias: bool, native: int = 48_000) -> np.ndarray:
    """Push a pure tone through the loopback resampler and return 16kHz output."""
    from app.core.config import settings as app_settings

    original = app_settings.audio_loopback_antialias
    app_settings.audio_loopback_antialias = antialias
    try:
        source = DeviceAudioSource(AudioChannel.LOOPBACK)
        source._loopback_sample_rate = native
        source._reset_loopback_buffers()
        t = np.arange(native) / native
        tone = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
        chunks = [
            source._resample_loopback(tone[i:i + FRAME_SAMPLES])
            for i in range(0, tone.size - FRAME_SAMPLES, FRAME_SAMPLES)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    finally:
        app_settings.audio_loopback_antialias = original


def _peak_amplitude(signal: np.ndarray, around_hz: float) -> float:
    from app.audio.base import SAMPLE_RATE

    signal = signal[100:]  # skip the filter's priming transient
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(signal.size, 1 / SAMPLE_RATE)
    band = (freqs >= around_hz - 150) & (freqs <= around_hz + 150)
    return float(np.max(spectrum[band])) / (np.sum(window) / 2)


@pytest.mark.parametrize("freq_hz", [500, 1000, 3000, 5000])
def test_antialias_filter_preserves_the_speech_band(freq_hz):
    """Whatever the filter rejects, it must not be speech."""
    filtered = _peak_amplitude(_resample_tone(freq_hz, antialias=True), freq_hz)
    assert filtered > 0.8, f"{freq_hz}Hz attenuated to {filtered:.3f}"


@pytest.mark.parametrize(
    "freq_hz,alias_hz",
    [(9_000, 7_000), (12_000, 4_000), (15_000, 1_000), (20_000, 4_000)],
)
def test_antialias_filter_rejects_content_above_the_output_nyquist(freq_hz, alias_hz):
    """Content above 8kHz must not reappear inside the speech band. Without
    the prefilter these come back at essentially full amplitude, and
    `alias_hz` is where each one lands."""
    aliased = _peak_amplitude(_resample_tone(freq_hz, antialias=False), alias_hz)
    filtered = _peak_amplitude(_resample_tone(freq_hz, antialias=True), alias_hz)

    assert aliased > 0.5, (
        f"expected {freq_hz}Hz to alias to {alias_hz}Hz without the filter, "
        f"got {aliased:.3f} -- the premise of this test no longer holds"
    )
    assert filtered < 0.02, f"{freq_hz}Hz still aliases to {alias_hz}Hz at {filtered:.4f}"


def test_antialias_filter_is_continuous_across_callback_boundaries(monkeypatch):
    """The filter keeps its own overlap state, like the resampler beside it.
    If it did not, every callback boundary would inject a discontinuity --
    which VAD reads as an onset and Whisper reads as a click."""
    from app.core.config import settings as app_settings

    monkeypatch.setattr(app_settings, "audio_loopback_antialias", True)
    source = DeviceAudioSource(AudioChannel.LOOPBACK)
    source._loopback_sample_rate = 48_000
    source._reset_loopback_buffers()

    native = 48_000
    t = np.arange(native) / native
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    out = np.concatenate([
        source._resample_loopback(tone[i:i + FRAME_SAMPLES])
        for i in range(0, tone.size - FRAME_SAMPLES, FRAME_SAMPLES)
    ])

    # A clean 1kHz sine at 16kHz steps by at most ~0.4 between samples; a
    # per-callback discontinuity would show up as a much larger jump.
    assert np.max(np.abs(np.diff(out[100:]))) < 0.5


def test_loopback_without_matching_device_raises_audio_error(monkeypatch):
    from app.audio import device_source

    manager = FakePyAudioManager([])
    monkeypatch.setattr(device_source, "_pyaudiowpatch", lambda: FakePyAudioModule([manager]))

    with pytest.raises(AudioError, match="No WASAPI loopback device"):
        DeviceAudioSource(AudioChannel.LOOPBACK).describe()
    assert manager.terminated


def test_missing_pyaudiowpatch_raises_actionable_audio_error(monkeypatch):
    from app.audio import device_source

    real_import = __import__

    def missing_dependency(name, *args, **kwargs):
        if name == "pyaudiowpatch":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(device_source.sys, "platform", "win32")
    monkeypatch.setattr("builtins.__import__", missing_dependency)

    with pytest.raises(AudioError, match="PyAudioWPatch"):
        device_source._pyaudiowpatch()


def test_loopback_startup_failure_closes_resources(monkeypatch):
    from app.audio import device_source

    manager = FakePyAudioManager([loopback_device()], start_error=RuntimeError("start failed"))
    monkeypatch.setattr(device_source, "_pyaudiowpatch", lambda: FakePyAudioModule([manager]))

    with pytest.raises(AudioError, match="Could not open LOOPBACK device"):
        DeviceAudioSource(AudioChannel.LOOPBACK).start()
    assert manager.stream.stopped
    assert manager.stream.closed
    assert manager.terminated
