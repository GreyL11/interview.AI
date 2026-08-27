"""Device enumeration and capture-source tests.

sounddevice/PortAudio is not installed in this environment, so these tests
exercise the abstraction and the failure paths. Real device capture is verified
manually — see README, "Pending hardware verification".
"""

import numpy as np
import pytest

from app.audio.base import FRAME_SAMPLES, AudioChannel, AudioError
from app.audio.devices import audio_available, list_devices
from tests.fakes import FakeAudioSource, silence_frames, speech_frames


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
