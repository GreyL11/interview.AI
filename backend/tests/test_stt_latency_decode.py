import numpy as np

from app.stt.faster_whisper_engine import FasterWhisperEngine


class _Info:
    language = "en"


class _Model:
    def __init__(self):
        self.kwargs = []

    def transcribe(self, audio, **kw):
        self.kwargs.append(kw)
        return iter([]), _Info()


def _run(is_final):
    eng = FasterWhisperEngine.__new__(FasterWhisperEngine)
    m = _Model()
    eng._run(m, np.zeros(16000, dtype=np.float32), is_final)
    return m.kwargs[0]


def test_interim_pins_temperature_final_short_ladder():
    interim, final = _run(False), _run(True)
    assert interim["temperature"] == 0.0 and interim["without_timestamps"] is True
    assert final["temperature"] == (0.0, 0.4) and final["without_timestamps"] is True
