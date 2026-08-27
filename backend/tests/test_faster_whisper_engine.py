import sys
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.stt.base import SttError
from app.stt.faster_whisper_engine import FasterWhisperEngine


def test_default_stt_configuration_is_cpu_int8(monkeypatch):
    monkeypatch.delenv("STT_DEVICE", raising=False)
    monkeypatch.delenv("STT_COMPUTE_TYPE", raising=False)

    configured = Settings(_env_file=None)

    assert configured.stt_device == "cpu"
    assert configured.stt_compute_type == "int8"


def test_explicit_cpu_configuration_is_used():
    engine = FasterWhisperEngine(device="cpu", compute_type="int8")

    assert engine._resolve_device() == ("cpu", "int8")


def test_explicit_cuda_configuration_is_passed_to_whisper_model(monkeypatch, tmp_path):
    calls = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 1),
    )

    engine = FasterWhisperEngine(
        model_name="mock-model",
        device="cuda",
        compute_type="float16",
        download_root=str(tmp_path),
    )
    engine._ensure_loaded()

    assert calls == [
        (("mock-model",), {
            "device": "cuda",
            "compute_type": "float16",
            "download_root": str(tmp_path),
        })
    ]


def test_explicit_cuda_without_a_usable_device_is_actionable(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 0),
    )

    with pytest.raises(SttError, match="STT_DEVICE=cpu"):
        FasterWhisperEngine(device="cuda", compute_type="float16")._resolve_device()
