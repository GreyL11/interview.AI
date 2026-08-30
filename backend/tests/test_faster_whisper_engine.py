import sys
from types import SimpleNamespace

import pytest

from app.core.config import Settings, resolved_stt_cpu_threads, settings
from app.stt.base import SttError
from app.stt.faster_whisper_engine import FasterWhisperEngine


def test_stt_defaults_to_automatic_hardware_selection(monkeypatch):
    """`auto`, not `cpu`: the same build should accelerate on an NVIDIA machine
    and fall back on an AMD or Intel one without being configured."""
    monkeypatch.delenv("STT_DEVICE", raising=False)
    monkeypatch.delenv("STT_COMPUTE_TYPE", raising=False)

    configured = Settings(_env_file=None)

    assert configured.stt_device == "auto"
    assert configured.stt_compute_type == "int8"        # used on CPU
    assert configured.stt_cuda_compute_type == "float16"  # used on a GPU


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
        SimpleNamespace(
            get_cuda_device_count=lambda: 1,
            # The probe touches the driver rather than trusting enumeration.
            get_supported_compute_types=lambda device: {"float16", "int8"},
        ),
    )
    from app.stt import faster_whisper_engine as engine_module

    engine_module.reset_cuda_probe()

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
            # Must match the scheduler's thread count, or CTranslate2 serialises
            # the extra threads internally where priority cannot reach them.
            "num_workers": settings.stt_inference_concurrency,
            # Resolved, not passed through: 0 means "decide for me", and
            # deferring to CTranslate2 measured ~2.5x slower than an explicit
            # count.
            "cpu_threads": resolved_stt_cpu_threads(),
        })
    ]


def test_explicit_cuda_without_a_usable_device_is_actionable(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(
            get_cuda_device_count=lambda: 0,
            get_supported_compute_types=lambda device: set(),
        ),
    )
    from app.stt import faster_whisper_engine as engine_module

    engine_module.reset_cuda_probe()

    # Points at `auto` now, which is the recovery an unconfigured user wants.
    with pytest.raises(SttError, match="STT_DEVICE=auto"):
        FasterWhisperEngine(device="cuda", compute_type="float16")._resolve_device()


def test_warmup_runs_the_throwaway_inference_pass_only_once(monkeypatch, tmp_path):
    """AudioPipeline.start() blocks on warmup() every time audio starts (a
    stop/start toggle, a second session, a reconnect). Re-running a real
    inference pass on an already-warm model on every one of those calls would
    burn CPU and delay capture for no benefit, so only the first call should
    actually transcribe."""
    transcribe_calls = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            transcribe_calls.append(audio)
            return [], SimpleNamespace(language="en")

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    engine = FasterWhisperEngine(
        model_name="mock-model", device="cpu", compute_type="int8",
        download_root=str(tmp_path),
    )

    engine.warmup()
    engine.warmup()
    engine.warmup()

    assert len(transcribe_calls) == 1


# ------------------------------------------------- automatic hardware selection


def test_auto_uses_the_gpu_when_one_genuinely_works(monkeypatch):
    """The point of `auto`: the same build accelerates on an NVIDIA laptop
    without anyone configuring it."""
    from app.stt import faster_whisper_engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (True, "cuda ready"))
    engine = FasterWhisperEngine(model_name="m", device="auto")

    assert engine._resolve_device() == ("cuda", settings.stt_cuda_compute_type)


def test_auto_falls_back_to_the_cpu_without_a_gpu(monkeypatch):
    """AMD, Intel, a VM, or a machine with a stale driver -- all of which must
    still transcribe rather than refuse."""
    from app.stt import faster_whisper_engine as engine_module

    monkeypatch.setattr(
        engine_module, "_cuda_usable", lambda: (False, "no nvidia gpu detected")
    )
    engine = FasterWhisperEngine(model_name="m", device="auto")

    assert engine._resolve_device() == ("cpu", settings.stt_compute_type)


def test_auto_never_raises_over_missing_hardware(monkeypatch):
    from app.stt import faster_whisper_engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (False, "whatever"))
    FasterWhisperEngine(model_name="m", device="auto")._resolve_device()


def test_an_explicit_cuda_request_still_fails_loudly(monkeypatch):
    """`auto` is forgiving, `cuda` is not: someone who asked for the GPU would
    read a silent 20x-slower CPU fallback as a performance bug."""
    from app.stt import faster_whisper_engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (False, "no gpu"))
    with pytest.raises(SttError) as caught:
        FasterWhisperEngine(model_name="m", device="cuda")._resolve_device()
    assert "STT_DEVICE=auto" in str(caught.value)


def test_an_explicit_cpu_request_never_probes(monkeypatch):
    """Probing loads CUDA libraries; someone who said cpu should not pay for
    that on every start."""
    from app.stt import faster_whisper_engine as engine_module

    def explode():
        raise AssertionError("cpu must not probe for a GPU")

    monkeypatch.setattr(engine_module, "_cuda_usable", explode)
    assert FasterWhisperEngine(model_name="m", device="cpu")._resolve_device() == (
        "cpu",
        settings.stt_compute_type,
    )


def test_the_probe_treats_an_unusable_driver_as_no_gpu(monkeypatch):
    """Enumeration is not enough. This machine reports a device count and then
    raises "CUDA driver version is insufficient" on the next call -- which is
    exactly the case that would otherwise crash a session."""
    from app.stt import faster_whisper_engine as engine_module

    class FakeCt2:
        @staticmethod
        def get_cuda_device_count():
            return 1

        @staticmethod
        def get_supported_compute_types(device):
            raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCt2)
    engine_module.reset_cuda_probe()
    try:
        usable, reason = engine_module._cuda_usable()
        assert usable is False
        assert "unusable" in reason
    finally:
        engine_module.reset_cuda_probe()


def test_the_probe_accepts_a_working_gpu(monkeypatch):
    from app.stt import faster_whisper_engine as engine_module

    class FakeCt2:
        @staticmethod
        def get_cuda_device_count():
            return 1

        @staticmethod
        def get_supported_compute_types(device):
            return {"float16", "int8"}

    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCt2)
    engine_module.reset_cuda_probe()
    try:
        usable, _ = engine_module._cuda_usable()
        assert usable is True
    finally:
        engine_module.reset_cuda_probe()


def test_the_probe_is_cached(monkeypatch):
    """It loads CUDA libraries, so it must not run per model load."""
    from app.stt import faster_whisper_engine as engine_module

    calls = []

    class FakeCt2:
        @staticmethod
        def get_cuda_device_count():
            calls.append(1)
            return 0

    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCt2)
    engine_module.reset_cuda_probe()
    try:
        engine_module._cuda_usable()
        engine_module._cuda_usable()
        engine_module._cuda_usable()
        assert len(calls) == 1
    finally:
        engine_module.reset_cuda_probe()


def test_an_auto_selected_gpu_that_fails_to_load_falls_back(monkeypatch, tmp_path):
    """The probe can only test the driver, not the whole model-construction
    path. If the GPU passes the probe and still fails to load, refusing to
    transcribe on a machine whose CPU would have worked is the wrong answer."""
    from app.stt import faster_whisper_engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (True, "cuda ready"))
    attempts = []

    class FlakyGpuModel:
        def __init__(self, name, **kwargs):
            attempts.append(kwargs["device"])
            if kwargs["device"] == "cuda":
                raise RuntimeError("cudnn missing")

    monkeypatch.setattr(engine_module, "WhisperModel", FlakyGpuModel, raising=False)
    monkeypatch.setitem(
        sys.modules, "faster_whisper", type("m", (), {"WhisperModel": FlakyGpuModel})
    )

    engine = FasterWhisperEngine(
        model_name="m", device="auto", download_root=str(tmp_path)
    )
    engine._ensure_loaded()

    assert attempts == ["cuda", "cpu"], "should have retried on the CPU"
    assert engine.active_device == "cpu"


def test_the_active_device_is_reported_only_after_loading(monkeypatch, tmp_path):
    from app.stt import faster_whisper_engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (False, "no gpu"))
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        type("m", (), {"WhisperModel": lambda name, **kw: object()}),
    )

    engine = FasterWhisperEngine(model_name="m", device="auto", download_root=str(tmp_path))
    assert engine.active_device is None  # nothing has run yet
    engine._ensure_loaded()
    assert engine.active_device == "cpu"


# --------------------------------------------- runtime CUDA failure -> CPU


def _fake_faster_whisper(monkeypatch, model_cls):
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=model_cls))


class _LazyCublasModel:
    """CTranslate2 loads cuBLAS on the first inference, not at construction --
    so a GPU build succeeds and only blows up once audio arrives."""

    built = []

    def __init__(self, name, **kwargs):
        self.device = kwargs["device"]
        _LazyCublasModel.built.append((kwargs["device"], kwargs["compute_type"]))

    def transcribe(self, audio, **kwargs):
        if self.device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return [SimpleNamespace(text=" hello ")], SimpleNamespace(language="en")


def test_missing_cuda_libraries_at_inference_fall_back_to_the_cpu(monkeypatch, tmp_path):
    """The bug this exists for: the model loads on the GPU, then the first real
    transcription dies on a missing cublas64_12.dll. It must retry on the CPU
    in place, without an app restart."""
    from app.stt import faster_whisper_engine as engine_module
    import numpy as np

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (True, "cuda ready"))
    _LazyCublasModel.built = []
    _fake_faster_whisper(monkeypatch, _LazyCublasModel)

    engine = FasterWhisperEngine(model_name="m", device="auto", download_root=str(tmp_path))
    result = engine.transcribe(np.zeros(1600, dtype=np.float32), is_final=True)

    assert result.text == "hello"
    assert engine.active_device == "cpu"
    assert _LazyCublasModel.built == [
        ("cuda", settings.stt_cuda_compute_type),
        ("cpu", "int8"),  # CPU wants int8, not the GPU's float16
    ]


def test_the_cpu_fallback_is_sticky_across_calls(monkeypatch, tmp_path):
    """Retrying the GPU per utterance would pay the same failure every time."""
    from app.stt import faster_whisper_engine as engine_module
    import numpy as np

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (True, "cuda ready"))
    _LazyCublasModel.built = []
    _fake_faster_whisper(monkeypatch, _LazyCublasModel)

    engine = FasterWhisperEngine(model_name="m", device="auto", download_root=str(tmp_path))
    for _ in range(3):
        engine.transcribe(np.zeros(1600, dtype=np.float32), is_final=False)

    assert [device for device, _ in _LazyCublasModel.built] == ["cuda", "cpu"]


def test_a_working_gpu_keeps_running_on_the_gpu(monkeypatch, tmp_path):
    """The fallback must not cost acceleration where CUDA actually works."""
    from app.stt import faster_whisper_engine as engine_module
    import numpy as np

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (True, "cuda ready"))
    built = []

    class WorkingGpuModel:
        def __init__(self, name, **kwargs):
            built.append(kwargs["device"])

        def transcribe(self, audio, **kwargs):
            return [SimpleNamespace(text="ok")], SimpleNamespace(language="en")

    _fake_faster_whisper(monkeypatch, WorkingGpuModel)

    engine = FasterWhisperEngine(model_name="m", device="auto", download_root=str(tmp_path))
    engine.transcribe(np.zeros(1600, dtype=np.float32), is_final=True)

    assert built == ["cuda"]
    assert engine.active_device == "cuda"


def test_an_explicit_cuda_request_does_not_silently_fall_back(monkeypatch, tmp_path):
    """`auto` is forgiving; someone who demanded the GPU gets the error."""
    from app.stt import faster_whisper_engine as engine_module
    import numpy as np

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (True, "cuda ready"))
    _LazyCublasModel.built = []
    _fake_faster_whisper(monkeypatch, _LazyCublasModel)

    engine = FasterWhisperEngine(
        model_name="m", device="cuda", compute_type="float16", download_root=str(tmp_path)
    )
    with pytest.raises(SttError, match="cublas64_12.dll"):
        engine.transcribe(np.zeros(1600, dtype=np.float32), is_final=True)

    assert [device for device, _ in _LazyCublasModel.built] == ["cuda"]


def test_warmup_survives_a_gpu_that_dies_on_first_inference(monkeypatch, tmp_path):
    """Warmup is the first inference, so it is where this fails in production --
    and AudioPipeline.start() blocks on it."""
    from app.stt import faster_whisper_engine as engine_module

    monkeypatch.setattr(engine_module, "_cuda_usable", lambda: (True, "cuda ready"))
    _LazyCublasModel.built = []
    _fake_faster_whisper(monkeypatch, _LazyCublasModel)

    engine = FasterWhisperEngine(model_name="m", device="auto", download_root=str(tmp_path))
    engine.warmup()

    assert engine.active_device == "cpu"
