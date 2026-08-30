import threading
import time

import numpy as np

from app.audio.base import SAMPLE_RATE
from app.core.config import resolved_stt_cpu_threads, settings
from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
from app.model_status import tracker
from app.stt.base import SttEngine, SttError, Transcript

logger = get_logger(__name__)


class FasterWhisperEngine(SttEngine):
    """faster-whisper (CTranslate2). No torch.

    Interim passes run with beam_size=1 and no VAD filter because latency
    dominates while text is still being displayed; the final pass uses the
    configured beam size, since that transcript is what the classifier and the
    LLM actually see.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        download_root: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.stt_model
        self._device = device or settings.stt_device
        self._compute_type = compute_type or settings.stt_compute_type
        self._cuda_compute_type = settings.stt_cuda_compute_type
        #: What was actually chosen, once the model has loaded. Reported to the
        #: UI so "GPU" is a fact rather than a hope.
        self.active_device: str | None = None
        self._download_root = download_root or str(settings.data_dir / "models" / "whisper")
        self._model = None
        self._lock = threading.Lock()
        self._warmed_up = False
        #: Set once CUDA has proved unusable at runtime. Sticky, so the
        #: next call does not retry the GPU and pay the failure again.
        self._cuda_demoted = False

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise SttError(
                    f"faster-whisper is not installed: {exc}. "
                    "Install the audio extras to enable speech-to-text."
                ) from exc

            device, compute_type = self._resolve_device()
            # WhisperModel downloads and loads in one call, so the UI is told
            # "downloading" for the whole of it. That is honest on a first run
            # and briefly pessimistic afterwards, which is the right way round:
            # the long wait is the download, and it is the one the user needs
            # explained.
            tracker.reset("stt")
            tracker.downloading("stt")
            try:
                self._model = self._build(WhisperModel, device, compute_type)
            except Exception as exc:
                # An auto-selected GPU that passed the probe and still failed to
                # load is the one case worth a second attempt: the probe can
                # only test the driver, not the whole model-construction path,
                # and falling back beats refusing to transcribe on a machine
                # whose CPU would have worked fine.
                if device == "cuda" and self._device == "auto":
                    self._demote_from_cuda("initialisation", exc)
                    device, compute_type = "cpu", self._compute_type
                    try:
                        self._model = self._build(WhisperModel, device, compute_type)
                    except Exception as cpu_exc:
                        raise self._load_failure(device, cpu_exc) from cpu_exc
                else:
                    raise self._load_failure(device, exc) from exc

            self.active_device = device
            tracker.ready("stt", device=device)
            logger.info(
                "stt_model_loaded model=%s device=%s compute=%s threads=%d",
                self._model_name, device, compute_type, resolved_stt_cpu_threads(),
            )
            return self._model

    def _build(self, whisper_model, device: str, compute_type: str):
        return whisper_model(
            self._model_name,
            device=device,
            compute_type=compute_type,
            download_root=self._download_root,
            # CTranslate2 serialises calls across num_workers slots. If this
            # were left at 1 while the scheduler ran two threads, the second
            # would block inside C++ where no priority applies — so the two
            # numbers must agree.
            num_workers=max(1, settings.stt_inference_concurrency),
            cpu_threads=resolved_stt_cpu_threads(),
        )

    def _load_failure(self, device: str, exc: Exception) -> SttError:
        if device == "cuda":
            message = (
                "CUDA speech recognition was explicitly selected but could not "
                "start. Install a compatible NVIDIA CUDA runtime with "
                "cuBLAS/cuDNN, or set STT_DEVICE=auto to fall back to the CPU "
                f"automatically. Cause: {exc}"
            )
        else:
            message = (
                f"Could not load the speech model '{self._model_name}'. The "
                f"first run needs internet access to download it. If it was "
                f"downloaded before, delete '{self._download_root}' and try "
                f"again to repair an incomplete copy. Cause: {exc}"
            )
        tracker.failed("stt", message)
        return SttError(message)

    def _resolve_device(self) -> tuple[str, str]:
        """Decide what to run on, and with what numeric precision.

        Three modes, and the difference between them is what happens when CUDA
        is asked for but unusable:

          "cuda" -- an explicit demand. Fail loudly, because someone chose it
                    and silently running 20x slower on the CPU would look like
                    a performance bug rather than a missing driver.
          "cpu"  -- an explicit demand. Never probes.
          "auto" -- the default. Use the GPU when it genuinely works, fall back
                    to the CPU when it does not, and say which in the log.

        Precision travels with the device: int8 is the right default on a CPU
        and float16 on an NVIDIA GPU, so picking the device implicitly has to
        pick the compute type too, or `auto` would land on a combination the
        hardware handles badly.
        """
        if self._device == "cpu" or self._cuda_demoted:
            return "cpu", self._compute_type

        if self._device == "cuda":
            usable, reason = _cuda_usable()
            if not usable:
                raise SttError(
                    f"CUDA speech recognition was explicitly requested but is not "
                    f"usable on this machine ({reason}). Install a compatible NVIDIA "
                    f"CUDA runtime with cuBLAS/cuDNN, or set STT_DEVICE=auto to fall "
                    f"back to the CPU automatically."
                )
            return "cuda", self._cuda_compute_type

        if self._device != "auto":
            return self._device, self._compute_type

        usable, reason = _cuda_usable()
        if usable:
            logger.info(
                "stt_gpu_candidate_detected device=cuda reason=%s; "
                "attempting cuda initialisation", reason,
            )
            return "cuda", self._cuda_compute_type
        # Not a warning: no GPU is the normal case on most laptops, and an
        # AMD or Intel machine would otherwise log a scary line on every start
        # about hardware it was never going to have.
        logger.info("stt_accelerator_selected device=cpu reason=%s", reason)
        return "cpu", self._compute_type

    def warmup(self) -> None:
        """Load the model and run one throwaway pass, the first time only.

        Loading alone is not enough on that first call: the first real
        `transcribe` still pays for CTranslate2's lazy graph setup, so half a
        second of silence buys that back before anyone speaks. But
        AudioPipeline.start() now blocks on this call every time audio starts
        (a stop/start toggle, a second session, a reconnect) — repeating a real
        inference pass on an already-warm model would just burn CPU and delay
        capture for no benefit, so every call after the first is a no-op.
        """
        if self._warmed_up:
            self._ensure_loaded()
            log_metric("stt_warmup_skipped", model=self._model_name)
            return
        started = time.monotonic()
        self._ensure_loaded()
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), is_final=False)
        self._warmed_up = True
        log_metric(
            "stt_warmup_completed",
            model=self._model_name,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )

    def transcribe(self, audio: np.ndarray, is_final: bool) -> Transcript:
        model = self._ensure_loaded()
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        try:
            return self._run(model, audio, is_final)
        except Exception as exc:
            # CTranslate2 loads cuBLAS/cuDNN lazily, on the first inference
            # rather than at model construction -- so a machine with an NVIDIA
            # GPU but no CUDA runtime builds the model happily and then dies
            # here with "Library cublas64_12.dll is not found". Rebuild on the
            # CPU and retry in place: the user gets their transcript, not a
            # restart.
            if self.active_device != "cuda" or self._device != "auto":
                raise SttError(f"Transcription failed: {exc}") from exc
            self._demote_from_cuda("inference", exc)
            try:
                return self._run(self._ensure_loaded(), audio, is_final)
            except Exception as cpu_exc:
                raise SttError(f"Transcription failed: {cpu_exc}") from cpu_exc

    def _run(self, model, audio: np.ndarray, is_final: bool) -> Transcript:
        segments, info = model.transcribe(
            audio,
            language=settings.stt_language or None,
            beam_size=settings.stt_beam_size if is_final else 1,
            vad_filter=False,  # segmentation already happened upstream
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()

        return Transcript(
            text=text,
            is_final=is_final,
            language=getattr(info, "language", "en") or "en",
            duration_ms=int(len(audio) / SAMPLE_RATE * 1000),
        )

    def _demote_from_cuda(self, stage: str, exc: Exception) -> None:
        """Give up on the GPU for the rest of the process and go to the CPU.

        Sticky on purpose: CUDA that is broken at 10:00 is still broken at
        10:01, and re-probing per utterance would pay the failure every time.
        """
        logger.warning(
            "stt_cuda_init_failed stage=%s error=%s: %s", stage, type(exc).__name__, exc
        )
        self._cuda_demoted = True
        self._model = None
        self.active_device = None
        logger.warning(
            "stt_cpu_fallback_activated compute=%s reason=cuda_%s_failed",
            self._compute_type, stage,
        )


#: Probed once. The answer cannot change while the process runs, and the probe
#: loads CUDA libraries, which is not free.
_cuda_probe: tuple[bool, str] | None = None
_cuda_probe_lock = threading.Lock()


def _cuda_usable() -> tuple[bool, str]:
    """Is there an NVIDIA GPU this machine can actually run inference on?

    Returns (usable, reason) -- the reason is for the log, never for the user.

    Counting devices is not enough on its own. A machine can report a GPU and
    still fail at model construction: this one does, with "CUDA driver version
    is insufficient for CUDA runtime version". So the probe asks CTranslate2
    what compute types the device supports, which is the cheapest call that
    actually touches the driver rather than just enumerating hardware.

    Anything unexpected counts as unusable. CPU transcription is slower but it
    always works, and the whole point of `auto` is that the app runs everywhere
    -- AMD, Intel, a VM, a laptop with a stale driver -- without being
    configured.
    """
    global _cuda_probe
    if _cuda_probe is not None:
        return _cuda_probe

    with _cuda_probe_lock:
        if _cuda_probe is not None:
            return _cuda_probe
        _cuda_probe = _probe_cuda()
        return _cuda_probe


def _probe_cuda() -> tuple[bool, str]:
    try:
        import ctranslate2
    except Exception as exc:  # pragma: no cover - packaging guard
        return False, f"ctranslate2 unavailable ({type(exc).__name__})"

    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False, "no nvidia gpu detected"
    except Exception as exc:
        return False, f"gpu enumeration failed ({type(exc).__name__})"

    try:
        # Touches the driver. Raises on a version mismatch, which enumeration
        # alone does not -- that is the case this probe exists to catch.
        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception as exc:
        return False, f"cuda driver unusable ({type(exc).__name__})"

    if not supported:
        return False, "gpu reports no usable compute types"
    return True, f"cuda ready ({','.join(sorted(supported))})"


def reset_cuda_probe() -> None:
    """Forget the cached probe. Tests use this; nothing in the app does."""
    global _cuda_probe
    _cuda_probe = None
