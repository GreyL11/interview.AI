import threading
import time

import numpy as np

from app.audio.base import SAMPLE_RATE
from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
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
        self._download_root = download_root or str(settings.data_dir / "models" / "whisper")
        self._model = None
        self._lock = threading.Lock()
        self._warmed_up = False

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
            try:
                self._model = WhisperModel(
                    self._model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=self._download_root,
                    # CTranslate2 serialises calls across num_workers slots. If
                    # this were left at 1 while the scheduler ran two threads,
                    # the second would block inside C++ where no priority
                    # applies — so the two numbers must agree.
                    num_workers=max(1, settings.stt_inference_concurrency),
                    cpu_threads=max(0, settings.stt_cpu_threads),
                )
            except Exception as exc:
                if device == "cuda":
                    raise SttError(
                        "CUDA STT was explicitly selected but could not start. "
                        "Install a compatible NVIDIA CUDA runtime with cuBLAS/cuDNN "
                        "libraries, or set STT_DEVICE=cpu and STT_COMPUTE_TYPE=int8. "
                        f"Cause: {exc}"
                    ) from exc
                raise SttError(
                    f"Could not load STT model '{self._model_name}'. First run needs "
                    f"network access to download it. Cause: {exc}"
                ) from exc

            logger.info(
                "stt_model_loaded model=%s device=%s compute=%s",
                self._model_name, device, compute_type,
            )
            return self._model

    def _resolve_device(self) -> tuple[str, str]:
        """Validate an explicit CUDA request; CPU is the safe default."""
        if self._device == "cpu":
            return "cpu", self._compute_type
        if self._device == "cuda":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    return "cuda", self._compute_type
            except Exception as exc:
                raise SttError(
                    "CUDA STT was explicitly selected but CUDA/cuBLAS libraries "
                    "are unavailable. Install a compatible NVIDIA CUDA runtime, or "
                    "set STT_DEVICE=cpu and STT_COMPUTE_TYPE=int8."
                ) from exc
            raise SttError(
                "CUDA STT was explicitly selected but no usable CUDA device was found. "
                "Install the NVIDIA CUDA/cuBLAS runtime, or set STT_DEVICE=cpu and "
                "STT_COMPUTE_TYPE=int8."
            )

        if self._device != "auto":
            return self._device, self._compute_type

        # Preserve compatibility with legacy STT_DEVICE=auto deployments, but
        # never opt into CUDA implicitly; use STT_DEVICE=cuda to request it.
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
            segments, info = model.transcribe(
                audio,
                language=settings.stt_language or None,
                beam_size=settings.stt_beam_size if is_final else 1,
                vad_filter=False,  # segmentation already happened upstream
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise SttError(f"Transcription failed: {exc}") from exc

        return Transcript(
            text=text,
            is_final=is_final,
            language=getattr(info, "language", "en") or "en",
            duration_ms=int(len(audio) / SAMPLE_RATE * 1000),
        )
