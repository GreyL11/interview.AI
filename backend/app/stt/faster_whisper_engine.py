import threading

import numpy as np

from app.audio.base import SAMPLE_RATE
from app.core.config import settings
from app.core.logging import get_logger
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
                )
            except Exception as exc:
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
        """CUDA needs cuDNN/cuBLAS DLLs we deliberately do not ship. Probe rather
        than assume, and fall back to CPU instead of failing the session."""
        if self._device != "auto":
            return self._device, self._compute_type
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"

    def warmup(self) -> None:
        self._ensure_loaded()

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
