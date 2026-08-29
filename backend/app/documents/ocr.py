"""Reading PDFs that contain pictures of text instead of text.

A scanned resume is still a resume. Before this, one produced

    No extractable text in PDF. Scanned or image-only documents are not
    supported (no OCR).

which told the user their document was unusable and offered nothing to do about
it -- for a file that looks completely normal when they open it.

Why this stack:

  * **pypdfium2** rasterises the page. A pure wheel with the PDF renderer
    statically linked, so there is no Poppler or Ghostscript to install, nothing
    to put on PATH, and no admin rights needed.
  * **rapidocr-onnxruntime** reads the pixels. It runs on onnxruntime, which the
    embedding model already depends on, and ships its own ~15MB of models inside
    the wheel -- so OCR needs no download and works offline on first run.

Both were chosen over the obvious alternatives for the same reason: Tesseract
needs a native installer, and EasyOCR/docTR need PyTorch, which the packaging
spec explicitly excludes because it would add well over a gigabyte to an
installer that is currently a few hundred megabytes.

OCR is the *fallback*, never the default. Native extraction is roughly a
thousand times faster and is exact rather than probabilistic, so it always runs
first and OCR only sees the pages it could not read.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric

logger = get_logger(__name__)

#: Progress callback: (pages_done, pages_total).
ProgressCallback = Callable[[int, int], None]


class OcrUnavailable(RuntimeError):
    """The OCR dependencies are not importable in this build."""


def ocr_available() -> bool:
    """Can this build OCR at all?

    Checked rather than assumed, because a packaging mistake that drops either
    dependency should degrade to the old, honest "cannot read this PDF" error
    instead of crashing mid-ingest.
    """
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("pypdfium2", "rapidocr_onnxruntime")
    )


class _Engine:
    """Lazily-built, process-wide OCR engine.

    Loading the detection and recognition models costs several seconds, so the
    engine is built once and reused. Guarded by a lock because ingestion runs on
    `asyncio.to_thread` worker threads and two uploads can arrive together.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine = None

    def get(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from rapidocr_onnxruntime import RapidOCR
            except Exception as exc:  # pragma: no cover - packaging guard
                # Logged with the underlying cause. The user gets a sentence
                # about reinstalling; whoever has to fix the *build* needs the
                # actual missing module, and without it this failure is
                # indistinguishable from every other packaging problem.
                logger.error(
                    "ocr_engine_unavailable error=%s detail=%s",
                    type(exc).__name__,
                    exc,
                )
                raise OcrUnavailable(
                    "This installation cannot read scanned documents because its "
                    "text-recognition files are missing. Reinstall Call Assistant."
                ) from exc

            started = time.monotonic()
            # Bound the ONNX thread pools. RapidOCR defaults to -1, meaning
            # "use every core", and it runs three models (detect, classify,
            # recognise). Ingestion can overlap a live interview, and an OCR
            # pass that saturates all cores would push Whisper inference --
            # which is on the critical path to an answer -- behind it.
            #
            # Document ingestion is background work and live transcription is
            # not, so OCR takes the smaller share and finishes a little later.
            threads = _ocr_threads()
            self._engine = RapidOCR(
                intra_op_num_threads=threads,
                inter_op_num_threads=1,
            )
            log_metric(
                "ocr_engine_loaded",
                threads=threads,
                duration_ms=elapsed_ms(started, time.monotonic()),
            )
            return self._engine


def _ocr_threads() -> int:
    """How many cores OCR may use.

    Configurable, and otherwise a quarter of the machine (at least one, at most
    four). The cap matters more than the fraction: past a handful of threads the
    per-page gain is small, while the contention it creates with live
    transcription is not.
    """
    configured = settings.ocr_threads
    if configured > 0:
        return configured
    import os

    return max(1, min(4, (os.cpu_count() or 4) // 4))


_engine = _Engine()


def _page_text(result) -> str:
    """Flatten RapidOCR's output into reading-order text.

    RapidOCR returns one entry per detected text box as
    `[box_coordinates, text, confidence]`, already in top-to-bottom order. Each
    box is roughly a line, so joining with newlines preserves the line structure
    the chunker splits on. `None` means the page held no readable text.
    """
    if not result:
        return ""
    lines: list[str] = []
    for entry in result:
        # Defensive: the shape is documented but this is third-party output on
        # a path where a crash would fail the user's whole upload.
        if len(entry) >= 2 and isinstance(entry[1], str):
            text = entry[1].strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def ocr_pdf_pages(
    file_path: Path,
    page_indices: list[int],
    progress: ProgressCallback | None = None,
) -> dict[int, str]:
    """Read the given pages of a PDF with OCR.

    Returns page index -> recognised text. Only the requested pages are
    rendered, so a mostly-digital PDF with two scanned inserts pays for two
    pages rather than the whole document.

    Runs synchronously and is CPU-bound; every caller must already be on a
    worker thread (`DocumentService` parses via `asyncio.to_thread`), because a
    multi-page OCR pass on the event loop would freeze every other request
    including `/health`.
    """
    if not page_indices:
        return {}

    try:
        import numpy as np
        import pypdfium2 as pdfium
    except Exception as exc:  # pragma: no cover - packaging guard
        logger.error(
            "ocr_renderer_unavailable error=%s detail=%s", type(exc).__name__, exc
        )
        raise OcrUnavailable(
            "This installation cannot read scanned documents because its PDF "
            "rendering files are missing. Reinstall Call Assistant."
        ) from exc

    engine = _engine.get()
    started = time.monotonic()
    out: dict[int, str] = {}
    total = len(page_indices)

    document = pdfium.PdfDocument(str(file_path))
    try:
        for done, index in enumerate(page_indices, start=1):
            page_started = time.monotonic()
            page = document[index]
            # Render well above the PDF's nominal 72dpi: below roughly 200dpi
            # recognition accuracy on ordinary body text falls off sharply, and
            # above ~300 the extra pixels cost time without improving it.
            bitmap = page.render(scale=settings.ocr_render_dpi / 72)
            image = np.asarray(bitmap.to_pil().convert("RGB"))

            result, _ = engine(image)
            out[index] = _page_text(result)

            log_metric(
                "ocr_page_completed",
                page=index + 1,
                chars=len(out[index]),
                duration_ms=elapsed_ms(page_started, time.monotonic()),
            )
            if progress is not None:
                progress(done, total)
    finally:
        # pdfium holds a native handle; leaving it open would keep the user's
        # file locked on Windows and they could not delete or replace it.
        document.close()

    log_metric(
        "ocr_completed",
        pages=total,
        chars=sum(len(text) for text in out.values()),
        duration_ms=elapsed_ms(started, time.monotonic()),
    )
    return out
