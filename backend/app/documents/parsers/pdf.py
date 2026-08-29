from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.documents.ocr import OcrUnavailable, ocr_available, ocr_pdf_pages
from app.documents.parsers.base import (
    DocumentParser,
    ParseError,
    ProgressCallback,
    normalize_whitespace,
)
from app.documents.schemas import FileType, NormalizedDocument

logger = get_logger(__name__)


def needs_ocr(page_text: str) -> bool:
    """Is this page's extracted text too thin to be the real content?

    Not simply `not text`. A scanned page routinely carries *some* embedded
    text -- a header, a page number, a footer stamped on by the scanner -- so an
    emptiness test passes a page that is, in substance, a picture. The threshold
    is a character count because it is the one signal that survives every
    scanner, language and layout.

    Being wrong in the generous direction is cheap: OCRing a page that already
    had text costs a second and the better of the two results is kept. Being
    wrong the other way means silently ingesting a document as one line of
    header text, which is exactly the failure worth avoiding.
    """
    return len(page_text.strip()) < settings.ocr_min_chars_per_page


class PdfParser(DocumentParser):
    """Native text first, OCR only for the pages that need it.

    Native extraction is exact and effectively instant; OCR is probabilistic and
    costs about a second per page. So every page is tried natively, and only the
    pages that come back too thin are rendered and recognised. A digital PDF
    never pays for OCR at all, and a PDF with two scanned inserts pays for two
    pages rather than for all of it.
    """

    def supports(self, file_type: FileType) -> bool:
        return file_type == FileType.PDF

    def parse(
        self,
        file_path: Path,
        document_id: str,
        progress: ProgressCallback | None = None,
    ) -> NormalizedDocument:
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(file_path))
            pages = [_extract(page) for page in reader.pages]
        except Exception as exc:
            raise ParseError(
                f"Could not read '{file_path.name}'. The file may be corrupted or "
                f"password protected. Cause: {exc}"
            ) from exc

        if not pages:
            raise ParseError(f"'{file_path.name}' has no pages.")

        scanned = [index for index, text in enumerate(pages) if needs_ocr(text)]
        # Two different facts, and conflating them produces the wrong error:
        # `attempted` decides what to tell the user when nothing was readable,
        # `recovered` records whether OCR text actually ended up in the result.
        attempted = False
        recovered = 0

        if scanned:
            pages, attempted, recovered = self._fill_in_with_ocr(
                file_path, pages, scanned, progress
            )

        text = normalize_whitespace("\n\n".join(page for page in pages if page.strip()))
        if not text.strip():
            raise ParseError(_nothing_readable(file_path, attempted))

        return NormalizedDocument(
            document_id=document_id,
            title=_title(reader) or file_path.stem,
            text=text,
            metadata={
                "parser": "pdf",
                "page_count": len(pages),
                # Recorded because it changes how much the text can be trusted:
                # OCR output has recognition errors that native extraction does
                # not, and anything auditing a bad answer should be able to see
                # which pages were read by machine vision.
                "ocr_used": recovered > 0,
                "ocr_pages": recovered,
            },
        )

    # ------------------------------------------------------------------ OCR

    def _fill_in_with_ocr(
        self,
        file_path: Path,
        pages: list[str],
        scanned: list[int],
        progress: ProgressCallback | None,
    ) -> tuple[list[str], bool, int]:
        """Replace thin pages with their OCR text.

        Returns (pages, ocr_was_attempted, pages_recovered). Never raises for a
        page that simply could not be read -- whether the *document* is a dead
        end is decided once, at the end, by the caller.
        """
        if not ocr_available():
            logger.warning("ocr_unavailable document=%s pages=%d", file_path.name, len(scanned))
            return pages, False, 0

        # A 400-page scanned book would take the better part of an hour and hold
        # the ingest lock for all of it. Cap it, and say so in the log rather
        # than silently truncating the document.
        limit = settings.ocr_max_pages
        targets = scanned[:limit]
        if len(scanned) > limit:
            logger.warning(
                "ocr_page_limit_reached document=%s scanned=%d limit=%d",
                file_path.name, len(scanned), limit,
            )

        logger.info(
            "ocr_started document=%s pages=%d of=%d",
            file_path.name, len(targets), len(pages),
        )
        if progress is not None:
            progress(f"Reading {len(targets)} scanned page{'s' if len(targets) > 1 else ''}…")

        def on_page(done: int, total: int) -> None:
            if progress is not None:
                progress(f"Reading scanned page {done} of {total}…")

        try:
            recognised = ocr_pdf_pages(file_path, targets, progress=on_page)
        except OcrUnavailable as exc:
            logger.warning("ocr_unavailable document=%s error=%s", file_path.name, exc)
            return pages, False, 0
        except Exception as exc:
            # A genuine OCR failure is actionable, unlike a missing dependency.
            raise ParseError(
                f"Could not read the scanned pages of '{file_path.name}'. "
                f"The file may be corrupted or use an unsupported image format. "
                f"Cause: {exc}"
            ) from exc

        merged = list(pages)
        for index, ocr_text in recognised.items():
            # Keep whichever read produced more. On a page with a real text
            # layer that merely tripped the threshold, native text is exact and
            # should win; on a truly scanned page it is a header and OCR wins.
            if len(ocr_text.strip()) > len(merged[index].strip()):
                merged[index] = ocr_text

        # Counted as recovered only where OCR text actually replaced what
        # native extraction produced -- a page where the native read won is not
        # an OCR success.
        recovered = sum(
            1 for index, text in recognised.items()
            if text.strip() and merged[index] == text
        )
        logger.info(
            "ocr_finished document=%s pages_recovered=%d chars=%d",
            file_path.name, recovered, sum(len(t) for t in recognised.values()),
        )
        return merged, True, recovered


def _extract(page) -> str:
    """One page's embedded text. A page that fails to extract is not a document
    failure -- the other pages, and OCR, can still carry it."""
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def _title(reader) -> str:
    try:
        if reader.metadata and reader.metadata.title:
            return str(reader.metadata.title).strip()
    except Exception:
        pass
    return ""


def _nothing_readable(file_path: Path, attempted_ocr: bool) -> str:
    """The one remaining dead end, phrased as something the user can act on.

    Keyed on whether OCR was *attempted*, not on whether it succeeded: if it ran
    and still found nothing, telling the user their file "may be empty" sends
    them to check a file they can see perfectly well.
    """
    if attempted_ocr:
        return (
            f"No readable text was found in '{file_path.name}', even after "
            f"scanning it as images. If the pages are handwritten, very low "
            f"resolution, or rotated, try a clearer scan or paste the text into "
            f"a .txt or .md file instead."
        )
    if not ocr_available():
        return (
            f"'{file_path.name}' appears to be a scanned document, and this "
            f"installation cannot read scanned pages. Reinstall Call Assistant, "
            f"or paste the text into a .txt or .md file instead."
        )
    return (
        f"No readable text was found in '{file_path.name}'. The file may be "
        f"empty or corrupted."
    )
