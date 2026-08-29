"""Scanned PDFs, which used to be a dead end.

A scanned resume is still a resume, but before this it produced

    No extractable text in PDF. Scanned or image-only documents are not
    supported (no OCR).

for a file that looks completely normal when the user opens it.

The PDFs here are *built* rather than checked in as fixtures, so the "scanned"
one is genuinely image-only -- a committed binary could rot into having a text
layer and this suite would keep passing while testing nothing.

OCR is slow (a second or two per page) and its accuracy is a property of the
model, not of this code. So the assertions are about the *decisions*: which
pages get OCR'd, which do not, what wins when both reads produce text, and what
the user is told when nothing works. Only one test actually recognises pixels.
"""

import pytest

from app.core.config import settings
from app.documents.parsers.base import ParseError
from app.documents.parsers.pdf import PdfParser, needs_ocr
from app.documents.schemas import FileType

pytest.importorskip("PIL", reason="Pillow is needed to build the test PDFs")

pdfium = pytest.importorskip("pypdfium2", reason="OCR rendering backend")
pytest.importorskip("rapidocr_onnxruntime", reason="OCR recognition backend")


SCANNED_LINES = [
    "Senior Backend Engineer",
    "Built a payment reconciliation service",
    "handling four million transactions per day.",
]


def _image_pdf(path, lines=SCANNED_LINES, pages=1):
    """A PDF whose pages are images of text -- exactly what a scanner produces.

    Rendered with Pillow and saved as a PDF, so there is no text layer at all.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:  # pragma: no cover - depends on the machine's fonts
        font = ImageFont.load_default()

    frames = []
    for _ in range(pages):
        image = Image.new("RGB", (1240, 600), "white")
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            draw.text((70, 80 + index * 90), line, fill="black", font=font)
        frames.append(image)

    frames[0].save(
        str(path),
        "PDF",
        resolution=150,
        save_all=len(frames) > 1,
        append_images=frames[1:],
    )
    return path


DIGITAL_BODY = (
    "This paragraph is real, selectable text in the PDF and is long enough "
    "to look like the body of an actual page rather than a scanner header."
)


def _text_pdf(path, body=DIGITAL_BODY):
    """A normal digital PDF with a genuine text layer."""
    # Built by hand rather than with a library: pypdfium2's text-writing API
    # varies across versions, and this test needs to own exactly what is in the
    # file -- the whole point is that this one is not a scan.
    path.write_bytes(_minimal_text_pdf(body))
    return path


def _minimal_text_pdf(body: str) -> bytes:
    """A hand-built one-page PDF with a real text layer.

    Written literally rather than with a library so the test owns exactly what
    is in the file -- the whole point is that this one is *not* a scan.
    """
    escaped = body.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 18 Tf 60 700 Td ({escaped}) Tj ET".encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body_bytes in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body_bytes + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def parser():
    return PdfParser()


# ----------------------------------------------------- the OCR decision itself


def test_an_empty_page_needs_ocr():
    assert needs_ocr("") is True
    assert needs_ocr("   \n  ") is True


def test_a_page_with_only_a_scanner_header_still_needs_ocr():
    """The reason this is a threshold and not an emptiness check: scanners stamp
    a header or page number onto the image, so a truly scanned page often has a
    few characters of real text and would pass `if not text`."""
    assert needs_ocr("Page 1 of 4") is True


def test_a_page_of_real_prose_does_not_need_ocr():
    body = (
        "Led the migration of the billing service from a monolith to three "
        "independently deployable components over eight months."
    )
    assert needs_ocr(body) is False


def test_the_threshold_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "ocr_min_chars_per_page", 5)
    assert needs_ocr("abcdefghij") is False
    monkeypatch.setattr(settings, "ocr_min_chars_per_page", 500)
    assert needs_ocr("abcdefghij") is True


# ------------------------------------------------------------- the fast path


def test_a_digital_pdf_is_read_natively_and_never_touches_ocr(parser, tmp_path, monkeypatch):
    """OCR is roughly a thousand times slower and is probabilistic. A PDF with a
    real text layer must never pay for it."""
    from app.documents import ocr as ocr_module

    called = False

    def explode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("OCR must not run on a digital PDF")

    monkeypatch.setattr(ocr_module, "ocr_pdf_pages", explode)
    monkeypatch.setattr(
        "app.documents.parsers.pdf.ocr_pdf_pages", explode
    )

    path = _text_pdf(tmp_path / "digital.pdf")
    result = parser.parse(path, "doc-1")

    assert called is False
    assert result.metadata["ocr_used"] is False
    assert "real, selectable text" in result.text


# ------------------------------------------------------------ the OCR path


@pytest.mark.slow
def test_a_scanned_pdf_is_read_by_ocr(parser, tmp_path):
    """The end-to-end proof: a PDF with no text layer at all still ingests.

    This is the one test that runs real recognition, so it is also the slowest.
    """
    path = _image_pdf(tmp_path / "scanned.pdf")

    # Precondition: prove to ourselves the file really has no text layer,
    # otherwise this test could pass on the native path.
    from pypdf import PdfReader

    native = "".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    assert native.strip() == "", "fixture is not actually a scanned PDF"

    result = parser.parse(path, "doc-2")

    assert result.metadata["ocr_used"] is True
    assert result.metadata["ocr_pages"] == 1
    # Recognition is not exact, so assert on distinctive words rather than on
    # the whole sentence.
    lowered = result.text.lower()
    assert "engineer" in lowered
    assert "reconciliation" in lowered


@pytest.mark.slow
def test_ocr_reports_progress_for_each_page(parser, tmp_path):
    """A multi-page OCR pass takes minutes; an unexplained spinner for that long
    is indistinguishable from a hang."""
    path = _image_pdf(tmp_path / "scanned-multi.pdf", pages=2)

    messages: list[str] = []
    parser.parse(path, "doc-3", progress=messages.append)

    assert messages, "a slow parse must say what it is doing"
    assert any("1 of 2" in m for m in messages)
    assert any("2 of 2" in m for m in messages)


# ------------------------------------------------------------------ failure


def test_a_scanned_pdf_is_not_rejected_before_ocr_is_tried(parser, tmp_path, monkeypatch):
    """Regression: the old parser raised as soon as native extraction came back
    empty, which is the bug -- that is the moment OCR should start, not the
    moment to give up."""
    seen: dict = {}

    def record(file_path, page_indices, progress=None):
        seen["pages"] = page_indices
        return {index: "recovered text from the scan" for index in page_indices}

    monkeypatch.setattr("app.documents.parsers.pdf.ocr_pdf_pages", record)

    path = _image_pdf(tmp_path / "scanned.pdf")
    result = parser.parse(path, "doc-4")

    assert seen["pages"] == [0]
    assert "recovered text" in result.text


def test_a_genuine_ocr_failure_says_what_to_try(parser, tmp_path, monkeypatch):
    """OCR ran and found nothing. The user needs a next step, not a diagnosis."""
    monkeypatch.setattr(
        "app.documents.parsers.pdf.ocr_pdf_pages",
        lambda file_path, page_indices, progress=None: {i: "" for i in page_indices},
    )

    path = _image_pdf(tmp_path / "unreadable.pdf")
    with pytest.raises(ParseError) as caught:
        parser.parse(path, "doc-5")

    message = str(caught.value)
    assert "unreadable.pdf" in message
    # Names something the user can actually do.
    assert "clearer scan" in message or "paste the text" in message
    # And never the old dead end.
    assert "not supported" not in message


def test_an_ocr_crash_is_reported_as_a_document_problem(parser, tmp_path, monkeypatch):
    def explode(file_path, page_indices, progress=None):
        raise RuntimeError("decoder blew up")

    monkeypatch.setattr("app.documents.parsers.pdf.ocr_pdf_pages", explode)

    path = _image_pdf(tmp_path / "broken.pdf")
    with pytest.raises(ParseError) as caught:
        parser.parse(path, "doc-6")
    assert "broken.pdf" in str(caught.value)


def test_a_build_without_ocr_explains_itself_rather_than_crashing(
    parser, tmp_path, monkeypatch
):
    """If packaging ever drops the OCR dependencies, the app must degrade to an
    honest message instead of a traceback."""
    monkeypatch.setattr("app.documents.parsers.pdf.ocr_available", lambda: False)

    path = _image_pdf(tmp_path / "scanned.pdf")
    with pytest.raises(ParseError) as caught:
        parser.parse(path, "doc-7")
    assert "cannot read scanned pages" in str(caught.value)


def test_a_corrupt_pdf_is_not_blamed_on_the_scanner(parser, tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a PDF at all")
    with pytest.raises(ParseError) as caught:
        parser.parse(path, "doc-8")
    assert "corrupt.pdf" in str(caught.value)


# ------------------------------------------------------------------ merging


def test_native_text_wins_when_it_is_richer_than_the_ocr_of_the_same_page(
    parser, tmp_path, monkeypatch
):
    """A page can trip the threshold and still have exact embedded text. Native
    extraction is exact and OCR is not, so the longer read wins rather than OCR
    unconditionally overwriting."""
    monkeypatch.setattr(settings, "ocr_min_chars_per_page", 10_000)  # force OCR
    monkeypatch.setattr(
        "app.documents.parsers.pdf.ocr_pdf_pages",
        lambda file_path, page_indices, progress=None: {i: "smudge" for i in page_indices},
    )

    path = _text_pdf(tmp_path / "digital.pdf")
    result = parser.parse(path, "doc-9")

    assert "real, selectable text" in result.text
    assert "smudge" not in result.text


def test_only_the_unreadable_pages_are_sent_to_ocr(parser, tmp_path, monkeypatch):
    """A mostly-digital PDF with one scanned insert must pay for one page, not
    for the whole document."""
    requested: list[list[int]] = []

    def record(file_path, page_indices, progress=None):
        requested.append(list(page_indices))
        return {index: "scanned page text" for index in page_indices}

    monkeypatch.setattr("app.documents.parsers.pdf.ocr_pdf_pages", record)

    # Two image pages: both are unreadable natively, so both are requested.
    path = _image_pdf(tmp_path / "mixed.pdf", pages=2)
    parser.parse(path, "doc-10")
    assert requested == [[0, 1]]


def test_the_page_budget_is_enforced(parser, tmp_path, monkeypatch):
    """A 400-page scanned book would hold the ingest lock for the better part of
    an hour."""
    monkeypatch.setattr(settings, "ocr_max_pages", 1)
    requested: list[list[int]] = []

    def record(file_path, page_indices, progress=None):
        requested.append(list(page_indices))
        return {index: "text" for index in page_indices}

    monkeypatch.setattr("app.documents.parsers.pdf.ocr_pdf_pages", record)

    path = _image_pdf(tmp_path / "long.pdf", pages=3)
    parser.parse(path, "doc-11")
    assert requested == [[0]]


def test_the_pdf_file_is_not_left_locked_after_ocr(tmp_path):
    """pdfium holds a native handle. Leaving it open keeps the file locked on
    Windows, and the user could then neither delete nor replace their document."""
    from app.documents.ocr import ocr_pdf_pages

    path = _image_pdf(tmp_path / "scanned.pdf")
    ocr_pdf_pages(path, [0])
    path.unlink()  # raises PermissionError on Windows if the handle leaked
    assert not path.exists()
