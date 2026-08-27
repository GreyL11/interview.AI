import pytest

from app.documents.parsers.base import ParseError
from app.documents.schemas import FileType
from tests.fixtures import RESUME_TEXT, write_docx, write_pdf, write_text


def test_text_parser(parsers, tmp_path):
    path = write_text(tmp_path / "resume.txt", RESUME_TEXT)
    doc = parsers.parse(path, FileType.TXT, "doc-1")
    assert "Kafka ingestion pipeline" in doc.text
    assert doc.title == "resume"
    assert doc.document_id == "doc-1"


def test_text_parser_reads_cp1252(parsers, tmp_path):
    path = write_text(tmp_path / "cv.txt", "Café résumé — naïve", encoding="cp1252")
    doc = parsers.parse(path, FileType.TXT, "doc-1")
    assert "Caf" in doc.text


def test_markdown_strips_syntax_but_keeps_headings(parsers, tmp_path):
    md = (
        "# My Projects\n\n"
        "I built a **streaming** pipeline with [Kafka](https://kafka.apache.org).\n\n"
        "![diagram](img.png)\n\n"
        "- First point\n- Second point\n"
    )
    path = write_text(tmp_path / "projects.md", md)
    doc = parsers.parse(path, FileType.MARKDOWN, "doc-1")

    assert doc.title == "My Projects"
    assert "**" not in doc.text
    assert "https://kafka.apache.org" not in doc.text
    assert "Kafka" in doc.text
    assert "img.png" not in doc.text
    assert "streaming" in doc.text


def test_markdown_preserves_code_fences(parsers, tmp_path):
    md = "# Notes\n\n```python\nx = a ** 2  # keep **\n```\n"
    path = write_text(tmp_path / "notes.md", md)
    doc = parsers.parse(path, FileType.MARKDOWN, "doc-1")
    assert "a ** 2" in doc.text


def test_pdf_parser(parsers, tmp_path):
    path = write_pdf(
        tmp_path / "resume.pdf",
        ["Jane Doe", "I built a Kafka ingestion pipeline.", "I led the migration."],
    )
    doc = parsers.parse(path, FileType.PDF, "doc-1")
    assert "Jane Doe" in doc.text
    assert "Kafka" in doc.text
    assert doc.metadata["page_count"] == 1


def test_pdf_without_text_layer_fails_clearly(parsers, tmp_path):
    path = write_pdf(tmp_path / "scanned.pdf", [])
    with pytest.raises(ParseError, match="No extractable text"):
        parsers.parse(path, FileType.PDF, "doc-1")


def test_docx_parser_includes_tables(parsers, tmp_path):
    path = write_docx(
        tmp_path / "resume.docx",
        ["Jane Doe", "I built a Kafka ingestion pipeline."],
        table=[["Company", "Role"], ["Acme", "Senior Data Engineer"]],
    )
    doc = parsers.parse(path, FileType.DOCX, "doc-1")
    assert "Kafka" in doc.text
    # Resumes keep half their content in tables; losing them loses job history.
    assert "Senior Data Engineer" in doc.text
    assert doc.metadata["table_count"] == 1


def test_empty_file_fails(parsers, tmp_path):
    path = write_text(tmp_path / "empty.txt", "   \n\n  ")
    with pytest.raises(ParseError):
        parsers.parse(path, FileType.TXT, "doc-1")


def test_detect_file_type_from_extension():
    from app.documents.service import detect_file_type

    assert detect_file_type("a.PDF") == FileType.PDF
    assert detect_file_type("a.docx") == FileType.DOCX
    assert detect_file_type("a.md") == FileType.MARKDOWN
    assert detect_file_type("a.markdown") == FileType.MARKDOWN
    assert detect_file_type("a.txt") == FileType.TXT


def test_unregistered_type_raises(parsers, tmp_path):
    from app.documents.parsers.base import ParserRegistry
    from app.documents.parsers.text import TextParser

    registry = ParserRegistry([TextParser()])
    with pytest.raises(ParseError, match="No parser registered"):
        registry.for_type(FileType.PDF)
