from pathlib import Path

from app.documents.parsers.base import (
    DocumentParser,
    ParseError,
    ensure_text,
    normalize_whitespace,
)
from app.documents.schemas import FileType, NormalizedDocument


def read_text_file(file_path: Path) -> str:
    # utf-8 first, then cp1252 — the two things a Windows user actually produces.
    for encoding in ("utf-8", "cp1252"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError(f"Could not decode '{file_path.name}' as UTF-8 or CP1252")


class TextParser(DocumentParser):
    def supports(self, file_type: FileType) -> bool:
        return file_type == FileType.TXT

    def parse(self, file_path: Path, document_id: str) -> NormalizedDocument:
        text = normalize_whitespace(read_text_file(file_path))
        ensure_text(text, file_path)
        return NormalizedDocument(
            document_id=document_id,
            title=file_path.stem,
            text=text,
            metadata={"parser": "text"},
        )
