from pathlib import Path

from app.documents.parsers.base import (
    DocumentParser,
    ParseError,
    ensure_text,
    normalize_whitespace,
)
from app.documents.schemas import FileType, NormalizedDocument


class PdfParser(DocumentParser):
    def supports(self, file_type: FileType) -> bool:
        return file_type == FileType.PDF

    def parse(self, file_path: Path, document_id: str) -> NormalizedDocument:
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise ParseError(f"Could not read PDF '{file_path.name}': {exc}") from exc

        text = normalize_whitespace("\n\n".join(pages))
        # A scanned PDF parses fine and yields nothing. Fail loudly here rather
        # than silently ingesting an empty document.
        ensure_text(text, file_path)

        title = ""
        try:
            if reader.metadata and reader.metadata.title:
                title = str(reader.metadata.title).strip()
        except Exception:
            pass

        return NormalizedDocument(
            document_id=document_id,
            title=title or file_path.stem,
            text=text,
            metadata={"parser": "pdf", "page_count": len(pages)},
        )
