import re
from abc import ABC, abstractmethod
from pathlib import Path

from app.documents.schemas import FileType, NormalizedDocument


class ParseError(Exception):
    """Raised when a document cannot be turned into usable text."""


class DocumentParser(ABC):
    @abstractmethod
    def supports(self, file_type: FileType) -> bool:
        ...

    @abstractmethod
    def parse(self, file_path: Path, document_id: str) -> NormalizedDocument:
        ...


def normalize_whitespace(text: str) -> str:
    """Collapse the layout noise every format produces, while preserving the
    blank lines that mark paragraph boundaries — the chunker splits on those."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_text(text: str, file_path: Path) -> str:
    if not text.strip():
        raise ParseError(
            f"No extractable text in '{file_path.name}'. "
            "Scanned or image-only documents are not supported (no OCR)."
        )
    return text


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers

    def for_type(self, file_type: FileType) -> DocumentParser:
        for parser in self._parsers:
            if parser.supports(file_type):
                return parser
        raise ParseError(f"No parser registered for {file_type.value}")

    def parse(self, file_path: Path, file_type: FileType, document_id: str) -> NormalizedDocument:
        return self.for_type(file_type).parse(file_path, document_id)
