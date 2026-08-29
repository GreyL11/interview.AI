import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from app.documents.schemas import FileType, NormalizedDocument


class ParseError(Exception):
    """Raised when a document cannot be turned into usable text."""


#: Reports what a slow parse is currently doing, as a sentence for the user.
#:
#: Only OCR needs this -- every other parser finishes far too fast to be worth
#: narrating -- but it is on the base contract so a caller can pass it
#: unconditionally rather than special-casing PDFs.
ProgressCallback = Callable[[str], None]


class DocumentParser(ABC):
    @abstractmethod
    def supports(self, file_type: FileType) -> bool:
        ...

    @abstractmethod
    def parse(
        self,
        file_path: Path,
        document_id: str,
        progress: ProgressCallback | None = None,
    ) -> NormalizedDocument:
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
    """Guard for the formats that either have text or are broken.

    PDFs do not use this: a PDF with no extractable text is usually a scan,
    which is recoverable, so `PdfParser` runs OCR and raises its own message
    only if that also comes back empty.
    """
    if not text.strip():
        raise ParseError(
            f"No readable text was found in '{file_path.name}'. "
            "The file may be empty or corrupted."
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

    def parse(
        self,
        file_path: Path,
        file_type: FileType,
        document_id: str,
        progress: ProgressCallback | None = None,
    ) -> NormalizedDocument:
        return self.for_type(file_type).parse(file_path, document_id, progress)
