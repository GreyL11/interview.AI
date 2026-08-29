import re
from pathlib import Path

from app.documents.parsers.base import (
    DocumentParser,
    ProgressCallback,
    ensure_text,
    normalize_whitespace,
)
from app.documents.parsers.text import read_text_file
from app.documents.schemas import FileType, NormalizedDocument

_CODE_FENCE = re.compile(r"^```", re.MULTILINE)
_ATX_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(.+?)\1", re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Strip syntax that would pollute embeddings, but keep headings as plain
    lines so the chunker can still split on them."""
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(-{3,}|\*{3,})\s*$", "", text, flags=re.MULTILINE)
    return text


class MarkdownParser(DocumentParser):
    def supports(self, file_type: FileType) -> bool:
        return file_type == FileType.MARKDOWN

    def parse(
        self,
        file_path: Path,
        document_id: str,
        progress: ProgressCallback | None = None,
    ) -> NormalizedDocument:
        # `progress` is unused: these formats parse in milliseconds, so
        # there is nothing worth narrating. It is accepted to keep one
        # parser signature across the registry.
        raw = read_text_file(file_path)

        # Code fences are content, not prose: leave anything inside them alone.
        segments = _CODE_FENCE.split(raw)
        cleaned = [
            _strip_markdown(seg) if i % 2 == 0 else seg for i, seg in enumerate(segments)
        ]
        text = normalize_whitespace("\n".join(cleaned))
        ensure_text(text, file_path)

        heading = _ATX_HEADING.search(raw)
        return NormalizedDocument(
            document_id=document_id,
            title=heading.group(1).strip() if heading else file_path.stem,
            text=text,
            metadata={"parser": "markdown"},
        )
