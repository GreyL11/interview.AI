import re
import uuid

from app.chunking.base import Chunker
from app.core.config import settings
from app.documents.schemas import Chunk, NormalizedDocument

# Heading-ish lines: markdown ATX, or a short line with no terminal punctuation
# (how headings survive PDF/DOCX extraction).
_HEADING = re.compile(r"^(#{1,6}\s+.+|[A-Z][^.!?\n]{2,60})$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    # ponytail: ~4 chars/token approximation instead of loading a tokenizer.
    # Only used for budgeting, never for correctness. Swap in the real tokenizer
    # if chunk sizes start hitting the model's limit in practice.
    return max(1, len(text) // 4)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries; fall back to a hard
    character split only for text with no sentence structure at all."""
    sentences = _SENTENCE_END.split(paragraph)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(
                sentence[i : i + limit].strip() for i in range(0, len(sentence), limit)
            )
            continue
        if len(current) + len(sentence) + 1 > limit and current:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current.strip())
    return [p for p in parts if p]


def _is_heading(paragraph: str) -> bool:
    return "\n" not in paragraph and bool(_HEADING.match(paragraph.strip()))


class SemanticChunker(Chunker):
    """Paragraph-first chunker.

    Packs whole paragraphs up to chunk_size, starts a new chunk at headings, and
    only splits mid-paragraph when a single paragraph exceeds the limit — then on
    sentence boundaries. Overlap is applied in whole sentences so chunks never
    begin mid-sentence.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk(self, document: NormalizedDocument, **metadata) -> list[Chunk]:
        units: list[str] = []
        for paragraph in _split_paragraphs(document.text):
            if len(paragraph) > self.chunk_size:
                units.extend(_split_long_paragraph(paragraph, self.chunk_size))
            else:
                units.append(paragraph)

        texts = self._pack(units)
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                chunk_index=index,
                text=text,
                token_count=estimate_tokens(text),
                metadata={
                    "document_id": document.document_id,
                    "chunk_index": index,
                    "title": document.title,
                    **metadata,
                },
            )
            for index, text in enumerate(texts)
        ]

    def _pack(self, units: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        length = 0

        for unit in units:
            starts_section = _is_heading(unit) and current
            too_long = length + len(unit) + 2 > self.chunk_size and current
            if starts_section or too_long:
                chunks.append("\n\n".join(current))
                carry = self._overlap_from("\n\n".join(current))
                current = [carry] if carry else []
                length = len(carry)
            current.append(unit)
            length += len(unit) + 2

        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _overlap_from(self, text: str) -> str:
        """Take up to chunk_overlap characters off the end, on a sentence
        boundary so the next chunk doesn't start mid-thought."""
        if self.chunk_overlap <= 0:
            return ""
        tail = text[-self.chunk_overlap :]
        sentences = _SENTENCE_END.split(tail)
        if len(sentences) > 1:
            return " ".join(sentences[1:]).strip()
        return tail.strip()
