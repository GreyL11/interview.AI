from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class KnowledgeType(StrEnum):
    RESUME = "RESUME"
    PERSONAL = "PERSONAL"
    EXPERIENCE = "EXPERIENCE"
    PROJECT = "PROJECT"
    BEHAVIORAL_STORY = "BEHAVIORAL_STORY"
    TECHNICAL = "TECHNICAL"
    REFERENCE = "REFERENCE"


class FileType(StrEnum):
    PDF = "PDF"
    DOCX = "DOCX"
    MARKDOWN = "MARKDOWN"
    TXT = "TXT"


class DocumentStatus(StrEnum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


#: Knowledge types describing things the user actually did, as opposed to things
#: they merely know. Retrieval for personal-experience questions is restricted to
#: these so technical reference material can never be passed off as lived experience.
PERSONAL_KNOWLEDGE_TYPES = (
    KnowledgeType.RESUME,
    KnowledgeType.PERSONAL,
    KnowledgeType.EXPERIENCE,
    KnowledgeType.PROJECT,
    KnowledgeType.BEHAVIORAL_STORY,
)

EXTENSION_TO_FILE_TYPE = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".txt": FileType.TXT,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    knowledge_type: KnowledgeType
    title: str = ""
    source: str = ""
    created_at: datetime
    ingested_at: datetime | None = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    error: str | None = None
    chunk_count: int = 0
    #: What a slow ingest is currently doing, as a sentence for the user
    #: ("Reading scanned page 3 of 12..."). Only set while PROCESSING; cleared
    #: on the way to READY or FAILED, where `error` carries the outcome.
    progress: str | None = None


class NormalizedDocument(BaseModel):
    """Parser output. Deliberately knows nothing about chunking or embedding."""

    document_id: str
    title: str
    text: str
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    vector_id: int | None = None
    token_count: int = 0
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk plus why it was retrieved. Carries provenance so the orchestrator
    can record retrieval_hits and decide context_found."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    knowledge_type: KnowledgeType
    title: str = ""

    def as_context(self) -> str:
        label = self.title or self.knowledge_type.value
        return f"[{self.knowledge_type.value}] {label}: {self.text}"


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus


class IngestResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    chunk_count: int
    error: str | None = None


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    chunks_removed: int
    vectors_removed: int
