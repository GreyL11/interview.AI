from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.deps import get_document_service
from app.documents.schemas import (
    DeleteResponse,
    Document,
    DocumentStatus,
    DocumentUploadResponse,
    IngestResponse,
    KnowledgeType,
)
from app.documents.service import DocumentError, DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post("", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    request: Request,
    filename: str,
    knowledge_type: KnowledgeType,
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Upload a document as a raw request body.

    Deliberately not multipart/form-data: the only client is our own desktop
    frontend, which can POST a file blob directly. That drops the
    python-multipart dependency and makes the client simpler too.
    """
    content = await request.body()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
        )

    document = await service.upload(filename, content, knowledge_type)
    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status,
    )


@router.post("/{document_id}/ingest", response_model=IngestResponse)
async def ingest_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> IngestResponse:
    return await service.ingest(document_id)


@router.get("", response_model=list[Document])
async def list_documents(
    knowledge_type: KnowledgeType | None = None,
    status: DocumentStatus | None = None,
    service: DocumentService = Depends(get_document_service),
) -> list[Document]:
    return service.list(knowledge_type=knowledge_type, status=status)


@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> Document:
    return service.get(document_id)


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DeleteResponse:
    return await service.delete(document_id)
