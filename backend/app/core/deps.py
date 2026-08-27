from functools import lru_cache

from app.chunking.semantic_chunker import SemanticChunker
from app.core.config import settings
from app.documents.parsers.base import ParserRegistry
from app.documents.parsers.docx import DocxParser
from app.documents.parsers.markdown import MarkdownParser
from app.documents.parsers.pdf import PdfParser
from app.documents.parsers.text import TextParser
from app.documents.service import DocumentService
from app.embeddings.base import EmbeddingProvider
from app.embeddings.onnx_embedder import OnnxEmbedder
from app.retrieval.base import Retriever
from app.retrieval.local_retriever import LocalRetriever
from app.storage.chunk_repository import ChunkRepository
from app.storage.database import Database
from app.storage.document_repository import DocumentRepository
from app.vector_store.base import VectorStore
from app.vector_store.faiss_store import FaissVectorStore

# Composition root. Everything is @lru_cache'd so the process shares one
# database, one FAISS index, and one loaded embedding model. Tests override the
# FastAPI dependencies rather than these.


@lru_cache
def get_database() -> Database:
    return Database(settings.db_path)


@lru_cache
def get_document_repository() -> DocumentRepository:
    return DocumentRepository(get_database())


@lru_cache
def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository(get_database())


@lru_cache
def get_embedder() -> EmbeddingProvider:
    return OnnxEmbedder()


@lru_cache
def get_vector_store() -> VectorStore:
    return FaissVectorStore(get_embedder().dimension, settings.faiss_path)


@lru_cache
def get_parsers() -> ParserRegistry:
    return ParserRegistry([TextParser(), MarkdownParser(), PdfParser(), DocxParser()])


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        db=get_database(),
        documents=get_document_repository(),
        chunks=get_chunk_repository(),
        parsers=get_parsers(),
        chunker=SemanticChunker(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_retriever() -> Retriever:
    return LocalRetriever(get_embedder(), get_vector_store(), get_chunk_repository())
