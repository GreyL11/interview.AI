import pytest

from app.chunking.semantic_chunker import SemanticChunker
from app.documents.schemas import NormalizedDocument


def make_doc(text: str) -> NormalizedDocument:
    return NormalizedDocument(document_id="doc-1", title="Test Doc", text=text)


def test_short_document_is_one_chunk():
    chunks = SemanticChunker(chunk_size=400, chunk_overlap=50).chunk(
        make_doc("A single short paragraph about Kafka.")
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_chunks_are_ordered_and_indexed():
    text = "\n\n".join(f"Paragraph number {i} with some filler text." * 4 for i in range(12))
    chunks = SemanticChunker(chunk_size=300, chunk_overlap=50).chunk(make_doc(text))
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_headings_start_new_chunks():
    text = "Experience\n\nI built a pipeline.\n\nEducation\n\nI studied physics."
    chunks = SemanticChunker(chunk_size=4000, chunk_overlap=0).chunk(make_doc(text))
    # Would fit in one chunk on size alone; the headings force the split.
    assert len(chunks) == 2
    assert chunks[0].text.startswith("Experience")
    assert chunks[1].text.startswith("Education")


def test_oversized_paragraph_splits_on_sentences():
    sentence = "This is a sentence about distributed systems. "
    chunks = SemanticChunker(chunk_size=200, chunk_overlap=0).chunk(
        make_doc(sentence * 20)
    )
    assert len(chunks) > 1
    for chunk in chunks:
        # Nothing should be chopped mid-word.
        assert not chunk.text.startswith("entence")


def test_overlap_carries_context_forward():
    paragraphs = [f"Paragraph {i}. It discusses topic number {i} in detail." for i in range(10)]
    chunks = SemanticChunker(chunk_size=200, chunk_overlap=80).chunk(
        make_doc("\n\n".join(paragraphs))
    )
    assert len(chunks) > 1
    overlapping = sum(
        1 for a, b in zip(chunks, chunks[1:]) if any(w in b.text for w in a.text.split()[-4:])
    )
    assert overlapping > 0


def test_no_overlap_when_configured_zero():
    text = "\n\n".join(f"Paragraph {i} text here." for i in range(10))
    chunks = SemanticChunker(chunk_size=120, chunk_overlap=0).chunk(make_doc(text))
    assert len(chunks) > 1


def test_metadata_is_attached():
    chunks = SemanticChunker(chunk_size=400, chunk_overlap=50).chunk(
        make_doc("Some text."), knowledge_type="RESUME", source="cv.pdf"
    )
    meta = chunks[0].metadata
    assert meta["knowledge_type"] == "RESUME"
    assert meta["source"] == "cv.pdf"
    assert meta["document_id"] == "doc-1"
    assert meta["title"] == "Test Doc"
    assert meta["chunk_index"] == 0


def test_chunking_is_deterministic():
    text = "\n\n".join(f"Paragraph {i} with content." for i in range(20))
    a = SemanticChunker(chunk_size=250, chunk_overlap=60).chunk(make_doc(text))
    b = SemanticChunker(chunk_size=250, chunk_overlap=60).chunk(make_doc(text))
    assert [c.text for c in a] == [c.text for c in b]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=100, chunk_overlap=100)


def test_token_count_is_populated():
    chunks = SemanticChunker(chunk_size=400, chunk_overlap=50).chunk(
        make_doc("Some reasonably long text about data engineering practices.")
    )
    assert chunks[0].token_count > 0
