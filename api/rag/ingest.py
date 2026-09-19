from pathlib import Path

from api.rag.chunking import chunk_text
from api.rag.chroma_store import get_chroma_client, upsert_chunks
from api.rag.embedder import embed_texts
from api.rag.pdf_loader import extract_text_from_pdf
from api.settings import settings


def ingest_pdf(pdf_path: Path | None = None) -> dict[str, object]:
    path = pdf_path or settings.pdf_path
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    text = extract_text_from_pdf(path)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No text extracted from PDF (empty or scanned image PDF?)")

    embeddings = embed_texts(settings.embedding_model, chunks)
    client = get_chroma_client()
    stored = upsert_chunks(
        client,
        settings.chroma_collection,
        chunks,
        embeddings,
        source_file=path.name,
    )

    return {
        "pdf": path.name,
        "collection": settings.chroma_collection,
        "chunks_stored": stored,
        "embedding_model": settings.embedding_model,
    }
