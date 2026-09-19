import chromadb
from chromadb.api import ClientAPI

from api.settings import settings


def get_chroma_client() -> ClientAPI:
    if not settings.chroma_api_key:
        raise ValueError("CHROMA_API_KEY is not set")
    if not settings.chroma_tenant or not settings.chroma_database:
        raise ValueError("CHROMA_TENANT and CHROMA_DATABASE must be set")

    return chromadb.CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )


def upsert_chunks(
    client: ClientAPI,
    collection_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    source_file: str,
) -> int:
    collection = client.get_or_create_collection(name=collection_name)
    ids = [f"{source_file}:{index}" for index in range(len(chunks))]
    metadatas = [{"source": source_file, "chunk_index": index} for index in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def query_similar_chunks(
    client: ClientAPI,
    collection_name: str,
    query_embedding: list[float],
    top_k: int,
) -> dict[str, list]:
    collection = client.get_or_create_collection(name=collection_name)
    total = collection.count()
    if total == 0:
        return {
            "documents": [],
            "metadatas": [],
            "distances": [],
        }

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )
    return {
        "documents": result["documents"][0] if result["documents"] else [],
        "metadatas": result["metadatas"][0] if result["metadatas"] else [],
        "distances": result["distances"][0] if result["distances"] else [],
    }
