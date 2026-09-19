import httpx
import logging
import json
from collections.abc import Iterator

from api.settings import settings
from api.rag.chroma_store import get_chroma_client, query_similar_chunks
from api.rag.embedder import embed_texts

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You answer questions using only the context from a resume/CV. "
    "If the context does not contain the answer, say you do not know."
)


def _build_context(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(chunks)


def _chat_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]


def _retrieval_only_answer(question: str, chunks: list[str], prefix: str) -> str:
    context = _build_context(chunks)
    return f"{prefix}\n\nQuestion: {question}\n\n{context}"


def _generate_with_ollama(question: str, context: str) -> str:
    base = settings.ollama_base_url.rstrip("/")
    response = httpx.post(
        f"{base}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": _chat_messages(question, context),
            # Return one JSON document so httpx.Response.json() can parse it.
            "stream": False,
            "options": {"temperature": 0.2},

        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


def _generate_with_openai(question: str, context: str) -> str:
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.openai_model,
            "messages": _chat_messages(question, context),
            "temperature": 0.2,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _generate_without_llm(question: str, chunks: list[str]) -> str:
    return _retrieval_only_answer(
        question,
        chunks,
        "LLM disabled (LLM_PROVIDER=none) — returning retrieved context only.",
    )


def _fallback_after_llm_error(
    question: str,
    chunks: list[str],
    provider: str,
    detail: str,
) -> tuple[str, str]:
    notice = f"{provider} unavailable ({detail}). Showing retrieved context instead."
    return (_retrieval_only_answer(question, chunks, notice), "llm_unavailable")


def generate_answer(question: str, chunks: list[str]) -> tuple[str, str | None]:
    """Returns (answer, optional notice when LLM was skipped or failed)."""
    if not chunks:
        return ("No matching content found. Run POST /embedding/ingest first.", None)

    provider = settings.llm_provider
    if provider == "none":
        return (_generate_without_llm(question, chunks), "retrieval_only")

    context = _build_context(chunks)

    try:
        if provider == "ollama":
            return (_generate_with_ollama(question, context), None)
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            return (_generate_with_openai(question, context), None)
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if provider == "openai" and status == 401:
            raise ValueError("Invalid OPENAI_API_KEY") from exc
        if status in (429, 500, 502, 503):
            return _fallback_after_llm_error(
                question, chunks, provider, f"HTTP {status}"
            )
        raise ValueError(f"{provider} request failed with status {status}") from exc
    except httpx.RequestError as exc:
        hint = (
            "Is Ollama running? Try: ollama serve && ollama pull llama3.2"
            if provider == "ollama"
            else "Check network and API key"
        )
        return _fallback_after_llm_error(question, chunks, provider, hint)


def ask_question(question: str, top_k: int) -> dict[str, object]:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question cannot be empty")

    logger.info("Generating embedding: model=%s", settings.embedding_model)
    query_vector = embed_texts(settings.embedding_model, [cleaned])[0]
    logger.info("Embedding generated: dimensions=%d", len(query_vector))

    logger.info("Connecting to Chroma: collection=%s", settings.chroma_collection)
    client = get_chroma_client()
    logger.info("Querying Chroma: top_k=%d", top_k)
    hits = query_similar_chunks(
        client,
        settings.chroma_collection,
        query_vector,
        top_k,
    )

    documents: list[str] = hits["documents"]
    metadatas: list[dict] = hits["metadatas"]
    distances: list[float] = hits["distances"]
    logger.info("Chroma returned %d documents", len(documents))

    sources: list[dict[str, object]] = []
    for index, text in enumerate(documents):
        meta = metadatas[index] if index < len(metadatas) else {}
        sources.append(
            {
                "text": text,
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    logger.info("Generating answer: provider=%s", settings.llm_provider)
    answer, notice = generate_answer(cleaned, documents)
    logger.info("Answer generated: notice=%s", notice)
    return {
        "question": cleaned,
        "answer": answer,
        "sources": sources,
        "notice": notice,
    }


def stream_answer(question: str, top_k: int) -> Iterator[str]:
    """Yield an Ollama response as Server-Sent Events."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question cannot be empty")
    if settings.llm_provider != "ollama":
        raise ValueError("Streaming is currently supported only with LLM_PROVIDER=ollama")

    query_vector = embed_texts(settings.embedding_model, [cleaned])[0]
    client = get_chroma_client()
    hits = query_similar_chunks(client, settings.chroma_collection, query_vector, top_k)
    documents: list[str] = hits["documents"]
    context = _build_context(documents)

    base = settings.ollama_base_url.rstrip("/")
    with httpx.stream(
        "POST",
        f"{base}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": _chat_messages(cleaned, context),
            "stream": True,
            "options": {"temperature": 0.2},
        },
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                yield f"data: {json.dumps({'token': content})}\n\n"
            if data.get("done"):
                yield "data: [DONE]\n\n"
