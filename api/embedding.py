from pathlib import Path
import logging

from fastapi import APIRouter

from api.exceptions import BadRequestError, NotFoundError
from api.rag.answer import ask_question
from api.rag.ingest import ingest_pdf
from api.schemas.query import AskRequest, AskResponse, SourceChunk

router = APIRouter(prefix="/embedding", tags=["embedding"])
logger = logging.getLogger(__name__)


@router.post("/ingest")
def ingest_resume_pdf():
    """Read Harish PDF → chunk → embed → store in Chroma Cloud."""
    try:
        result = ingest_pdf()
    except FileNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    return result


@router.post("/ingest/{file_name}")
def ingest_named_pdf(file_name: str):
    from api.settings import PROJECT_ROOT

    path = PROJECT_ROOT / file_name
    try:
        result = ingest_pdf(path)
    except FileNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    return result


@router.post("/ask", response_model=AskResponse)
def ask_from_resume(body: AskRequest):
    """Retrieve similar chunks from Chroma and answer the question (RAG)."""
    logger.info("/embedding/ask received: question_length=%d top_k=%d", len(body.question), body.top_k)
    try:
        result = ask_question(body.question, body.top_k)
    except ValueError as exc:
        logger.exception("/embedding/ask rejected with ValueError")
        raise BadRequestError(str(exc)) from exc
    except Exception:
        logger.exception("/embedding/ask failed unexpectedly")
        raise

    logger.info("/embedding/ask completed: sources=%d notice=%s", len(result["sources"]), result.get("notice"))

    return AskResponse(
        question=str(result["question"]),
        answer=str(result["answer"]),
        sources=[SourceChunk(**item) for item in result["sources"]],
        notice=result.get("notice"),
    )
