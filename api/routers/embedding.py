import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.exceptions import BadRequestError, NotFoundError
from api.schemas.query import AskRequest, AskResponse, SourceChunk
from api.services.rag_service import ask_question, ingest_pdf, stream_answer

router = APIRouter(prefix="/embedding", tags=["embedding"])
logger = logging.getLogger(__name__)


@router.post("/ingest")
def ingest_resume_pdf():
    try:
        return ingest_pdf()
    except FileNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.post("/ingest/{file_name}")
def ingest_named_pdf(file_name: str):
    from api.settings import PROJECT_ROOT

    try:
        return ingest_pdf(PROJECT_ROOT / file_name)
    except FileNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask_from_resume(body: AskRequest):
    logger.info("/embedding/ask received: question_length=%d top_k=%d", len(body.question), body.top_k)
    try:
        result = ask_question(body.question, body.top_k)
    except ValueError as exc:
        logger.exception("/embedding/ask rejected with ValueError")
        raise BadRequestError(str(exc)) from exc

    return AskResponse(
        question=str(result["question"]),
        answer=str(result["answer"]),
        sources=[SourceChunk(**item) for item in result["sources"]],
        notice=result.get("notice"),
    )


@router.post("/ask/stream")
def stream_from_resume(body: AskRequest):
    return StreamingResponse(
        stream_answer(body.question, body.top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
