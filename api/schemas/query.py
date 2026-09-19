from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


class SourceChunk(BaseModel):
    text: str
    source: str | None = None
    chunk_index: int | None = None
    distance: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    notice: str | None = None
