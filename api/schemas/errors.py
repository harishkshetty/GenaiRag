from pydantic import BaseModel, Field


class ErrorItem(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    errors: list[ErrorItem] = Field(min_length=1)
