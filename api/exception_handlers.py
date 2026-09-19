from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.exceptions import AppError
from api.schemas.errors import ErrorItem, ErrorResponse


def _json_errors(status_code: int, items: list[ErrorItem]) -> JSONResponse:
    body = ErrorResponse(errors=items).model_dump()
    return JSONResponse(status_code=status_code, content=body)


def _field_from_loc(loc: tuple[str | int, ...]) -> str | None:
    if not loc:
        return None
    return ".".join(str(part) for part in loc)


def _http_exception_items(detail: object) -> list[ErrorItem]:
    if isinstance(detail, str):
        return [ErrorItem(code="HTTP_ERROR", message=detail)]
    if isinstance(detail, list):
        items: list[ErrorItem] = []
        for entry in detail:
            if isinstance(entry, dict):
                items.append(
                    ErrorItem(
                        code=str(entry.get("code", "HTTP_ERROR")),
                        message=str(entry.get("msg", entry.get("message", "Error"))),
                        field=_field_from_loc(tuple(entry.get("loc", ())))
                        if entry.get("loc")
                        else entry.get("field"),
                    )
                )
            else:
                items.append(ErrorItem(code="HTTP_ERROR", message=str(entry)))
        return items or [ErrorItem(code="HTTP_ERROR", message="Error")]
    if isinstance(detail, dict):
        if "message" in detail:
            return [
                ErrorItem(
                    code=str(detail.get("code", "HTTP_ERROR")),
                    message=str(detail["message"]),
                    field=detail.get("field"),
                )
            ]
        return [ErrorItem(code="HTTP_ERROR", message=str(detail))]
    return [ErrorItem(code="HTTP_ERROR", message="Error")]


async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
    return _json_errors(
        exc.status_code,
        [
            ErrorItem(
                code=exc.code,
                message=exc.message,
                field=exc.field,
            )
        ],
    )


async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
    return _json_errors(exc.status_code, _http_exception_items(exc.detail))


async def handle_validation_error(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    items = [
        ErrorItem(
            code="VALIDATION_ERROR",
            message=err.get("msg", "Invalid value"),
            field=_field_from_loc(tuple(err.get("loc", ()))),
        )
        for err in exc.errors()
    ]
    return _json_errors(422, items)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
