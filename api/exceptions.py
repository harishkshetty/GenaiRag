class AppError(Exception):
    """Raise in routes/services; mapped to HTTP by register_exception_handlers()."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "APP_ERROR",
        field: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        self.field = field
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(
        self,
        message: str = "Not found",
        field: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=404,
            code="NOT_FOUND",
            field=field,
        )


class BadRequestError(AppError):
    def __init__(
        self,
        message: str = "Bad request",
        field: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=400,
            code="BAD_REQUEST",
            field=field,
        )
