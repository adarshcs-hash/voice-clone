"""Request middleware: correlation ids, access logging, error mapping.

Three things every request gets:

*   **A request id**, taken from ``X-Request-Id`` if the caller supplied one and
    generated otherwise, bound into the structlog context so that *every* log
    line emitted while handling the request carries it, and echoed in the
    response header. This is what makes a user's bug report ("the voice said
    this wrong at 14:32") traceable.
*   **A timed access log line**, emitted once per request with method, path,
    status, duration and the request id.
*   **Uniform error bodies.** :class:`~mlvoice.errors.MlvoiceError` carries its
    own HTTP status and code, so the mapping lives in one handler instead of
    being repeated in every route.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from mlvoice.errors import MlvoiceError
from mlvoice.logging import get_logger

__all__ = ["RequestContextMiddleware", "install_exception_handlers"]

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-Id"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a request id and emits one access log line per request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Bind context, time the request, log the outcome."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers below build the body; this only records
            # that the request died so the access log has no silent gaps.
            log.exception(
                "request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response


def _body(error: MlvoiceError, request_id: str | None) -> dict[str, object]:
    payload = error.to_dict()
    payload["request_id"] = request_id
    return payload


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers that give every failure the same body shape."""

    @app.exception_handler(MlvoiceError)
    async def _handle_known(request: Request, exc: MlvoiceError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        # Client mistakes are not incidents; server-side failures are.
        if exc.http_status >= 500:
            log.error("request error", code=exc.code, message=exc.message, **exc.context)
        else:
            log.info("request rejected", code=exc.code, message=exc.message, **exc.context)
        return JSONResponse(status_code=exc.http_status, content=_body(exc, request_id))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "request body failed validation",
                "context": {"errors": exc.errors()},
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Deliberately does not echo the exception text: an unexpected error
        # can carry file paths, prompts or reference-audio details.
        log.exception("unhandled error", error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "an unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )
