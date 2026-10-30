"""The server's one exception policy: classify, scrub, and render per protocol.

Three envelopes share one port, discriminated by path prefix: OpenAI
``{"error": {message, type, param, code}}`` under ``/v1/``, Ollama
``{"error": "<msg>"}`` under ``/api/``, and ``{"detail": "<msg>"}`` everywhere
else (the native ``/drowse/v1/`` tree and the dashboard shell).

:func:`describe_error` is the single classifier every surface uses — the
FastAPI exception handlers, the SSE/JSON operation runner, the polled
preparation status, and the WebSocket error frames — so one failure reads the
same everywhere and no surface leaks a raw exception string:

- a :class:`DrowseError` speaks through ``user_message()`` (its text is the
  user-facing contract);
- ``HTTPException`` keeps its status and flattened detail — or, when it wraps
  a typed or filesystem failure, that failure's safe text;
- a plain ``ValueError`` is a rejected request (400, verbatim; its trace is
  logged at debug);
- filesystem errors keep a status but get a fixed message (their text is a
  path);
- a missing third-party module (an optional extra) is a logged 503;
- anything else — including a failed import of drowse's own modules — is
  logged with its traceback and scrubbed to a generic 500.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from drowse.errors import DrowseError

log = logging.getLogger("drowse.api")

NATIVE_PREFIX = "/drowse/v1/"
OLLAMA_PREFIX = "/api/"
OPENAI_PREFIX = "/v1/"
#: Every path prefix whose requests are authenticated API calls.
PRIVATE_PREFIXES = (NATIVE_PREFIX, OLLAMA_PREFIX, OPENAI_PREFIX)

_FILESYSTEM_MESSAGES = {
    404: "Requested artifact not found",
    409: "An artifact already exists at the destination",
}


@dataclass(frozen=True)
class ErrorInfo:
    """A classified failure: HTTP status, user-safe message, machine code."""

    status: int
    message: str
    code: str

    def frame(self) -> dict[str, Any]:
        """The ``{message, code, status}`` body of an SSE/polled error."""
        return {"message": self.message, "code": self.code, "status": self.status}


def openai_error_type(status: int) -> str:
    """Map an HTTP status onto OpenAI's ``error.type`` vocabulary."""
    if status == 409:
        return "conflict"
    if 400 <= status < 500:
        return "invalid_request_error"
    return "server_error"


def detail_text(detail: Any) -> str:
    """Flatten an ``HTTPException.detail`` (str, dict, or list) to one string."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        for key in ("message", "msg", "detail"):
            value = detail.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(detail, default=str)
    if isinstance(detail, (list, tuple)):
        return "; ".join(detail_text(item) for item in detail)
    return str(detail)


def validation_message(errors: list[Any] | ValidationError | RequestValidationError) -> str:
    """Render every pydantic error as ``"<field path>: <message>"`` joined by ``"; "``.

    The one rendering for REST bodies, preparation fields, and WebSocket
    frames.  Every error is reported: a union field's real failure is often
    the second branch error.  The ``body`` prefix FastAPI adds is dropped.
    """
    if isinstance(errors, (ValidationError, RequestValidationError)):
        errors = list(errors.errors())
    parts: list[str] = []
    for error in errors:
        loc = [str(part) for part in error.get("loc", ())]
        if loc and loc[0] in {"body", "query", "path"}:
            loc = loc[1:]
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{'.'.join(loc)}: {message}" if loc else message)
    return "; ".join(parts) or "invalid request"


def _drowse_status(exc: DrowseError) -> tuple[int, str]:
    status, message = exc.user_message()
    if status == 500:
        # A class that declares no status reports the base default; its stdlib
        # parent still says what kind of failure it is.
        if isinstance(exc, KeyError):
            status = 404
            message = str(exc.args[0]) if exc.args else message
        elif isinstance(exc, ValueError):
            status = 400
    return status, message


def _missing_extra(exc: ImportError) -> str | None:
    """The third-party module a failed import names, else ``None``.

    A re-raised install hint (``raise ImportError(...) from e``) carries no
    ``name`` of its own, so the chained import failure is consulted too.  A
    failed import of drowse's own modules is never a missing extra.
    """
    cause = exc.__cause__
    name = exc.name or (cause.name if isinstance(cause, ImportError) else None)
    if not name or name.partition(".")[0] == "drowse":
        return None
    return name


def _filesystem_message(status: int) -> str:
    return _FILESYSTEM_MESSAGES.get(status, "Filesystem operation failed. Check the server log for details.")


def describe_error(exc: BaseException, *, failure: str = "Request failed") -> ErrorInfo:
    """Classify ``exc`` into a user-safe :class:`ErrorInfo` (see module doc).

    ``failure`` labels the generic message an unexpected exception is
    scrubbed to.  Unexpected and wrapped failures are logged here, so callers
    never log the same exception twice.
    """
    code = type(exc).__name__
    if isinstance(exc, DrowseError):
        if exc.__cause__ is not None:
            log.error("%s", failure, exc_info=exc)
        status, message = _drowse_status(exc)
        return ErrorInfo(status, message, code)
    if isinstance(exc, StarletteHTTPException):
        # A route that wraps a typed or filesystem failure keeps its status,
        # but the text still comes from the safe source, never the raw detail.
        cause = exc.__cause__
        if isinstance(cause, DrowseError):
            if cause.__cause__ is not None:
                log.error("%s", failure, exc_info=cause)
            return ErrorInfo(exc.status_code, _drowse_status(cause)[1], code)
        if isinstance(cause, OSError):
            log.error("%s", failure, exc_info=cause)
            return ErrorInfo(exc.status_code, _filesystem_message(exc.status_code), code)
        return ErrorInfo(exc.status_code, detail_text(exc.detail), code)
    if isinstance(exc, ValidationError):
        return ErrorInfo(400, validation_message(exc), code)
    if isinstance(exc, OSError):
        log.error("%s", failure, exc_info=exc)
        status = 404 if isinstance(exc, FileNotFoundError) else 409 if isinstance(exc, FileExistsError) else 500
        return ErrorInfo(status, _filesystem_message(status), code)
    if isinstance(exc, ValueError):
        # Usually a rejected argument, occasionally an engine bug: keep the
        # trace reachable without blaming the client in the error log.
        log.debug("%s: rejected as a bad request", failure, exc_info=exc)
        return ErrorInfo(400, str(exc) or code, code)
    if isinstance(exc, ImportError) and (missing := _missing_extra(exc)) is not None:
        log.error("%s", failure, exc_info=exc)
        return ErrorInfo(503, f"optional dependency not installed: {missing}", code)
    # Includes an ImportError of drowse's own modules: a packaging bug, not
    # a missing extra.
    log.error("%s", failure, exc_info=exc)
    return ErrorInfo(500, f"{failure}. Check the server log for details.", code)


def classify_extraction_error(exc: Exception) -> ErrorInfo | None:
    """Tag the RBF poisedness failure so the client can offer its layout hint.

    The two routes that drive the extraction pipeline (``POST /extract`` and
    manifold ``fit``) pass this to the operation runner; every other failure
    falls through to :func:`describe_error`.
    """
    if isinstance(exc, ValueError) and not isinstance(exc, DrowseError) and "poisedness" in str(exc).lower():
        return ErrorInfo(400, str(exc), "PoisednessError")
    return None


def protocol_error(
    path: str, status: int, message: str, headers: dict[str, str] | None = None,
) -> Response:
    """Render one error message in the envelope the request's protocol owns."""
    if status in {204, 304}:
        return Response(status_code=status, headers=headers)
    if path.startswith(OLLAMA_PREFIX):
        content: dict[str, Any] = {"error": message}
    elif path.startswith(OPENAI_PREFIX):
        content = {"error": {
            "message": message, "type": openai_error_type(status),
            "param": None, "code": status,
        }}
    else:
        content = {"detail": message}
    return JSONResponse(status_code=status, content=content, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    """Route every failure a handler can raise through :func:`describe_error`."""

    async def _render(request: Request, exc: Exception) -> Response:
        if isinstance(exc, RequestValidationError):
            info = ErrorInfo(400, validation_message(exc), "RequestValidationError")
        else:
            info = describe_error(exc)
        headers = getattr(exc, "headers", None) if isinstance(exc, StarletteHTTPException) else None
        return protocol_error(request.url.path, info.status, info.message, headers)

    async def _render_unexpected(request: Request, exc: Exception) -> Response:
        # Starlette re-raises after this response is sent, so the server logs
        # the traceback once; render the scrubbed envelope only.
        return protocol_error(
            request.url.path, 500, "Request failed. Check the server log for details.",
        )

    for exc_type in (
        StarletteHTTPException, RequestValidationError, DrowseError,
        ValueError, OSError, ImportError,
    ):
        app.add_exception_handler(exc_type, _render)
    app.add_exception_handler(Exception, _render_unexpected)
