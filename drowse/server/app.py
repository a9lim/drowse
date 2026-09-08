"""OpenAI-compatible API server backed by DrowseSession."""

from __future__ import annotations

import asyncio
import base64
import hmac
import ipaddress
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, cast
from urllib.parse import parse_qsl, urlencode, urlsplit

if TYPE_CHECKING:
    from drowse.core.results import GenerationResult

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, model_validator
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from drowse.core.errors import DrowseError
from drowse.core.session import ConcurrentGenerationError, GenerationStream, DrowseSession
from drowse.core.steering import Steering
from drowse.server.request_helpers import (
    UnsupportedContentError,
    build_sampling_config,
    flatten_content,
    merge_steering,
    parse_request_steering,
    probe_token_readings,
    strict_model_enabled,
)
from drowse.server.streaming import (
    ClosingStreamingResponse,
    run_in_thread,
    stream_events,
    probe_reading_aggregate,
    stream_finalizer,
    usage_dict,
)


SESSION_LOCK_TIMEOUT_SECONDS = 300
DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024

#: Route-prefix discriminators for the three protocols served on one port.
#: Each owns an error envelope: OpenAI ``{"error": {message, type, param,
#: code}}``, Ollama ``{"error": "<msg>"}``, native ``{"detail": "<msg>"}``.
NATIVE_PREFIX = "/drowse/v1/"
OLLAMA_PREFIX = "/api/"


@asynccontextmanager
async def acquire_session_lock(session: DrowseSession) -> AsyncIterator[bool]:
    """Acquire ``session.lock`` with a 5-minute bound.

    Yields ``True`` if the lock was obtained (released on exit) and
    ``False`` on timeout.  Callers branch on the result to emit their
    protocol-specific 503.  Serializes all generation routes across both
    the OpenAI and Ollama protocols on the same session.
    """
    try:
        async with asyncio.timeout(SESSION_LOCK_TIMEOUT_SECONDS):
            await session.lock.acquire()
    except (TimeoutError, asyncio.TimeoutError):
        yield False
        return
    try:
        yield True
    finally:
        session.lock.release()


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content parts
    name: str | None = None

    @model_validator(mode="after")
    def _normalize_content(self):
        # Accept OpenAI multimodal content-part arrays for text-only use:
        # concatenate text parts, reject anything else with a clear error.
        self.content = flatten_content(self.content)
        return self


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingBase(BaseModel):
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    # Canonical native steering field — a steering expression string
    # parsed through the shared grammar in
    # :mod:`drowse.core.steering_expr`.  Merged over the server's default
    # :class:`Steering` and resolved through ``session.steering()`` so pole
    # aliases and events fire via the single canonical resolver site.
    steering: str | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: bool | int | None = None  # chat: bool; completions: int
    top_logprobs: int | None = None
    user: str | None = None
    # Native thinking override.  None = auto (honours supports_thinking).
    thinking: bool | None = None
    # LangChain compat: accept no-op shapes, reject anything real.
    tools: list[Any] | None = None
    tool_choice: Any = None
    # Fields accepted and ignored:
    n: int | None = None
    response_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _unify_max_tokens(self):
        if self.max_completion_tokens is not None and self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        return self

    @model_validator(mode="after")
    def _check_langchain_compat(self):
        # Accept `tools: []` / None silently; reject non-empty.
        if self.tools:
            raise UnsupportedContentError(
                "tool calling is not supported by drowse"
            )
        # tool_choice: accept None, "none", "auto"; reject "required" and dicts.
        tc = self.tool_choice
        if tc is not None and tc not in ("none", "auto"):
            raise UnsupportedContentError(
                "tool_choice values other than 'none'/'auto' are not supported"
            )
        # response_format: accept None or {"type": "text"}; reject json modes.
        rf = self.response_format
        if rf is not None:
            rf_type = rf.get("type")
            if rf_type not in (None, "text"):
                raise UnsupportedContentError(
                    "response_format types other than 'text' are not supported"
                )
        return self

    def to_steering(
        self, default_steering: "Steering | None",
    ) -> "Steering | None":
        """Compose ``self.steering`` (expression string) over the server default.

        ``None`` inherits the server default; an explicit empty string
        clears it.  Non-empty per-request expressions override the default
        at the key level: alphas for concepts named in both the default
        and the request come from the request; alphas only in the default
        pass through. Returns ``None`` when the composed result is empty
        and no ``thinking`` override was requested. Pole aliasing happens
        inside ``session.steering()`` — the server does not resolve poles
        here.
        """
        req_steering, explicit_clear = parse_request_steering(self.steering)
        thinking: bool | None = self.thinking
        if req_steering is not None and req_steering.thinking is not None:
            thinking = req_steering.thinking
        return merge_steering(
            req_steering, default_steering, explicit_clear, thinking,
        )


class ChatCompletionRequest(_SamplingBase):
    messages: list[ChatMessage]


class CompletionRequest(_SamplingBase):
    prompt: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return f"drowse-{uuid.uuid4().hex[:12]}"


def openai_error_type(status: int) -> str:
    """Map an HTTP status onto OpenAI's ``error.type`` vocabulary.

    One table for both OpenAI surfaces — the non-streaming exception handler
    and the in-band SSE error frame — so a 404 / 409 / 422 reads the same on
    either.
    """
    if status == 409:
        return "conflict"
    if 400 <= status < 500:
        return "invalid_request_error"
    return "server_error"


def _error(status: int, message: str, error_type: str = "error",
           param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type,
                           "param": param, "code": status}},
    )


def _detail_text(detail: Any) -> str:
    """Flatten an ``HTTPException.detail`` to the native envelope's string."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        for key in ("message", "msg", "detail"):
            value = detail.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(detail, default=str)
    if isinstance(detail, (list, tuple)):
        return "; ".join(_detail_text(item) for item in detail)
    return str(detail)


def _protocol_error(path: str, status: int, message: str) -> JSONResponse:
    """Render one error message in the envelope the request's protocol owns."""
    if path.startswith(OLLAMA_PREFIX):
        return JSONResponse(status_code=status, content={"error": message})
    if path.startswith(NATIVE_PREFIX):
        return JSONResponse(status_code=status, content={"detail": message})
    return _error(status, message, openai_error_type(status))


_bearer = HTTPBearer(auto_error=False)


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return address.is_loopback
    except ValueError:
        return False


def _origin(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment
            or any(char.isspace() for char in value) or "\\" in value
        ):
            return None
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        return parsed.scheme, parsed.hostname.lower(), port
    except ValueError:
        return None


def _local_host_ok(conn: Request | WebSocket) -> bool:
    client = conn.scope.get("client")
    if client:
        try:
            ipaddress.ip_address(client[0])
        except ValueError:
            pass  # In-process ASGI transports can use non-IP client labels.
        else:
            if not is_loopback_host(client[0]):
                return False
    hosts = conn.headers.getlist("host")
    target = _origin("http://" + hosts[0]) if len(hosts) == 1 else None
    if target is None:
        return False
    server = conn.scope.get("server")
    return is_loopback_host(target[1]) or bool(server and target[1] == server[0].lower())


def _check_bearer(headers: Headers, expected: str) -> bool:
    """Return True iff a correct ``Authorization: Bearer <expected>`` header is present."""
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return False
    scheme, _, token = auth.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.encode(), expected.encode())


def _require_auth(request: Request = None,  # pyright: ignore[reportArgumentType]  # FastAPI injects Request/WebSocket by type; None default is a sentinel, not a real argument
                  websocket=None):  # pyright: ignore[reportMissingParameterType]  # FastAPI special-cases bare WebSocket/Request injection; an explicit `WebSocket | None` annotation makes it build a request field and raises at app-construction time
    """Bearer-token auth gate for HTTP routes.

    Accepts either a ``Request`` or a ``WebSocket`` — FastAPI resolves the
    non-None one based on the route type. On WebSocket connections we can't
    raise ``HTTPException(401)`` (the handshake hasn't completed), so the
    dep returns silently and the handler uses ``ws_auth_ok()`` + ``close(1008)``
    before accepting the connection.
    """
    conn = request if request is not None else websocket
    if conn is None:
        return
    expected = getattr(conn.app.state, "api_key", None)
    if not expected:
        if request is not None and not _local_host_ok(request):
            raise HTTPException(403, "Untrusted host; configure an API key for remote access")
        return
    if request is None:
        # WS path: handler calls ws_auth_ok() before websocket.accept().
        return
    if not _check_bearer(request.headers, expected):
        raise HTTPException(
            status_code=401,
            detail={"message": "Invalid API key", "type": "invalid_request_error",
                    "param": None, "code": 401},
        )
    return


def _browser_origin_ok(conn: Request | WebSocket) -> bool:
    origins = conn.headers.getlist("origin")
    if origins:
        origin = _origin(origins[0]) if len(origins) == 1 else None
        hosts = conn.headers.getlist("host")
        scheme = "https" if conn.scope["scheme"] in {"https", "wss"} else "http"
        target = _origin(scheme + "://" + hosts[0]) if len(hosts) == 1 else None
        allowed = conn.app.state.ws_origins
        if origin is None or (origin != target and origin not in allowed):
            return False
    return True


def ws_auth_ok(websocket: WebSocket) -> bool:
    """Validate browser Origin and bearer auth before accepting a WebSocket."""
    query = parse_qsl(websocket.scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
    tokens = [value for name, value in query if name == "token"]
    if tokens:
        websocket.scope["query_string"] = urlencode([(name, value) for name, value in query if name != "token"]).encode()
    expected = getattr(websocket.app.state, "api_key", None)
    if not expected and not _local_host_ok(websocket):
        return False
    if not _browser_origin_ok(websocket):
        return False
    if not expected:
        return True
    if _check_bearer(websocket.headers, expected):
        return True
    credentials = [protocol.removeprefix("drowse.auth.") for protocol in websocket.scope.get("subprotocols", [])
                   if protocol.startswith("drowse.auth.")]
    if credentials:
        encoded = base64.urlsafe_b64encode(expected.encode()).rstrip(b"=")
        return len(credentials) == 1 and hmac.compare_digest(credentials[0].encode(), encoded)
    return len(tokens) == 1 and hmac.compare_digest(tokens[0].encode(), expected.encode())


class _HttpSecurityMiddleware:
    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        path = scope["path"]
        private = path.startswith((NATIVE_PREFIX, OLLAMA_PREFIX, "/v1/"))

        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
                headers["Content-Security-Policy"] = (
                    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' https://www.neuronpedia.org https://drowse.ai/api/contact; "
                    "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'"
                )
                if private:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        if private:
            try:
                _require_auth(request)
            except HTTPException as exc:
                handler = request.app.exception_handlers[StarletteHTTPException]
                response = await handler(request, exc)
                await response(scope, receive, secure_send)
                return
        try:
            check_origin = private or scope["method"] not in {"GET", "HEAD", "OPTIONS"}
            if check_origin and scope["method"] != "OPTIONS" and not _browser_origin_ok(request):
                raise HTTPException(403, "Untrusted request origin")
            lengths = request.headers.getlist("content-length")
            if lengths:
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise HTTPException(400, "Invalid Content-Length")
                length = lengths[0].lstrip("0") or "0"
                if len(length) > len(str(self.max_request_bytes)) or int(length) > self.max_request_bytes:
                    raise HTTPException(413, "Request body too large")
        except HTTPException as exc:
            response = _protocol_error(path, exc.status_code, _detail_text(exc.detail))
            await response(scope, receive, secure_send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_request_bytes:
                    raise HTTPException(413, "Request body too large")
            return message

        await self.app(scope, limited_receive, secure_send)


def _sampling_kwargs(
    req: _SamplingBase, default_steering: "Steering | None",
) -> dict[str, Any]:
    """Build the kwargs dict passed to session.generate / generate_stream.

    Returns ``sampling=SamplingConfig(...)`` + ``steering=Steering(...)``
    / None + ``thinking=`` + ``stateless=True``.  The server never mutates
    ``session.config``.

    Composes ``req.steering`` (expression string) over
    ``default_steering``: per-request keys override defaults. ``thinking``
    is the native request override; ``None`` triggers
    ``supports_thinking`` auto-detect inside ``_generate_core``.
    """
    # ``stop`` (str or list) and the two OpenAI ``logprobs`` shapes (chat's
    # bool + ``top_logprobs`` count, completions' bare int) are normalized
    # inside the shared constructor.
    sc = build_sampling_config(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        seed=req.seed,
        stop=req.stop,
        logit_bias=req.logit_bias,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        logprobs=req.logprobs,
        top_logprobs=req.top_logprobs,
    )

    steering = req.to_steering(default_steering)

    thinking_kwarg: bool | None = req.thinking
    if thinking_kwarg is None and steering is not None and steering.thinking is not None:
        thinking_kwarg = steering.thinking

    return {
        "sampling": sc,
        "steering": steering,
        "thinking": thinking_kwarg,
        "stateless": True,
    }


def _token_bytes(text: str) -> list[int]:
    try:
        return list(text.encode("utf-8"))
    except Exception:
        return []


def _render_logprobs_chat(result: GenerationResult, session: DrowseSession) -> dict[str, Any] | None:
    if result.logprobs is None:
        return None
    tok = session.tokenizer
    content = []
    # Inner ``top`` is now ``list[TokenAlt]`` (id/text/logprob triples
    # decoded by the engine at top-K capture time); the previous
    # ``list[tuple[int, float]]`` pair shape is retired, so we no longer
    # re-tokenize the alt ids here.  The
    # chosen-token text still goes through ``tok.decode`` because the
    # engine emits its id via ``result.tokens`` without the streaming
    # text representation alongside.
    for tid, lp, top in result.logprobs:
        tok_str: str = tok.decode([tid])  # pyright: ignore[reportAssignmentType]  # transformers stub returns str | list[str] but single-list input always yields str
        content.append({
            "token": tok_str,
            "logprob": lp,
            "bytes": _token_bytes(tok_str),
            "top_logprobs": [
                {"token": alt.text, "logprob": alt.logprob,
                 "bytes": _token_bytes(alt.text)}
                for alt in top
            ],
        })
    return {"content": content}


def _render_logprobs_completions(result: GenerationResult, session: DrowseSession) -> dict[str, Any] | None:
    """OpenAI /v1/completions logprobs shape (flat, token-parallel arrays).

    https://platform.openai.com/docs/api-reference/completions/object#completions/object-logprobs

    Inner ``top`` is ``list[TokenAlt]`` after the logit-alternative pass — alt
    text comes off the dataclass rather than a redundant tokenizer
    decode.
    """
    if result.logprobs is None:
        return None
    tok = session.tokenizer
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    for tid, lp, top in result.logprobs:
        tok_str: str = tok.decode([tid])  # pyright: ignore[reportAssignmentType]  # transformers stub returns str | list[str] but single-list input always yields str
        tokens.append(tok_str)
        token_logprobs.append(lp)
        top_logprobs.append({alt.text: alt.logprob for alt in top})
        text_offset.append(offset)
        offset += len(tok_str)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


async def _stream_generation(
    session: DrowseSession,
    stream_factory: Callable[[], GenerationStream], rid: str, model_id: str, object_type: str,
    format_delta: Callable[[Any], dict[str, Any]], empty_delta: dict[str, Any],
    include_usage: bool = False, role_delta: bool = False,
    request: Request | None = None,
):
    """Shared SSE generator for chat and completion streaming.

    Serializes against other requests via ``session.lock`` for the full
    stream lifetime (streams inherit queue semantics rather than 409).
    Per-request sampling overrides are carried in the iterator's own
    ``sampling=`` kwarg (bound at caller site) — no session.config rebind.

    The inner ``stream_iter`` is always ``.close()``d in a ``finally`` so the
    engine's worker-thread teardown (stop-flag + join) fires deterministically
    on early exit rather than waiting on GC.  When a ``request`` is wired in,
    a periodic ``is_disconnected()`` check stops generation early once the
    client is gone, so the GPU isn't spent on a stream nobody reads.
    """
    created_ts = int(time.time())

    def _error_frames(status: int, message: str) -> list[str]:
        """One in-band error frame plus the terminating ``[DONE]`` sentinel.

        ``openai-python``'s ``Stream`` and LangChain's ``ChatOpenAI`` treat an
        SSE body that ends without ``[DONE]`` the way ollama-python treats a
        missing terminating ``done`` frame — they wait for it.  The Ollama
        NDJSON path already terminates after an in-band error chunk; this is
        the same contract on the SSE side.
        """
        err = {
            "error": {
                "message": message,
                "type": openai_error_type(status),
                "code": status,
            },
        }
        return [f"data: {json.dumps(err)}\n\n", "data: [DONE]\n\n"]

    async with acquire_session_lock(session) as acquired:
        if not acquired:
            for frame in _error_frames(503, "Server busy"):
                yield frame
            return

        if role_delta:
            chunk = {
                "id": rid, "object": object_type, "created": created_ts,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        stream_iter = None
        try:
            stream_iter = stream_factory()
            async for event in stream_events(stream_iter):
                # Bail out if the client has hung up — close the inner
                # generator (handled in ``finally``) and stop spending the
                # GPU on tokens nobody is reading.
                if request is not None and await request.is_disconnected():
                    return
                choice: dict[str, Any] = {
                    "index": 0, **format_delta(event), "finish_reason": None,
                }
                # Per-token manifold readings ride under a vendor-
                # prefixed extension on the choice so OpenAI clients
                # that don't read the field stay unaffected.  Populated
                # only when at least one manifold probe is attached
                # and ``live_scores`` is True on the stream.
                mf_token = probe_token_readings(event)
                if mf_token is not None:
                    choice["x-drowse-probe-readings"] = mf_token
                chunk = {
                    "id": rid,
                    "object": object_type,
                    "created": created_ts,
                    "model": model_id,
                    "choices": [choice],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        except ConcurrentGenerationError:
            for frame in _error_frames(409, "Generation already in progress"):
                yield frame
            return
        except DrowseError as e:
            status, msg = e.user_message()
            for frame in _error_frames(status, msg):
                yield frame
            return
        finally:
            # Deterministically tear down the engine worker thread (stop-flag
            # + join) on every exit — normal completion (no-op on an exhausted
            # generator), an in-band error, or an early client-disconnect
            # ``return`` — rather than leaving it to GC.
            if stream_iter is not None:
                await run_in_thread(stream_iter.close)

        assert stream_iter is not None
        last_result = stream_iter.result
        finish_reason, usage, mf_agg = stream_finalizer(session, last_result)
        final_choice: dict[str, Any] = {
            "index": 0, **empty_delta, "finish_reason": finish_reason,
        }
        if mf_agg:
            final_choice["x-drowse-probe-readings"] = mf_agg
        final = {
            "id": rid,
            "object": object_type,
            "created": created_ts,
            "model": model_id,
            "choices": [final_choice],
        }
        yield f"data: {json.dumps(final)}\n\n"

        if include_usage and usage is not None:
            usage_chunk = {
                "id": rid, "object": object_type, "created": created_ts,
                "model": model_id, "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session: DrowseSession,
               default_steering: "Steering | None" = None,
               cors_origins: list[str] | None = None,
               api_key: str | None = None,
               *,
               web: bool = False,
               max_request_bytes: int | None = None) -> FastAPI:
    if max_request_bytes is None:
        max_request_bytes = int(os.environ.get("DROWSE_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be a positive integer")
    app = FastAPI(
        title="drowse",
        description="OpenAI-compatible API with activation steering",
        dependencies=[Depends(_require_auth)],
    )
    app.state.session = session
    app.state.default_steering = default_steering
    app.state.created_ts = int(time.time())
    app.state.api_key = api_key if api_key is not None else os.environ.get("DROWSE_API_KEY")
    app.state.ws_origins = {origin for value in (cors_origins or []) if (origin := _origin(value)) is not None}
    app.add_middleware(_HttpSecurityMiddleware, max_request_bytes=max_request_bytes)
    # Generation serialization lives on ``session.lock`` (asyncio.Lock)
    # so both the OpenAI and Ollama route families share a single FIFO
    # queue.  Requests wait rather than 409 on contention.

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(DrowseError)
    async def _on_drowse_error(request: Request, exc: DrowseError):
        if exc.__cause__ is not None:
            logging.getLogger("drowse.api").error("Request failed", exc_info=exc)
        status, msg = exc.user_message()
        return _protocol_error(request.url.path, status, msg)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError):
        errs = exc.errors()
        first = errs[0] if errs else {}
        loc = first.get("loc", ())
        param = ".".join(str(p) for p in loc[1:]) if len(loc) > 1 else (str(loc[0]) if loc else None)
        msg = first.get("msg", "Invalid request")
        path = request.url.path
        if path.startswith(NATIVE_PREFIX) or path.startswith(OLLAMA_PREFIX):
            # The native tree and the Ollama shim each own an envelope; a
            # body-validation failure has to speak the same one the rest of
            # that protocol's failures do.
            return _protocol_error(
                path, 400, f"{param}: {msg}" if param else msg,
            )
        return _error(400, msg, "invalid_request_error", param=param)

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_exception(request: Request, exc: StarletteHTTPException):
        """Normalize the native tree's ``HTTPException``s to one envelope.

        Almost every native route raises a bare ``HTTPException``, which
        Starlette renders as ``{"detail": …}`` — but ``detail`` could be a
        string, the auth dependency's dict, or a list of pydantic error dicts,
        so a client had to guess.  On ``/drowse/v1/*`` it is always a string;
        every other prefix keeps FastAPI's default rendering.
        """
        if isinstance(exc.__cause__, DrowseError):
            _status, detail = exc.__cause__.user_message()
            if exc.__cause__.__cause__ is not None:
                logging.getLogger("drowse.api").error("Request failed", exc_info=exc.__cause__)
            return _protocol_error(request.url.path, exc.status_code, detail)
        if isinstance(exc.__cause__, OSError):
            logging.getLogger("drowse.api").error("Filesystem request failed", exc_info=exc.__cause__)
            detail = {404: "Requested artifact not found", 409: "An artifact already exists at the destination"}.get(
                exc.status_code, "Filesystem operation failed. Check the server log for details.",
            )
            return _protocol_error(request.url.path, exc.status_code, detail)
        response = await http_exception_handler(request, exc)
        detail = cast(object, exc.detail)
        if (
            request.url.path.startswith(NATIVE_PREFIX)
            and not isinstance(detail, str)
            # A bodiless status (204 / 304) comes back as a bare ``Response``
            # from the default handler — leave those alone.
            and isinstance(response, JSONResponse)
        ):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": _detail_text(detail)},
                headers=getattr(exc, "headers", None),
            )
        return response

    _register_routes(app)

    # Mount Ollama-compatible /api/* routes alongside OpenAI routes so any
    # Ollama client (Open WebUI, Enchanted, ollama-python, etc.) talks to
    # drowse as a drop-in replacement.
    from drowse.server.ollama import register_ollama_routes
    register_ollama_routes(app)

    from drowse.server.native_routes import register_drowse_routes
    register_drowse_routes(app)

    # Mount the Svelte+Vite SPA dashboard last so its catch-all route
    # doesn't shadow any of the API routes registered above.  CLI
    # default-on (``drowse serve``); ``--no-web`` opts out for
    # production / proxied deployments where ``/`` already belongs to
    # something else.  Library callers using ``create_app`` directly
    # default-off so embedded API surfaces don't accidentally pick up
    # the dashboard.
    if web:
        from drowse.web import register_web_routes

        register_web_routes(app)

    return app


def _openai_known_model_names(session: DrowseSession) -> set[str]:
    """Names accepted for OpenAI routes in strict mode.

    Includes the HF id plus any Ollama-style aliases (`<family>:<size>`)
    — the OpenAI catalogue is a superset so clients hitting either
    protocol with the same name keep working.
    """
    from drowse.server.model_names import known_model_names

    return known_model_names(session)


def _check_openai_model_strict(session: DrowseSession, name: str | None) -> None:
    if not strict_model_enabled():
        return
    if not name:
        return
    if name.lower() not in _openai_known_model_names(session):
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model '{name}' not found",
                "type": "invalid_request_error",
                "param": "model",
                "code": 404,
            },
        )


def _register_routes(app: FastAPI) -> None:
    session: DrowseSession = app.state.session

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": session.model_id,
                    "object": "model",
                    "created": app.state.created_ts,
                    "owned_by": "local",
                }
            ],
        }

    @app.get("/v1/models/{model_id:path}")
    def get_model(model_id: str):
        if model_id != session.model_id:
            raise HTTPException(404, f"Model '{model_id}' not found")
        return {
            "id": session.model_id,
            "object": "model",
            "created": app.state.created_ts,
            "owned_by": "local",
        }

    # -----------------------------------------------------------------------
    # Chat completions
    # -----------------------------------------------------------------------

    async def _run_blocking(req: _SamplingBase, prompt_or_messages: Any, *, raw: bool) -> Any:
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)
        # Bounded lock so a non-streaming request can't queue forever behind
        # a stuck generation — it 503s like the streaming paths do.  Returns
        # a ``JSONResponse`` on timeout; callers return it verbatim.
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                return _error(503, "Server busy", "server_error")
            return (await run_in_thread(session.generate, prompt_or_messages, raw=raw, **gen_kwargs)).first

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        _check_openai_model_strict(session, req.model)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)

        if req.stream:
            def _chat_delta(event: Any) -> dict[str, Any]:
                d: dict[str, str] = {}
                if event.thinking:
                    d["reasoning_content"] = event.text
                else:
                    d["content"] = event.text
                return {"delta": d}

            def stream_factory() -> GenerationStream:
                return session.generate_stream(
                    messages, live_scores=False, live_readouts=False, **gen_kwargs,
                )
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return ClosingStreamingResponse(
                _stream_generation(session,
                                   stream_factory, rid, model_id,
                                   "chat.completion.chunk", _chat_delta, {"delta": {}},
                                   include_usage=include_usage, role_delta=True,
                                   request=request),
                media_type="text/event-stream",
            )
        try:
            result = await _run_blocking(req, messages, raw=False)
        except ConcurrentGenerationError:
            return _error(409, "Generation already in progress", "conflict")
        if isinstance(result, JSONResponse):  # bounded-lock timeout → 503
            return result

        chat_choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "logprobs": _render_logprobs_chat(result, session),
            "finish_reason": result.finish_reason,
        }
        mf_chat = probe_reading_aggregate(session, result)
        if mf_chat:
            chat_choice["x-drowse-probe-readings"] = mf_chat
        body = {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [chat_choice],
            "usage": usage_dict(result),
        }
        return body

    # -----------------------------------------------------------------------
    # Text completions
    # -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        _check_openai_model_strict(session, req.model)
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)

        if req.stream:
            def stream_factory() -> GenerationStream:
                return session.generate_stream(
                    req.prompt, raw=True, live_scores=False, live_readouts=False,
                    **gen_kwargs,
                )
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return ClosingStreamingResponse(
                _stream_generation(session,
                                   stream_factory, rid, model_id,
                                   "text_completion", lambda e: {"text": e.text}, {"text": ""},
                                   include_usage=include_usage, role_delta=False,
                                   request=request),
                media_type="text/event-stream",
            )
        try:
            result = await _run_blocking(req, req.prompt, raw=True)
        except ConcurrentGenerationError:
            return _error(409, "Generation already in progress", "conflict")
        if isinstance(result, JSONResponse):  # bounded-lock timeout → 503
            return result

        completion_choice: dict[str, Any] = {
            "index": 0,
            "text": result.text,
            "logprobs": _render_logprobs_completions(result, session),
            "finish_reason": result.finish_reason,
        }
        mf_completion = probe_reading_aggregate(session, result)
        if mf_completion:
            completion_choice["x-drowse-probe-readings"] = mf_completion
        body = {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [completion_choice],
            "usage": usage_dict(result),
        }
        return body
