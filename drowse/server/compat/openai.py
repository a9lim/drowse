"""OpenAI-compatible ``/v1/*`` routes.

``GET /v1/models``, ``GET /v1/models/{id}``, ``POST /v1/chat/completions``,
``POST /v1/completions``.  Requests lower onto the shared
:class:`~drowse.server.compat.CompatGeneration`; this module owns only the
request models and the OpenAI wire shapes (chunks, logprobs, usage).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, FiniteFloat, NonNegativeInt, model_validator

from drowse.generation.results import TokenEvent
from drowse.server.compat import (
    CompatGeneration,
    StreamEnd,
    StreamStarted,
    check_strict_model,
    probe_reading_aggregate,
    run_generation,
    stream_generation,
    usage_dict,
)
from drowse.server.errors import ErrorInfo, openai_error_type
from drowse.server.request_helpers import (
    UnsupportedContentError,
    build_sampling_config,
    compose_request_steering,
    flatten_content,
    validate_logit_bias,
)
from drowse.server.runtime import ClosingStreamingResponse

if TYPE_CHECKING:
    from drowse.generation.results import GenerationResult
    from drowse.session.session import DrowseSession

_PROBE_KEY = "x-drowse-probe-readings"


class ChatMessage(BaseModel):
    role: str
    content: Any  # str or a list of OpenAI content parts
    name: str | None = None

    @model_validator(mode="after")
    def _normalize_content(self):
        # Text parts of a multimodal array concatenate; any other part rejects.
        self.content = flatten_content(self.content)
        return self


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingBase(BaseModel):
    model: str | None = None
    temperature: FiniteFloat | None = None
    top_p: FiniteFloat | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    stream: bool = False
    stream_options: StreamOptions | None = None
    #: Native extension: a steering expression composed over the server default.
    steering: str | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    logit_bias: dict[NonNegativeInt, FiniteFloat] | None = None
    presence_penalty: FiniteFloat = 0.0
    frequency_penalty: FiniteFloat = 0.0
    logprobs: bool | int | None = None  # chat: bool; completions: int
    top_logprobs: int | None = None
    user: str | None = None
    #: Native extension: thinking override; ``None`` = auto.
    thinking: bool | None = None
    # LangChain compatibility: accept the no-op shapes, reject anything real.
    tools: list[Any] | None = None
    tool_choice: Any = None
    # Accepted and ignored.
    n: int | None = None
    response_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _unify_max_tokens(self):
        if self.max_completion_tokens is not None and self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        return self

    @model_validator(mode="after")
    def _check_langchain_compat(self):
        if self.tools:
            raise UnsupportedContentError("tool calling is not supported by drowse")
        if self.tool_choice is not None and self.tool_choice not in ("none", "auto"):
            raise UnsupportedContentError(
                "tool_choice values other than 'none'/'auto' are not supported"
            )
        if self.response_format is not None and self.response_format.get("type") not in (None, "text"):
            raise UnsupportedContentError(
                "response_format types other than 'text' are not supported"
            )
        return self

    def lower(
        self, session: "DrowseSession", default: Any, payload: Any, *, raw: bool,
    ) -> CompatGeneration:
        sampling = build_sampling_config(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=self.stop,
            logit_bias=self.logit_bias,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
        validate_logit_bias(sampling, session)
        return CompatGeneration(
            input=payload,
            raw=raw,
            sampling=sampling,
            steering=compose_request_steering(self.steering, default),
            thinking=self.thinking,
        )


class ChatCompletionRequest(_SamplingBase):
    messages: list[ChatMessage]


class CompletionRequest(_SamplingBase):
    prompt: str


def _token_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def render_logprobs_chat(result: "GenerationResult", session: "DrowseSession") -> dict[str, Any] | None:
    """The chat-completions ``logprobs.content`` array (alt text off ``TokenAlt``)."""
    if result.logprobs is None:
        return None
    content = []
    for tid, lp, top in result.logprobs:
        text: str = session.tokenizer.decode([tid])  # pyright: ignore[reportAssignmentType]  # single-id decode returns str
        content.append({
            "token": text,
            "logprob": lp,
            "bytes": _token_bytes(text),
            "top_logprobs": [
                {"token": alt.text, "logprob": alt.logprob, "bytes": _token_bytes(alt.text)}
                for alt in top
            ],
        })
    return {"content": content}


def render_logprobs_completions(result: "GenerationResult", session: "DrowseSession") -> dict[str, Any] | None:
    """The completions token-parallel ``logprobs`` arrays."""
    if result.logprobs is None:
        return None
    tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    for tid, lp, top in result.logprobs:
        text: str = session.tokenizer.decode([tid])  # pyright: ignore[reportAssignmentType]  # single-id decode returns str
        tokens.append(text)
        token_logprobs.append(lp)
        top_logprobs.append({alt.text: alt.logprob for alt in top})
        text_offset.append(offset)
        offset += len(text)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


async def _sse_stream(
    app: FastAPI,
    gen: CompatGeneration,
    request: Request,
    *,
    object_type: str,
    delta: Callable[[TokenEvent], dict[str, Any]],
    empty_delta: dict[str, Any],
    role_delta: bool,
    include_usage: bool,
) -> AsyncIterator[str]:
    """The OpenAI SSE framing over the shared driver.

    Always ends with ``data: [DONE]`` — after an in-band error frame too,
    because OpenAI clients wait for the sentinel.
    """
    session = app.state.session
    created = int(time.time())
    head = {"id": f"drowse-{uuid.uuid4().hex[:12]}", "object": object_type,
            "created": created, "model": session.model_id}

    def frame(**body: Any) -> str:
        return f"data: {json.dumps({**head, **body})}\n\n"

    async with stream_generation(app, gen, request) as events:
        async for item in events:
            if isinstance(item, StreamStarted):
                if role_delta:
                    yield frame(choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
            elif isinstance(item, ErrorInfo):
                error = {"message": item.message, "type": openai_error_type(item.status), "code": item.status}
                yield f"data: {json.dumps({'error': error})}\n\n"
            elif isinstance(item, StreamEnd):
                choice: dict[str, Any] = {"index": 0, **empty_delta, "finish_reason": item.finish_reason}
                if readings := probe_reading_aggregate(session, item.result):
                    choice[_PROBE_KEY] = readings
                yield frame(choices=[choice])
                if include_usage and item.result is not None:
                    yield frame(choices=[], usage=usage_dict(item.result))
            else:
                yield frame(choices=[{"index": 0, **delta(item), "finish_reason": None}])
    yield "data: [DONE]\n\n"


def _chat_delta(event: TokenEvent) -> dict[str, Any]:
    return {"delta": {"reasoning_content" if event.thinking else "content": event.text}}


def register_openai_routes(app: FastAPI) -> None:
    session: "DrowseSession" = app.state.session

    def model_card() -> dict[str, Any]:
        return {"id": session.model_id, "object": "model",
                "created": app.state.created_ts, "owned_by": "local"}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [model_card()]}

    @app.get("/v1/models/{model_id:path}")
    def get_model(model_id: str):
        if model_id != session.model_id:
            raise HTTPException(404, f"Model '{model_id}' not found")
        return model_card()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        check_strict_model(session, req.model)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        gen = req.lower(session, app.state.default_steering, messages, raw=False)
        if req.stream:
            return ClosingStreamingResponse(_sse_stream(
                app, gen, request,
                object_type="chat.completion.chunk", delta=_chat_delta,
                empty_delta={"delta": {}}, role_delta=True,
                include_usage=bool(req.stream_options and req.stream_options.include_usage),
            ), media_type="text/event-stream")
        result, _elapsed = await run_generation(app, gen)
        choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "logprobs": render_logprobs_chat(result, session),
            "finish_reason": result.finish_reason,
        }
        if readings := probe_reading_aggregate(session, result):
            choice[_PROBE_KEY] = readings
        return {
            "id": f"drowse-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": session.model_id,
            "choices": [choice], "usage": usage_dict(result),
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        check_strict_model(session, req.model)
        gen = req.lower(session, app.state.default_steering, req.prompt, raw=True)
        if req.stream:
            return ClosingStreamingResponse(_sse_stream(
                app, gen, request,
                object_type="text_completion", delta=lambda event: {"text": event.text},
                empty_delta={"text": ""}, role_delta=False,
                include_usage=bool(req.stream_options and req.stream_options.include_usage),
            ), media_type="text/event-stream")
        result, _elapsed = await run_generation(app, gen)
        choice: dict[str, Any] = {
            "index": 0,
            "text": result.text,
            "logprobs": render_logprobs_completions(result, session),
            "finish_reason": result.finish_reason,
        }
        if readings := probe_reading_aggregate(session, result):
            choice[_PROBE_KEY] = readings
        return {
            "id": f"drowse-{uuid.uuid4().hex[:12]}", "object": "text_completion",
            "created": int(time.time()), "model": session.model_id,
            "choices": [choice], "usage": usage_dict(result),
        }
