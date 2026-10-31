"""Ollama-compatible ``/api/*`` routes.

Any Ollama client (Open WebUI, Enchanted, Msty, ollama-python, LangChain's
ChatOllama) can talk to drowse as a drop-in replacement.  Requests lower onto
the shared :class:`~drowse.server.compat.CompatGeneration`; this module owns
the typed request bodies, option translation, and the Ollama wire shapes.

Differences from real Ollama:

- One model per server.  ``/api/tags`` advertises its HF repo id and any
  recognized Ollama alias; ``model`` on a request is accepted unless
  ``DROWSE_STRICT_MODEL`` is set.
- ``/api/pull`` is a no-op success for the loaded model and 404 otherwise;
  ``push`` / ``create`` / ``copy`` / ``delete`` / ``embeddings`` / ``embed``
  are 501.
- Streaming is NDJSON (``application/x-ndjson``); ``/api/generate`` omits
  ``context`` (drowse cannot round-trip it).
- Steering rides a non-standard ``steer`` expression string inside
  ``options`` or at the top level.  A top-level ``think`` (a bool, or one of
  Ollama's reasoning levels ``"low"``/``"medium"``/``"high"``) overrides
  thinking; drowse has no effort knob, so any level turns thinking on.  Probe
  readings ride the top-level ``x-drowse-probe-readings`` key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from drowse.errors import RequestError
from drowse.generation.results import GenerationResult
from drowse.server.compat import (
    CompatGeneration,
    StreamEnd,
    StreamStarted,
    check_strict_model,
    probe_reading_aggregate,
    run_generation,
    stream_generation,
)
from drowse.server.compat.model_names import aliases_for_session, known_model_names
from drowse.server.errors import ErrorInfo
from drowse.server.request_helpers import (
    build_sampling_config,
    compose_request_steering,
    flatten_content,
)
from drowse.server.runtime import ClosingStreamingResponse
from drowse.session.session import DrowseSession
from drowse.steering.terms import Steering

log = logging.getLogger(__name__)

_PROBE_KEY = "x-drowse-probe-readings"


class OllamaBadRequest(RequestError):
    """400-grade Ollama compatibility error for a malformed request field."""


# ---------------------------------------------------------------------------
# Request bodies — typed, but open: Ollama clients send fields drowse ignores,
# and an explicit ``null`` reads as the field's unset default.
# ---------------------------------------------------------------------------

#: Ollama's ``think`` levels (gpt-oss reasoning effort).
ThinkLevel = Literal["low", "medium", "high"]

class _OllamaBody(BaseModel):
    model_config = ConfigDict(extra="allow")


class OllamaModelRequest(_OllamaBody):
    """``/api/show`` and ``/api/pull``: a model selected by ``model`` or ``name``."""

    model: str | None = None
    name: str | None = None

    @property
    def requested(self) -> str:
        return self.model or self.name or ""


class OllamaMessage(_OllamaBody):
    role: str = "user"
    content: Any = None


class OllamaGenerationRequest(_OllamaBody):
    model: str | None = None
    #: Ollama streams unless told otherwise; an explicit ``null`` does not stream.
    stream: bool | None = True
    options: dict[str, Any] | None = None
    system: str | None = None
    think: bool | ThinkLevel | None = None
    steer: Any = None
    stop: Any = None
    num_predict: Any = None

    @property
    def thinking(self) -> bool | None:
        """The engine's thinking override: a reasoning level means on."""
        return self.think if self.think is None or isinstance(self.think, bool) else True


class OllamaChatRequest(OllamaGenerationRequest):
    messages: list[OllamaMessage] | None = Field(default_factory=list)


class OllamaGenerateRequest(OllamaGenerationRequest):
    prompt: Any = ""
    raw: bool | None = False


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """ISO 8601 timestamp with microseconds and trailing Z, matching Ollama."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _estimate_bytes(session: DrowseSession) -> int:
    info = session.model_info
    params = int(info.get("param_count", 0) or 0)
    dtype = str(info.get("dtype", ""))
    bytes_per = 4 if "float32" in dtype else (1 if "int8" in dtype or "fp8" in dtype else 2)
    return params * bytes_per


def _param_size_label(session: DrowseSession) -> str:
    params = int(session.model_info.get("param_count", 0) or 0)
    if params <= 0:
        return ""
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.0f}M"
    return str(params)


def _quant_label(session: DrowseSession) -> str:
    dtype = str(session.model_info.get("dtype", "")).lower()
    for needles, label in (
        (("int4", "4bit"), "Q4_0"),
        (("int8", "8bit"), "Q8_0"),
        (("bfloat16", "bf16"), "BF16"),
        (("float16", "fp16"), "F16"),
        (("float32", "fp32"), "F32"),
    ):
        if any(needle in dtype for needle in needles):
            return label
    return "unknown"


def _model_details(session: DrowseSession) -> dict[str, Any]:
    family = str(session.model_info.get("model_type", "unknown"))
    return {
        "parent_model": "",
        "format": "safetensors",
        "family": family,
        "families": [family],
        "parameter_size": _param_size_label(session),
        "quantization_level": _quant_label(session),
    }


def _tag_entries(session: DrowseSession) -> list[dict[str, Any]]:
    """The ``/api/tags`` rows: the HF repo id plus each recognized Ollama alias."""
    common = {
        "modified_at": _now_iso(),
        "size": _estimate_bytes(session),
        "digest": "sha256:" + hashlib.sha256(session.model_id.encode("utf-8")).hexdigest(),
        "details": _model_details(session),
    }
    names = dict.fromkeys([session.model_id, *aliases_for_session(session)])
    return [{"name": name, "model": name, **common} for name in names]


# ---------------------------------------------------------------------------
# Option translation
# ---------------------------------------------------------------------------

_PROCESSED_OPTIONS: frozenset[str] = frozenset({
    "temperature", "top_p", "top_k", "seed", "num_predict",
    "stop", "presence_penalty", "frequency_penalty", "repeat_penalty",
    "steer",
})


def _number_option(opts: dict[str, Any], name: str, default: float | None = None) -> float | None:
    raw = opts.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as e:
        raise OllamaBadRequest(f"Ollama option '{name}' must be a finite number") from e
    if not math.isfinite(value):
        raise OllamaBadRequest(f"Ollama option '{name}' must be a finite number")
    return value


def _int_option(name: str, raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
        if isinstance(raw, float) and raw != value:
            raise ValueError
        return value
    except (TypeError, ValueError, OverflowError) as e:
        raise OllamaBadRequest(f"Ollama option '{name}' must be an integer") from e


def resolve_options(
    body: OllamaGenerationRequest, default_steering: Steering | None, payload: Any, *, raw: bool,
) -> CompatGeneration:
    """Translate Ollama ``options`` + top-level fields into a generation.

    Recognized: ``temperature``, ``top_p``, ``top_k``, ``seed``,
    ``num_predict`` → ``max_tokens``, ``stop``, ``presence_penalty``,
    ``frequency_penalty``, and ``repeat_penalty`` → ``presence_penalty`` via
    ``ln(repeat_penalty)`` (Ollama divides positive logits by the penalty,
    which subtracts a constant per seen token).  Other options (``min_p``,
    ``mirostat*``, ``num_ctx``, …) are logged at debug and dropped.
    """
    opts = dict(body.options or {})
    stop_raw = opts.get("stop") or body.stop
    stop_spec = stop_raw if isinstance(stop_raw, (str, list)) else None

    top_p = _number_option(opts, "top_p")
    if top_p is not None and not 0 <= top_p <= 1:
        raise OllamaBadRequest("Ollama option 'top_p' must be between 0 and 1")
    top_k = _int_option("top_k", opts.get("top_k"))
    if top_k is not None and top_k <= 0:
        top_k = None
    presence_penalty = _number_option(opts, "presence_penalty", 0.0) or 0.0
    repeat_raw = opts.get("repeat_penalty")
    if repeat_raw is not None and presence_penalty == 0.0:
        try:
            penalty = float(repeat_raw)
            if not math.isfinite(penalty):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as e:
            raise OllamaBadRequest("Ollama option 'repeat_penalty' must be a finite number") from e
        if penalty > 1.0:
            presence_penalty = math.log(penalty)

    # Option names come from the client: strip line breaks so one can't forge log lines.
    ignored = sorted(
        str(k).replace("\r", "").replace("\n", "")
        for k in opts if k not in _PROCESSED_OPTIONS
    )
    if ignored:
        log.debug("ollama: unsupported options dropped: %s", ", ".join(ignored))

    steer = opts["steer"] if "steer" in opts else body.steer
    if steer is not None and not isinstance(steer, str):
        raise OllamaBadRequest(
            "Ollama 'steer' must be a steering expression string, "
            "e.g. \"0.5 honest + 0.3 warm\""
        )
    num_predict = opts["num_predict"] if "num_predict" in opts else body.num_predict
    sampling = build_sampling_config(
        temperature=_number_option(opts, "temperature"),
        top_p=top_p,
        top_k=top_k,
        max_tokens=_int_option("num_predict", num_predict),
        seed=_int_option("seed", opts.get("seed")),
        stop=stop_spec,
        presence_penalty=presence_penalty,
        frequency_penalty=_number_option(opts, "frequency_penalty", 0.0) or 0.0,
    )
    return CompatGeneration(
        input=payload,
        raw=raw,
        sampling=sampling,
        steering=compose_request_steering(steer, default_steering),
        thinking=body.thinking,
    )


def _chat_payload(body: OllamaChatRequest) -> list[dict[str, str]]:
    messages = [{"role": m.role, "content": flatten_content(m.content)} for m in body.messages or ()]
    if body.system:
        messages.insert(0, {"role": "system", "content": body.system})
    return messages


def _generate_payload(body: OllamaGenerateRequest) -> tuple[Any, bool]:
    prompt = flatten_content(body.prompt)
    if body.system:
        # ``system`` belongs at the top of the chat template, so a prompt with
        # a system field routes through the chat path.
        return [{"role": "system", "content": body.system}, {"role": "user", "content": prompt}], False
    return prompt, bool(body.raw)


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------

def _duration_stats(result: GenerationResult | None, elapsed_ns: int) -> dict[str, int]:
    """Ollama's ``*_duration`` / ``*_count`` fields, in nanoseconds.

    Measured wall time splits proportionally between prompt eval and eval.
    """
    prompt_tokens = result.prompt_tokens if result is not None else 0
    completion_tokens = result.token_count if result is not None else 0
    total = max(prompt_tokens + completion_tokens, 1)
    prompt_ns = elapsed_ns * prompt_tokens // total
    return {
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": prompt_ns,
        "eval_count": completion_tokens,
        "eval_duration": max(elapsed_ns - prompt_ns, 1),
    }


def _finish_to_done_reason(finish_reason: str | None) -> str:
    if finish_reason == "stop_sequence":
        return "stop"
    return finish_reason or "stop"


def _content(is_chat: bool, text: str, *, thinking: bool = False) -> dict[str, Any]:
    """The text-bearing fields of one chat / generate frame."""
    if is_chat:
        message: dict[str, Any] = {"role": "assistant", "content": "" if thinking else text}
        if thinking:
            message["thinking"] = text
        return {"message": message}
    return {"response": "", "thinking": text} if thinking else {"response": text}


def _final_frame(
    session: DrowseSession, is_chat: bool, head: dict[str, Any], text: str,
    result: GenerationResult | None, finish_reason: str | None, elapsed_ns: int,
) -> dict[str, Any]:
    frame = {
        **head, **_content(is_chat, text),
        "done_reason": _finish_to_done_reason(finish_reason), "done": True,
        **_duration_stats(result, elapsed_ns),
    }
    if readings := probe_reading_aggregate(session, result):
        frame[_PROBE_KEY] = readings
    return frame


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_ollama_routes(app: FastAPI) -> None:
    """Mount the ``/api/*`` routes; they share the session lock and the default steering."""
    session: DrowseSession = app.state.session

    @app.get("/api/version")
    def api_version():
        from drowse import __version__

        return {"version": f"drowse-{__version__}"}

    @app.get("/api/tags")
    def api_tags():
        return {"models": _tag_entries(session)}

    @app.get("/api/ps")
    def api_ps():
        vram = int((session.model_info.get("vram_used_gb") or 0) * 1024**3)
        return {"models": [
            {**entry, "expires_at": "9999-12-31T23:59:59Z", "size_vram": vram}
            for entry in _tag_entries(session)
        ]}

    @app.post("/api/show")
    def api_show(body: OllamaModelRequest):
        info = session.model_info
        family = info.get("model_type", "unknown")
        # The honest HF Jinja template: clients that parse it as a Go template
        # fail either way, so it beats a meaningless placeholder.
        template = getattr(session.tokenizer, "chat_template", None) or "{{ .Prompt }}"
        return {
            "license": "See upstream model card.",
            "modelfile": f"# drowse: {session.model_id}\nFROM {session.model_id}\n",
            "parameters": "",
            "template": template,
            "details": _model_details(session),
            "model_info": {
                "general.architecture": family,
                "general.parameter_count": info.get("param_count", 0),
                "general.quantization_version": 0,
                f"{family}.block_count": info.get("num_layers", 0),
                f"{family}.embedding_length": info.get("hidden_dim", 0),
                "drowse.loaded_model": session.model_id,
                "drowse.requested_name": body.requested or session.model_id,
            },
            "capabilities": ["completion", "chat"],
        }

    @app.post("/api/pull")
    async def api_pull(body: OllamaModelRequest):
        # Drowse loads its one model at startup, so a pull is a no-op success
        # for that model (by HF id or alias) and a 404 for anything else.
        name = body.requested
        if name and name.lower() not in known_model_names(session):
            hosted = ", ".join(sorted({session.model_id, *aliases_for_session(session)}))
            raise HTTPException(
                404,
                f"model '{name}' not found. drowse currently hosts: {hosted}. "
                "To serve a different model, restart with: drowse serve <model>",
            )

        async def _stream():
            yield json.dumps({"status": "pulling manifest"}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"
        return ClosingStreamingResponse(_stream(), media_type="application/x-ndjson")

    def _unsupported(path: str, method: str, message: str) -> None:
        async def _route():
            raise HTTPException(501, message)
        app.add_api_route(path, _route, methods=[method])

    _unsupported("/api/push", "POST", "drowse does not implement /api/push. Use `drowse pack push` to publish a manifold.")
    _unsupported("/api/create", "POST", "drowse does not implement /api/create.")
    _unsupported("/api/copy", "POST", "drowse does not implement /api/copy.")
    _unsupported("/api/delete", "DELETE", "drowse does not implement /api/delete.")
    _unsupported("/api/embeddings", "POST", "drowse does not implement /api/embeddings. Use the model's native embedding API.")
    _unsupported("/api/embed", "POST", "drowse does not implement /api/embed.")

    @app.head("/")
    def api_head_root():
        # Ollama clients probe liveness with a bodyless ``HEAD /``.
        return Response(status_code=200)

    async def _ndjson(
        gen: CompatGeneration, request: Request, is_chat: bool, head: dict[str, Any],
    ) -> AsyncIterator[str]:
        async with stream_generation(app, gen, request) as events:
            async for item in events:
                if isinstance(item, StreamStarted):
                    continue
                if isinstance(item, ErrorInfo):
                    # ollama-python / ChatOllama wait for a terminating done
                    # frame even after an in-band error.
                    yield json.dumps({**head, "error": item.message}) + "\n"
                    frame = {**head, "done": True, "done_reason": "error"}
                elif isinstance(item, StreamEnd):
                    frame = _final_frame(session, is_chat, head, "", item.result, item.finish_reason, item.elapsed_ns)
                else:
                    frame = {**head, **_content(is_chat, item.text, thinking=item.thinking), "done": False}
                yield json.dumps(frame) + "\n"

    async def _respond(
        body: OllamaGenerationRequest, request: Request, gen: CompatGeneration, is_chat: bool,
    ):
        head = {"model": body.model or session.model_id, "created_at": _now_iso()}
        if body.stream:
            return ClosingStreamingResponse(
                _ndjson(gen, request, is_chat, head), media_type="application/x-ndjson",
            )
        result, elapsed_ns = await run_generation(app, gen)
        return _final_frame(session, is_chat, head, result.text, result, result.finish_reason, elapsed_ns)

    @app.post("/api/chat")
    async def api_chat(body: OllamaChatRequest, request: Request):
        check_strict_model(session, body.model)
        # Lowered before the response starts, so a bad option or steering
        # expression is a clean 400 rather than a cut stream.
        gen = resolve_options(body, app.state.default_steering, _chat_payload(body), raw=False)
        return await _respond(body, request, gen, is_chat=True)

    @app.post("/api/generate")
    async def api_generate(body: OllamaGenerateRequest, request: Request):
        check_strict_model(session, body.model)
        payload, raw = _generate_payload(body)
        gen = resolve_options(body, app.state.default_steering, payload, raw=raw)
        return await _respond(body, request, gen, is_chat=False)
