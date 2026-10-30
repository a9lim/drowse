"""Session route schemas and serializers for ``/drowse/v1/sessions``."""

from __future__ import annotations

from pydantic import Field, FiniteFloat

from drowse.session.session import DrowseSession
from drowse.steering.terms import Steering
from drowse.server.instrument_routes import instrument_families
from drowse.server.runtime import NativeRequest
from drowse.server.response_models import SamplingFields, SessionInfo


class PatchSessionRequest(NativeRequest):
    temperature: FiniteFloat | None = None
    top_p: FiniteFloat | None = Field(default=None, ge=0, le=1)
    top_k: int | None = None
    max_tokens: int | None = Field(default=None, ge=0)
    system_prompt: str | None = None
    thinking: bool | None = None


class ValidateSteeringRequest(NativeRequest):
    expression: str


def session_config_dict(session: DrowseSession) -> SamplingFields:
    cfg = session.config
    return {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "max_tokens": cfg.max_new_tokens,
        "system_prompt": cfg.system_prompt,
        "thinking": cfg.thinking,
    }


def role_support(session: DrowseSession) -> tuple[bool, bool]:
    """``(assistant_supported, user_supported)`` for the loaded model family."""
    from drowse.model.role_templates import resolve_seat_headers

    mt = session.renderer.model_type
    seats = resolve_seat_headers(session.tokenizer, mt) if mt is not None else None
    if seats is None:
        return (False, False)
    return (seats.assistant is not None, seats.user is not None)


def default_role_labels(
    session: DrowseSession,
) -> tuple[str | None, str | None]:
    """The family-standard assistant/user role labels, if substitutable."""
    from drowse.model.role_templates import resolve_seat_headers

    mt = session.renderer.model_type
    seats = resolve_seat_headers(session.tokenizer, mt) if mt is not None else None
    if seats is None:
        return (None, None)
    asst, usr = seats.assistant, seats.user
    return (
        asst.label if asst is not None else None,
        usr.label if usr is not None else None,
    )


def device_dtype(session: DrowseSession) -> tuple[str, str]:
    info = session.model_info
    return str(info["device"]), str(info["dtype"])


def _scene_capabilities(session: DrowseSession) -> tuple[bool, bool, bool]:
    """The current session's validated scene grammar, as three flags."""
    grammar = session.scene_grammar
    return (
        grammar is not None,
        bool(grammar is not None and isinstance(grammar.think_open, str)),
        bool(grammar is not None and grammar.strips_history_thinking is True),
    )


def session_info(
    session: DrowseSession,
    default_steering: Steering | None,
    created_ts: int,
) -> SessionInfo:
    device, dtype = device_dtype(session)
    traits = session.chat_traits
    thinks = traits.supports_thinking
    thinks_optional = traits.thinking_optional
    is_base = bool(session.is_base_model)
    default_expr = str(default_steering) if default_steering is not None else None
    assistant_role_ok, user_role_ok = role_support(session)
    default_assistant_role, default_user_role = default_role_labels(session)
    jlens_fitted = session.has_compatible_jlens()
    scene_mode, thinking_input, strips_history_thinking = _scene_capabilities(session)
    return {
        "id": "default",
        "model_id": session.model_id,
        "device": device,
        "dtype": dtype,
        "created": created_ts,
        "config": session_config_dict(session),
        "profiles": sorted(session.profiles.keys()),
        "probes": sorted(session.probes.keys()),
        "history_length": len(session.tree.messages_for()),
        "supports_thinking": thinks,
        "thinking_is_optional": thinks_optional,
        "is_base_model": is_base,
        "jlens_fitted": jlens_fitted,
        # The read plane's one representation: a {family, live, source,
        # probes, capabilities} block per instrument family.
        "instruments": instrument_families(session),
        "default_steering": default_expr,
        "role_substitution_supported": assistant_role_ok,
        "user_role_supported": user_role_ok,
        "default_assistant_role": default_assistant_role,
        "default_user_role": default_user_role,
        # Scene-grammar capabilities (the cast model): scene_mode gates
        # the seat toggle + free commit seating, thinking_input_supported
        # the committed-thinking box, strips_history_thinking its
        # "lasts one turn" pre-submit warning.
        "scene_mode": scene_mode,
        "thinking_input_supported": thinking_input,
        "strips_history_thinking": strips_history_thinking,
    }
