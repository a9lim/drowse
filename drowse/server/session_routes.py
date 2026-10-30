"""Native ``/drowse/v1/sessions`` routes: session info, sampling defaults, steering validation."""

from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI

from drowse.errors import DrowseError
from drowse.server.response_models import (
    SessionInfo,
    SessionListResponse,
    SteeringValidationResponse,
)
from drowse.server.runtime import session_lock, session_router
from drowse.server.session_models import (
    PatchSessionRequest,
    ValidateSteeringRequest,
    session_info,
)
from drowse.steering.grammar import format_expr, parse_expr


def register_session_routes(app: FastAPI) -> None:
    session = app.state.session
    router = session_router()

    def info() -> SessionInfo:
        return session_info(session, app.state.default_steering, app.state.created_ts)

    @app.get("/drowse/v1/sessions")
    def list_sessions() -> SessionListResponse:
        return {"sessions": [info()]}

    @router.get("")
    def get_session() -> SessionInfo:
        return info()

    @router.patch("")
    def patch_session(req: PatchSessionRequest) -> SessionInfo:
        config = session.config
        fields = req.model_fields_set
        session.config = replace(
            config,
            temperature=req.temperature if req.temperature is not None else config.temperature,
            top_p=req.top_p if req.top_p is not None else config.top_p,
            # An explicit ``null`` disables top-k / clears the system prompt;
            # an omitted field preserves the current value.
            top_k=req.top_k if "top_k" in fields else config.top_k,
            max_new_tokens=req.max_tokens if req.max_tokens is not None else config.max_new_tokens,
            system_prompt=req.system_prompt if "system_prompt" in fields else config.system_prompt,
            thinking=req.thinking if req.thinking is not None else config.thinking,
        )
        return info()

    @router.post("/steering/validate")
    async def validate_steering(req: ValidateSteeringRequest) -> SteeringValidationResponse:
        """Parse, resolve, and dry-install a dashboard-authored expression.

        Entering ``session.steering(parsed)`` resolves selectors, projections,
        variants, and atoms and composes the hooks exactly as a generation
        would, then rolls straight back — so a green validation cannot hide a
        rack-time failure.  It touches shared hook state, so it serializes on
        the session lock.  A rejected expression is an in-band form result
        (``valid: false``), not a failed request.
        """
        expression = req.expression.strip()
        if not expression:
            return {"valid": True, "expression": "", "error": None}
        try:
            parsed = parse_expr(expression)
        except DrowseError as exc:
            return {"valid": False, "expression": expression, "error": exc.user_message()[1]}
        async with session_lock(app):
            try:
                with session.steering(parsed):
                    pass
            except DrowseError as exc:
                return {"valid": False, "expression": expression, "error": exc.user_message()[1]}
        return {"valid": True, "expression": format_expr(parsed), "error": None}

    app.include_router(router)
