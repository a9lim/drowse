"""The FastAPI factory: one model, three protocols, one port."""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from drowse.server.compat.ollama import register_ollama_routes
from drowse.server.compat.openai import register_openai_routes
from drowse.server.errors import install_error_handlers
from drowse.server.instrument_routes import register_instrument_routes
from drowse.server.manifold_routes import register_manifold_routes
from drowse.server.operations import register_operation_routes
from drowse.server.probe_routes import register_probe_routes
from drowse.server.profile_routes import register_profile_routes
from drowse.server.security import DEFAULT_MAX_REQUEST_BYTES, HttpSecurityMiddleware, parse_origin
from drowse.server.session_routes import register_session_routes
from drowse.server.template_routes import register_template_routes
from drowse.server.tree_routes import register_tree_routes
from drowse.server.ws_stream import register_ws_stream
from drowse.session.session import DrowseSession
from drowse.steering.terms import Steering


def create_app(
    session: DrowseSession,
    default_steering: Steering | None = None,
    cors_origins: list[str] | None = None,
    api_key: str | None = None,
    *,
    web: bool = False,
    max_request_bytes: int | None = None,
) -> FastAPI:
    """Serve ``session`` over OpenAI ``/v1``, Ollama ``/api``, and native ``/drowse/v1``.

    ``api_key`` falls back to ``DROWSE_API_KEY``; without one, only loopback
    clients are served.  ``web=True`` mounts the dashboard at ``/`` (last, so
    its catch-all cannot shadow an API path).  Every generation and model read
    across the three protocols serializes on ``app.state.session_lock``.
    """
    if max_request_bytes is None:
        max_request_bytes = int(os.environ.get("DROWSE_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be a positive integer")
    app = FastAPI(title="drowse", description="OpenAI- and Ollama-compatible API with activation steering")
    app.state.session = session
    app.state.session_lock = asyncio.Lock()
    app.state.default_steering = default_steering
    app.state.created_ts = int(time.time())
    app.state.api_key = api_key if api_key is not None else os.environ.get("DROWSE_API_KEY")
    app.state.ws_origins = {
        origin for value in (cors_origins or []) if (origin := parse_origin(value)) is not None
    }
    app.add_middleware(HttpSecurityMiddleware, max_request_bytes=max_request_bytes)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"],
        )
    install_error_handlers(app)

    register_openai_routes(app)
    register_ollama_routes(app)
    for register in (
        register_operation_routes,
        register_manifold_routes,
        register_template_routes,
        register_session_routes,
        register_tree_routes,
        register_profile_routes,
        register_probe_routes,
        register_instrument_routes,
        register_ws_stream,
    ):
        register(app)
    if web:
        from drowse.web import register_web_routes

        register_web_routes(app)
    return app
