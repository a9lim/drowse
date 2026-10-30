"""The request-error policy: a typed error keeps its status and text, a bare
``ValueError`` is an engine invariant scrubbed to a logged 500.

Three tables pin it.  Every family maps through
:func:`~drowse.server.errors.describe_error` to its status, text, and class
name (the wire ``code``); every non-WebSocket surface — the FastAPI handler
under each protocol envelope, the progress runner as JSON and SSE, and the
polled preparation status — renders the typed and the bare case the same
way (the WebSocket frame is pinned beside its fixture in
``test_drowse_api.py``); and the validation sites a request reaches raise the
typed family, not a bare ``ValueError``.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from drowse.base.hf import HFError
from drowse.errors import (
    ConcurrentGenerationError,
    DrowseError,
    GenerationRequestError,
    ManifoldSpecError,
    PoisednessError,
    RequestError,
    SourceSpecError,
    SteeringCompositionError,
    SteeringExprError,
    UnknownManifoldLabelError,
)
from drowse.server.errors import describe_error, install_error_handlers
from drowse.server.operations import Preparation, ProgressCallback, register_operation_routes, run_operation
from drowse.store.selectors import SelectorError

SECRET = "/home/user/.drowse/models/x/neutral.safetensors: shape (3,) != (4,)"
SCRUBBED = "Job failed. Check the server log for details."


# ---------------------------------------------------------------------------
# describe_error: one row per family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("exc", "status", "text"), [
    (RequestError("top_k must be in [1, 256]"), 400, "top_k must be in [1, 256]"),
    (GenerationRequestError("n must be >= 1, got 0"), 400, "n must be >= 1, got 0"),
    (SourceSpecError("bad Hugging Face corpus 'x@'"), 400, "bad Hugging Face corpus 'x@'"),
    (ManifoldSpecError("discover manifold 'm': picked k=2 needs >= 5 nodes"), 400,
     "discover manifold 'm': picked k=2 needs >= 5 nodes"),
    (PoisednessError("RBF poisedness failure: 3 nodes"), 400, "RBF poisedness failure: 3 nodes"),
    (SteeringExprError("unexpected token"), 400, "unexpected token"),
    (SelectorError("bad selector"), 400, "bad selector"),
    (SteeringCompositionError("project_profile: no layers produced"), 422, "project_profile: no layers produced"),
    (UnknownManifoldLabelError("no node labeled 'x'"), 404, "no node labeled 'x'"),
    (ConcurrentGenerationError("already running"), 409, "already running"),
    (HFError("could not resolve immutable revision for org/x"), 502,
     "could not resolve immutable revision for org/x"),
    (ValueError(SECRET), 500, SCRUBBED),
    (KeyError(SECRET), 500, SCRUBBED),
    (TypeError(SECRET), 500, SCRUBBED),
])
def test_each_family_keeps_its_status_text_and_code(
    exc: Exception, status: int, text: str, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("DEBUG", logger="drowse.api"):
        info = describe_error(exc, failure="Job failed")
    assert (info.status, info.message, info.code) == (status, text, type(exc).__name__)
    if status == 500:
        # The traceback is the only place the bare text survives.
        [record] = caplog.records
        assert record.levelname == "ERROR" and record.exc_info is not None
    else:
        assert not caplog.records


def test_a_request_error_stays_a_value_error() -> None:
    """Every ``except ValueError`` written before the family keeps catching."""
    for cls in (RequestError, GenerationRequestError, SourceSpecError, ManifoldSpecError, PoisednessError):
        assert issubclass(cls, ValueError) and issubclass(cls, DrowseError)
        assert issubclass(cls, RequestError)


# ---------------------------------------------------------------------------
# Every surface: the typed case keeps its text, the bare case is scrubbed
# ---------------------------------------------------------------------------

_CASES = [
    pytest.param(SourceSpecError("unknown lens fetch source 'x'"), 400, "unknown lens fetch source 'x'", id="typed"),
    pytest.param(ValueError(SECRET), 500, SCRUBBED, id="bare"),
]


def _app() -> FastAPI:
    app = FastAPI()
    app.state.session = SimpleNamespace(model_id="model/test")
    app.state.session_lock = asyncio.Lock()
    install_error_handlers(app)
    register_operation_routes(app)
    return app


@pytest.mark.parametrize(("error", "status", "text"), _CASES)
@pytest.mark.parametrize(("path", "envelope"), [
    ("/drowse/v1/boom", lambda message, status: {"detail": message}),
    ("/api/boom", lambda message, status: {"error": message}),
    ("/v1/boom", lambda message, status: {"error": {
        "message": message,
        "type": "invalid_request_error" if status < 500 else "server_error",
        "param": None, "code": status,
    }}),
])
def test_handler_renders_both_cases_in_every_protocol_envelope(
    error: Exception, status: int, text: str, path: str,
    envelope: Callable[[str, int], dict[str, Any]],
) -> None:
    app = _app()

    @app.get(path)
    def boom() -> None:
        raise error

    resp = TestClient(app).get(path)
    message = "Request failed. Check the server log for details." if status == 500 else text
    assert resp.status_code == status
    assert resp.json() == envelope(message, status)
    assert "/home/user" not in resp.text


def _runner(error: Exception) -> TestClient:
    app = _app()

    async def job(_progress: ProgressCallback) -> dict[str, Any]:
        raise error

    @app.post("/drowse/v1/run")
    async def run_route(request: Request):
        return await run_operation(request, job, failure="Job failed")

    return TestClient(app)


@pytest.mark.parametrize(("error", "status", "text"), _CASES)
def test_progress_runner_renders_both_cases_as_json_and_sse(error: Exception, status: int, text: str) -> None:
    client = _runner(error)
    resp = client.post("/drowse/v1/run")
    assert (resp.status_code, resp.json()) == (status, {"detail": text})
    with client.stream("POST", "/drowse/v1/run", headers={"accept": "text/event-stream"}) as stream:
        frames = b"".join(stream.iter_bytes()).decode("utf-8")
    assert "event: error" in frames
    payload = json.loads(frames.split("data: ", 1)[1])
    assert payload == {"message": text, "code": type(error).__name__, "status": status}
    assert "/home/user" not in frames


@pytest.mark.parametrize(("error", "status", "text"), _CASES)
def test_polled_preparation_renders_both_cases(error: Exception, status: int, text: str) -> None:
    async def scenario() -> dict[str, Any]:
        prep = Preparation("fetch", label="Job", fields=("source",))

        async def body() -> None:
            raise error

        prep.start(body, message="starting…", source="neuronpedia")
        assert prep.task is not None
        await prep.task
        return prep.wire()

    status_doc = asyncio.run(scenario())
    assert status_doc["state"] == "error"
    assert status_doc["error"] == text
    assert "/home/user" not in json.dumps(status_doc)


# ---------------------------------------------------------------------------
# The validation sites a request reaches raise the typed family
# ---------------------------------------------------------------------------

class _CoveringWhitener:
    def covers_all(self, _layers: Any) -> bool:
        return True


def _parse_lens_corpus(*raw: str) -> Any:
    from drowse.lens.corpus import parse_lens_corpus

    return parse_lens_corpus(list(raw))


def _layers(text: str) -> Any:
    from drowse.model.layers import parse_layer_selector

    return parse_layer_selector(text)


def _model_layers(selector: Any, n_layers: int) -> Any:
    from drowse.model.layers import resolve_model_layers

    return resolve_model_layers(selector, n_layers)


def _lens_source(text: str) -> Any:
    from drowse.lens.sources import LENS_SOURCES

    return LENS_SOURCES.parse(text)


def _sae_source(text: str) -> Any:
    from drowse.sae.sources import SAE_SOURCES

    return SAE_SOURCES.parse(text)


def _lens_fit_request(**fields: Any) -> Any:
    from drowse.lens.lifecycle import LensFitRequest

    return LensFitRequest(**fields)


def _install(target: str, as_: str | None = None) -> Any:
    from drowse.store.install import install_manifold

    return install_manifold(target, as_)


def _rbf(node_params: torch.Tensor) -> Any:
    from drowse.geometry.rbf import RbfFitPlan

    return RbfFitPlan.build(node_params)


def _pca_coords(k: int) -> Any:
    from drowse.geometry.topology import derive_pca_coords

    return derive_pca_coords(torch.eye(k))


def _project_disjoint() -> Any:
    from drowse.geometry.fold import project_profile

    return project_profile({0: torch.ones(4)}, {1: torch.ones(4)}, "~", whitener=_CoveringWhitener())


@pytest.mark.parametrize(("call", "family", "match"), [
    (lambda: _parse_lens_corpus("org/data@"), SourceSpecError, "bad Hugging Face corpus"),
    (lambda: _parse_lens_corpus("org/data,weight=2"), SourceSpecError, "only to a mix"),
    (lambda: _parse_lens_corpus("org/data,files=x"), SourceSpecError, "take integers"),
    (lambda: _layers("1,x"), RequestError, "malformed layers"),
    (lambda: _model_layers([7], 4), RequestError, r"must lie in \[0, 4\)"),
    (lambda: _model_layers("sample", 4), RequestError, "unknown layer mode"),
    (lambda: _lens_source("bogus"), SourceSpecError, "source must be"),
    (lambda: _lens_source("local:Not A Name"), SourceSpecError, "invalid local"),
    (lambda: _sae_source("local:Not A Name"), SourceSpecError, "invalid local"),
    (lambda: _lens_fit_request(kind="Q"), RequestError, "lens kind must be"),
    (lambda: _lens_fit_request(seq_len=0), RequestError, "seq_len must be > 0"),
    (lambda: _install("not-a-coordinate"), SourceSpecError, "install target must be"),
    (lambda: _install("org/repo", "nonamespace"), RequestError, "as_ must be"),
    (lambda: _rbf(torch.zeros(2, 2)), PoisednessError, "RBF poisedness failure"),
    (lambda: _pca_coords(1), ManifoldSpecError, "needs >= 2 centroids"),
    (_project_disjoint, SteeringCompositionError, "no layers produced"),
])
def test_request_validation_raises_the_typed_family(
    call: Callable[[], Any], family: type[DrowseError], match: str,
) -> None:
    with pytest.raises(family, match=match):
        call()
