"""Native profile routes under ``/drowse/v1/sessions/{id}``.

The steering-profile registry (list / get), concept extraction, and the two
Mahalanobis comparison reads: the cross-layer ``pairwise`` matrix and the N×N
``correlation`` matrix across loaded profiles and probes.  Comparisons read
``session.analytics_profile`` — cached CPU snapshots built once under the
exclusive-GPU lock — so a polled endpoint never copies off the model device
while a model op runs.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from drowse.geometry.profile import Profile
from drowse.readout.analytics import cosine_matrix, cross_layer_cosine
from drowse.server.operations import ProgressCallback, run_operation
from drowse.server.profile_models import ExtractRequest, extract_registry_name, profile_to_json
from drowse.server.response_models import (
    CorrelationData,
    ExtractResponse,
    PairwiseCompareResponse,
    ProfileListResponse,
    VectorInfo,
)
from drowse.server.runtime import run_in_thread, session_router

_SNAPSHOT_BUSY = (
    "a model operation is in flight; the comparison snapshot is not ready yet — retry shortly"
)


def register_profile_routes(app: FastAPI) -> None:
    session = app.state.session
    router = session_router()

    def whitener_or_409(purpose: str) -> Any:
        whitener = session.whitener
        if whitener is None:
            raise HTTPException(
                409,
                f"{purpose} requires a Mahalanobis whitener; regenerate the neutral "
                "activation cache for this model",
            )
        return whitener

    @router.get("/profiles")
    def list_profiles() -> ProfileListResponse:
        return {"profiles": [
            profile_to_json(name, Profile(tensors)) for name, tensors in sorted(session.profiles.items())
        ]}

    # Registered before ``/profiles/{name}`` so the literal path wins.
    @router.get("/profiles/pairwise")
    def pairwise_compare(a: str, b: str) -> PairwiseCompareResponse:
        """Cross-layer whitened cosine matrix between two named profiles or probes.

        ``matrix[i][j]`` is the Mahalanobis cosine between ``a``'s layer
        ``layers_a[i]`` and ``b``'s layer ``layers_b[j]``, framed in ``a``'s
        row-layer covariance.  404 for an unknown name; 409 while a
        registered name's snapshot is not built yet, or when the whitener does
        not cover every layer of ``a``.
        """
        prof_a, prof_b = session.analytics_profile(a), session.analytics_profile(b)
        if prof_a is None or prof_b is None:
            known = set(session.analytics_names())
            missing = [n for n, p in ((a, prof_a), (b, prof_b)) if p is None and n not in known]
            if missing:
                raise HTTPException(404, f"names not loaded: {missing}")
            raise HTTPException(409, _SNAPSHOT_BUSY)
        whitener = whitener_or_409("pairwise compare")
        if not whitener.covers_all(prof_a.keys()):
            raise HTTPException(
                409,
                f"pairwise compare requires a whitener covering every layer of '{a}'; "
                "regenerate the neutral activation cache for this model",
            )
        layers_a, layers_b, matrix = cross_layer_cosine(prof_a, prof_b, whitener)
        return {
            "a": a, "b": b, "metric": "mahalanobis",
            "layers_a": layers_a, "layers_b": layers_b, "matrix": matrix,
            "model": session.model_id,
        }

    @router.get("/profiles/{name}")
    def get_profile(name: str) -> VectorInfo:
        if name not in session.profiles:
            raise HTTPException(404, f"profile '{name}' not found")
        return profile_to_json(name, Profile(session.profiles[name]))

    @router.get("/correlation")
    def correlation_matrix(names: str | None = None) -> CorrelationData:
        """N×N Mahalanobis cosine across loaded profiles and active probes.

        ``?names=a,b,c`` restricts the matrix; by default it covers every
        registered profile and attached probe (a profile wins a name
        collision).  A name whose snapshot is not ready, or a pair the
        whitener does not cover, reads ``null`` until a later poll.
        """
        available = session.analytics_names()
        ordered = available
        if names is not None and names.strip():
            ordered = [n.strip() for n in names.split(",") if n.strip()]
            missing = [n for n in ordered if n not in available]
            if missing:
                raise HTTPException(404, f"names not loaded: {missing}")
        pool = {
            name: snap for name in ordered if (snap := session.analytics_profile(name)) is not None
        }
        matrix, layers_shared = cosine_matrix(pool, ordered, whitener_or_409("correlation"))
        return {"names": ordered, "matrix": matrix, "layers_shared": layers_shared}

    @router.post("/extract", response_model=ExtractResponse)
    async def extract_profile(req: ExtractRequest, request: Request):
        """Fit a concept and register its folded profile on the live session.

        SSE progress on ``Accept: text/event-stream``, one JSON body otherwise
        (which also carries the collected ``progress`` lines).
        """
        if req.kind == "custom" and not req.custom_system:
            raise HTTPException(
                400, "kind='custom' requires custom_system (a system template with a {c} placeholder)",
            )

        async def job(on_progress: ProgressCallback) -> ExtractResponse:
            canonical, profile = await run_in_thread(
                session.extract, req.concept, req.baseline,
                kind=req.kind, custom_system=req.custom_system, on_progress=on_progress,
                sae=req.sae, role=req.role, namespace=req.namespace, force=req.force,
            )
            name = extract_registry_name(canonical, req.namespace)
            session.registry.put_profile(name, profile)
            return {"done": True, "profile": profile_to_json(name, profile), "canonical": name}

        return await run_operation(
            request, job, failure="Extract failed", progress_key="progress",
        )

    app.include_router(router)
