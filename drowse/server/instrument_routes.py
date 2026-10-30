"""Native ``/instruments`` routes — the unified read-side surface.

One route tree under ``/drowse/v1/sessions/{id}/instruments`` over the three
read families ``session.instruments`` registers (``geometry`` / ``lens`` /
``sae``):

- ``POST {family}/live`` — uniform live toggle.
- ``GET {family}/sources`` — prepared sources (lens) / prepared sources plus
  provider release candidates (sae).
- ``PUT {family}/source`` — synchronous source switch (lens only).
- ``POST/GET/DELETE {family}/preparations`` — polled background preparations
  (lens ``fetch``/``fit``, sae ``fetch``/``train``), one status shape.
- ``GET {family}/token-readout`` — the loom token-drilldown replay, wrapped in
  the ``measurements`` replay envelope the instrument builds.
- Family extras: ``lens/token/validate``, ``sae/features/validate``,
  ``sae/features/metadata``.

What a family supports is declared once in :data:`CAPABILITIES`; the same
declaration rides every ``session_info`` instrument block.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from pydantic import Field

from drowse.lens.jlens import JacobianLensCancelled, LensNotFittedError, resolve_token_atom
from drowse.lens.layers import parse_layer_selector
from drowse.readout.measurements import MeasurementsEnvelope
from drowse.sae.training import SaeTrainingCancelled
from drowse.server.operations import Preparation, PreparationGroup
from drowse.server.response_models import (
    InstrumentCapabilities,
    InstrumentFamilyBlock,
    InstrumentLiveState,
    InstrumentSourceJSON,
    LensTokenValidationJSON,
    PreparationStatusJSON,
    SaeFeatureMetaResponse,
    SaeFeatureValidationJSON,
    SaeReleaseJSON,
    SourceSwitchResponse,
    SourcesResponse,
)
from drowse.server.runtime import NativeRequest, run_in_thread, session_lock, session_router

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class FamilyCapabilities:
    """What one read family supports over HTTP, and why not when it doesn't.

    ``*_error`` is the ``(status, message)`` an unsupported operation returns.
    """

    sources: bool
    preparations: tuple[str, ...]
    token_readout: bool
    source_switch: bool
    sources_error: "tuple[int, str] | None" = None
    source_switch_error: "tuple[int, str] | None" = None
    preparations_error: "tuple[int, str] | None" = None

    def to_dict(self) -> InstrumentCapabilities:
        return {
            "sources": self.sources,
            "preparations": list(self.preparations),
            "token_readout": self.token_readout,
            "source_switch": self.source_switch,
        }


#: Per-family capability declaration, keyed like ``session.instruments``.
CAPABILITIES: dict[str, FamilyCapabilities] = {
    "geometry": FamilyCapabilities(
        sources=False,
        preparations=(),
        token_readout=True,
        source_switch=False,
        sources_error=(
            404,
            "geometry has no source lifecycle (Monitor probes attach directly; "
            "there is nothing to fetch or switch)",
        ),
        source_switch_error=(404, "geometry has no source to switch"),
        preparations_error=(404, "geometry has no background preparations"),
    ),
    "lens": FamilyCapabilities(
        sources=True,
        preparations=("fetch", "fit"),
        token_readout=True,
        source_switch=True,
    ),
    "sae": FamilyCapabilities(
        sources=True,
        preparations=("fetch", "train"),
        token_readout=True,
        source_switch=False,
        source_switch_error=(
            409,
            "switching the SAE source loads weights — run it as a background "
            "preparation (POST .../instruments/sae/preparations with operation='fetch')",
        ),
    ),
}


def family_block(session: Any, family: str) -> InstrumentFamilyBlock:
    """``{family, live, source, probes, capabilities}`` — the one per-family wire block.

    ``source`` is ``instrument.active_source``, the same resolver that stamps
    every measurement binding, so the block agrees with persisted rows.
    """
    instrument = session.instruments[family]
    return {
        "family": family,
        "live": instrument.live_state.to_dict(),
        "source": instrument.active_source,
        "probes": list(instrument.names),
        "capabilities": CAPABILITIES[family].to_dict(),
    }


def instrument_families(session: Any) -> list[InstrumentFamilyBlock]:
    """Every read family's block, in registry order."""
    return [family_block(session, family) for family in session.instruments]


def _capabilities(session: Any, family: str) -> FamilyCapabilities:
    """404 unless ``family`` names a registered read instrument."""
    if family not in session.instruments:
        raise HTTPException(
            404,
            f"unknown instrument family {family!r} (want one of {', '.join(session.instruments)})",
        )
    return CAPABILITIES[family]


def _refuse(error: "tuple[int, str] | None", fallback: str) -> None:
    status, message = error if error is not None else (404, fallback)
    raise HTTPException(status, message)


def _parse_layers(layers: str | None) -> list[int] | str | None:
    """``"3,7,11"`` → ``[3, 7, 11]``; ``workspace`` / ``sample`` / ``all`` pass
    through (:func:`~drowse.lens.layers.parse_layer_selector`)."""
    try:
        return parse_layer_selector(layers)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class InstrumentLiveRequest(NativeRequest):
    """Body for ``POST .../instruments/{family}/live``.

    ``layers`` applies to the lens.  Readout width is generation state shared
    with ``return_top_k``, so ``top_k`` is refused for every family.
    """

    enabled: bool
    layers: list[int] | None = None
    top_k: int | None = Field(default=None, ge=1, le=256)


class SourceRequest(NativeRequest):
    source: str = Field(min_length=1)


class LensTokenValidationRequest(NativeRequest):
    word: str


class SaeFeatureRequest(NativeRequest):
    id: int = Field(ge=0)


class SaeFeatureMetaRequest(NativeRequest):
    ids: list[int] = Field(min_length=1, max_length=64)


class PreparationRequest(NativeRequest):
    """The ``{operation, …fields}`` envelope; fields re-parse per operation."""

    model_config = NativeRequest.model_config | {"extra": "allow"}

    operation: str


class LensFetchRequest(NativeRequest):
    source: str = "neuronpedia"
    force: bool = False


class LensFitRequest(NativeRequest):
    """Defaults mirror CLI ``lens fit``.  A matching partial fit resumes unless
    ``force``; ``hf_corpus`` lists ``REPO[:CONFIG][@REV][,KEY=VALUE…]``
    sources (several make a weighted mix), else the default web-text sample."""

    prompts: int = Field(default=100, ge=1, le=5000)
    hf_corpus: list[str] | None = Field(default=None, min_length=1, max_length=16)
    seq_len: int | None = Field(default=None, ge=32, le=4096)
    prompt_batch: int | None = Field(default=None, ge=1, le=64)
    layers: str = "all"
    force: bool = False
    relp: bool = True


class SaeFetchRequest(NativeRequest):
    release: str = Field(min_length=1)
    layer: int | None = Field(default=None, ge=0)


class SaeTrainRequest(NativeRequest):
    name: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,63}$")
    layer: int | None = Field(default=None, ge=0)
    tokens: int = Field(default=1_000_000, ge=1, le=100_000_000)
    seq_len: int = Field(default=128, ge=8, le=4096)
    batch_size: int = Field(default=8, ge=1, le=256)
    width: int | None = Field(default=None, ge=1)
    expansion: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=3e-4, gt=0)
    l1: float = Field(default=1e-3, ge=0)
    dead_threshold: float = Field(default=1e-6, ge=0)
    seed: int = 0
    force: bool = False


#: ``fit_jacobian_lens`` per-prompt progress — "prompt 12/100 (…)".
_FIT_PROGRESS_RE = re.compile(r"prompt (\d+)/(\d+)")
#: ``train_residual_sae`` token progress — "trained 12,345/1,000,000 tokens".
_TRAIN_PROGRESS_RE = re.compile(r"trained ([\d,]+)/([\d,]+) tokens")


def register_instrument_routes(app: FastAPI) -> None:
    session = app.state.session
    router = session_router()

    lens_fetch = Preparation("fetch", label="J-lens fetch", fields=("source", "live_layers"))
    lens_fit = Preparation(
        "fit", label="lens fit", fields=("live_layers",),
        cancel_exc=JacobianLensCancelled, progress_unit="prompts", progress_pattern=_FIT_PROGRESS_RE,
    )
    sae_fetch = Preparation("fetch", label="SAE fetch", fields=("release", "info"))
    sae_train = Preparation(
        "train", label="SAE training", fields=("name", "info"),
        cancel_exc=SaeTrainingCancelled, progress_unit="tokens", progress_pattern=_TRAIN_PROGRESS_RE,
    )
    preparations = {
        "lens": PreparationGroup("a J-lens artifact operation is already running", lens_fetch, lens_fit),
        "sae": PreparationGroup("an SAE artifact operation is already running", sae_fetch, sae_train),
    }

    async def stop_preparations() -> None:
        for group in preparations.values():
            await group.stop()

    app.router.on_shutdown.append(stop_preparations)

    async def enable_lens_live() -> list[int]:
        return list((await run_in_thread(session.lens.set_live, True)).layers or ())

    async def enable_live_after(preparation: Preparation, enable: Any) -> None:
        """Best-effort: turn the family's live readout on once its artifact landed.

        The preparation itself succeeded, so a busy lock or a live-toggle
        failure is logged rather than reported as a failed preparation.
        """
        try:
            async with session_lock(app):
                await enable()
        except Exception:  # noqa: BLE001 - the artifact is in place; live is optional
            log.exception("could not enable live readout after %s", preparation.label)

    async def activate_lens_source(source: str) -> list[int]:
        async with session_lock(app):
            session.lens.set_live(False)
            try:
                await run_in_thread(session.lens.select_source, source)
            except FileNotFoundError as exc:
                if exc.filename is not None:
                    raise  # an OS-level miss names a path: scrubbed by the policy
                # The source registry's own "<kind>:<name> is not available".
                raise HTTPException(404, str(exc)) from None
            return await enable_lens_live()

    # -- live / sources / source -------------------------------------------

    @router.post("/instruments/{family}/live")
    async def instrument_live(family: str, body: InstrumentLiveRequest) -> InstrumentLiveState:
        """``instrument.set_live(enabled, **extras)``; the instrument rejects an
        extra it cannot honor (``TypeError`` → 400)."""
        _capabilities(session, family)
        if body.top_k is not None:
            raise HTTPException(
                400, "live takes no top_k; readout width follows the generation's alternatives (return_top_k)",
            )
        extras = {"layers": body.layers} if body.layers is not None else {}
        async with session_lock(app):
            try:
                state = await run_in_thread(session.instruments[family].set_live, body.enabled, **extras)
            except TypeError as e:
                raise HTTPException(400, str(e)) from e
        return state.to_dict()

    @router.get("/instruments/{family}/sources")
    async def instrument_sources(family: str) -> SourcesResponse:
        caps = _capabilities(session, family)
        if not caps.sources:
            _refuse(caps.sources_error, f"{family} has no source lifecycle")
        if family == "lens":
            from drowse.lens.sources import list_lens_sources

            rows = list_lens_sources(session.model_id)
            return {"sources": [
                cast(InstrumentSourceJSON, {k: v for k, v in row.items() if k != "path"}) for row in rows
            ]}
        from drowse.sae.backend import list_sae_releases
        from drowse.sae.sources import list_sae_sources

        rows = await run_in_thread(list_sae_sources, session.model_id)
        releases = await run_in_thread(list_sae_releases, session.model_id)
        return {
            "sources": [
                cast(InstrumentSourceJSON, {k: v for k, v in row.items() if k != "path"}) for row in rows
            ],
            "releases": cast(list[SaeReleaseJSON], releases),
        }

    @router.put("/instruments/{family}/source")
    async def instrument_source(family: str, body: SourceRequest) -> SourceSwitchResponse:
        caps = _capabilities(session, family)
        if not caps.source_switch:
            _refuse(caps.source_switch_error, f"{family} has no synchronous source switch")
        preparations["lens"].refuse_if_busy()
        try:
            layers = await activate_lens_source(body.source)
        except LensNotFittedError as exc:
            # The route exists and the source is selected, but its artifact
            # cannot load for this model: a conflict with the artifact state.
            raise HTTPException(409) from exc
        return {"source": body.source, "live_layers": layers}

    # -- preparations ---------------------------------------------------------

    async def lens_fetch_body(body: LensFetchRequest) -> None:
        from drowse.lens.sources import fetch_lens_source

        lens_fetch.report("fetching external lens into the Hugging Face cache…")
        binding = await run_in_thread(
            fetch_lens_source, session.model_id, body.source, force=body.force, activate=False,
        )
        lens_fetch.report("activating…")
        lens_fetch.state["live_layers"] = await activate_lens_source(binding.name)
        lens_fetch.report(
            f"active: {binding.name} ({len(binding.files[binding.key].layers)} layers)"
        )

    def start_lens_fetch(fields: dict[str, Any]) -> None:
        from drowse.lens.sources import LENS_PROVIDERS

        body = LensFetchRequest(**fields)
        if body.source not in LENS_PROVIDERS:
            raise HTTPException(
                400, "J-lens source must be one of " + ", ".join(sorted(LENS_PROVIDERS)),
            )
        lens_fetch.start(lambda: lens_fetch_body(body), message="starting…", source=body.source, live_layers=None)

    async def lens_fit_body(body: LensFitRequest, source_layers: list[int] | str, corpus: Any) -> None:
        from drowse.lens.corpus import stream_lens_corpus

        lens_fit.report(f"streaming {body.prompts} documents from {corpus}…")
        docs, spec = await run_in_thread(
            stream_lens_corpus, body.prompts, corpus, cancel_event=lens_fit.cancel_event,
        )
        lens_fit.report("fitting…")
        await run_in_thread(
            session.lens.fit, docs,
            kind="R" if body.relp else "J",
            corpus_spec=spec, source_layers=source_layers, seq_len=body.seq_len,
            prompt_batch=body.prompt_batch, force=body.force,
            on_progress=lens_fit.report, cancel_event=lens_fit.cancel_event,
        )
        async def enable() -> None:
            lens_fit.state["live_layers"] = await enable_lens_live()

        await enable_live_after(lens_fit, enable)
        lens_fit.report("done")

    def start_lens_fit(fields: dict[str, Any]) -> None:
        from drowse.lens.corpus import parse_lens_corpus

        body = LensFitRequest(**fields)
        corpus = parse_lens_corpus(body.hf_corpus)
        source_layers = _parse_layers(body.layers) or "workspace"
        if source_layers == "sample":
            raise HTTPException(
                400,
                "layers='sample' is not fittable (debug readout only) — "
                "use 'workspace', 'all', or an explicit csv list",
            )
        lens_fit.start(
            lambda: lens_fit_body(body, source_layers, corpus),
            message="starting…", current=0, total=body.prompts,
        )

    async def sae_fetch_body(body: SaeFetchRequest, source: str, release: str) -> None:
        async with session_lock(app):
            info = await run_in_thread(session.load_sae, release, layer=body.layer)
            await run_in_thread(session.sae.set_live, True)
        sae_fetch.state["info"] = info
        sae_fetch.report(
            f"loaded {source} · live at L{info.get('layer')} ({info.get('width')} features)",
        )

    def start_sae_fetch(fields: dict[str, Any]) -> None:
        body = SaeFetchRequest(**fields)
        source = body.release.strip()
        release = source.removeprefix("saelens:")
        if not release:
            raise HTTPException(400, "SAE source must not be empty")
        sae_fetch.start(
            lambda: sae_fetch_body(body, source, release),
            message=f"loading {source}", release=source, info=None,
        )

    async def sae_train_body(body: SaeTrainRequest, layer: int) -> None:
        from drowse.lens.corpus import stream_lens_corpus

        n_docs = max(1, math.ceil(body.tokens / body.seq_len))
        sae_train.report(f"streaming {n_docs:,} corpus documents…")
        docs, spec = await run_in_thread(stream_lens_corpus, n_docs)
        result = await run_in_thread(
            session.train_sae, body.name, docs,
            layer=layer, corpus_spec=spec, tokens=body.tokens, seq_len=body.seq_len,
            batch_size=body.batch_size, d_sae=body.width, expansion=body.expansion,
            learning_rate=body.learning_rate, l1_coefficient=body.l1,
            dead_feature_threshold=body.dead_threshold, seed=body.seed, force=body.force,
            on_progress=sae_train.report, cancel_event=sae_train.cancel_event,
        )
        sae_train.state["info"] = result["runtime"]
        sae_train.state["current"] = int(result["metrics"]["tokens_trained"])
        sae_train.report(f"active: {result['source']}")
        await enable_live_after(sae_train, lambda: run_in_thread(session.sae.set_live, True))

    def start_sae_train(fields: dict[str, Any]) -> None:
        body = SaeTrainRequest(**fields)
        n_layers = len(session.layers)
        layer = body.layer if body.layer is not None else round(0.65 * max(n_layers - 1, 0))
        if not 0 <= layer < n_layers:
            raise HTTPException(400, f"SAE layer {layer} is outside model layers 0..{n_layers - 1}")
        sae_train.start(
            lambda: sae_train_body(body, layer),
            message="starting…", name=body.name, current=0, total=body.tokens, info=None,
        )

    #: Starters keyed by ``(family, operation)`` — the CLI verbs on both surfaces.
    starters = {
        ("lens", "fetch"): start_lens_fetch,
        ("lens", "fit"): start_lens_fit,
        ("sae", "fetch"): start_sae_fetch,
        ("sae", "train"): start_sae_train,
    }

    def preparation_group(family: str) -> PreparationGroup:
        caps = _capabilities(session, family)
        if not caps.preparations:
            _refuse(caps.preparations_error, f"{family} has no background preparations")
        return preparations[family]

    @router.post("/instruments/{family}/preparations", status_code=202)
    async def preparations_start(family: str, body: PreparationRequest) -> PreparationStatusJSON:
        group = preparation_group(family)
        starter = starters.get((family, body.operation))
        if starter is None:
            valid = [op for (fam, op) in starters if fam == family]
            raise HTTPException(400, f"{family} preparations support {valid}, not {body.operation!r}")
        group.refuse_if_busy()
        starter(body.model_extra or {})
        return cast(PreparationStatusJSON, group.status())

    @router.get("/instruments/{family}/preparations")
    def preparations_status(family: str) -> PreparationStatusJSON:
        return cast(PreparationStatusJSON, preparation_group(family).status())

    @router.delete("/instruments/{family}/preparations")
    async def preparations_cancel(family: str) -> PreparationStatusJSON:
        """lens: cancel a running fit.  sae: cancel a running train, else unload the resident SAE."""
        group = preparation_group(family)
        if family == "lens":
            lens_fit.request_cancel()
        elif sae_train.running:
            sae_train.request_cancel()
        else:
            async with session_lock(app):
                session.unload_sae()
        return cast(PreparationStatusJSON, group.status())

    # -- token readout ----------------------------------------------------------

    @router.get("/instruments/{family}/token-readout")
    async def instrument_token_readout(
        family: str,
        node_id: str,
        raw_index: int,
        top_k: int | None = None,
        steered: bool = True,
        raw: bool = False,
        layers: str | None = None,
    ) -> MeasurementsEnvelope:
        """The instrument owns the whole replay, envelope included; a knob it
        cannot honor is a ``ValueError`` → 400, never silently dropped."""
        caps = _capabilities(session, family)
        if not caps.token_readout:
            raise HTTPException(404, f"{family} has no token readout")
        async with session_lock(app):
            return await run_in_thread(
                session.instruments[family].token_readout,
                node_id, raw_index,
                top_k=top_k, layers=_parse_layers(layers), apply_steering=steered, raw=raw,
            )

    # -- family extras ---------------------------------------------------------

    @router.post("/instruments/lens/token/validate")
    def validate_lens_token(body: LensTokenValidationRequest) -> LensTokenValidationJSON:
        """Read-only single-token check for the J-lens steer/probe forms
        (``word`` or ``#<id>``, the same contract as the atoms)."""
        word = body.word.strip()
        if not word:
            raise HTTPException(400, "word must not be empty")
        vocab_size = int(session.lens.runtime.readout_modules()[0].shape[0])
        return {
            "word": word,
            "token_id": resolve_token_atom(session.tokenizer, word, vocab_size=vocab_size),
        }

    @router.post("/instruments/sae/features/validate")
    def sae_feature_validate(body: SaeFeatureRequest) -> SaeFeatureValidationJSON:
        return session.validate_sae_feature(body.id)

    @router.post("/instruments/sae/features/metadata")
    async def sae_features_metadata(body: SaeFeatureMetaRequest) -> SaeFeatureMetaResponse:
        """Neuronpedia label + ``maxActApprox`` backfill.  Network and disk
        cache only, so it takes no session lock."""
        if any(feature_id < 0 for feature_id in body.ids):
            raise HTTPException(400, "feature ids must be non-negative")
        return {"features": await run_in_thread(session.fetch_sae_feature_meta, body.ids)}

    app.include_router(router)
