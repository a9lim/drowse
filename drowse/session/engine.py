"""The session's generation engine: one call from kwargs to a finalized result.

:class:`GenerationEngine` owns everything a single generation needs that
outlives one call: the decode loop's :class:`GenerationState`, the
:class:`GenState` phase, the prefix KV cache, the compile/StaticCache routing
(:class:`~drowse.generation.compile_runtime.CompiledRuntime`), and the last
result.  One generation runs in stages:

1. :meth:`GenerationEngine.resolve_call` — the call's kwargs, the cast roster,
   a regen's recipe override and the same-role continuation rule fold into
   one frozen :class:`~drowse.generation.types.CallSpec`.
2. :meth:`GenerationEngine._plan_consumers` — which per-token consumers this
   generation feeds; the session's read plane turns that into a capture plan.
3. :class:`TokenTap` — the decode loop's per-token callback: logprob rollup,
   the token's measurement payload, the loom token row, and one
   :class:`~drowse.generation.results.TokenEvent` (or the public six-argument
   callback) per token.
4. :meth:`GenerationEngine.decode` — prefix-cache hit, StaticCache /
   compiled-module routing, and the decode loop itself.
5. :meth:`GenerationEngine.finalize` — the result, recorded as ``last_result``.

Everything runs inside :meth:`GenerationEngine.transaction`, the single-flight
guard with an exception-hard teardown.

The engine calls the session's read plane (``_resolve_read_demand``,
``_begin_capture``, ``_end_capture`` and the capture state they set) and its
prompt-render and loom-node helpers through the ``session`` it was built
with; those move to their own collaborators separately.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from drowse.errors import ConcurrentGenerationError, DrowseError
from drowse.events import GenerationFinished, GenerationStarted
from drowse.generation.compile_runtime import CompiledRuntime
from drowse.generation.decode import (
    GenerationConfig,
    GenerationState,
    build_chat_input,
    generate_steered,
)
from drowse.generation.finalize import finalize_generation
from drowse.generation.prefix_cache import PrefixCache
from drowse.generation.results import GenerationResult, TokenAlt, TokenEvent
from drowse.generation.sampling import SamplingConfig
from drowse.generation.static_cache import make_static_cache
from drowse.generation.token_callback import (
    TokenCallback,
    TokenConsumer,
    TokenConsumerOptions,
    consumer_options,
)
from drowse.generation.token_payloads import TokenProbePayload
from drowse.generation.types import (
    CallSpec,
    GenerationStream,
    GenerationTransaction,
    GenState,
)
from drowse.loom.recipe import (
    INHERIT_SYSTEM_PROMPT,
    ComposedCall,
    InheritSystemPrompt,
    Recipe,
    compose_call,
)
from drowse.steering.terms import Steering

if TYPE_CHECKING:
    from drowse.generation.results import ProbeReading
    from drowse.readout.capture_buffer import HiddenCapture
    from drowse.session.session import DrowseSession, _AuthoredPromptCapture
    from drowse.steering.hooks import SteeringManager

_log = logging.getLogger(__name__)

#: Shared logit / J-lens / SAE top-k width when logit alternatives are off.
DEFAULT_READOUT_TOP_K = 8


class PrefixCacheStateError(RuntimeError, DrowseError):
    """``cache_prefix`` was called in a state where its prefill would be wrong."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


@dataclass(frozen=True, slots=True)
class ConsumerPlan:
    """Which per-token consumers one generation feeds, decided once.

    ``live_scores_on`` is the geometry instrument's live toggle (off masks
    every per-token monitor consumer at the source).  ``lens`` / ``sae`` mean
    the family's live readout is on and the callback asked for it.
    ``need_tap`` is false when nothing would consume a per-token callback, so
    the decode loop skips the tap (and the per-step log-softmax it forces).
    """

    live_scores_on: bool
    has_monitor_probes: bool
    wants_live_token_scores: bool
    persists_layer_scores: bool
    persists_subspace_coords: bool
    persists_probe_row: bool
    lens: bool
    sae: bool
    perplexity_requested: bool
    need_tap: bool


class TokenTap:
    """One generation's per-token callback (the decode loop's ``on_token``).

    Per token it rolls up chosen-token logprobs, builds the token's
    measurement payload (geometry readings plus the live lens/SAE readouts),
    appends the loom token row, and hands the consumer either one
    :class:`TokenEvent` (a :class:`TokenConsumer`) or the public six-argument
    :data:`TokenCallback`.  Everything that is constant for the generation is
    resolved here once, after capture is attached and the instrument runs are
    bound, so the per-token path does no lookups it can avoid.
    """

    def __init__(
        self,
        session: "DrowseSession",
        *,
        state: GenerationState,
        consumer: "TokenCallback | TokenConsumer | None",
        plan: ConsumerPlan,
        assistant_node_id: str | None,
        collect_logprobs: bool,
        readout_top_k: int,
        recipe_steering: str | None,
    ) -> None:
        self.logprobs: list[tuple[int, float | None, list[Any]]] | None = (
            [] if collect_logprobs else None
        )
        self._lp_sum = 0.0
        self._lp_count = 0
        self._lp_missing = False
        self._index = 0
        self._state = state
        self._node_id = assistant_node_id
        self._append_token = (
            session.tree.append_token if assistant_node_id is not None else None
        )
        if isinstance(consumer, TokenConsumer):
            self._on_event: Callable[[TokenEvent], None] | None = consumer.on_event
            self._callback: TokenCallback | None = None
            self._probe_readings = consumer.options.probe_readings
        else:
            self._on_event = None
            self._callback = consumer
            self._probe_readings = False

        # Geometry: one monitor reading per token, from the capture sink's
        # row when it already scored this forward, else from the latest
        # captured slices through the bound geometry run.
        self._needs_scores = bool(
            plan.live_scores_on
            and plan.has_monitor_probes
            and (
                assistant_node_id is not None
                or plan.wants_live_token_scores
                or plan.persists_subspace_coords
            )
        )
        self._per_layer = bool(
            assistant_node_id is not None and plan.persists_layer_scores
        )
        self._capture = session._capture
        self._capture_state = session._capture_state
        self._incremental_rows = session._incremental_readings
        self._geometry_run = session.geometry.current_run

        lens, sae = session.lens, session.sae
        self._lens = lens
        self._sae = sae
        self._lens_step = session._live_lens_readout_step if plan.lens else None
        self._sae_step = session._live_sae_readout_step if plan.sae else None
        self._readout_top_k = readout_top_k
        self._lens_source = lens.live.get("source") if lens.live is not None else None
        self._sae_source = sae.live.get("source") if sae.live is not None else None
        self._sae_layer = int(sae.live["layer"]) if sae.live is not None else None
        self._recipe_steering = recipe_steering

    def mean_logprob(self) -> tuple[float | None, float | None]:
        """``(mean_logprob, mean_surprise)`` over the response span.

        ``None`` when nothing was captured or any response token lacked a
        logprob (a partial mean would mis-rank siblings).
        """
        if self._lp_count == 0 or self._lp_missing:
            return None, None
        mean = self._lp_sum / self._lp_count
        return mean, -mean

    def _geometry_readings(self, step_id: int) -> "dict[str, ProbeReading] | None":
        state = self._capture_state
        rows = self._incremental_rows
        if (state.incremental or state.lean) and rows:
            # The capture sink already scored this forward (rows align 1:1
            # with forwards): full rows are complete readings, lean rows
            # deliberately coords-only.  Read the row, never rescore.
            return rows[-1] or None
        latest = {
            layer: bucket[-1]
            for layer, bucket in self._capture.per_layer_buckets().items()
            if bucket
        }
        if not latest:
            return None
        return self._geometry_run.observe(step_id, latest) or None

    def __call__(
        self,
        text: str,
        is_thinking: bool,
        tid: int | None,
        lp: float | None,
        top_alts: list[TokenAlt] | None,
        perplexity: float | None = None,
        step_id: int = -1,
    ) -> None:
        # ``step_id`` is the decode loop's forward index — the same value
        # this forward's gate callback received, so the lens/SAE display
        # steps can reuse the gate's step-keyed stash rows.
        response_token = tid is not None and tid >= 0 and not is_thinking
        if response_token:
            if self.logprobs is not None:
                self.logprobs.append((tid, lp, top_alts or []))
            if lp is None:
                self._lp_missing = True
            else:
                self._lp_sum += lp
                self._lp_count += 1

        payload: TokenProbePayload | None = None
        if self._needs_scores:
            readings = self._geometry_readings(step_id)
            if readings:
                payload = TokenProbePayload.from_geometry(
                    readings, per_layer=self._per_layer,
                )
        lens_step = (
            self._lens_step(top_k=self._readout_top_k, step_id=step_id)
            if self._lens_step is not None else None
        )
        sae_step = (
            self._sae_step(top_k=self._readout_top_k, step_id=step_id)
            if self._sae_step is not None else None
        )
        # Pinned lens/SAE probes tick on the same step (their readings come
        # out of the readout step) and merge into the same payload, so the
        # loom row, the event and the WS frame carry them uniformly.
        if lens_step is not None and self._lens.last_step_readings:
            if payload is None:
                payload = TokenProbePayload()
            payload.merge_readings(
                self._lens.last_step_readings, family="lens",
                per_layer=self._per_layer,
            )
        if sae_step is not None and self._sae.last_step_readings:
            if payload is None:
                payload = TokenProbePayload()
            payload.merge_readings(
                self._sae.last_step_readings, family="sae",
                per_layer=self._per_layer,
            )

        lens_payload = lens_step[0] if lens_step is not None else None
        lens_aggregate = lens_step[1] if lens_step is not None else None
        measurements: dict[str, Any] | None = None
        if (
            lens_payload
            or lens_aggregate
            or sae_step
            or (
                payload is not None
                and (payload.scores or payload.per_layer_scores or payload.has_readings)
            )
        ):
            if payload is None:
                payload = TokenProbePayload()
            measurements = payload.to_token_payload(
                lens=lens_payload,
                lens_aggregate=lens_aggregate,
                lens_token_ids=(
                    lens_step[2]
                    if lens_step is not None and len(lens_step) > 2 else None
                ),
                lens_source=self._lens_source,
                sae=sae_step,
                sae_source=self._sae_source,
                sae_layer=self._sae_layer,
                steering=self._recipe_steering,
            ).get("measurements")

        # The decode loop records this emission's ``generated_ids`` index just
        # before the tap runs: the join key a logit fork slices
        # ``raw_token_ids`` with.
        emit_map = self._state.emit_map
        raw_index = int(emit_map[-1][0]) if emit_map else None

        if self._append_token is not None and tid is not None:
            row: dict[str, Any] = {
                "token_id": int(tid),
                "text": text,
                "logprob": float(lp) if lp is not None else None,
                "perplexity": float(perplexity) if perplexity is not None else None,
            }
            if raw_index is not None:
                row["raw_index"] = raw_index
            if top_alts:
                row["top_alts"] = [
                    {"id": int(a.id), "text": a.text, "logprob": float(a.logprob)}
                    for a in top_alts
                ]
            if payload is not None and payload.scores:
                row["probes"] = {
                    name: round(float(value), 6)
                    for name, value in payload.scores.items()
                }
            if payload is not None and payload.per_layer_scores:
                row["per_layer_scores"] = payload.per_layer_scores
            if measurements:
                row["measurements"] = measurements
            self._append_token(self._node_id, row, thinking=bool(is_thinking))

        if self._on_event is not None:
            self._on_event(TokenEvent(
                text=text,
                token_id=tid if tid is not None else -1,
                index=self._index,
                thinking=is_thinking,
                logprob=lp,
                top_alts=top_alts,
                # The cross-family ``ProbeReading`` projection is the
                # compat channel; it runs only for a consumer that asked.
                probe_readings=(
                    payload.all_readings
                    if self._probe_readings
                    and payload is not None
                    and payload.has_readings
                    else None
                ),
                perplexity=perplexity,
                measurements=measurements,
                raw_index=raw_index,
            ))
            self._index += 1
        elif self._callback is not None:
            self._callback(text, is_thinking, tid, lp, top_alts, perplexity)


class _WorkerStream:
    """One generation on a worker thread, iterated as its ``TokenEvent``s.

    Closing the stream (or exhausting it) cancels the worker through the
    generation state's thread-local cancellation scope, joins it, and
    re-raises the worker's exception when it produced no result.
    """

    def __init__(
        self,
        run: Callable[[TokenConsumer], GenerationResult],
        options: TokenConsumerOptions,
        state: GenerationState,
    ) -> None:
        self._events: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._done = object()
        self._cancelled = threading.Event()
        self._result: GenerationResult | None = None
        self._error: BaseException | None = None
        self._finished = False
        consumer = TokenConsumer(self._events.put, options)

        def _worker() -> None:
            try:
                with state.cancellation_scope(self._cancelled):
                    self._result = run(consumer)
            except BaseException as exc:
                self._error = exc
            finally:
                self._events.put(self._done)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        self._iterator = self._iterate()

    def _iterate(self) -> Iterator[TokenEvent]:
        try:
            while True:
                item = self._events.get()
                if item is self._done:
                    break
                yield item
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._cancelled.set()
        self._thread.join()
        while not self._events.empty():
            self._events.get_nowait()
        self._finished = True
        if self._error is not None and self._result is None:
            raise self._error

    def __iter__(self) -> "_WorkerStream":
        return self

    def __next__(self) -> TokenEvent:
        return next(self._iterator)

    def close(self) -> None:
        try:
            self._iterator.close()
        finally:
            self._finish()

    @property
    def result(self) -> GenerationResult | None:
        return self._result


class GenerationEngine:
    """Runs the session's generations; owns their lifecycle state.

    ``lock`` is the session's exclusive-GPU re-entrant lock (shared with
    every other model op).  ``runtime`` is the model's compile / StaticCache
    routing.  ``steering`` and ``capture`` are the session's installed
    steering manager and hidden-state capture, which the decode routing
    consults.
    """

    def __init__(
        self,
        session: "DrowseSession",
        *,
        lock: "threading.RLock",
        runtime: CompiledRuntime,
        steering: "SteeringManager",
        capture: "HiddenCapture",
        release: Callable[[], None],
        default_return_top_k: int = 0,
    ) -> None:
        self._session = session
        self._lock = lock
        self._steering = steering
        self._capture = capture
        self._release = release
        self.runtime = runtime
        #: Session default for ``SamplingConfig.return_top_k`` (a per-call
        #: value > 0 wins; the sampler bounds it by the candidate pool).
        self.default_return_top_k = max(0, int(default_return_top_k))
        #: The decode loop's mutable state (finish reason, emit map, stop).
        self.state = GenerationState()
        #: The lifecycle phase (``IDLE`` between generations).
        self.phase = GenState.IDLE
        self.prefix_cache = PrefixCache()
        self.last_result: GenerationResult | None = None
        self.last_per_token_scores: dict[str, list[float]] | None = None
        #: This generation's capture may ride the persistent compile-clean
        #: buffers (read by the session's capture planner).
        self.compiled_clean_eligible = False

    # ---------------------------------------------------------- lifecycle

    @property
    def is_generating(self) -> bool:
        return self.phase is not GenState.IDLE

    @contextmanager
    def transaction(self) -> Iterator[GenerationTransaction]:
        """The generation boundary: single-flight guard in, hard teardown out.

        On entry: the non-blocking lock, the ``IDLE`` re-entry guard, and the
        ``PREAMBLE`` transition.  On exit, whatever raised: pop any steering
        scope the body left open, run the session's generation release (detach
        capture, close the instrument runs, reset the read plane, free the
        loom reservation), and release the phase and the lock.

        The body owns the steering scope through ``txn.steering_cm``: it
        assigns the scope when it enters one and clears the slot if it pops
        the scope itself (the decode path pops before finalizing, so the
        finalize reads happen unsteered).  A scope still present here is
        popped with ``swallow=txn.failed`` — a teardown ``Exception`` must not
        mask the failure already propagating.
        """
        if not self._lock.acquire(blocking=False):
            raise ConcurrentGenerationError("Generation already in progress")
        if self.phase is not GenState.IDLE:
            self._lock.release()
            raise ConcurrentGenerationError("session generation already in flight")
        self.phase = GenState.PREAMBLE
        txn = GenerationTransaction()
        try:
            yield txn
        except BaseException:
            txn.failed = True
            raise
        finally:
            try:
                try:
                    if txn.steering_cm is not None:
                        self._session._exit_internal_steering(
                            txn.steering_cm, swallow=txn.failed,
                        )
                finally:
                    try:
                        self._release()
                    finally:
                        self.compiled_clean_eligible = False
            finally:
                # Unconditional: an earlier teardown exception must not leave
                # the session phase-wedged or the lock held.
                self.phase = GenState.IDLE
                self._lock.release()

    def stop(self) -> None:
        """Ask the in-flight decode loop to stop after its current step."""
        self.state.request_stop()

    def close(self) -> None:
        """Drop the prefix cache, the reusable StaticCache and the last result."""
        self.prefix_cache.clear()
        self.runtime.close()
        self.last_result = None
        self.last_per_token_scores = None

    # ----------------------------------------------------- call resolution

    def gen_config(self, sampling: SamplingConfig | None) -> GenerationConfig:
        """The session :class:`GenerationConfig` with ``sampling``'s overrides.

        Never mutates ``session.config``; ``None`` fields fall through.
        """
        config = self._session.config
        if sampling is None:
            return config
        overrides: dict[str, Any] = {}
        if sampling.temperature is not None:
            overrides["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            overrides["top_p"] = sampling.top_p
        if sampling.top_k is not None:
            overrides["top_k"] = sampling.top_k
        if sampling.max_tokens is not None:
            overrides["max_new_tokens"] = sampling.max_tokens
        return replace(config, **overrides) if overrides else config

    def return_top_k(self, sampling: SamplingConfig | None) -> int:
        """The logit-alternative width: per-call when positive, else the default."""
        top_k = sampling.return_top_k if sampling is not None else 0
        return int(top_k or self.default_return_top_k)

    def readout_top_k(self, sampling: SamplingConfig | None) -> int:
        """The shared logit / J-lens / SAE top-k width.

        Read-side instruments keep the canonical eight-wide view when logit
        alternatives are off, and match the ``alts`` width otherwise.
        """
        return self.return_top_k(sampling) or DEFAULT_READOUT_TOP_K

    def compose_call(
        self,
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        system_prompt: str | None | InheritSystemPrompt,
        raw: bool,
        gen_seat: str,
        parent_node_id: str | None,
        recipe_override: "Recipe | str | None",
    ) -> ComposedCall:
        """Compose a generation's effective kwargs in one :func:`compose_call`.

        Tiers, weakest first: the gen label's cast member, the call's own
        kwargs, and for a regen the recipe of the node it reproduces
        (:meth:`LoomTree.anchor_recipe`) plus the modifier
        ``recipe_override`` derives from it — a :class:`Recipe` partial, a
        built-in mode, or a comma-separated partial-recipe string
        (:meth:`Recipe.compose_modifier`).  All four go through the one rule
        together, so the seed order it documents (modifier > call > anchor >
        cast) holds across them.

        The cast label rides the call's sampling box (``user_role`` for a
        user-seat gen, ``assistant_role`` otherwise), falling back to the
        structural seat name, so a bare continuation of a cast turn doesn't
        re-trigger the recipe.  Raw / flat mode renders no labels, so the
        roster doesn't apply there.
        """
        tree = self._session.tree
        cast_recipe: Recipe | None = None
        if not raw:
            member = tree.cast.get(_seat_label(sampling, gen_seat) or gen_seat)
            cast_recipe = member.recipe if member is not None else None
        anchor: Recipe | None = None
        modifier: Recipe | None = None
        if recipe_override is not None:
            anchor = tree.anchor_recipe(parent_node_id)
            modifier = (anchor or Recipe()).compose_modifier(recipe_override)
        if cast_recipe is None and modifier is None:
            return ComposedCall(steering, sampling, thinking, system_prompt)
        return compose_call(
            steering=steering, sampling=sampling, thinking=thinking,
            system_prompt=system_prompt,
            cast=cast_recipe, anchor=anchor, modifier=modifier,
        )

    def resolve_call(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        parent_node_id: str | None = None,
        recipe_override: "Recipe | str | None" = None,
        forced_prefix: list[int] | None = None,
        gen_seat: str = "assistant",
        append_same_role: bool = False,
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> CallSpec:
        """Fold one generation call into a frozen :class:`CallSpec`.

        Reads the tree (cast roster, regen anchor, the continuation
        candidate, the active node) but mutates nothing.
        """
        session = self._session
        steering, sampling, thinking, system_prompt = self.compose_call(
            steering=steering, sampling=sampling, thinking=thinking,
            system_prompt=system_prompt, raw=raw, gen_seat=gen_seat,
            parent_node_id=parent_node_id, recipe_override=recipe_override,
        )
        # ``""`` is the composition tiers' explicit "unsteered" (it stops a
        # cast default from filling the slot, and is what the ``unsteered`` /
        # ``inverted`` modifiers produce); once composed it means no
        # steering, which the grammar spells ``None``.
        if isinstance(steering, str) and not steering.strip():
            steering = None

        # One-shot continuation is message-oriented: when the selected leaf
        # already occupies the same seat under the same role label, the
        # generation regenerates that node in place with its current text as
        # the forced prefix.  Fan, fork and prefill paths leave this off.
        continuation_node_id: str | None = None
        history_anchor = parent_node_id
        if (
            append_same_role
            and not stateless
            and forced_prefix is None
            and (input is None or (raw and input == ""))
        ):
            tree = session.tree
            candidate = tree.get(parent_node_id or tree.active_node_id)
            seat = "user" if gen_seat == "user" else "assistant"
            label = None if raw else _seat_label(sampling, gen_seat)
            if candidate.role == seat and candidate.role_label == label:
                continuation_node_id = candidate.id
                history_anchor = candidate.parent_id
                forced_prefix = list(session.tokenizer.encode(
                    candidate.text, add_special_tokens=False,
                ))
                if forced_prefix:
                    base = sampling if sampling is not None else SamplingConfig()
                    base_max = base.max_tokens or session.config.max_new_tokens
                    sampling = replace(
                        base, max_tokens=len(forced_prefix) + base_max,
                    )
                # A same-message continuation is a body prefill, not a
                # request to open a fresh thinking channel.
                thinking = False
        # Snapshot the chat-history anchor before the loom node exists:
        # otherwise the new (empty) node becomes the active leaf and the
        # render walks into it.
        if (
            history_anchor is None
            and not stateless
            and (isinstance(input, str) or input is None)
        ):
            history_anchor = session.tree.active_node_id

        steering_obj = Steering.from_value(
            steering, profile_names=set(session.profiles),
        )
        if thinking is not None:
            use_thinking_req = bool(thinking)
        elif steering_obj is not None and steering_obj.thinking is not None:
            use_thinking_req = bool(steering_obj.thinking)
        elif getattr(session.config, "thinking", None) is not None:
            use_thinking_req = bool(session.config.thinking)
        else:
            use_thinking_req = session.chat_traits.supports_thinking

        logprobs = sampling.logprobs if sampling is not None else None
        top_k = self.return_top_k(sampling)
        if top_k > 0:
            lp_count = max(top_k, logprobs) if logprobs is not None else top_k
        else:
            lp_count = logprobs
        stop = list(sampling.stop) if sampling is not None and sampling.stop else None
        if isinstance(system_prompt, InheritSystemPrompt):
            system_prompt = session.config.system_prompt
        return CallSpec(
            input=input,
            steering_obj=steering_obj,
            sampling=sampling,
            system_prompt=system_prompt,
            stateless=stateless,
            raw=raw,
            gen_seat=gen_seat,
            parent_node_id=parent_node_id,
            history_anchor=history_anchor,
            use_thinking_req=use_thinking_req,
            gen_config=self.gen_config(sampling),
            lp_count=lp_count,
            seed=sampling.seed if sampling is not None else None,
            stop_list=session._seat_stop_augmentation(
                stop, gen_seat=gen_seat, raw=raw,
            ),
            logit_bias=sampling.logit_bias if sampling is not None else None,
            presence_penalty=(
                sampling.presence_penalty if sampling is not None else 0.0
            ),
            frequency_penalty=(
                sampling.frequency_penalty if sampling is not None else 0.0
            ),
            collect_logprobs=logprobs is not None,
            readout_top_k=self.readout_top_k(sampling),
            user_role=sampling.user_role if sampling is not None else None,
            assistant_role=(
                sampling.assistant_role if sampling is not None else None
            ),
            return_probe_readings=bool(
                sampling is None or sampling.return_probe_readings
            ),
            want_hidden=bool(sampling is not None and sampling.return_hidden),
            persist_per_layer_scores=bool(
                sampling is not None and sampling.persist_per_layer_scores
            ),
            persist_subspace_coords=bool(
                sampling is not None and sampling.persist_subspace_coords
            ),
            forced_prefix=forced_prefix,
            continuation_node_id=continuation_node_id,
        )

    def _plan_consumers(
        self,
        spec: CallSpec,
        on_token: "TokenCallback | TokenConsumer | None",
    ) -> ConsumerPlan:
        session = self._session
        options = consumer_options(on_token)
        # The geometry live toggle: when off, every per-token monitor
        # consumer is masked at the source (probes still report the
        # end-of-generation aggregate; only gates can force a subset).
        live = session.geometry.is_live
        has_monitor_probes = bool(session.monitor.probe_names)
        persists_probe_row = bool(
            live and spec.return_probe_readings and not spec.stateless
            and has_monitor_probes
        )
        return ConsumerPlan(
            live_scores_on=live,
            has_monitor_probes=has_monitor_probes,
            wants_live_token_scores=bool(
                live and on_token is not None and options.live_scores
            ),
            persists_layer_scores=bool(
                live and (
                    spec.persist_per_layer_scores
                    or (on_token is not None and options.per_layer_scores)
                )
            ),
            persists_subspace_coords=bool(live and spec.persist_subspace_coords),
            persists_probe_row=persists_probe_row,
            lens=bool(
                session.lens.live is not None
                and on_token is not None and options.lens_readout
            ),
            sae=bool(
                session.sae.live is not None
                and on_token is not None and options.sae_readout
            ),
            perplexity_requested=bool(on_token is not None and options.perplexity),
            # Without a consumer, logprobs, a loom probe row or a stop list
            # the tap would be a no-op whose presence still forces the
            # per-step log-softmax + entropy sync; stop-sequence matching only
            # runs under a tap.
            need_tap=bool(
                on_token is not None
                or spec.collect_logprobs
                or persists_probe_row
                or spec.stop_list is not None
            ),
        )

    # ----------------------------------------------------------- generate

    def generate_one(
        self,
        input: Any,
        *,
        on_token: "TokenCallback | TokenConsumer | None" = None,
        **call: Any,
    ) -> GenerationResult:
        """Run one generation; ``call`` takes :meth:`resolve_call`'s kwargs.

        Inside the transaction: resolve the call, open the internal steering
        scope and the loom nodes, render, plan the reads, decode with capture
        attached, pop the scope, and finalize.
        """
        session = self._session
        with self.transaction() as txn:
            spec = self.resolve_call(input, **call)
            steering_obj = spec.steering_obj
            steered = steering_obj is not None and bool(steering_obj.alphas)
            # Steering synthesis is Mahalanobis-normalized and needs the
            # whitener.  On a cold ``probes=[]`` session it may not exist yet:
            # build it here, while this generation owns the lock but before
            # the scope is entered (the public lazy property rejects a
            # non-idle phase).
            if steered and session._whitener is None:
                session._install_whitener_if_missing()
            plan = self._plan_consumers(spec, on_token)
            # The probe-inspector post-pass is per generation; the release
            # resets it.
            session.monitor.set_subspace_coords(plan.persists_subspace_coords)
            self.compiled_clean_eligible = self.runtime.compiled_clean_capture(
                session.device, want_hidden=spec.want_hidden,
            )
            if steered:
                txn.steering_cm = session.steering(steering_obj)

            assistant_node_id = session._start_loom_assistant(
                spec.input,
                stateless=spec.stateless,
                raw=spec.raw,
                parent_node_id=spec.parent_node_id,
                sampling=spec.sampling,
                steering_obj=steering_obj,
                use_thinking_req=spec.use_thinking_req,
                gen_seat=spec.gen_seat,
                system_prompt=spec.system_prompt,
                continuation_node_id=spec.continuation_node_id,
            )
            if txn.steering_cm is not None:
                txn.steering_cm.__enter__()
            use_thinking = spec.use_thinking_req and session.chat_traits.supports_thinking
            input_ids = session._prepare_input(
                spec.input,
                raw=spec.raw,
                thinking=use_thinking,
                stateless=spec.stateless,
                parent_node_id=spec.history_anchor,
                user_role=spec.user_role,
                assistant_role=spec.assistant_role,
                gen_seat=spec.gen_seat,
                system_prompt=spec.system_prompt,
            )
            self.state.reset()
            vector_snapshot = session._steering_composer.snapshot_steering_alphas()

            session.lens.active_for_generation = plan.lens
            session.sae.active_for_generation = plan.sae
            recipe_steering = self._recipe_steering(assistant_node_id)
            monitor_active = bool(plan.live_scores_on and plan.has_monitor_probes)
            authored = (
                session._plan_authored_prompt_capture(
                    spec,
                    assistant_node_id=assistant_node_id,
                    use_thinking=use_thinking,
                    monitor_active=monitor_active,
                    lens_active=plan.lens,
                    sae_active=plan.sae,
                    persist_per_layer_scores=plan.persists_layer_scores,
                    steering=recipe_steering,
                )
                if monitor_active or plan.lens or plan.sae else None
            )
            demand = session._resolve_read_demand(
                want_hidden=spec.want_hidden,
                live_scores_on=plan.live_scores_on,
                has_monitor_probes=plan.has_monitor_probes,
                loom_attached=assistant_node_id is not None,
                wants_live_token_scores=plan.wants_live_token_scores,
                persists_layer_scores=plan.persists_layer_scores,
                persists_probe_row=plan.persists_probe_row,
                persists_subspace_coords=plan.persists_subspace_coords,
                live_lens_active=plan.lens,
                live_sae_active=plan.sae,
                final_probe_aggregate=spec.return_probe_readings,
                capture_prompt=authored is not None,
            )
            session.events.emit(
                GenerationStarted(input=input, stateless=spec.stateless),
            )
            tap: TokenTap | None = None
            try:
                # Capture attach and the trigger reset live inside the inner
                # try so a BaseException between them still detaches.
                session._begin_capture(demand, widen=spec.want_hidden)
                if plan.need_tap:
                    tap = TokenTap(
                        session,
                        state=self.state,
                        consumer=on_token,
                        plan=plan,
                        assistant_node_id=assistant_node_id,
                        collect_logprobs=spec.collect_logprobs,
                        readout_top_k=spec.readout_top_k,
                        recipe_steering=recipe_steering,
                    )
                # ``generate_steered`` mutates the trigger context at
                # lifecycle boundaries; every manifold foot-follower re-seeds
                # at the origin instead of inheriting the prior run's foot.
                self._steering.ctx.reset()
                self._steering.reset_manifold_feet()
                self.phase = GenState.RUNNING
                generated_ids, elapsed = self.decode(
                    spec,
                    input_ids,
                    tap=tap,
                    use_thinking=use_thinking,
                    # Per-token perplexity costs one host sync per token;
                    # compute it only when something surfaces it (a loom
                    # row, an interactive generation, or a consumer asking).
                    want_perplexity=(
                        not spec.stateless
                        or assistant_node_id is not None
                        or plan.perplexity_requested
                    ),
                    authored_capture=authored,
                )
            finally:
                session._end_capture()
                if txn.steering_cm is not None:
                    # ``swallow=False``: a teardown failure surfaces here.
                    # Clearing the slot hands the transaction an
                    # already-popped scope.
                    session._exit_internal_steering(txn.steering_cm, swallow=False)
                    txn.steering_cm = None
                self.phase = GenState.FINALIZING

            mean_logprob, mean_surprise = (
                tap.mean_logprob() if tap is not None else (None, None)
            )
            result = self.finalize(
                generated_ids, elapsed, vector_snapshot,
                prompt_tokens=int(input_ids.shape[1]),
                stateless=spec.stateless,
                logprobs_list=tap.logprobs if tap is not None else None,
                applied_steering=(
                    str(steering_obj) if steering_obj is not None else None
                ),
                return_hidden=spec.want_hidden,
                return_probe_readings=spec.return_probe_readings,
                assistant_node_id=assistant_node_id,
                mean_logprob=mean_logprob,
                mean_surprise=mean_surprise,
            )
            session.events.emit(GenerationFinished(result=result))
            return result

    def _recipe_steering(self, node_id: str | None) -> str | None:
        """The steering string recorded on the generating node's recipe."""
        if node_id is None:
            return None
        node = self._session.tree.nodes.get(node_id)
        if node is None or node.recipe is None:
            return None
        return node.recipe.steering

    def decode(
        self,
        spec: CallSpec,
        input_ids: torch.Tensor,
        *,
        tap: TokenTap | None,
        use_thinking: bool,
        want_perplexity: bool,
        authored_capture: "_AuthoredPromptCapture | None" = None,
    ) -> tuple[list[int], float]:
        """Run the decode loop once capture and steering are installed.

        Routing: a prefix-cache hit decodes only the suffix; StaticCache
        (with compile fusion on MPS and CUDA-graph capture on CUDA) is
        eligible when steering is absent or every steered layer is the static
        single-affine fast path; the compiled module runs only when the live
        hook topology matches the one it was traced with.
        """
        session = self._session
        runtime = self.runtime
        steering = self._steering
        capture = self._capture
        composer = session._steering_composer
        gen_config = spec.gen_config
        static_eligible = bool(
            runtime.static_cache_active
            and runtime.graph_thread_safe()
            and (steering.all_fast_path() or steering.static_steerable())
        )
        # The cached prefix KV is unsteered: reusable whenever the active
        # steering doesn't touch the prefill (``@response`` / ``@generated`` /
        # probe-gated).  ``return_hidden`` needs every prefix hidden state and
        # thinking takes a different decode path, so both re-prefill.
        hit = None
        if (
            self.prefix_cache
            and not spec.want_hidden
            and not use_thinking
            and not composer.steering_active_in_prefill()
        ):
            hit = self.prefix_cache.hit(
                input_ids,
                static_eligible=static_eligible,
                required_max_new_tokens=gen_config.max_new_tokens,
            )
        cache_position_offset = hit.prefix_len if hit is not None else 0
        decode_ids = hit.suffix_ids if hit is not None else input_ids

        step_callback = (
            capture.ingest_persistent
            if session._capture_state.persistent
            else capture.fire_step_sink
        )
        if authored_capture is not None:
            matches, prompt_positions = session._match_authored_prompt_targets(
                authored_capture.targets,
                input_ids,
                cache_position_offset=cache_position_offset,
                search_end=authored_capture.search_end,
            )
            capture.set_prompt_positions(prompt_positions)
            if matches:
                step_callback = _authored_first_step(
                    session, matches, authored_capture, step_callback,
                )

        start = time.monotonic()
        gating_callback = (
            composer.build_gating_score_callback()
            if composer.steering_needs_probe_gating()
            else None
        )
        use_static_cache = static_eligible and (hit is None or hit.static)
        # Compiled routing (CUDA/MPS).  The graph was traced with only the
        # persistent offset + capture hooks attached, so the compiled module
        # is correct exactly when the live hook topology matches: no
        # transient capture hooks and either unsteered or steering lowered to
        # the persistent offsets.  Everything else runs the eager original
        # (same persistent hooks, so offsets and capture still apply) on a
        # DynamicCache, which also avoids a Dynamo retrace per hook topology.
        gen_model = session.model
        uses_offsets = bool(steering.uses_compiled_offsets)
        if runtime.compiled_on(session.device):
            compiled_clean = not capture.is_transient() and (
                uses_offsets or steering.all_fast_path()
            )
            if compiled_clean and runtime.graph_thread_safe():
                use_static_cache = runtime.static_cache_active and (
                    hit is None or hit.static
                )
            else:
                gen_model = getattr(gen_model, "_orig_mod", gen_model)
                use_static_cache = False
        past_key_values = hit.past_key_values if hit is not None else None
        if use_static_cache and past_key_values is None:
            # Reserve at least the session default decode window and bucket
            # the total length, so most requests keep one cache allocation and
            # one set of Dynamo guards whatever their per-call max_tokens.
            reserve = max(
                int(gen_config.max_new_tokens),
                int(session.config.max_new_tokens),
                1,
            )
            required = cache_position_offset + int(decode_ids.shape[1]) + reserve
            try:
                past_key_values = runtime.acquire_static_cache(
                    gen_model,
                    max_cache_len=((required + 255) // 256) * 256,
                    device=session.device,
                )
            except Exception as exc:
                _log.warning(
                    "Reusable StaticCache allocation failed (%s: %s); "
                    "falling back to DynamicCache",
                    type(exc).__name__,
                    exc,
                )
                use_static_cache = False
        generated_ids = generate_steered(
            gen_model, session.tokenizer, decode_ids,
            gen_config, self.state, thinking=use_thinking,
            on_token=tap,
            seed=spec.seed, stop=spec.stop_list, logit_bias=spec.logit_bias,
            presence_penalty=spec.presence_penalty,
            frequency_penalty=spec.frequency_penalty,
            logprobs=spec.lp_count,
            trigger_ctx=steering.ctx,
            past_key_values=past_key_values,
            cache_position_offset=cache_position_offset,
            score_callback=gating_callback,
            # Per-token capture ingest + scoring fires post-forward, never
            # inside a capture hook; it no-ops when no per-token sink exists.
            step_callback=step_callback,
            use_static_cache=use_static_cache,
            forced_prefix=spec.forced_prefix,
            steering_active=bool(steering.hooks or uses_offsets),
            want_perplexity=want_perplexity,
            traits=session.chat_traits,
        )
        elapsed = time.monotonic() - start
        if hit is not None:
            self.prefix_cache.release(hit)
        return generated_ids, elapsed

    def finalize(
        self,
        generated_ids: list[int],
        elapsed: float,
        vector_snapshot: dict[str, float],
        **kwargs: Any,
    ) -> GenerationResult:
        """Build the result (:func:`finalize_generation`) and record it."""
        return finalize_generation(
            self._session, generated_ids, elapsed, vector_snapshot,
            state=self.state, record=self.record, **kwargs,
        )

    def record(
        self,
        result: GenerationResult,
        per_token_scores: dict[str, list[float]] | None,
    ) -> None:
        """Publish a finished result as ``last_result``."""
        self.last_result = result
        self.last_per_token_scores = per_token_scores

    def stream(
        self,
        input: Any,
        *,
        live_scores: bool = True,
        live_readouts: bool = True,
        **call: Any,
    ) -> GenerationStream:
        """Run :meth:`generate_one` on a worker thread; iterate its tokens."""
        options = TokenConsumerOptions(
            live_scores=bool(live_scores),
            lens_readout=bool(live_readouts),
            sae_readout=bool(live_readouts),
            probe_readings=bool(live_scores),
        )
        return _WorkerStream(
            lambda consumer: self.generate_one(input, on_token=consumer, **call),
            options,
            self.state,
        )

    # -------------------------------------------------------- prefix cache

    def cache_prefix(
        self,
        messages: "list[dict[str, str]] | torch.Tensor | None",
        *,
        max_new_tokens: int | None = None,
        prefer_static: bool = False,
    ) -> int:
        """Prefill a shared prefix for later generations to reuse.

        See :meth:`DrowseSession.cache_prefix` for the contract.
        """
        if messages is None:
            self.prefix_cache.clear()
            return 0
        session = self._session
        if session._steering_composer._stack:
            raise PrefixCacheStateError(
                "cache_prefix called inside an active session.steering() "
                "scope; prefill must run with the neutral baseline so the "
                "cached prefix is consume-regime independent. Cache before "
                "entering any steering scope."
            )
        if self.is_generating:
            raise ConcurrentGenerationError(
                "cache_prefix called while a generation is in flight"
            )
        if isinstance(messages, torch.Tensor):
            prefix_ids = messages if messages.dim() > 1 else messages.unsqueeze(0)
            prefix_ids = prefix_ids.to(device=session.device, dtype=torch.long)
        else:
            # ``add_generation_prompt=False``: the assistant-turn opener
            # belongs to each call's suffix, so one prefix serves multi-turn
            # variants.
            prefix_ids = build_chat_input(
                session.tokenizer, list(messages),
                session.config.system_prompt,
                thinking=False,
                add_generation_prompt=False,
                scene=session.scene_grammar,
                traits=session.chat_traits,
            ).to(session.device)
        # Suspend capture so the prefix tokens don't fill the buckets the next
        # generation's scoring reads.
        session._end_capture()
        static_len: int | None = None
        if prefer_static and self.runtime.static_cache_active:
            headroom = int(
                max_new_tokens if max_new_tokens is not None
                else session.config.max_new_tokens
            )
            static_len = int(prefix_ids.shape[1]) + max(headroom, 1)
        model = session.model
        # The entry owns its cache: never the reusable generation StaticCache,
        # which the next ordinary generation resets in place.
        return self.prefix_cache.build(
            model,
            prefix_ids,
            static_max_cache_len=static_len,
            make_static=lambda length: make_static_cache(
                model,
                max_cache_len=length,
                device=session.device,
                dtype=next(model.parameters()).dtype,
            ),
        )


def _seat_label(sampling: SamplingConfig | None, gen_seat: str) -> str | None:
    """The role label the generated turn carries (the seat's sampling box)."""
    if sampling is None:
        return None
    return sampling.user_role if gen_seat == "user" else sampling.assistant_role


def _authored_first_step(
    session: "DrowseSession",
    matches: list[Any],
    authored: "_AuthoredPromptCapture",
    step_callback: Callable[[int], None],
) -> Callable[[int], None]:
    """Wrap the step callback to persist authored prompt rows on step one.

    The first callback follows the full/suffix prefill: score the visible
    authored positions in prompt order before the ordinary step sink advances
    curved monitor feet to the state that produces generated token 0.
    """
    pending = True

    def _step(step_id: int) -> None:
        nonlocal pending
        if pending:
            pending = False
            session._persist_authored_prompt_captures(
                matches,
                monitor_active=authored.monitor_active,
                lens_active=authored.lens_active,
                sae_active=authored.sae_active,
                readout_top_k=authored.readout_top_k,
                persist_per_layer_scores=authored.persist_per_layer_scores,
                steering=authored.steering,
            )
        step_callback(step_id)

    return _step


__all__ = [
    "DEFAULT_READOUT_TOP_K",
    "ConsumerPlan",
    "GenerationEngine",
    "PrefixCacheStateError",
    "TokenTap",
]
