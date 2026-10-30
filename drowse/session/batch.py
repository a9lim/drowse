"""Multi-row generation: sibling fans, prompt batches and alpha sweeps.

:class:`BatchRunner` owns every surface that turns one call into several
generations: ``generate(n>1)`` sibling fans, :meth:`DrowseSession.generate_batch`
and :meth:`DrowseSession.generate_sweep`.  Rows run one at a time through the
session's single-generation engine (``generate_one``), except a compatible
stateless batch, which :class:`FastBatchPath` serves with one left-padded
``transformers`` ``generate`` call.

A user :meth:`~drowse.generation.decode.GenerationState.request_stop` ends the
whole series; finishing a row does not (the runner compares
``GenerationState.stop_count`` across rows rather than reading the per-row
stop event, which every generation sets on its way out).
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ContextManager, cast

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from drowse.events import EventBus, GenerationFinished, GenerationStarted
from drowse.fit.capture import last_content_index
from drowse.generation.decode import (
    GenerationState,
    hf_generation_config,
    hf_unneutralizable_processors,
)
from drowse.generation.finalize import MIN_ELAPSED_FOR_RATE, FamilyAggregates
from drowse.generation.results import GenerationResult, RunSet
from drowse.generation.sampling import SamplingConfig
from drowse.loom.recipe import (
    INHERIT_SYSTEM_PROMPT,
    ComposedCall,
    InheritSystemPrompt,
    Recipe,
    derive_seed_schedule,
)
from drowse.readout.types import AGG_TAIL_DEPTH, ReadRequest
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger

if TYPE_CHECKING:
    from drowse.generation.token_callback import TokenCallback
    from drowse.lens.instrument import LensInstrument
    from drowse.loom.tree import LoomTree
    from drowse.model.chat import ChatTraits
    from drowse.readout.capture_buffer import HiddenCapture
    from drowse.readout.geometry import GeometryInstrument
    from drowse.readout.types import InstrumentPlan
    from drowse.sae.instrument import SaeInstrument
    from drowse.session.session import GenState, PreparedCall
    from drowse.steering.hooks import SteeringManager

# Floor on the shared token prefix a serial batch will KV-cache.  Below this
# the saved prefill is in the noise and the double render plus cache
# management isn't worth it.  The cached run is also capped at
# ``min_row_len - 1`` so every row keeps a non-empty suffix to sample from.
PREFIX_CACHE_MIN_TOKENS = 16

OnRow = Callable[[int, GenerationResult, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class GenerationJob:
    """One row of a multi-row run: the kwargs for one engine generation."""

    input: Any
    steering: Any = None
    sampling: SamplingConfig | None = None
    raw: bool = False
    thinking: bool | None = None
    on_token: "TokenCallback | None" = None
    parent_node_id: str | None = None
    recipe_override: Recipe | str | None = None
    system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT
    grid: dict[str, Any] | None = None
    gen_seat: str = "assistant"


class BatchStopCriteria(StoppingCriteria):
    """Bridge :meth:`DrowseSession.stop` into ``transformers`` ``generate``."""

    def __init__(self, state: GenerationState) -> None:
        self._state = state

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | tuple[torch.FloatTensor, ...] | None,
        **kwargs: Any,
    ) -> Any:
        del scores, kwargs
        stop = 1 if self._state.is_stop_requested() else 0
        return input_ids.new_full((int(input_ids.shape[0]),), stop).bool()


# ---------------------------------------------------------------------------
# Fast-path eligibility and the HF batch I/O helpers (pure).
# ---------------------------------------------------------------------------


def steering_always_on(steering: Steering | None) -> bool:
    """HF ``generate`` can batch only trigger-free / ``Trigger.BOTH`` terms.

    One batched call has no per-step hook re-evaluation, so a phased or
    probe-gated term would mean nothing there.
    """
    return steering is None or steering.triggers() <= {Trigger.BOTH}


def fast_batch_blocked(
    prepared: "PreparedCall", sampling: SamplingConfig | None,
) -> bool:
    """True iff the call needs the row-shaped decode loop.

    The one eligibility predicate for the batched ``model.generate`` path:
    thinking, logprobs or top-k alternatives (per call or the session
    default), stop strings, logit bias, penalties, per-row seeded sampling,
    hidden-state returns, per-layer/subspace persistence, and phased or gated
    steering all stay on the serial loop.
    """
    if prepared.use_thinking_req or prepared.gen_config.max_new_tokens < 1:
        return True
    if (
        prepared.lp_count is not None
        or prepared.stop_list is not None
        or prepared.logit_bias is not None
        or prepared.presence_penalty != 0.0
        or prepared.frequency_penalty != 0.0
    ):
        return True
    if prepared.seed is not None and prepared.gen_config.temperature > 0:
        return True
    if sampling is not None and (
        sampling.return_hidden
        or sampling.persist_per_layer_scores
        or sampling.persist_subspace_coords
    ):
        return True
    return not steering_always_on(prepared.steering_obj)


def generate_capable_model(model: Any) -> Any | None:
    """The object exposing ``generate``; unwraps a ``torch.compile`` module."""
    if callable(getattr(model, "generate", None)):
        return model
    orig = getattr(model, "_orig_mod", None)
    if orig is not None and callable(getattr(orig, "generate", None)):
        return orig
    return None


def pad_token_id(tokenizer: Any) -> int:
    """Pad id for left-padded batches (pad, else the first eos, else 0)."""
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if value:
                return int(value[0])
        else:
            return int(value)
    return 0


def left_pad(
    rows: Sequence[torch.Tensor], pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad 1-D token rows into ``(input_ids, attention_mask, lengths)``."""
    lengths = [int(row.shape[0]) for row in rows]
    width = max(lengths)
    batch = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(batch)
    for index, row in enumerate(rows):
        length = lengths[index]
        batch[index, width - length:] = row
        attention[index, width - length:] = 1
    return batch, attention, lengths


def trim_generated(
    row: torch.Tensor,
    *,
    eos_ids: set[int],
    pad_id: int,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    """One HF output row as ``(generated_ids, finish_reason)``.

    Cuts at the first EOS (``"stop"``).  A row with no EOS ends in ``"length"``
    unless it stopped early, which shows as trailing pad or a short row
    (``"stop"``).
    """
    raw_ids = [int(tok) for tok in row.detach().to("cpu").tolist()]
    tokens: list[int] = []
    finish_reason = "length"
    for token_id in raw_ids:
        if token_id in eos_ids:
            finish_reason = "stop"
            break
        tokens.append(token_id)
    else:
        stripped_pad = False
        while tokens and tokens[-1] == pad_id:
            tokens.pop()
            stripped_pad = True
        if stripped_pad or len(tokens) < max_new_tokens:
            finish_reason = "stop"
    return tokens, finish_reason


def common_prefix(
    rows: Sequence[torch.Tensor], *, min_tokens: int = PREFIX_CACHE_MIN_TOKENS,
) -> torch.Tensor | None:
    """The longest shared leading run of 1-D CPU token rows, or ``None``.

    Capped at ``min_row_len - 1`` so every row keeps a suffix; ``None`` when
    the run is shorter than ``min_tokens``.
    """
    min_len = min(int(row.shape[0]) for row in rows)
    if min_len <= min_tokens:
        return None
    ref = rows[0]
    shared = 0
    for i in range(min_len):
        token = int(ref[i])
        if all(int(row[i]) == token for row in rows):
            shared += 1
        else:
            break
    shared = min(shared, min_len - 1)
    if shared < min_tokens:
        return None
    return ref[:shared].clone()


def _blank_to_none(steering: Any) -> Any:
    """``""`` is the composition tiers' explicit "unsteered"; as a call it
    means no steering, which the grammar spells ``None``."""
    if isinstance(steering, str) and not steering.strip():
        return None
    return steering


# ---------------------------------------------------------------------------
# The batched ``model.generate`` path.
# ---------------------------------------------------------------------------


class FastBatchPath:
    """Serve a compatible stateless batch with one HF ``generate`` call.

    Shares the engine's generation transaction, steering scope, call
    composition (cast roster included) and finalize, and the read plane's
    instrument binding.  Only the body differs from the decode loop: one
    left-padded ``generate`` for every row, a batched tail ring for probe
    aggregates, and no loom attachment.  :meth:`run` returns ``None`` when the
    call needs the row-shaped decode loop.
    """

    def __init__(
        self,
        *,
        model: Callable[[], Any],
        tokenizer: Any,
        device: torch.device,
        chat_traits: "ChatTraits",
        layers: Any,
        gen_state: GenerationState,
        steering: "SteeringManager",
        events: EventBus,
        capture: "HiddenCapture",
        geometry: "GeometryInstrument",
        lens: "LensInstrument",
        sae: "SaeInstrument",
        sae_layer: Callable[[], int | None],
        compose_call: Callable[..., ComposedCall],
        prepare_call: Callable[..., "PreparedCall"],
        render: Callable[..., torch.Tensor],
        transaction: Callable[[], ContextManager[Any]],
        open_scope: Callable[[Steering], ContextManager[Any]],
        snapshot_alphas: Callable[[], dict[str, float]],
        set_phase: Callable[["GenState"], None],
        bind_runs: Callable[
            [ReadRequest, ReadRequest, ReadRequest],
            "tuple[InstrumentPlan, InstrumentPlan, InstrumentPlan]",
        ],
        score_geometry: Callable[..., dict[str, Any]],
        finalize: Callable[..., GenerationResult],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._chat_traits = chat_traits
        self._layers = layers
        self._gen_state = gen_state
        self._steering = steering
        self._events = events
        self._capture = capture
        self._geometry = geometry
        self._lens = lens
        self._sae = sae
        self._sae_layer = sae_layer
        self._compose_call = compose_call
        self._prepare_call = prepare_call
        self._render = render
        self._transaction = transaction
        self._open_scope = open_scope
        self._snapshot_alphas = snapshot_alphas
        self._set_phase = set_phase
        self._bind_runs = bind_runs
        self._score_geometry = score_geometry
        self._finalize = finalize

    def run(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        stateless: bool,
        raw: bool,
        on_result: Callable[[int, GenerationResult], None] | None,
    ) -> RunSet | None:
        """The batch's ``RunSet``, or ``None`` to fall back to the serial loop."""
        if len(prompts) < 2 or not stateless:
            return None
        # The same composition the decode loop applies (cast roster weakest,
        # then the call's own kwargs), so a cast member's defaults reach
        # batched rows too.
        composed = self._compose_call(
            steering=steering,
            sampling=sampling,
            thinking=thinking,
            system_prompt=INHERIT_SYSTEM_PROMPT,
            raw=raw,
            gen_seat="assistant",
            parent_node_id=None,
            recipe_override=None,
        )
        call_sampling = composed.sampling
        prepared = self._prepare_call(
            _blank_to_none(composed.steering), call_sampling, composed.thinking,
        )
        if fast_batch_blocked(prepared, call_sampling):
            return None
        want_readings = call_sampling is None or call_sampling.return_probe_readings
        # An SAE feature probe reads a resident SAE layer; with none loaded
        # the serial loop is the one that reports it.
        if want_readings and self._sae.names and self._sae_layer() is None:
            return None
        model = generate_capable_model(self._model())
        if model is None or hf_unneutralizable_processors(model):
            return None
        return self._execute(
            prompts,
            model=model,
            prepared=prepared,
            sampling=call_sampling,
            system_prompt=composed.system_prompt,
            raw=raw,
            want_readings=want_readings,
            on_result=on_result,
        )

    def _render_batch(
        self,
        prompts: list[Any],
        *,
        sampling: SamplingConfig | None,
        system_prompt: str | None | InheritSystemPrompt,
        raw: bool,
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Render each prompt statelessly and left-pad the batch."""
        rows: list[torch.Tensor] = []
        for prompt in prompts:
            ids = self._render(
                prompt,
                raw=raw,
                thinking=False,
                stateless=True,
                user_role=sampling.user_role if sampling is not None else None,
                assistant_role=(
                    sampling.assistant_role if sampling is not None else None
                ),
                system_prompt=system_prompt,
                to_device=False,
            )
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise RuntimeError(
                    "generate_batch fast path expected rendered prompt shape [1, T]"
                )
            rows.append(ids[0].detach().to(device="cpu", dtype=torch.long))
        input_ids, attention, lengths = left_pad(rows, pad_id)
        return input_ids.to(self._device), attention.to(self._device), lengths

    def _attach_aggregate_ring(self, want_readings: bool) -> bool:
        """Bind the instrument runs and arm the batched tail ring.

        Returns whether any family will read a per-row aggregate.
        """
        captures = want_readings and bool(
            self._geometry.names or self._lens.names or self._sae.names
        )
        # One request serves all three families: the lens prepare takes the
        # disk refresh + pin and snapshots specs against that identity, so one
        # pinned lens serves every row aggregate.
        request = ReadRequest(final_aggregate=want_readings)
        plans = self._bind_runs(request, request, request)
        if captures:
            layer_set: set[int] = set()
            for plan in plans:
                layer_set |= plan.latest_layers
                layer_set |= plan.tail_layers
            self._capture.clear()
            if layer_set:
                self._capture.attach_batch_tail(
                    self._layers, sorted(layer_set), depth=AGG_TAIL_DEPTH,
                )
        return captures

    def _row_aggregates(
        self, row: int, generated_ids: list[int], *, geometry: bool,
    ) -> FamilyAggregates:
        """Score one row's aggregates from its slice of the batched tail ring.

        The lens/SAE runs measure against their bind-time binding, so an
        ``on_result`` callback detaching a probe cannot change later rows.
        """
        pooled = (
            self._capture.batch_tail_slice_at(
                row, last_content_index(generated_ids, self._tokenizer),
            )
            if generated_ids else {}
        )
        return FamilyAggregates(
            geometry=(
                self._score_geometry(generated_ids, pooled=pooled)
                if geometry else {}
            ),
            lens=self._lens.score_aggregate(generated_ids, pooled=pooled),
            sae=self._sae.score_aggregate(generated_ids, pooled=pooled),
        )

    def _execute(
        self,
        prompts: list[Any],
        *,
        model: Any,
        prepared: "PreparedCall",
        sampling: SamplingConfig | None,
        system_prompt: str | None | InheritSystemPrompt,
        raw: bool,
        want_readings: bool,
        on_result: Callable[[int, GenerationResult], None] | None,
    ) -> RunSet:
        from drowse.session.session import GenState

        steering_obj = prepared.steering_obj
        gen_config = prepared.gen_config
        with self._transaction() as txn:
            if steering_obj is not None and steering_obj.alphas:
                txn.steering_cm = self._open_scope(steering_obj)
                txn.steering_cm.__enter__()
            vector_snapshot = self._snapshot_alphas()
            pad_id = pad_token_id(self._tokenizer)
            input_ids, attention_mask, prompt_lengths = self._render_batch(
                prompts,
                sampling=sampling,
                system_prompt=system_prompt,
                raw=raw,
                pad_id=pad_id,
            )
            score_geometry = want_readings and bool(self._geometry.names)
            captures = self._attach_aggregate_ring(want_readings)
            self._gen_state.reset()
            self._steering.ctx.reset()
            self._steering.reset_manifold_feet()
            self._set_phase(GenState.RUNNING)
            for prompt in prompts:
                self._events.emit(GenerationStarted(input=prompt, stateless=True))

            # The full EOS set (generation_config + tokenizer + added
            # end-of-turn specials), matching the decode loop; the explicit
            # config pins the decode loop's sampler so the checkpoint's own
            # ``generation_config`` cannot leak in.
            eos_ids = set(self._chat_traits.eos_ids)
            text_cfg = getattr(model.config, "text_config", model.config)
            hf_config = hf_generation_config(
                gen_config,
                vocab=int(text_cfg.vocab_size),
                eos_ids=sorted(eos_ids),
                pad_id=pad_id,
                steering_active=bool(
                    self._steering.hooks or self._steering.uses_compiled_offsets
                ),
            )
            start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=hf_config,
                    stopping_criteria=StoppingCriteriaList([
                        BatchStopCriteria(self._gen_state),
                    ]),
                )
            elapsed = time.monotonic() - start
            if captures:
                self._capture.detach()
            sequences = getattr(generated, "sequences", generated)
            if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                raise RuntimeError(
                    "generate_batch fast path expected model.generate to return "
                    "a [batch, sequence] tensor"
                )

            prompt_width = int(input_ids.shape[1])
            self._set_phase(GenState.FINALIZING)
            applied_steering = str(steering_obj) if steering_obj is not None else None
            no_aggregates = FamilyAggregates({}, {}, {})
            results: list[GenerationResult] = []
            for idx, prompt_tokens in enumerate(prompt_lengths):
                generated_ids, finish_reason = trim_generated(
                    sequences[idx, prompt_width:],
                    eos_ids=eos_ids,
                    pad_id=pad_id,
                    max_new_tokens=int(gen_config.max_new_tokens),
                )
                # Finalize reads the per-row generation state.
                state = self._gen_state
                state.thinking_end_idx = 0
                state.emit_map = []
                state.response_text = None
                state.response_aggregate_index = None
                state.finish_reason = finish_reason
                result = self._finalize(
                    generated_ids,
                    elapsed,
                    vector_snapshot,
                    prompt_tokens=prompt_tokens,
                    stateless=True,
                    applied_steering=applied_steering,
                    return_probe_readings=want_readings,
                    aggregates=(
                        self._row_aggregates(
                            idx, generated_ids, geometry=score_geometry,
                        )
                        if captures else no_aggregates
                    ),
                )
                results.append(result)
                self._events.emit(GenerationFinished(result=result))
                if on_result is not None:
                    on_result(idx, result)
            total_tokens = sum(r.token_count for r in results)
            return RunSet(
                results,
                node_ids=[None] * len(results),
                grid=[{"prompt_index": idx} for idx in range(len(results))],
                kind="batch",
                metrics={
                    "batch_elapsed": elapsed,
                    "batch_token_count": total_tokens,
                    "batch_tok_per_sec": (
                        total_tokens / elapsed
                        if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
                    ),
                },
            )


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------


class BatchRunner:
    """Fans, batches and sweeps over the session's single-generation engine.

    ``generate_one`` is the engine entry (``_generate_core``'s keyword
    surface); ``tree`` reads the live loom tree (it is replaced on restore).
    A serial batch whose rows share a long rendered prefix prefills it once
    through ``cache_prefix`` and clears it afterwards.
    """

    def __init__(
        self,
        *,
        generate_one: Callable[..., GenerationResult],
        tree: Callable[[], "LoomTree"],
        gen_state: GenerationState,
        render: Callable[..., torch.Tensor],
        cache_prefix: Callable[..., int],
        prefill_inactive: Callable[[Any], bool],
        static_cache_active: Callable[[], bool],
        default_max_tokens: Callable[[], int],
        fast: FastBatchPath | None = None,
    ) -> None:
        self._generate_one = generate_one
        self._tree = tree
        self._gen_state = gen_state
        self._render = render
        self._cache_prefix = cache_prefix
        self._prefill_inactive = prefill_inactive
        self._static_cache_active = static_cache_active
        self._default_max_tokens = default_max_tokens
        self._fast = fast

    # -- public surfaces ----------------------------------------------------

    def fan(
        self,
        input: Any,
        *,
        n: int,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: "TokenCallback | None" = None,
        parent_node_id: str | None = None,
        recipe_override: Recipe | str | None = None,
        gen_seat: str = "assistant",
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> RunSet:
        """``n`` sibling generations under derived per-sibling seeds.

        A stateful fan pins its anchor to the node active when it starts, so
        each row lands beside the first (the engine's user-turn dedup reuses
        the first row's user turn) instead of under the previous row.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not stateless and parent_node_id is None:
            parent_node_id = self._tree().active_node_id
        base = sampling if sampling is not None else SamplingConfig()
        schedule = derive_seed_schedule(base.seed, n)
        jobs = [
            GenerationJob(
                input=input,
                steering=steering,
                sampling=replace(base, seed=seed),
                raw=raw,
                thinking=thinking,
                on_token=on_token,
                parent_node_id=parent_node_id,
                recipe_override=recipe_override,
                gen_seat=gen_seat,
                system_prompt=system_prompt,
            )
            for seed in schedule
        ]
        return self.run_jobs(jobs, stateless=stateless, kind="fan")

    def batch(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        stateless: bool = True,
        raw: bool = False,
        on_result: Callable[[int, GenerationResult], None] | None = None,
    ) -> RunSet:
        """N prompts under one steering; see :meth:`DrowseSession.generate_batch`."""
        prompts_in: Any = prompts
        if not isinstance(prompts_in, list) or not prompts_in:
            raise ValueError("generate_batch: prompts must be a non-empty list")
        if self._fast is not None:
            fast = self._fast.run(
                prompts,
                steering=steering,
                sampling=sampling,
                thinking=thinking,
                stateless=stateless,
                raw=raw,
                on_result=on_result,
            )
            if fast is not None:
                return fast

        prefix_cached = self._cache_shared_prefix(
            prompts, steering=steering, sampling=sampling,
            thinking=thinking, stateless=stateless, raw=raw,
        )
        try:
            jobs = [
                GenerationJob(
                    input=prompt,
                    steering=steering,
                    sampling=sampling,
                    raw=raw,
                    thinking=thinking,
                    grid={"prompt_index": idx},
                )
                for idx, prompt in enumerate(prompts)
            ]
            return self.run_jobs(
                jobs,
                stateless=stateless,
                kind="batch",
                on_result=(
                    (lambda idx, result, _row: on_result(idx, result))
                    if on_result is not None else None
                ),
            )
        finally:
            if prefix_cached:
                self._cache_prefix(None)

    def sweep(
        self,
        prompt: Any,
        sweep: dict[str, list[float]],
        *,
        base_steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        stateless: bool = True,
        raw: bool = False,
        on_result: Callable[[int, GenerationResult, dict[str, float]], None] | None = None,
        parent_node_id: str | None = None,
    ) -> RunSet:
        """Cartesian alpha sweep; see :meth:`DrowseSession.generate_sweep`."""
        sweep_in: Any = sweep
        if not isinstance(sweep_in, dict) or not sweep_in:
            raise ValueError("generate_sweep: sweep dict must be non-empty")
        for name, alphas in sweep.items():
            alphas_in: Any = alphas
            if not isinstance(alphas_in, (list, tuple)) or not alphas_in:
                raise ValueError(
                    f"generate_sweep: sweep['{name}'] must be a non-empty "
                    f"list of alpha values"
                )

        concept_names = list(sweep)
        alpha_lists = [list(sweep[name]) for name in concept_names]
        total = 1
        for values in alpha_lists:
            total *= len(values)
        base = sampling if sampling is not None else SamplingConfig()
        seed_schedule = derive_seed_schedule(base.seed, total)
        # A pre-built Steering renders through the canonical formatter so the
        # swept terms compose by string concatenation.
        base_str = None if base_steering is None else str(base_steering)

        # Stateful sweeps land every row as a sibling under one shared user
        # turn: spawn it with the label the rows will send so the engine's
        # label-aware dedup reuses it, and anchor the rows at its parent.
        gen_parent_id = parent_node_id
        if not stateless and isinstance(prompt, str):
            tree = self._tree()
            anchor_id = tree.add_user_turn(
                prompt,
                parent_id=parent_node_id,
                role_label=(
                    sampling.user_role
                    if sampling is not None and not raw else None
                ),
            )
            gen_parent_id = tree.nodes[anchor_id].parent_id

        jobs: list[GenerationJob] = []
        for idx, combo in enumerate(itertools.product(*alpha_lists)):
            alpha_values = {
                name: float(alpha)
                for name, alpha in zip(concept_names, combo, strict=True)
            }
            expr = " + ".join(f"{alpha} {name}" for name, alpha in alpha_values.items())
            if base_str:
                expr = f"{base_str} + {expr}"
            jobs.append(GenerationJob(
                input=prompt,
                steering=expr,
                sampling=replace(base, seed=seed_schedule[idx]),
                raw=raw,
                thinking=thinking,
                parent_node_id=gen_parent_id,
                grid=alpha_values,
            ))
        return self.run_jobs(
            jobs,
            stateless=stateless,
            kind="fan",
            on_result=(
                (lambda idx, result, row: on_result(
                    idx, result, cast(dict[str, float], row),
                ))
                if on_result is not None else None
            ),
        )

    # -- serial execution ---------------------------------------------------

    def run_jobs(
        self,
        jobs: Iterable[GenerationJob],
        *,
        stateless: bool,
        kind: str,
        on_result: OnRow | None = None,
    ) -> RunSet:
        """Run ``jobs`` one at a time; a user stop ends the series."""
        stops = self._gen_state.stop_count
        results: list[GenerationResult] = []
        node_ids: list[str | None] = []
        grid_rows: list[dict[str, Any]] = []
        for idx, job in enumerate(jobs):
            result = self._generate_one(
                job.input,
                steering=job.steering,
                sampling=job.sampling,
                stateless=stateless,
                raw=job.raw,
                thinking=job.thinking,
                on_token=job.on_token,
                parent_node_id=job.parent_node_id,
                recipe_override=job.recipe_override,
                system_prompt=job.system_prompt,
                gen_seat=job.gen_seat,
            )
            row = dict(job.grid or {})
            results.append(result)
            node_ids.append(None if stateless else self._tree().active_node_id)
            grid_rows.append(row)
            if on_result is not None:
                on_result(idx, result, row)
            if self._gen_state.stop_count != stops:
                break
        return RunSet(results, node_ids=node_ids, grid=grid_rows, kind=kind)

    # -- shared-prefix KV reuse ---------------------------------------------

    def _cache_shared_prefix(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        stateless: bool,
        raw: bool,
    ) -> bool:
        """Prefill and cache the rows' shared token prefix when reuse is valid.

        Returns ``True`` iff a prefix was cached (the caller clears it after
        the batch).  Needs at least two stateless rows (per-row history breaks
        the shared prefix), prefill-inactive steering (else the prompt KV
        differs per row), no thinking and no ``return_hidden`` (both force a
        full re-prefill), and a shared run of at least
        :data:`PREFIX_CACHE_MIN_TOKENS`.  Lifecycle, rendering and prefill
        failures propagate.
        """
        if len(prompts) < 2 or not stateless or thinking:
            return False
        if sampling is not None and sampling.return_hidden:
            return False
        if not self._prefill_inactive(steering):
            return False
        # Prefix discovery walks tokens one by one: keep it on CPU so MPS/CUDA
        # don't pay a device sync per comparison.
        rows = [
            self._render(prompt, raw=raw, stateless=True, to_device=False)[0]
            .detach().to("cpu")
            for prompt in prompts
        ]
        shared = common_prefix(rows)
        if shared is None:
            return False
        max_new_tokens = (
            sampling.max_tokens
            if sampling is not None and sampling.max_tokens is not None
            else self._default_max_tokens()
        )
        cached = self._cache_prefix(
            shared,
            max_new_tokens=max_new_tokens,
            prefer_static=bool(self._static_cache_active() and steering is None),
        )
        return cached >= PREFIX_CACHE_MIN_TOKENS
