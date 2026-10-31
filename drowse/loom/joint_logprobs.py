"""Cross-branch joint logprobs for generated loom siblings.

Given two assistant LoomNodes A and B that share a parent, replay each
branch under the recipe stamped on that node and report, for each aligned
assistant-token position pair:

* ``lp_a_in_a`` / ``lp_b_in_b`` — chosen-token logprob under each
  branch's own distribution.  Mirrors what the engine already captures
  at generation time, recomputed here so the cross-evaluation and the
  self-evaluation use bit-identical math.
* ``lp_a_in_b`` / ``lp_b_in_a`` — chosen-token logprob under the *other*
  branch's distribution at the byte-aligned position.  Answers "what
  would B have given the token A picked here?" and vice versa.
* ``rank_changed`` — true iff the argmax token differs between the two
  distributions at this aligned position.  This is the canonical
  "steering shifted the head of the distribution, not just the
  argmax" signal.
* ``approx_kl`` — top-K-truncated KL(P_A || P_B), summed over the union
  of each side's top-K tokens.  The tail is unobserved, so this is
  documented as an approximate signal, not a full-distribution measurement.

Every replayed row is kept as the sampler's **support pool** — the
fixed-width top-k candidate set with its renormalized logprobs
(:class:`SamplerRows`).  Every token outside the pool has logprob ``-inf``
under the sampler, so the pool is the exact distribution at a fraction of a
dense ``[rows, vocab]`` table, and it crosses to the host once per branch.

A branch whose steering is position-independent (unsteered, compiled
offsets, or the static single-affine fast path) and carries no repetition
penalty replays teacher-forced: the forced stream runs in KV-cached chunks
that keep only the predictor logits.  Gated, phased, or curved steering and
penalties depend on the decode step, so those branches replay one forced
token per forward, with capture bound only when a probe gate reads it.

:class:`JointLogprobReplayer` runs the replays over explicit collaborators
(the tree, renderer, model, steering scope and manager, and the read plane for
probe-gated branches).  The route is called lazily when the comparison UI
opens; results cache on the session keyed by sorted ``(a_id, b_id)`` for the
session lifetime, and tree edits invalidate the entries.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from drowse.generation.decode import (
    GenerationConfig,
    _PenaltyState,
    _effective_topk,
    _sampler_candidates,
    build_chat_input,
)
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, SamplingConfig
from drowse.generation.scoring import shared_prefix_len
from drowse.loom.diff import per_token_diff
from drowse.model.chat import logits_keep_kwarg
from drowse.steering.terms import Steering

if TYPE_CHECKING:
    from drowse.generation.read_plane import ReadPlane
    from drowse.generation.render import PromptRenderer
    from drowse.loom.recipe import Recipe
    from drowse.loom.tree import LoomTree
    from drowse.model.chat import ChatTraits
    from drowse.steering.hooks import SteeringManager


# Truncation budget for the approximate KL.  ~32 covers the practical
# mass at typical sampler temperatures; we don't try to estimate the
# tail because the engine isn't shipping it.
_KL_TOP_K = 32

# Predictor rows per teacher-forced forward.  Bounds the kept
# ``[chunk, vocab]`` head output (~67 MB in bf16 at a 262k vocab).
_REPLAY_CHUNK = 128


@dataclass(frozen=True)
class JointLogprobRow:
    """One aligned-position record in a :class:`JointLogprobs` result.

    Indices are positions in each branch's *assistant-only* token list
    (i.e. relative to the divergence point), so they line up with the
    drawer's per-token row rendering.  Text fields carry the decoded
    token strings, ready for display without re-tokenization.
    """

    a_index: int
    b_index: int
    a_text: str
    b_text: str
    aligned: bool
    lp_a_in_a: float | None
    lp_b_in_b: float | None
    lp_a_in_b: float | None
    lp_b_in_a: float | None
    rank_changed: bool
    approx_kl: float | None


@dataclass(frozen=True)
class JointLogprobs:
    a_id: str
    b_id: str
    parent_id: str | None
    rows: tuple[JointLogprobRow, ...]
    n_rank1_changed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_id": self.a_id,
            "b_id": self.b_id,
            "parent_id": self.parent_id,
            "n_rank1_changed": self.n_rank1_changed,
            "rows": [
                {
                    "a_index": r.a_index,
                    "b_index": r.b_index,
                    "a_text": r.a_text,
                    "b_text": r.b_text,
                    "aligned": r.aligned,
                    "lp_a_in_a": r.lp_a_in_a,
                    "lp_b_in_b": r.lp_b_in_b,
                    "lp_a_in_b": r.lp_a_in_b,
                    "lp_b_in_a": r.lp_b_in_a,
                    "rank_changed": r.rank_changed,
                    "approx_kl": r.approx_kl,
                }
                for r in self.rows
            ],
        }


# ---------------------------------------------------------------------------
# Sampler-distribution rows (pure tensor, host side)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerRows:
    """Sampler support pools keyed by predictor row of the visible sequence.

    ``ids[i]`` / ``logps[i]`` are pool ``i``: candidate token ids sorted by
    descending logit and their sampler logprobs (``-inf`` where top-p cut an
    entry).  ``index`` maps a predictor row (position ``k`` predicts the token
    at ``k + 1``) to its pool.  Tokens absent from a pool have logprob
    ``-inf``.
    """

    index: dict[int, int]
    ids: torch.Tensor     # [M, W] long
    logps: torch.Tensor   # [M, W] fp32

    @classmethod
    def empty(cls) -> "SamplerRows":
        return cls(
            {}, torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.float32),
        )

    def pool(self, row: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        pool_idx = self.index.get(row)
        if pool_idx is None:
            return None
        return self.ids[pool_idx], self.logps[pool_idx]


def _lookup(pool: tuple[torch.Tensor, torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    """Sampler logprobs of ``tokens`` under one pool (``-inf`` off-pool)."""
    ids, logps = pool
    match = ids.unsqueeze(0) == tokens.unsqueeze(1)          # [U, W]
    found = match.any(dim=1)
    picked = logps[match.to(torch.int8).argmax(dim=1)]
    return torch.where(found, picked, torch.full_like(picked, float("-inf")))


def _pool_logprob(
    pool: tuple[torch.Tensor, torch.Tensor] | None, token_id: int,
) -> float | None:
    if pool is None or token_id < 0:
        return None
    value = float(_lookup(pool, torch.tensor([token_id]))[0])
    return value if math.isfinite(value) else None


def _pool_argmax(pool: tuple[torch.Tensor, torch.Tensor]) -> int:
    """Lowest token id among the pool's maximal entries (dense ``argmax``)."""
    ids, logps = pool
    return int(ids[logps == logps.max()].min())


def _approx_kl_topk(
    pool_a: tuple[torch.Tensor, torch.Tensor],
    pool_b: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
) -> float:
    """Truncated KL(P_A || P_B) over the union of each side's top-K.

    Within the union, contributions to ``Σ P_A(t) (log P_A(t) − log P_B(t))``
    are summed exactly; the tail (tokens neither side ranked in its top-K)
    is dropped — that's the "approx" in ``approx_kl``.  For typical
    post-sampler distributions K=32 captures the bulk; the residual is
    well under 0.1 nats in practice.  A pool is sorted by descending logit,
    so its first K entries are its top-K.
    """
    union = torch.unique(torch.cat([pool_a[0][:top_k], pool_b[0][:top_k]]))
    la = _lookup(pool_a, union)
    lb = _lookup(pool_b, union)
    support = torch.isfinite(la)
    if not bool(support.any()):
        return 0.0
    if bool((~torch.isfinite(lb[support])).any()):
        return float("inf")
    return float((la[support].exp() * (la[support] - lb[support])).sum())


def _compute_rows(
    rows_a: SamplerRows,
    rows_b: SamplerRows,
    token_ids_a: list[int],      # full sequence (prefix + assistant) for A
    token_ids_b: list[int],
    token_strs_a: list[str],     # decoded text per id (display) for A's full seq
    token_strs_b: list[str],
    prefix_len: int,             # shared prefix length (in tokens)
    *,
    kl_top_k: int = _KL_TOP_K,
) -> list[JointLogprobRow]:
    """Pure-tensor inner loop — no session / tokenizer / IO.

    Aligns A's assistant tail and B's assistant tail via the shared
    :func:`per_token_diff` byte-offset walker, then looks up logprobs in the
    replayed sampler pools.  Position ``prefix_len + i`` in the full sequence
    is *predicted* by the row at position ``prefix_len + i - 1`` — that's the
    row we read on each record.
    """
    assistant_ids_a = token_ids_a[prefix_len:]
    assistant_ids_b = token_ids_b[prefix_len:]
    spans = per_token_diff(token_strs_a[prefix_len:], token_strs_b[prefix_len:])

    rows: list[JointLogprobRow] = []
    for sp in spans:
        a_idx = sp.a_index
        b_idx = sp.b_index
        # ``max(0, …)`` guards the degenerate ``prefix_len == 0`` first row.
        pool_a = rows_a.pool(max(0, prefix_len + a_idx - 1))
        pool_b = rows_b.pool(max(0, prefix_len + b_idx - 1))
        has_a = 0 <= a_idx < len(assistant_ids_a)
        has_b = 0 <= b_idx < len(assistant_ids_b)

        lp_a_in_a = _pool_logprob(pool_a, assistant_ids_a[a_idx]) if has_a else None
        lp_b_in_b = _pool_logprob(pool_b, assistant_ids_b[b_idx]) if has_b else None

        # Cross-evaluation: only meaningful when the positions actually
        # align (byte-equal context up to here).  On divergent rows the
        # cross-prob is ambiguous (which prior position do we score
        # against?) so we leave it null.
        lp_a_in_b: float | None = None
        lp_b_in_a: float | None = None
        rank_changed = False
        approx_kl: float | None = None
        if sp.aligned and has_a and has_b and pool_a is not None and pool_b is not None:
            lp_a_in_b = _pool_logprob(pool_b, assistant_ids_a[a_idx])
            lp_b_in_a = _pool_logprob(pool_a, assistant_ids_b[b_idx])
            rank_changed = _pool_argmax(pool_a) != _pool_argmax(pool_b)
            kl = _approx_kl_topk(pool_a, pool_b, kl_top_k)
            approx_kl = kl if math.isfinite(kl) else None

        rows.append(JointLogprobRow(
            a_index=a_idx,
            b_index=b_idx,
            a_text=sp.a_text,
            b_text=sp.b_text,
            aligned=sp.aligned,
            lp_a_in_a=lp_a_in_a,
            lp_b_in_b=lp_b_in_b,
            lp_a_in_b=lp_a_in_b,
            lp_b_in_a=lp_b_in_a,
            rank_changed=rank_changed,
            approx_kl=approx_kl,
        ))
    return rows


# ---------------------------------------------------------------------------
# IO wrapper — talks to the tree, renderer, tokenizer, model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReplayBranch:
    prompt_ids: list[int]
    response_ids: list[int]
    token_ids: list[int]
    token_strs: list[str]
    thinking_ids: list[int]
    sampling: SamplingConfig
    steering: Steering | None


def _decode_each(tokenizer: Any, ids: list[int]) -> list[str]:
    # Decode per id rather than ``decode(ids)`` to keep the per-position
    # list aligned 1:1 with the id list.
    return [str(tokenizer.decode([tid])) for tid in ids]


def _row_token_ids(rows: list[dict[str, Any]] | None) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    texts: list[str] = []
    for row in rows or []:
        try:
            tid = int(row.get("token_id"))  # pyright: ignore[reportArgumentType]  # None raises TypeError, caught below
        except (TypeError, ValueError):
            continue
        if tid < 0:
            # Buffered partial UTF-8 rows do not correspond to a single
            # model token and cannot be forced through replay.
            continue
        ids.append(tid)
        text = row.get("text")
        texts.append(str(text) if text is not None else "")
    return ids, texts


def _sampling_from_recipe(recipe: "Recipe | None") -> SamplingConfig:
    sampling = recipe.sampling if recipe is not None else None
    return sampling if sampling is not None else SamplingConfig()


def _sampler_pool(
    logits: torch.Tensor, config: GenerationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ``[1, V]`` row's sampler support pool on device: ``(ids, logps)``."""
    ids, probs = _sampler_candidates(
        logits, config, _effective_topk(config, int(logits.shape[-1])),
    )
    logps = torch.where(
        probs > 0,
        probs.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.full_like(probs, float("-inf")),
    )
    return ids, logps


class _LogitAdjust:
    """The recipe's logit-space edits before the sampler transform."""

    def __init__(
        self, sampling: SamplingConfig, n_forced: int, device: torch.device,
    ) -> None:
        self.presence = sampling.presence_penalty
        self.frequency = sampling.frequency_penalty
        self.penalties = (
            _PenaltyState(max(n_forced, 1), device, torch.float32)
            if self.presence != 0.0 or self.frequency != 0.0 else None
        )
        bias = sampling.logit_bias or {}
        self.bias_idx = (
            torch.tensor(list(bias), dtype=torch.long, device=device)
            if bias else None
        )
        self.bias_val = (
            torch.tensor(list(bias.values()), dtype=torch.float32, device=device)
            if bias else None
        )

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Adjust ``[rows, V]`` logits in place and return them."""
        logits.nan_to_num_(nan=0.0, posinf=100.0, neginf=-100.0)
        if self.penalties is not None:
            self.penalties.apply(
                logits,
                presence_penalty=self.presence,
                frequency_penalty=self.frequency,
            )
        if self.bias_idx is not None:
            assert self.bias_val is not None
            logits[:, self.bias_idx] += self.bias_val.to(logits.dtype)
        return logits


def _prefill_kwargs(input_ids: torch.Tensor) -> dict[str, Any]:
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "use_cache": True,
    }


def _teacher_forced_pools(
    model: Any,
    full: torch.Tensor,
    first_row: int,
    *,
    adjust: _LogitAdjust,
    config: GenerationConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Predictor rows ``[first_row, len(full) - 1)`` in KV-cached chunks.

    Valid only for position-independent steering: every forced token runs
    through the same hooks whether it is prefill or decode.  A model that
    returns no cache re-feeds the prefix per chunk.
    """
    end = int(full.shape[1]) - 1
    keep_kwarg = logits_keep_kwarg(model)
    pools: list[tuple[torch.Tensor, torch.Tensor]] = []
    past: Any = None
    for start in range(first_row, end, _REPLAY_CHUNK):
        stop = min(start + _REPLAY_CHUNK, end)
        if past is None:
            kwargs = _prefill_kwargs(full[:, :stop])
        else:
            kwargs = {
                "input_ids": full[:, start:stop],
                "past_key_values": past,
                "use_cache": True,
            }
        if keep_kwarg is not None:
            kwargs[keep_kwarg] = stop - start
        out = model(**kwargs)
        past = getattr(out, "past_key_values", None)
        logits = adjust(out.logits[0, -(stop - start):, :])
        pools.extend(
            _sampler_pool(logits[r:r + 1], config)
            for r in range(stop - start)
        )
        del out, logits
    return pools


def _stepwise_pools(
    ctx: Any,
    model: Any,
    full: torch.Tensor,
    branch: _ReplayBranch,
    *,
    adjust: _LogitAdjust,
    config: GenerationConfig,
    gating_callback: Any,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One forced token per forward, driving the per-step trigger context."""
    ctx.reset()
    prompt_len = len(branch.prompt_ids)
    n_thinking = len(branch.thinking_ids)
    forced = branch.thinking_ids + branch.response_ids
    keep_kwarg = logits_keep_kwarg(model)
    pools: list[tuple[torch.Tensor, torch.Tensor]] = []
    past: Any = None
    cached = True
    for step, token_id in enumerate(forced):
        ctx.is_prefill = step == 0
        ctx.thinking = step < n_thinking
        ctx.gen_step = step
        stop = prompt_len + step
        if past is None:
            kwargs = _prefill_kwargs(full[:, :stop])
        else:
            kwargs = {
                "input_ids": full[:, stop - 1:stop],
                "past_key_values": past,
                "use_cache": True,
            }
        if keep_kwarg is not None:
            kwargs[keep_kwarg] = 1
        out = model(**kwargs)
        if gating_callback is not None:
            # The replay's forward index IS the step identity the live
            # decode loop threads to the gate sink.
            ctx.probe_scores = gating_callback(step)
        if cached:
            past = getattr(out, "past_key_values", None)
            cached = past is not None
        logits = adjust(out.logits[:, -1, :])
        if step >= n_thinking:
            pools.append(_sampler_pool(logits, config))
        if adjust.penalties is not None:
            adjust.penalties.add(token_id)
        del out, logits
    return pools


class JointLogprobReplayer:
    """Cross-evaluates two generated loom siblings by force-replaying both.

    Dependencies:

    - ``tree``: the live loom tree (a callable: the session swaps its tree
      on restore);
    - ``model`` / ``tokenizer``: the loaded model;
    - ``traits``: the chat facts (whether the family can think);
    - ``renderer``: rebuilds each node's prompt render (and resolves the
      default system prompt, the scene grammar and the model type);
    - ``profile_names``: the registered profile names a recipe's steering
      expression may reference;
    - ``gen_config``: the session :class:`GenerationConfig` under a recipe's
      sampling overrides;
    - ``open_scope``: opens a steering scope for a recipe's expression;
    - ``steering``: the :class:`SteeringManager` (the per-step trigger
      context and the static-path predicates);
    - ``needs_probe_gating``: whether the open steering reads probe gates;
    - ``reads``: the read plane that binds a gate-only capture for those.
    """

    def __init__(
        self,
        *,
        tree: Callable[[], "LoomTree"],
        model: Any,
        tokenizer: Any,
        traits: "ChatTraits",
        renderer: "PromptRenderer",
        profile_names: Callable[[], Iterable[str]],
        gen_config: Callable[[SamplingConfig | None], GenerationConfig],
        open_scope: Callable[[Steering], AbstractContextManager[Any]],
        steering: "SteeringManager",
        needs_probe_gating: Callable[[], bool],
        reads: "ReadPlane",
    ) -> None:
        self._tree = tree
        self._model = model
        self._tokenizer = tokenizer
        self._traits = traits
        self._renderer = renderer
        self._profile_names = profile_names
        self._gen_config = gen_config
        self._open_scope = open_scope
        self._steering = steering
        self._needs_probe_gating = needs_probe_gating
        self._reads = reads

    def compute(self, a_id: str, b_id: str) -> JointLogprobs:
        """Run cross-evaluation between two generated sibling nodes.

        Builds each branch's prompt through the chat template, force-replays
        its stored response tokens under that node's recipe, and assembles
        per-aligned-position records.  The caller serializes it against any
        concurrent generation on the same session (the server holds its
        session lock, ``app.state.session_lock``, around the call).

        Raises ``KeyError`` when either node id is unknown to the tree.
        Returns an empty-rows :class:`JointLogprobs` when the branches share
        no divergent generated tokens (e.g. one node is empty), which is the
        least-surprising shape for the drawer to render.
        """
        tree = self._tree()
        a_node = tree.nodes[a_id]
        b_node = tree.nodes[b_id]
        parent_id = a_node.parent_id if a_node.parent_id == b_node.parent_id else None
        branch_a = self._branch_inputs(tree, a_id)
        branch_b = self._branch_inputs(tree, b_id)
        ids_a = branch_a.token_ids
        ids_b = branch_b.token_ids
        prefix_len = shared_prefix_len(ids_a, ids_b)

        # If neither side has any assistant tokens past the prefix, return
        # an empty result — nothing to align.
        if prefix_len >= len(ids_a) and prefix_len >= len(ids_b):
            return JointLogprobs(
                a_id=a_id, b_id=b_id, parent_id=parent_id,
                rows=(), n_rank1_changed=0,
            )

        rows = _compute_rows(
            self._replay(branch_a),
            self._replay(branch_b),
            ids_a, ids_b, branch_a.token_strs, branch_b.token_strs, prefix_len,
        )
        n_changed = sum(1 for r in rows if r.aligned and r.rank_changed)
        return JointLogprobs(
            a_id=a_id, b_id=b_id, parent_id=parent_id,
            rows=tuple(rows), n_rank1_changed=n_changed,
        )

    def _branch_inputs(self, tree: "LoomTree", node_id: str) -> _ReplayBranch:
        tokenizer = self._tokenizer
        node = tree.nodes[node_id]
        recipe = node.recipe
        sampling = _sampling_from_recipe(recipe)
        steering = Steering.from_value(
            recipe.steering if recipe is not None else None,
            profile_names=set(self._profile_names()),
        )

        stamped_thinking = recipe.thinking if recipe is not None else None
        if stamped_thinking is not None:
            thinking = bool(stamped_thinking)
        elif steering is not None and steering.thinking is not None:
            thinking = bool(steering.thinking)
        else:
            thinking = self._traits.supports_thinking

        system_prompt = self._renderer.resolve_system_prompt(
            recipe.system_prompt if recipe is not None else INHERIT_SYSTEM_PROMPT
        )
        # Rebuild the exact prompt that opened this generated node.  Recipe
        # presence is the capability boundary; the structural seat may be
        # either human or model, and scene histories need not alternate.
        prompt_tensor = self._renderer.prepare_input(
            None,
            thinking=thinking,
            parent_node_id=node.parent_id,
            user_role=node.role_label if node.role == "user" else None,
            assistant_role=(
                node.role_label if node.role == "assistant" else None
            ),
            gen_seat=node.role,
            to_device=False,
            system_prompt=system_prompt,
        )
        prompt_ids = [int(t) for t in prompt_tensor[0].tolist()]

        response_ids, response_texts = _row_token_ids(node.tokens)
        thinking_ids, _thinking_texts = _row_token_ids(node.thinking_tokens)

        if not response_ids:
            # A committed (non-streamed) node: recover its tokens from the closed
            # render of the path through it.
            messages = tree.messages_for(node_id, with_labels=True)
            has_labels = any(m.get("label") for m in messages)
            full_input = build_chat_input(
                tokenizer,
                messages,
                system_prompt=system_prompt,
                thinking=thinking,
                add_generation_prompt=False,
                model_type=self._renderer.model_type if has_labels else None,
                scene=self._renderer.scene_grammar,
                gen_seat=node.role,
            )
            full_ids = [int(t) for t in full_input[0].tolist()]
            cut = shared_prefix_len(prompt_ids, full_ids)
            prompt_ids = full_ids[:cut]
            response_ids = full_ids[cut:]
            response_texts = _decode_each(tokenizer, response_ids)
        elif len(response_texts) != len(response_ids) or any(t == "" for t in response_texts):
            response_texts = _decode_each(tokenizer, response_ids)

        return _ReplayBranch(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            token_ids=prompt_ids + response_ids,
            token_strs=_decode_each(tokenizer, prompt_ids) + response_texts,
            thinking_ids=thinking_ids,
            sampling=sampling,
            steering=steering,
        )

    def _replay(self, branch: _ReplayBranch) -> SamplerRows:
        """Force-replay one branch and return its visible response-row pools.

        Rows are keyed by predictor position in the visible (prompt +
        response) sequence; prompt and thinking-only rows are never kept.
        """
        if not branch.response_ids:
            return SamplerRows.empty()
        if not branch.prompt_ids:
            raise ValueError("joint-logprob replay requires a non-empty prompt")

        model = self._model
        device = next(model.parameters()).device
        forced = branch.thinking_ids + branch.response_ids
        full = torch.tensor(
            [branch.prompt_ids + forced], dtype=torch.long, device=device,
        )
        config = self._gen_config(branch.sampling)
        adjust = _LogitAdjust(branch.sampling, len(forced), device)

        steering_cm: Any = contextlib.nullcontext()
        if branch.steering is not None and branch.steering.alphas:
            steering_cm = self._open_scope(branch.steering)

        with steering_cm, torch.inference_mode():
            manager = self._steering
            if adjust.penalties is None and (
                manager.all_fast_path() or manager.static_steerable()
            ):
                pools = _teacher_forced_pools(
                    model, full,
                    len(branch.prompt_ids) + len(branch.thinking_ids) - 1,
                    adjust=adjust, config=config,
                )
            elif self._needs_probe_gating():
                reads = self._reads
                gate_only = reads.resolve_demand(
                    want_hidden=False, live_scores_on=False,
                    has_monitor_probes=False, loom_attached=False,
                    wants_live_token_scores=False, persists_layer_scores=False,
                    persists_probe_row=False, persists_subspace_coords=False,
                    live_lens_active=False, live_sae_active=False,
                    final_probe_aggregate=False, capture_prompt=False,
                )
                try:
                    # Inside the try: ``begin`` binds every family's
                    # per-generation run, so a partial-bind failure must
                    # still reach the teardown below.
                    reads.begin(gate_only)
                    pools = _stepwise_pools(
                        manager.ctx, model, full, branch,
                        adjust=adjust, config=config,
                        gating_callback=reads.gate_score_callback(),
                    )
                finally:
                    # A bound run must not leak past the request (a stale
                    # lens pin suppresses disk refresh between generations;
                    # a stale SAE binding keeps serving frozen units to idle
                    # reads) — even when the hook detach itself raises.
                    try:
                        reads.end()
                    finally:
                        reads.close_runs()
            else:
                pools = _stepwise_pools(
                    manager.ctx, model, full, branch,
                    adjust=adjust, config=config,
                    gating_callback=None,
                )

        ids = torch.stack([pool_ids for pool_ids, _ in pools]).cpu()
        logps = torch.stack([pool_logps for _, pool_logps in pools]).cpu()
        first_visible = len(branch.prompt_ids) - 1
        return SamplerRows(
            {first_visible + j: j for j in range(len(pools))}, ids, logps,
        )

# ---------------------------------------------------------------------------
# Cache helpers — symmetric key so (a, b) and (b, a) dedupe
# ---------------------------------------------------------------------------


def _cache_key(a_id: str, b_id: str) -> tuple[str, str]:
    """Symmetric (a, b) ↔ (b, a) cache key.

    Sorted so the drawer hits the same entry regardless of which node
    the user right-clicked first.  Result layout still respects the
    caller's ``(a, b)`` orientation — see :func:`reorient_for_request`.
    """
    return (a_id, b_id) if a_id <= b_id else (b_id, a_id)


def reorient_for_request(
    result: JointLogprobs, requested_a_id: str, requested_b_id: str,
) -> JointLogprobs:
    """Flip the result's a/b labelling to match the caller's request.

    Cache stores under the sorted key, so a request for ``(B, A)`` after
    ``(A, B)`` was already computed needs the columns swapped before
    the drawer renders them.  Pure metadata work — no recomputation.
    """
    if (result.a_id, result.b_id) == (requested_a_id, requested_b_id):
        return result
    swapped_rows = tuple(
        JointLogprobRow(
            a_index=r.b_index,
            b_index=r.a_index,
            a_text=r.b_text,
            b_text=r.a_text,
            aligned=r.aligned,
            lp_a_in_a=r.lp_b_in_b,
            lp_b_in_b=r.lp_a_in_a,
            lp_a_in_b=r.lp_b_in_a,
            lp_b_in_a=r.lp_a_in_b,
            rank_changed=r.rank_changed,
            approx_kl=r.approx_kl,
        )
        for r in result.rows
    )
    return JointLogprobs(
        a_id=requested_a_id,
        b_id=requested_b_id,
        parent_id=result.parent_id,
        rows=swapped_rows,
        n_rank1_changed=result.n_rank1_changed,
    )


__all__ = [
    "JointLogprobReplayer",
    "JointLogprobRow",
    "JointLogprobs",
    "SamplerRows",
    "reorient_for_request",
]
