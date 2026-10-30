"""Residual capture for fitting.

- Pooled extraction capture: render ``[system, *history, assistant]`` rows,
  locate each row's last content token, and run right-padded batched forwards
  that gather ``(B, D)`` per layer inside the hook and stop at the last
  requested block (:class:`_ReusablePooledCapture`,
  :func:`_encode_and_capture_all_batch`, :func:`compute_neutral_activations`).
- The shared elicitation corpora (baseline prompts, neutral statements).
- Discriminative Layer Selection (:func:`compute_dls_axes`).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from importlib import resources as _resources
from pathlib import Path
from typing import Any

import torch

from drowse.model.chat import takes_enable_thinking
from drowse.model.tokens import last_content_index, special_token_ids


# Rows per batched capture forward.  Capture starts here and halves on OOM
# (never grows back); length bucketing keeps a wide batch from turning one long
# response into quadratic padding work for every neighbour.
_CAPTURE_BATCH_MAX = 64


class _CaptureComplete(BaseException):
    """Private non-error control flow: the last requested block has fired.

    Causal-LM wrappers run final norm + the full vocabulary head after the
    transformer stack.  Fitting consumes only residual-stream activations, so
    the terminal capture hook aborts the wrapper before that unused projection.
    ``BaseException`` keeps broad ``except Exception`` blocks inside model
    wrappers from accidentally swallowing the sentinel.
    """


class _ReusablePooledCapture:
    """Layer hooks for the per-row pooled forward.

    Owns the whole per-row path: clamp each row's pool position into range,
    gather ``(B, D)`` inside the hook so the ``(B, T, D)`` stack is never
    retained, optionally promote to fp32, and abort the wrapper at the last
    requested block.  Fit-wide capture holds one instance across many forwards
    so the hooks are registered once; :func:`_capture_all_hidden_states` runs a
    single-use instance for a one-off per-row capture.
    """

    def __init__(
        self, model: torch.nn.Module, layers: torch.nn.ModuleList,
        layer_indices: Sequence[int],
    ) -> None:
        self.model = model
        self.layers = layers
        self.layer_indices = [int(idx) for idx in layer_indices]
        if not self.layer_indices:
            raise ValueError("layer_indices must select at least one layer")
        self.terminal_layer = max(self.layer_indices)
        self.captured: dict[int, torch.Tensor] = {}
        self._rows: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._promote = True
        self._handles: list[Any] = []

        def make_hook(idx: int):
            def hook(_module: torch.nn.Module, _input: Any, output: Any) -> None:
                h = output if isinstance(output, torch.Tensor) else output[0]
                assert self._rows is not None and self._positions is not None
                positions = self._positions.clamp(max=h.shape[1] - 1)
                pooled = h[self._rows, positions, :].detach()
                self.captured[idx] = (
                    pooled.to(torch.float32)
                    if self._promote and pooled.dtype != torch.float32
                    else pooled
                )
                if idx == self.terminal_layer:
                    raise _CaptureComplete()

            return hook

        try:
            for idx in self.layer_indices:
                self._handles.append(
                    layers[idx].register_forward_hook(make_hook(idx)),
                )
        except BaseException:
            self.close()
            raise

    def capture(
        self, input_ids: torch.Tensor, *,
        attention_mask: torch.Tensor | None,
        pool_index: torch.Tensor,
        promote_pooled: bool,
    ) -> dict[int, torch.Tensor]:
        self.captured = {}
        self._rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        self._positions = pool_index.to(input_ids.device)
        self._promote = promote_pooled
        try:
            with torch.inference_mode():
                try:
                    if attention_mask is not None:
                        self.model(
                            input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False,
                        )
                    else:
                        self.model(input_ids=input_ids, use_cache=False)
                except _CaptureComplete:
                    pass
            if input_ids.device.type == "mps":
                torch.mps.synchronize()
            return self.captured
        finally:
            self._rows = None
            self._positions = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "_ReusablePooledCapture":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _capture_all_hidden_states(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    pool_index: "int | torch.Tensor | None" = None,
    layer_indices: Sequence[int] | None = None,
    promote_pooled: bool = True,
    capture_context: _ReusablePooledCapture | None = None,
):
    """Run one forward pass capturing hidden states at selected layers.

    Uses ``use_cache=False`` to avoid polluting any persistent KV cache.
    ``attention_mask`` (when supplied) is forwarded so right-padded batches
    don't attend to pad tokens.

    ``pool_index`` selects what is retained per layer:

    - ``None`` — the full ``(B, seq, dim)`` hidden state (the default).
    - ``int`` — the single-sequence (``B == 1``) last-content pool: a ``(dim,)``
      fp32 vector at that position.
    - ``Tensor`` ``(B,)`` — the **per-row** batched pool: a ``(B, dim)`` fp32
      stack gathered at each row's position *inside the hook*, so only ``(B, D)``
      is retained per layer rather than the full ``(B, T, D)``.

    ``layer_indices`` narrows hook registration to a subset; ``None`` captures
    every layer. Returns a dict mapping layer index to the retained tensor.

    The per-row shape is served by a single-use :class:`_ReusablePooledCapture`
    — the same hook the fit-wide reusable path runs — so there is exactly one
    implementation of the in-hook gather. This function's own hook covers the
    ``None`` and scalar-``int`` shapes only.
    """
    capture_layers = (
        list(range(len(layers)))
        if layer_indices is None
        else [int(idx) for idx in layer_indices]
    )
    if capture_context is not None:
        if not isinstance(pool_index, torch.Tensor):
            raise ValueError("reusable capture requires a per-row pool_index tensor")
        if capture_layers != capture_context.layer_indices:
            raise ValueError(
                "reusable capture layer selection does not match this forward"
            )
        return capture_context.capture(
            input_ids, attention_mask=attention_mask, pool_index=pool_index,
            promote_pooled=promote_pooled,
        )

    if not capture_layers:
        raise ValueError("layer_indices must select at least one layer")
    if isinstance(pool_index, torch.Tensor):
        with _ReusablePooledCapture(model, layers, capture_layers) as ctx:
            return ctx.capture(
                input_ids, attention_mask=attention_mask,
                pool_index=pool_index, promote_pooled=promote_pooled,
            )

    captured_hidden: dict[int, torch.Tensor] = {}
    terminal_layer = max(capture_layers)

    def _make_hook(idx: int):
        def _hook(module: torch.nn.Module, input: Any, output: Any):
            h = output if isinstance(output, torch.Tensor) else output[0]
            if pool_index is not None:
                pos = min(max(int(pool_index), 0), h.shape[1] - 1)
                pooled = h[0, pos, :].detach()
                captured_hidden[idx] = (
                    pooled.to(torch.float32)
                    if promote_pooled and pooled.dtype != torch.float32
                    else pooled.clone()
                )
                if idx == terminal_layer:
                    raise _CaptureComplete()
                return
            # No .clone() — with use_cache=False and inference_mode() the
            # residual-stream tensors are fresh allocations at each layer
            # boundary (residual add produces a new tensor).  Detach severs
            # the autograd graph reference so the rest of the forward pass
            # can't invalidate the data.
            captured_hidden[idx] = h.detach()
            if idx == terminal_layer:
                raise _CaptureComplete()
        return _hook

    handles = [
        layers[idx].register_forward_hook(_make_hook(idx))
        for idx in capture_layers
    ]
    try:
        with torch.inference_mode():
            try:
                if attention_mask is not None:
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                else:
                    model(input_ids=input_ids, use_cache=False)
            except _CaptureComplete:
                pass
        # Single sync after the full forward pass — lazy backends (MPS)
        # may not have materialised tensor data yet.
        if input_ids.device.type == "mps":
            torch.mps.synchronize()
    finally:
        for h in handles:
            h.remove()

    return captured_hidden


# Shared one-paragraph length directive — a system prompt applied to BOTH
# generation (``session.generate_responses`` / ``generate_neutral_responses``)
# AND capture (:func:`_render_capture_text`), for node and neutral
# corpora alike.  Identical and symmetric across all four sites, so it is
# common-mode: it cancels in ``node − neutral`` (and ``concept − ν``) instead of
# leaking a "be brief" offset into the extracted directions, and it matches the
# capture framing to the generation framing so the pooled responses are not
# out-of-distribution.  Without it the model rambles past the token cap and
# truncates mid-thought.  (For a node, generation also prepends the persona to
# this system prompt; capture keeps only the directive — the persona stays
# generation-only, its signal carried in the response text, pooled in
# standard-assistant space.)
_LENGTH_DIRECTIVE = "Answer in one short paragraph."


def _encode_and_capture_all_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: "Sequence[str | list[dict[str, str]]]",
    responses: Sequence[str],
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    roles: Sequence[str | None] | None = None,
    model_type: str | None = None,
    layer_indices: Sequence[int] | None = None,
    rendered: "Sequence[tuple[torch.Tensor, int]] | None" = None,
    promote_pooled: bool = True,
    capture_context: _ReusablePooledCapture | None = None,
) -> dict[int, torch.Tensor]:
    """Batched conversational capture — one forward over a chunk of pairs.

    Each ``[system, user: prompt, assistant: response]`` turn is rendered +
    tokenized by :func:`_prepare_capture_batch` (or arrives pre-rendered), then
    the rows are **right-padded** to a common length and run through ONE forward
    pass.  Right-padding is the correct choice for capture (not generation):
    causal attention at a row's last-content position only sees positions to its
    *left*, all real tokens, so the pooled hidden state is identical to the
    unpadded single-sequence forward — the per-row ``content_end`` indices need
    no adjustment, and an ``attention_mask`` keeps the trailing pads out of every
    real token's context.  Pooling happens **inside the capture hook** (per-row
    gather at ``content_end``), so only ``(B, D)`` is retained per layer, never
    the full ``(B, T, D)``.

    The attention mask is built only when the chunk actually has ragged lengths;
    a uniform-length (or ``B == 1``) chunk runs full attention, bit-identical to
    the single-pair path.

    Returns ``{layer_idx: (B, D)}`` in fp32 (on the model device).

    Argument agreement (``responses`` / ``roles`` against ``prompts``) is
    checked by :func:`_prepare_capture_batch`; the pre-rendered branch checks
    the one alignment it consumes.
    """
    if rendered is None:
        rendered_rows = _prepare_capture_batch(
            tokenizer, prompts, responses, roles=roles, model_type=model_type,
        )
    else:
        if len(rendered) != len(prompts):
            raise ValueError(
                "rendered capture rows must align with prompts "
                f"({len(rendered)} != {len(prompts)})"
            )
        rendered_rows = list(rendered)
    # Keep the whole render/tokenize/pool-index front half on CPU.  Moving every
    # row to the accelerator and immediately calling ``tolist()`` to find its
    # content end forced one H2D + blocking D2H round-trip per corpus item before
    # the actual batched forward.  Padding on CPU also avoids a Python loop of
    # tiny device slice writes; the finished batch crosses the boundary once.
    seqs = [ids[0] for ids, _ in rendered_rows]          # each (L_i,) on CPU
    ends = [content_end for _, content_end in rendered_rows]
    lengths = [int(s.shape[0]) for s in seqs]
    max_len = max(lengths)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    input_ids_cpu = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=int(pad_id),
    )
    input_ids = input_ids_cpu.to(device)
    # An attention mask is only needed when padding is actually present; a
    # ragged chunk masks the pads, a uniform/B=1 chunk runs full attention
    # (matching the single-pair path exactly).
    if min(lengths) != max_len:
        lengths_cpu = torch.tensor(lengths, dtype=torch.long)
        attn = (
            torch.arange(max_len).unsqueeze(0) < lengths_cpu.unsqueeze(1)
        ).to(dtype=torch.long, device=device)
    else:
        attn = None
    pool_index = torch.tensor(ends, dtype=torch.long, device=device)
    return _capture_all_hidden_states(
        model, layers, input_ids,
        attention_mask=attn, pool_index=pool_index,
        layer_indices=layer_indices,
        promote_pooled=promote_pooled,
        capture_context=capture_context,
    )


def _prepare_capture_batch(
    tokenizer: Any,
    prompts: "Sequence[str | list[dict[str, str]]]",
    responses: Sequence[str],
    *,
    roles: Sequence[str | None] | None = None,
    model_type: str | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """Render/tokenize capture rows once for reuse across packing and retries.

    Returns ``(ids [1, T] on CPU, content_end)`` per row, where
    ``content_end`` is the response's last non-special token — chat templates
    append trailing markers (Llama's ``<|eot_id|>``, Gemma's ``<end_of_turn>``,
    Qwen's ``<|im_end|>``) whose hidden states are disconnected from content.
    """
    if len(prompts) != len(responses):
        raise ValueError(
            "capture preparation needs len(prompts) == len(responses) "
            f"({len(prompts)} != {len(responses)})"
        )
    if roles is not None and len(roles) != len(prompts):
        raise ValueError(
            "capture preparation needs len(roles) == len(prompts) "
            f"({len(roles)} != {len(prompts)})"
        )
    row_roles = roles if roles is not None else [None] * len(prompts)
    skip = special_token_ids(tokenizer)
    texts = [
        _render_capture_text(
            tokenizer, prompt, response, role=row_role, model_type=model_type,
        )
        for prompt, response, row_role in zip(
            prompts, responses, row_roles, strict=True,
        )
    ]
    # One batched call amortizes normalization/pretokenization over the corpus.
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    rows: list[tuple[torch.Tensor, int]] = []
    for token_row in encoded:
        values = [int(token) for token in token_row]
        if not values:
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                bos_id = tokenizer.eos_token_id or 0
            values = [int(bos_id)]
        rows.append((
            torch.tensor([values], dtype=torch.long),
            last_content_index(values, tokenizer, skip=skip),
        ))
    return rows


def _render_capture_text(
    tokenizer: Any,
    prompt: "str | list[dict[str, str]]",
    response: str,
    *,
    role: str | None = None,
    model_type: str | None = None,
) -> str:
    """Render one ``[system, *history, assistant]`` capture row, untokenized.

    The system turn is the shared :data:`_LENGTH_DIRECTIVE` (the framing the
    corpus was generated under, common-mode against neutral).  ``prompt`` is a
    single user turn or a multi-turn history (templated capture); ``role``
    substitutes the assistant label for a persona-baselined fit.
    """
    history: list[dict[str, str]] = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str)
        else [dict(t) for t in prompt]
    )
    if getattr(tokenizer, "chat_template", None) is not None:
        kwargs: dict[str, Any] = {}
        if takes_enable_thinking(tokenizer):
            kwargs["enable_thinking"] = False
        messages = [{"role": "system", "content": _LENGTH_DIRECTIVE}, *history]
        messages.append({"role": "assistant", "content": response})
        if role is None:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs,
            )
        if model_type is None:
            raise ValueError(
                "capture rendering: role= requires model_type= "
                "so the family's role-header registry entry can be looked up"
            )
        from drowse.model.role_templates import apply_with_role

        return apply_with_role(
            tokenizer, messages, role=role, model_type=model_type,
            add_generation_prompt=False, tokenize=False, **kwargs,
        )
    hist_text = "\n".join(turn["content"] for turn in history)
    return f"{_LENGTH_DIRECTIVE}\n{hist_text}\n{response}"


def _load_json_with_override(user_path: Path, resource_name: str) -> list[str]:
    """Read a bundled ``drowse.data`` JSON list, or the user's override of it.

    A file at ``user_path`` wins outright — the bundled resource is the
    fallback, never a merge base.
    """
    if user_path.exists():
        with open(user_path) as f:
            return json.load(f)
    with _resources.files("drowse.data").joinpath(resource_name).open() as f:
        return json.load(f)


def _load_neutral_prompts() -> list[str]:
    """Load neutral prompts, preferring a user override at ~/.drowse/neutral_statements.json."""
    from drowse.base.paths import neutral_statements_path
    return _load_json_with_override(
        neutral_statements_path(), "neutral_statements.json",
    )


def _load_baseline_prompts() -> list[str]:
    """Load the shared A2 baseline user prompts.

    Prefers a user override at ``~/.drowse/baseline_prompts.json``, else the
    bundled ``drowse/data/baseline_prompts.json`` (48 affect-neutral prompts).
    Conversational capture pairs each node response with its prompt by
    ``response[i] -> prompt[i % len(prompts)]``.
    """
    from drowse.base.paths import baseline_prompts_path
    return _load_json_with_override(
        baseline_prompts_path(), "baseline_prompts.json",
    )


def _neutral_pairs() -> list[tuple[str, str]]:
    """The neutral baseline as ``(prompt, response)`` pairs for conversational capture.

    The neutral corpus (``neutral_statements.json``) is the model's organic,
    no-persona / no-role responses to the shared baseline prompts (generated and
    captured under the shared length directive only — see :data:`_LENGTH_DIRECTIVE`);
    pairing is positional, ``response[i] -> baseline[i % k]`` — the same alignment
    a node corpus uses.  Raises if the corpus length is not a multiple of the
    prompt set (regenerate neutrals against the shared baseline).
    """
    responses = _load_neutral_prompts()
    baseline = _load_baseline_prompts()
    k = len(baseline)
    if k == 0:
        raise ValueError("no baseline prompts available for neutral capture")
    if len(responses) % k != 0:
        raise ValueError(
            f"neutral corpus ({len(responses)}) must be a multiple of the "
            f"baseline prompt set ({k}); regenerate neutrals against it"
        )
    return [(baseline[i % k], r) for i, r in enumerate(responses)]


def compute_neutral_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    layers: torch.nn.ModuleList,
    device: torch.device | None = None,
    *,
    rendered: Sequence[tuple[torch.Tensor, int]] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[int, torch.Tensor]:
    """Per-layer ``[N, D]`` stack across the neutral corpus.

    Last non-special-token pooling, fp32.  The ``(prompt, response)`` pairs are
    captured in **batched chunks** (:func:`_encode_and_capture_all_batch`, one
    forward per up to ``_CAPTURE_BATCH_MAX`` pairs) rather than one per pair, and
    the MPS allocator flush is amortized to once per chunk.  Returns one stacked
    tensor per layer (rows = pairs, original order preserved).  This is the
    single per-model neutral artifact: the Mahalanobis whitener's covariance is
    built from the stack and the probe-centering means are its per-layer
    ``X.mean(0)`` (:class:`drowse.session.neutral.NeutralBaseline`).
    Used by cross-model alignment (:func:`drowse.geometry.procrustes.fit_alignment`)
    which needs paired observations to fit Procrustes; the means alone (N=1)
    are degenerate for that fit.

    ``on_progress`` receives one line per captured chunk (the same narration
    surface :meth:`~drowse.session.session.DrowseSession.fit` and
    ``generate_responses`` expose), so a caller building the whitener can show
    the neutral pass rather than blocking silently.

    Storage cost: ~90 · n_layers · hidden_dim · 4B in fp32 (≈ 56MB on
    a 4096-dim, 80-layer model).  Callers persist this through
    :func:`drowse.fit.neutral.load_or_compute_neutral_activations`.
    """
    if device is None:
        device = next(model.parameters()).device
    assert device is not None  # device is always set by this point

    n_layers = len(layers)
    chunks_by_layer: dict[int, list[torch.Tensor]] = {
        idx: [] for idx in range(n_layers)
    }
    _mps = device.type == "mps"

    pairs = _neutral_pairs()
    prompts = [prompt for prompt, _ in pairs]
    responses = [response for _, response in pairs]
    if rendered is not None and len(rendered) != len(pairs):
        raise ValueError(
            "prepared neutral rows must align with the neutral corpus "
            f"({len(rendered)} != {len(pairs)})"
        )
    n = len(pairs)

    active_batch = _CAPTURE_BATCH_MAX
    capture_layers = list(range(n_layers))
    with _ReusablePooledCapture(
        model, layers, capture_layers,
    ) as capture_context:
        start = 0
        while start < n:
            end = min(start + active_batch, n)
            if on_progress is not None:
                on_progress(
                    f"Capturing neutral activations {start + 1}-{end}/{n}..."
                )
            try:
                per_layer = _encode_and_capture_all_batch(
                    model, tokenizer,
                    prompts[start:end], responses[start:end],
                    layers, device,
                    rendered=None if rendered is None else rendered[start:end],
                    capture_context=capture_context,
                )
            except RuntimeError as exc:
                from drowse.errors import is_out_of_memory_error

                if not is_out_of_memory_error(exc) or active_batch <= 1:
                    raise
                active_batch = max(1, active_batch // 2)
                if _mps:
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            # Move each chunk's (B, D) to CPU before discarding device rows.
            for idx in range(n_layers):
                chunks_by_layer[idx].append(per_layer[idx].detach().to("cpu"))
            del per_layer
            start = end
            if _mps:
                torch.mps.empty_cache()

    return {
        idx: torch.cat(chunks, dim=0)  # (N, D), fp32 on cpu
        for idx, chunks in chunks_by_layer.items()
    }


def compute_dls_axes(
    node_centroids: dict[int, torch.Tensor],
    bases: dict[int, torch.Tensor],
    layer_means: dict[int, torch.Tensor] | None,
) -> dict[int, set[int]]:
    """Per-axis Discriminative-Layer-Selection over **N node centroids**.

    The N-node generalization of the bipolar opposite-sign test (Dang & Ngo
    2026, Eq. 9) — the form a flat *subspace* of any node count uses.  Each
    layer carries a basis ``[R, D]`` (its ``LayerSubspace.basis`` rows) and
    ``node_centroids[L]`` is the ``[K, D]`` stack of that layer's per-node
    mean activations.  Every row ``d̂_r`` is kept iff the node projections
    relative to the neutral baseline, ``{(c_i − ν)·d̂_r : i}``, **straddle
    zero** (both a negative and a positive present) — the axis then
    discriminates node *position* across the baseline rather than encoding a
    common offset / intensity every node shares.

    **N=2 parity (the spine invariant).**  With
    ``node_centroids[L] = stack([μ_pos, μ_neg])`` the straddle
    ``min < 0 < max`` is exactly the bipolar opposite-sign product test, bit
    for bit, so a 2-node subspace reproduces a difference-of-means steering
    vector's DLS keep set exactly (the ``R = 1`` layer set), and an N-node
    subspace (e.g. ``personas``) prunes per-axis the same way.

    Returns ``{layer: {kept axis indices}}``; a layer absent / mapped to an
    empty set is fully dropped.  ``layer_means`` ``None``/empty disables DLS
    (every axis of every layer kept).  A layer whose baseline ``ν`` is missing
    keeps every checkable axis conservatively (over-include rather than drop a
    real discriminative axis for missing data).  Degenerate (zero-norm) rows
    are dropped and excluded from the all-fail fallback; when no axis anywhere
    passes, falls back to the full checkable set with a one-time warning.

    Args:
        node_centroids: per-layer ``[K, D]`` stack of node mean activations.
        bases: per-layer ``[R, D]`` basis (rows are the candidate axes).
        layer_means: per-layer neutral baseline; ``None``/empty disables.
    """
    if layer_means is None or not layer_means:
        out: dict[int, set[int]] = {}
        for L in node_centroids:
            B = bases.get(L)
            r = int(B.shape[0]) if B is not None else 1
            out[L] = set(range(r))
        return out
    keep: dict[int, set[int]] = {}
    # ``checkable`` collects the (layer, axis) pairs we *attempted* the test
    # on — rows with a valid (non-degenerate) direction.  Rows skipped for
    # zero norm are excluded from the all-failed fallback.
    checkable: dict[int, set[int]] = {}
    any_pass = False
    for L in node_centroids:
        B = bases.get(L)
        if B is None:
            continue
        B32 = B.to(dtype=torch.float32, device="cpu").reshape(int(B.shape[0]), -1)
        mu_n = layer_means.get(L)
        mu_n_cpu = (
            mu_n.to(dtype=torch.float32, device="cpu").reshape(-1)
            if mu_n is not None
            else None
        )
        C = node_centroids[L].to(dtype=torch.float32, device="cpu")
        if C.ndim == 1:
            C = C.reshape(1, -1)
        C = C.reshape(int(C.shape[0]), -1)              # (K, D)
        norms = torch.linalg.vector_norm(B32, dim=1)
        valid = norms >= 1e-12
        layer_checkable = set(valid.nonzero(as_tuple=False).reshape(-1).tolist())
        layer_keep: set[int]
        if mu_n_cpu is None:
            # Baseline doesn't cover this layer. Conservative: keep every
            # non-degenerate axis rather than drop genuine signal.
            layer_keep = set(layer_checkable)
        elif layer_checkable:
            normalized = B32[valid] / norms[valid].unsqueeze(1)
            projs = (C - mu_n_cpu) @ normalized.transpose(0, 1)  # (K, R_valid)
            straddles = (projs.amin(dim=0) < 0.0) & (projs.amax(dim=0) > 0.0)
            valid_axes = valid.nonzero(as_tuple=False).reshape(-1)
            layer_keep = set(valid_axes[straddles].tolist())
        else:
            layer_keep = set()
        if layer_checkable:
            checkable[L] = layer_checkable
        if layer_keep:
            keep[L] = layer_keep
            any_pass = True
    if not any_pass:
        # Every checkable axis failed — probe degenerate on this model (the
        # diagnostics warning fires separately).  Keep the checkable set
        # rather than every layer; when nothing is checkable the fallback is
        # empty and the share-bake all-zero handler kicks in.
        warnings.warn(
            "DLS: no layers pass the discriminative check; falling back "
            "to keep-all-checkable.  Probe likely degenerate on this "
            "model — review the diagnostics warning above.",
            UserWarning,
            stacklevel=3,
        )
        return checkable
    return keep
