"""``torch.compile`` with a load-time probe of the call shapes drowse uses."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)


def run_compile_probes(compiled: Any, model: PreTrainedModel, device: str | torch.device, bos_token_id: int, *, mode: str) -> None:
    """Trigger compilation of the call shapes drowse actually uses.

    For ``mode="reduce-overhead"`` drowse's session generates through
    a ``StaticCache`` with an explicit ``cache_position`` kwarg — that
    is a *distinct dynamo call shape* from a plain DynamicCache
    forward, so a DynamicCache probe will not surface inductor
    codegen bugs that fire only under the static-cache path
    (observed: ``TypeError: Pointer argument must be either uint64 or
    have data_ptr method`` on Qwen3.5 + torch 2.12).  We mimic both
    code paths here so any per-shape codegen failure raises during
    load, not on the user's first generation.

    Each forward is single-shot — we materialize a scalar after each
    call before releasing the output, so the CUDA-graph capture path
    has nothing to alias even under ``mode="reduce-overhead"``.  Two
    distinct prefill lengths force dynamo's automatic-shape promotion
    so the dynamic-shape artifact (which the user will hit) is also
    compiled here.
    """
    def _probe_dynamic(prefill_len: int) -> None:
        ids = torch.full(
            (1, prefill_len), bos_token_id, device=device, dtype=torch.long,
        )
        attn = torch.ones_like(ids)
        out = compiled(input_ids=ids, attention_mask=attn, use_cache=True)
        _ = out.logits[:, -1, :].argmax().item()

    def _probe_static(prefill_len: int) -> None:
        from drowse.generation.static_cache import make_static_cache
        dtype = next(model.parameters()).dtype
        try:
            cache = make_static_cache(
                model, max_cache_len=prefill_len + 2,
                device=device, dtype=dtype,
            )
        except Exception as cache_exc:
            # If StaticCache can't be built for this model, the
            # generation loop won't use it either — session
            # construction has already gated on
            # ``is_cuda_graphs_supported`` before settling on
            # ``mode="reduce-overhead"``.  In load_model called
            # standalone or from incomplete mocks the gate is
            # absent, so skip the static probe silently.
            log.info("static-cache probe skipped: %s", cache_exc)
            return
        # Prefill — drowse's generation passes attention_mask on prefill.
        ids = torch.full(
            (1, prefill_len), bos_token_id, device=device, dtype=torch.long,
        )
        attn = torch.ones_like(ids)
        cache_position = torch.arange(prefill_len, device=device)
        out = compiled(
            input_ids=ids, attention_mask=attn,
            past_key_values=cache, use_cache=True,
            cache_position=cache_position,
        )
        # Materialize next-token onto a fresh CPU-side tensor so the
        # decode forward below has nothing aliased to a cudagraph
        # buffer in ``out``.
        next_id = int(out.logits[:, -1, :].argmax().item())
        del out
        # Decode — drowse passes ``attention_mask=None`` on decode steps;
        # that is a *distinct* dynamo call signature from the prefill
        # call, so it gets a separate compile artifact.  Inductor
        # codegen bugs that only fire under the decode-shape path
        # (observed: Qwen3.5 + torch 2.12) won't be caught by a
        # prefill-only probe.
        next_ids = torch.tensor([[next_id]], device=device, dtype=torch.long)
        cache_position = torch.tensor([prefill_len], device=device)
        out2 = compiled(
            input_ids=next_ids, attention_mask=None,
            past_key_values=cache, use_cache=True,
            cache_position=cache_position,
        )
        _ = out2.logits[:, -1, :].argmax().item()

    # Two prefill lengths so dynamo promotes to dynamic-shape codegen.
    for n in (2, 16):
        _probe_dynamic(n)
    if mode == "reduce-overhead":
        for n in (2, 16):
            _probe_static(n)


def compile_with_probe(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str | torch.device,
    *,
    mode: str = "default",
):
    """``torch.compile`` the model and verify the compiled artifact runs.

    Triggers compilation of both prefill and decode shapes via a tiny
    probe forward pair so that inductor / Triton failures surface here
    rather than at the user's first generation. On failure, warns and
    returns the *un-compiled* model so generation still works.

    Forces ``torch._inductor.config.use_static_cuda_launcher = False``
    for the process before compiling. Torch 2.12's static CUDA launcher
    dereferences kernel args without validation, so an inductor codegen
    bug (e.g. passing a symbolic shape where a tensor pointer is
    expected — observed on Gemma-4) segfaults uncatchably. Routing
    through Triton's regular launcher instead turns that into a
    catchable ``TypeError``, which the fallback below can handle.
    """
    try:
        import torch._dynamo  # noqa: F401
    except ImportError:
        log.info("torch.compile unavailable (no _dynamo); skipping")
        return model

    import torch._inductor.config as _ic
    _ic.use_static_cuda_launcher = False
    previous_matmul_precision: str | None = None
    if torch.device(device).type == "cuda":
        # Ada tensor cores accelerate the FP32 matmuls surrounding BF16 model
        # inference (notably live J-LENS transport/readout) with TF32 while
        # retaining FP32 range.  ``high`` is PyTorch's explicit opt-in.
        previous_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")

    log.info("Compiling model with torch.compile(mode=%r)", mode)
    compiled = torch.compile(model, mode=mode, dynamic=None)

    bos = (
        getattr(tokenizer, "bos_token_id", None)
        or getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )
    try:
        with torch.inference_mode():
            run_compile_probes(compiled, model, device, bos, mode=mode)
    except Exception as exc:
        if previous_matmul_precision is not None:
            torch.set_float32_matmul_precision(previous_matmul_precision)
        warnings.warn(
            f"torch.compile probe failed during warmup "
            f"({type(exc).__name__}: {str(exc)[:200]}); falling back to "
            "eager mode. Drop --compile (CLI) or compile=False "
            "(DrowseSession.from_pretrained) to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        log.info("torch.compile probe failed; using eager", exc_info=True)
        return model
    return compiled
