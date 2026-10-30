"""Load a Hugging Face causal LM and its tokenizer.

:func:`load_model` resolves a :class:`LoadPlan` from configs alone (no weight
I/O), materializes weights through the text-extraction path or the
``from_pretrained`` fallback ladder, then finalizes identity, frozen state and
the chat stop-token set.  Compilation is the session's job
(:mod:`drowse.model.compile`), not the loader's.
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, cast

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from drowse.model.arch import text_extractable
from drowse.model.checkpoint import NoTextWeightsExtracted, load_text_from_multimodal
from drowse.model.device import detect_device, memory_gb, patch_torch_for_mps, resolve_dtype
from drowse.model.fingerprint import (
    _model_state_signature,
    _pinned_revision,
    model_source_fingerprint,
)

log = logging.getLogger(__name__)

_MODEL_LOAD_LOCK = threading.RLock()


def load_model(
    model_id: str,
    quantize: str | None = None,
    device: str = "auto",
    dtype: torch.dtype | str | None = None,
    *,
    trust_remote_code: bool | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_id: Hub ID or local path.
        quantize: None for bf16/fp16/fp32, "4bit", or "8bit".
        device: "auto" (detect), "cuda", "mps", or "cpu".
        dtype: torch.dtype or string ("bfloat16"/"float16"/"float32"). None
            picks the device default — bf16 on CUDA/MPS, fp32 on CPU.
        trust_remote_code: Permit repository Python only when explicitly True,
            or when unset and DROWSE_TRUST_REMOTE_CODE is enabled. Defaults off.
        on_progress: Optional per-step status sink.  The loader's own
            status goes to ``log.info``, which is invisible unless the
            embedding application configured logging — a library must not
            configure it globally — so a CLI/server caller passes a callback
            to surface the device / weights / memory lines instead.

    Returns:
        (model, tokenizer) tuple.
    """
    if trust_remote_code is None:
        trust_remote_code = os.environ.get("DROWSE_TRUST_REMOTE_CODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if type(trust_remote_code) is not bool:
        raise ValueError("trust_remote_code must be a boolean")
    device = detect_device(device)
    if device == "mps":
        patch_torch_for_mps()
    log.info("Device: %s", device)
    if on_progress is not None:
        on_progress(f"Device: {device}")

    plan = _resolve_load_plan(model_id, quantize=quantize, device=device, dtype=dtype, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, **plan.tokenizer_kwargs, **plan.pin_kwargs,
    )
    if on_progress is not None:
        on_progress(f"Loading weights ({plan.dtype})...")
    model = _materialize_model(plan)
    model = _finalize_model(
        model, tokenizer, plan, on_progress=on_progress,
    )
    return cast(PreTrainedModel, model), tokenizer


@dataclass
class LoadPlan:
    """Every decision :func:`load_model` makes before a weight is read.

    Produced by :func:`_resolve_load_plan` from the model id and the caller's
    knobs plus two config reads — no weight I/O — so the decisions (which
    attention impl, which dtype, whether to extract a text sub-model, whether
    to trust remote code) are inspectable and unit-testable without loading a
    checkpoint.  :func:`_materialize_model` consumes it and owns the fallback
    ladder; :func:`_finalize_model` owns the post-load stamping.

    ``load_kwargs`` is mutable on purpose: the materialize step's fallback
    ladder rewrites ``attn_implementation`` / ``dtype`` / ``device_map`` in it
    as each retry narrows.
    """

    model_id: str
    device: str
    dtype: torch.dtype
    quantize: str | None
    revision: str | None
    pin_kwargs: dict[str, str]
    tokenizer_kwargs: dict[str, Any]
    load_kwargs: dict[str, Any]
    config: Any
    text_config: Any
    extract_text_model: bool


def _resolve_load_plan(
    model_id: str,
    *,
    quantize: str | None,
    device: str,
    dtype: torch.dtype | str | None,
    trust_remote_code: bool = False,
) -> LoadPlan:
    """Decide how to load ``model_id``; read configs, never weights."""
    resolved_dtype = resolve_dtype(dtype, device)

    # Resolve the Hub commit once, then pin every config, tokenizer, custom
    # shard, and model read to it. Without a single pin, a mutable Hub branch
    # can advance between independent ``from_pretrained`` calls and leave
    # caches stamped with config A over weights B. Local paths carry no commit
    # and remain path-hash identified.
    resolved_revision = _pinned_revision(model_id)
    pin_kwargs: dict[str, str] = (
        {"revision": resolved_revision} if resolved_revision else {}
    )
    probe_config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, **pin_kwargs,
    )

    # HF-distributed Mistral checkpoints ship a buggy pre-tokenizer regex
    # that mis-splits ~1% of tokens (e.g. ``"'The'"`` → ``["'", "T", "he", "'"]``
    # instead of ``["'", "The", "'"]``).  ``fix_mistral_regex=True`` swaps in the
    # correct regex from ``mistral_common``.  Substring-match on the model_id
    # so it fires for any mistralai/* repo (Mistral-Small, Ministral, etc.) and
    # third-party finetunes whose name carries the family.
    # https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if "mistral" in model_id.lower():
        tokenizer_kwargs["fix_mistral_regex"] = True

    # --- quantization config ---
    if quantize and device != "cuda":
        warnings.warn(
            f"bitsandbytes quantization ({quantize}) requires CUDA. "
            f"Ignoring --quantize on {device}, loading in {resolved_dtype}.",
            stacklevel=2,
        )
        quantize = None

    # --- attention implementation ---
    attn_impl = "sdpa"  # safe default, works everywhere
    if device == "cuda":
        try:
            # Availability check only — presence of the package flips
            # transformers onto the flash-attention-2 kernel. We never
            # call into flash_attn ourselves.
            import flash_attn  # pyright: ignore[reportMissingImports]  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

    # --- device map ---
    # device_map="auto" requires accelerate and works best on CUDA.
    # For MPS/CPU, place the whole model on a single device.
    device_map = "auto" if device == "cuda" else {"": device}

    # --- trust_remote_code gating ---
    # Some repos (e.g. deepseek-ai/DeepSeek-V2-Lite-Chat) ship an
    # ``auto_map`` pointing to a stale ``modeling_*.py`` that breaks
    # against newer transformers.  When the architecture is already
    # supported natively, skip the custom code entirely.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    native_type = getattr(probe_config, "model_type", None)
    native_text_type = getattr(getattr(probe_config, "text_config", None),
                               "model_type", None)
    trust = trust_remote_code and not (
        (native_type and native_type in CONFIG_MAPPING)
        or (native_text_type and native_text_type in CONFIG_MAPPING)
    )

    # MLA architectures (DeepSeek-V2/V3) decompose attention into a
    # query/key head_dim = qk_nope_head_dim + qk_rope_head_dim (192 on
    # V2-Lite) and a separate value head_dim = v_head_dim (128).  PyTorch
    # 2.11's MPS scaled_dot_product_attention kernel returns the output
    # with the *query* head_dim instead of the value head_dim when the
    # two differ — attn_output ends up (B, T, n_heads * 192) instead of
    # (B, T, n_heads * 128), and o_proj rejects it with
    # ``linear(): input and weight.T shapes cannot be multiplied``.
    # CPU SDPA returns the correct shape; CUDA SDPA also handles this
    # via the flash backend.  Narrowest-correct fix: force eager on MPS
    # for these models.  The ``torch.histc`` MPS patch already installed by
    # ``load_model`` covers the MoE-routing issue eager would otherwise hit.
    _MLA_TYPES = {"deepseek_v2", "deepseek_v3"}
    if device == "mps" and (
        native_type in _MLA_TYPES or native_text_type in _MLA_TYPES
    ) and attn_impl != "eager":
        log.info(
            "forcing eager attention on MPS for MLA model %r "
            "(PyTorch MPS SDPA mishandles mismatched q/v head_dim)",
            native_type or native_text_type,
        )
        attn_impl = "eager"

    load_kwargs: dict[str, Any] = {
        "attn_implementation": attn_impl,
        "trust_remote_code": trust,
        "device_map": device_map,
        **pin_kwargs,
    }
    if quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = resolved_dtype

    # --- check for multimodal configs wrapping a text-only model ---
    # Some text-only models ship with a multimodal config whose
    # model_type isn't registered with AutoModelForCausalLM (e.g.
    # Ministral tagged as Mistral3).  If the config has a text_config
    # that IS a known causal-LM type, use that instead.
    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust, **pin_kwargs,
    )
    text_cfg = getattr(config, "text_config", None)
    # A non-None text_config means a multimodal wrapper. When it wraps a
    # text model drowse supports, load *only* the text submodel: it skips
    # the unused vision tower and — crucially — loads on-device shard-by-
    # shard so peak memory stays ~1× the model size instead of the ~2× the
    # standard CPU-stage-then-device-copy path costs on unified memory (the
    # >128 GB gemma-4-31B blowup). This fires even when the outer composite
    # is itself loadable (gemma3/gemma4): the full
    # multimodal model is still usable when handed to DrowseSession
    # directly, but the from_pretrained path prefers text-only.
    #
    # Under bitsandbytes quantization the extraction path is skipped: its
    # shard streamer copies raw weights into a ``from_config`` skeleton and
    # cannot apply a quantizer, so it would silently produce an unquantized
    # model stamped as quantized. The standard ``from_pretrained`` path
    # quantizes the full multimodal model correctly (quantize survives only
    # on CUDA, where the unified-memory discipline is moot).
    extractable = text_extractable(text_cfg)
    extract_text_model = extractable and quantize is None
    if extractable and quantize is not None:
        log.info(
            "quantization requested; loading the full multimodal model "
            "(the text-extraction path cannot quantize)",
        )

    return LoadPlan(
        model_id=model_id,
        device=device,
        dtype=resolved_dtype,
        quantize=quantize,
        revision=resolved_revision,
        pin_kwargs=pin_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        load_kwargs=load_kwargs,
        config=config,
        text_config=text_cfg,
        extract_text_model=extract_text_model,
    )


@contextmanager
def _model_load_environment(device: str) -> Iterator[None]:
    # Transformers' concurrent dtype/device copies can segfault on MPS.
    # Scope its supported synchronous-loading flag to the load, not inference.
    with _MODEL_LOAD_LOCK:
        previous = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
        if device == "mps":
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        try:
            yield
        finally:
            if device == "mps":
                if previous is None:
                    os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
                else:
                    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = previous


class _IncompleteCheckpointError(ValueError):
    """A checkpoint would leave required model parameters newly initialized."""


def _load_pretrained_checked(plan: LoadPlan) -> Any:
    model, loading_info = cast(
        tuple[Any, dict[str, Any]],
        AutoModelForCausalLM.from_pretrained(
            plan.model_id, output_loading_info=True, **plan.load_kwargs,
        ),
    )
    parameters = {name for name, _ in model.named_parameters()}
    missing = parameters.intersection(loading_info["missing_keys"])
    if missing:
        raise _IncompleteCheckpointError(
            f"checkpoint is missing model parameters: {sorted(missing)!r}"
        )
    return model


def _load_with_fallbacks(plan: LoadPlan) -> Any:
    """One ``from_pretrained`` attempt plus the attn / dtype retries.

    Each retry narrows ``plan.load_kwargs`` in place so a caller that retries
    this whole function (the device-conversion fallback) inherits the
    already-narrowed kwargs rather than repeating a known-bad attempt.
    """
    load_kwargs = plan.load_kwargs
    with _model_load_environment(plan.device):
        try:
            return _load_pretrained_checked(plan)
        except _IncompleteCheckpointError:
            raise
        except ValueError as e:
            if "does not support an attention implementation" not in str(e):
                raise
            log.info("attn_implementation %r unsupported, falling back to eager",
                     load_kwargs.get("attn_implementation"))
            load_kwargs["attn_implementation"] = "eager"
            return _load_pretrained_checked(plan)
        except Exception:
            if plan.quantize is not None:
                raise
            fallback = torch.float16 if plan.device == "cuda" else torch.float32
            load_kwargs["dtype"] = fallback
            return _load_pretrained_checked(plan)


def _materialize_model(plan: LoadPlan) -> Any:
    """Read weights per ``plan`` — text extraction, then the fallback ladder."""
    model = None
    text_cfg = plan.text_config
    if plan.extract_text_model:
        # Weights live under a "language_model." path segment that doesn't
        # match the text-only model's parameter names.  Load manually.
        # Propagate _name_or_path so cache paths resolve correctly.
        assert text_cfg is not None  # guaranteed by plan.extract_text_model
        if not getattr(text_cfg, "_name_or_path", ""):
            text_cfg._name_or_path = plan.model_id
        log.info("extracting text model (%s) from multimodal checkpoint (%s)",
                 text_cfg.model_type, plan.config.model_type)
        try:
            model = load_text_from_multimodal(
                plan.model_id, text_cfg,
                plan.load_kwargs.get("dtype", plan.dtype),
                plan.device, revision=plan.revision,
            )
        except NoTextWeightsExtracted as e:
            # Unexpected weight layout — don't ship random init; fall through
            # to the standard full-model load below.
            log.warning("%s; falling back to full multimodal load", e)
            model = None

    if model is not None:
        return model

    # --- standard load (with attention, dtype, and device fallbacks) ---
    try:
        return _load_with_fallbacks(plan)
    except _IncompleteCheckpointError:
        raise
    except (RuntimeError, ValueError) as e:
        if (
            isinstance(e, ValueError)
            and plan.quantize is not None
            and text_extractable(text_cfg)
            and "Unrecognized configuration class" in str(e)
        ):
            # A composite AutoModelForCausalLM cannot load (Ministral-as-
            # Mistral3) and the only working route — text extraction — cannot
            # quantize.  Fail with the actionable cause, not HF's registry
            # error.
            raise RuntimeError(
                f"{plan.model_id!r} is a multimodal checkpoint drowse can "
                "only load through the text-extraction path, which does not "
                "support bitsandbytes quantization; load without quantize=."
            ) from e
        if plan.device in ("cuda", "cpu") or "CONVERSION" not in str(e):
            raise
        log.info("weight conversion failed on %s, retrying on CPU", plan.device)
        plan.load_kwargs["device_map"] = {"": "cpu"}
        if "dtype" not in plan.load_kwargs:
            plan.load_kwargs["dtype"] = torch.float32
        model = _load_with_fallbacks(plan)
        return model.to(plan.device)  # pyright: ignore[reportArgumentType]  # transformers stub: .to(str) overload missing


def _finalize_model(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    plan: LoadPlan,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    """Stamp identity, freeze, and reconcile the chat stop-token set."""
    # The live config's ``_commit_hash`` is the slot every identity check reads
    # (external J-lens bindings, capture fingerprints).  Transformers 5.18 and
    # later never fill it, and on older releases the text-only multimodal path
    # lost it anyway (``from_config`` builds a fresh child config), which made
    # an official J-lens fetched for the same immutable checkpoint look stale.
    # The revision was resolved before any weights were read and every read
    # above was pinned to it, so stamp that proven identity uniformly.
    if plan.revision:
        live_config = getattr(model, "config", None)
        if live_config is not None:
            live_config._commit_hash = plan.revision
            live_text_config = getattr(live_config, "text_config", None)
            if live_text_config is not None:
                live_text_config._commit_hash = plan.revision

    model.requires_grad_(False)
    model.train(False)
    # Mark only drowse-owned loads as source-trusted.  Arbitrary modules whose
    # configs merely claim the same commit must still take the exact full-state
    # fingerprint path.  ``dtype=plan.dtype`` is what the text-extraction path
    # handed ``from_config`` (immutable against the fallback ladder), so the
    # payload canonicalization's dtype write no-ops against the mutated config.
    source_fingerprint = model_source_fingerprint(
        plan.model_id, quantize=plan.quantize, device=plan.device,
        dtype=plan.dtype, config=plan.config, revision=plan.revision,
        parameter_dtype=(
            "quantized" if plan.quantize is not None
            else next(model.parameters()).dtype
        ),
    )
    if source_fingerprint is not None:
        setattr(model, "_drowse_source_fingerprint", source_fingerprint)
        setattr(model, "_drowse_loaded_state_signature", _model_state_signature(model))
        setattr(model, "_drowse_state_invalidated", False)

    # --- reconcile the chat stop-token set -------------------------------
    # A multimodal checkpoint's *text* sub-config can under-specify the
    # end-of-turn tokens its chat template actually emits.  Gemma-4 is the
    # live case: the composite config lists ``eos_token_id = [<eos>, <turn|>]``
    # but the extracted ``gemma4_unified_text`` sub-config (what
    # ``load_text_from_multimodal`` builds from via ``from_config``) carries
    # only ``<eos>``, so the loaded model's ``generation_config`` claims the
    # sole stop token is ``<eos>`` while the model ends every turn with
    # ``<turn|>``.  HF ``model.generate`` (the native-generate corpus path,
    # ``session._run_generator``) reads ``generation_config.eos_token_id``, so
    # without this it runs past every ``<turn|>`` to ``max_new_tokens`` — the
    # model loops and leaks reasoning-channel name tokens (``thought``) that
    # ``skip_special_tokens`` can't strip.  ``_get_eos_ids`` is the
    # chat-correct set (config eos ∪ tokenizer eos ∪ the template's added
    # end-of-turn tokens) and is exactly what the steered loop already stops
    # on, so adopting it as the generation default unifies both paths.
    from drowse.generation.decode import _get_eos_ids
    chat_eos = sorted(_get_eos_ids(model, tokenizer))
    gen_cfg = getattr(model, "generation_config", None)
    if chat_eos and gen_cfg is not None:
        gen_cfg.eos_token_id = chat_eos if len(chat_eos) > 1 else chat_eos[0]

    # --- memory report ---
    mem_gb = memory_gb(plan.device)
    if mem_gb > 0:
        log.info("Memory used: %.2f GB", mem_gb)
        if on_progress is not None:
            on_progress(f"Memory used: {mem_gb:.2f} GB")

    return model
