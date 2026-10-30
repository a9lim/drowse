"""Architecture facts: the residual-block list, tested families, model shape.

Drowse needs one structural fact from every decoder: where its sequential
residual blocks live.  That is discovered from the module tree rather than
looked up per ``model_type`` — every causal LM in ``transformers`` keeps its
blocks as a ``ModuleList`` at one of a handful of attribute paths, so a
per-family accessor table only restated those paths (and drifted: a
"tested" entry once had no accessor).  What *is* per family is whether drowse
has end-to-end coverage for it, which is :data:`TESTED_ARCHS`; any other
architecture loads with a one-time "untested" warning.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizerBase

from drowse.model.device import memory_gb

# Architectures with end-to-end coverage (smoke + session).  ``mistral`` is
# what a Mistral-Small-3 checkpoint loads as: the composite ``mistral3``
# config carries ``text_config.model_type == "mistral"``, so the text
# sub-model is extracted and the composite type never reaches
# :func:`get_layers`.  Ministral-3 has no ``text_config``, so its composite
# type is what loads.  The gemma VLM composites (``gemma3``/``gemma4``/
# ``gemma4_unified``) are listed because a caller may hand the full
# multimodal model to ``DrowseSession`` directly.
TESTED_ARCHS: frozenset[str] = frozenset({
    "llama",
    "mistral", "ministral3",
    "gemma2", "gemma3", "gemma3_text",
    "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text",
    "qwen2", "qwen3", "qwen3_5", "qwen3_5_text", "qwen3_5_moe",
    "glm",
    "gpt_oss",
    "talkie",
})

# Where decoders keep their residual blocks, in lookup order.  A text model
# keeps them under its base model (``model.layers``); a multimodal composite
# under its language model (``model.language_model.layers``); the GPT-2 /
# Bloom / Falcon / Qwen-1 shape under ``transformer.h``; MPT / DBRX under
# ``transformer.blocks``; GPT-NeoX / Pythia under ``gpt_neox.layers``; OPT
# under ``model.decoder.layers``; Talkie under ``model.blocks``.
_LAYER_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "layers"),
    ("model", "language_model", "layers"),
    ("model", "blocks"),
    ("model", "decoder", "layers"),
    ("transformer", "h"),
    ("transformer", "blocks"),
    ("gpt_neox", "layers"),
)

_warned: set[str] = set()


def _walk(model: Any, path: tuple[str, ...]) -> Any:
    node = model
    for attr in path:
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def get_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the sequential transformer blocks, found structurally.

    The first non-empty ``nn.ModuleList`` along :data:`_LAYER_PATHS` is the
    block list.  An architecture outside :data:`TESTED_ARCHS` warns once.
    """
    model_type = getattr(model.config, "model_type", None)
    for path in _LAYER_PATHS:
        layers = _walk(model, path)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            break
    else:
        raise ValueError(
            f"model_type {model_type!r}: no transformer block list found at "
            f"any known path ({', '.join('.'.join(p) for p in _LAYER_PATHS)})"
        )
    if model_type not in TESTED_ARCHS and model_type not in _warned:
        _warned.add(str(model_type))
        warnings.warn(
            f"architecture {model_type!r} is untested — "
            "report issues at https://github.com/stdlogits/drowse",
            UserWarning,
            stacklevel=2,
        )
    return layers


def text_extractable(text_cfg: Any) -> bool:
    """Whether a multimodal wrapper's ``text_config`` loads as its own causal LM.

    The text-extraction path builds the text model with
    ``AutoModelForCausalLM.from_config(text_cfg)``, so the predicate is exactly
    that precondition.  The loader's plan and the source fingerprint's payload
    canonicalization share it so a pre-load recompute reproduces the stamp.
    """
    if text_cfg is None:
        return False
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    return getattr(text_cfg, "model_type", None) in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def text_config(model: PreTrainedModel) -> Any:
    """Return the text-specific config, handling multimodal wrappers."""
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def config_model_shape(model_id: str) -> tuple[int, int]:
    """``(n_layers, hidden_dim)`` for ``model_id`` without loading weights.

    Reads the published config and unwraps a multimodal ``text_config``, so the
    numbers match what a loaded session would report — which is what every
    weight-free preflight compares against (the SAE's covered-layer and
    residual-width checks, the manifold fit no-op's requested-layer set).
    Raises when the config declares neither, because a *guessed* layer count
    would silently widen or narrow a proof.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    text_cfg = getattr(config, "text_config", config)
    n_layers = getattr(
        text_cfg, "num_hidden_layers", getattr(text_cfg, "n_layer", None),
    )
    hidden = getattr(
        text_cfg, "hidden_size", getattr(text_cfg, "n_embd", None),
    )
    if not isinstance(n_layers, int) or isinstance(n_layers, bool):
        raise ValueError(f"{model_id} config declares no layer count")
    if not isinstance(hidden, int) or isinstance(hidden, bool):
        raise ValueError(f"{model_id} config declares no hidden size")
    return int(n_layers), int(hidden)


def workspace_layer_indices(
    candidates: "list[int] | range", n_layers: int,
) -> list[int]:
    """Canonical 40–90% workspace-band filter with shallow-model fallback."""
    available = [int(layer) for layer in candidates]
    band = [
        layer for layer in available
        if 0.40 <= layer / max(n_layers - 1, 1) <= 0.90
    ]
    return band or available


def get_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """Summary dict: model_type, num_layers, hidden_dim, device, dtype, vram_used_gb, param_count."""
    layers = get_layers(model)
    first_param = next(model.parameters())
    model_id = getattr(model.config, "_name_or_path", "unknown")
    param_count = model.num_parameters()
    return {
        "model_id": model_id,
        "model_type": model.config.model_type,
        "num_layers": len(layers),
        "hidden_dim": text_config(model).hidden_size,
        "device": str(first_param.device),
        "dtype": str(first_param.dtype),
        "vram_used_gb": memory_gb(str(first_param.device)),
        "param_count": param_count,
    }
