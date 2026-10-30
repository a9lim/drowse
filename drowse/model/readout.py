"""The final-residual-to-logits path: unembedding, final norm, softcap."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from drowse.model.arch import get_layers, text_config


def get_unembedding(model: PreTrainedModel) -> torch.Tensor:
    """Return the unembedding matrix ``W_U``, shape ``[vocab, head_width]``.

    Uses the HF-standard ``get_output_embeddings()`` (the lm_head module,
    whether or not its weight is tied to the input embedding). Nothing else in
    drowse touches the unembedding outside the model's own forward; this
    accessor exists for the Jacobian lens readout. Projected OPT models have
    a head width distinct from the residual width; see get_output_projection.
    """
    out = model.get_output_embeddings()
    weight = getattr(out, "weight", None)
    if weight is None:
        raise ValueError(
            f"model_type {model.config.model_type!r} exposes no output "
            "embedding weight (get_output_embeddings() returned "
            f"{type(out).__name__}); cannot build a vocabulary readout"
        )
    return weight


_FINAL_NORM_ATTRS = ("norm", "final_layernorm", "final_layer_norm", "ln_f", "final_norm")


def get_logit_softcap(model: PreTrainedModel) -> float | None:
    return getattr(text_config(model), "final_logit_softcapping", None)


def get_output_projection(model: PreTrainedModel) -> torch.Tensor | None:
    if model.config.model_type == "opt":
        projection = model.model.decoder.project_out
        return projection.weight if projection is not None else None
    return None


class LogitReadout(nn.Module):
    """The model's final-residual-to-logits path, without another forward."""

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.norm = get_final_norm(model)
        self.head = cast(nn.Module, model.get_output_embeddings())
        self.projection = model.model.decoder.project_out if model.config.model_type == "opt" else None
        self.softcap = get_logit_softcap(model)
        self.scale = model.logit_scale if model.config.model_type in {"cohere", "cohere2"} else 1.0
        if model.config.model_type == "falcon_h1":
            self.scale = model.model.lm_head_multiplier
        self.divisor = model.config.logits_scaling if model.config.model_type in {"granite", "granitemoe"} else 1.0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = cast(torch.Tensor, self.head.weight)
        hidden = self.norm(hidden.to(weight.dtype))
        if self.projection is not None:
            hidden = self.projection(hidden)
        logits = self.head(hidden)
        if self.scale != 1.0:
            logits = logits * self.scale
        if self.divisor != 1.0:
            logits = logits / self.divisor
        if self.softcap is not None:
            logits = (logits / self.softcap).tanh() * self.softcap
        return logits


def get_final_norm(model: PreTrainedModel) -> nn.Module:
    """Return the final pre-unembedding normalization module.

    Every supported decoder keeps its final norm as a sibling of the
    transformer block list (``.norm`` on the llama/qwen/gemma families,
    ``.ln_f`` on the GPT-2 family, ``.final_layernorm`` on GLM-shaped
    models), so the owner of the ``get_layers`` ModuleList is searched for
    the first matching attribute rather than adding a second per-architecture
    accessor table.
    """
    layers = get_layers(model)
    for module in model.modules():
        if any(child is layers for child in module.children()):
            for attr in _FINAL_NORM_ATTRS:
                cand = getattr(module, attr, None)
                if isinstance(cand, nn.Module):
                    return cand
            if model.config.model_type == "opt" and module.final_layer_norm is None:
                return nn.Identity()
    raise ValueError(
        f"model_type {model.config.model_type!r}: no final norm found next to "
        f"the layer list (tried attributes {', '.join(_FINAL_NORM_ATTRS)})"
    )
