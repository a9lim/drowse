"""LRP-modified backward rules for the Jacobian-lens estimator (RelP).

The R-lens ("R-lens: making J-lens more faithful on early layers",
LessWrong 2026) keeps the J-lens estimator and its forward values
bit-identical while reading the Jacobian through an LRP-modified backward
graph (RelP, arXiv:2508.21258).  Three rules, dense-model scope:

- **LN-rule** — an RMSNorm backpropagates as if its ``rsqrt`` denominator
  were a constant: the Jacobian becomes the elementwise diagonal
  ``weight·rsqrt(mean(x²)+eps)`` and the rank-one ``d rms / d x`` term is
  dropped.
- **identity-rule** — the gated activation backpropagates as its detached
  per-element linear factor (``sigmoid(x)`` for SiLU, the tanh/erf CDF
  factor for GELU) instead of ``act'(x)``.
- **half-rule** — the ``act(gate) * up`` product splits relevance evenly
  (β = 0.5) across its two factors instead of the full product rule.

Bit-exact forwards by construction: every rewrite routes the module's own
output tensor through a ``torch.autograd.Function`` that returns it
untouched and only substitutes the backward factor — there is no forward
arithmetic to round, overflow, or lose ``-0`` on.  Backward factors are
computed division-free in fp32 (so ``x = 0`` coordinates keep their exact
detached-factor gradient rather than a spurious zero), and each patched
module verifies its reconstructed factor against its real output on the
first forward — a norm or MLP whose true forward is not the assumed shape
raises :class:`RelpUnsupportedError` instead of fitting a silently wrong
artifact.

Scope and caveats:

- Only *direct* RMSNorm children of each decoder block are rewritten.
  Norms nested inside attention (q/k norms, gated linear-attention norms)
  keep their true gradients, matching ``include_qk_norms: false`` in the
  reference fits.
- MoE experts and fused gate-up projections need per-arch rule arms this
  module does not implement; :func:`relp_backward_rules` raises rather
  than producing a partial fit.
- Patching is instance-scoped and strictly reversible, but process-visible:
  the caller must hold the model exclusively for the duration (the session
  fit runs under ``_model_exclusive``), and must pass the *eager* module
  list (a ``torch.compile`` wrapper would recompile; the fit already
  resolves ``_orig_mod``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import MethodType
from typing import Any, cast

import torch
from torch import nn

from saklas.core.errors import RelpUnsupportedError

RELP_RULES = {
    "ln_rule": True,
    "identity_rule": True,
    "half_rule": True,
    "half_rule_beta": 0.5,
    "include_qk_norms": False,
}

_HALF_RULE_BETA = 0.5
_VERIFIED_FLAG = "_saklas_relp_verified"
_MLP_VERIFIED_FLAG = "_saklas_relp_mlp_verified"
#: Structural verification tolerance — the factor reconstruction may differ
#: from the module's own output by op-order rounding, while a wrong shape
#: (missing ``(1+w)``, wrong eps, an unexpected extra term) errs at O(1).
_VERIFY_RTOL = 1e-2
_VERIFY_ATOL = 1e-2


class _DetachedFactorProduct(torch.autograd.Function):
    """Return ``value`` untouched; backpropagate ``grad · factor`` to ``x``."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, value: torch.Tensor, factor: torch.Tensor,
    ) -> torch.Tensor:
        del x
        ctx.save_for_backward(factor)
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (grad,) = grad_outputs
        (factor,) = ctx.saved_tensors
        # Multiply in the factor's fp32 and round the product once — a
        # pre-cast factor would round twice on a low-precision grad.
        return (grad.float() * factor).to(grad.dtype), None, None


class _HalfProduct(torch.autograd.Function):
    """Forward ``a · b`` exactly; backward halves each factor (β = 0.5)."""

    @staticmethod
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (grad,) = grad_outputs
        a, b = ctx.saved_tensors
        return _HALF_RULE_BETA * (grad * b), _HALF_RULE_BETA * (grad * a)


_NORM_OFFSET_FLAG = "_saklas_relp_norm_offset"


def _norm_backward_factor(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor,
) -> torch.Tensor:
    """The LN-rule diagonal ``weight·rsqrt(mean(x²)+eps)``, fp32, zero-safe.

    The weight convention — plain ``w`` (llama/qwen2/qwen3) vs zero-centered
    ``1 + w`` (gemma, and gemma-style adopters like qwen3_5) — is detected
    empirically on the first forward by reconstructing the module's own
    output under both candidates; a name heuristic would silently mis-factor
    an adopter.  The winning offset is cached on the module for the patch's
    lifetime, and a module matching neither shape (or an input that cannot
    discriminate) is refused.
    """
    eps = getattr(module, "variance_epsilon", None)
    if eps is None:
        eps = getattr(module, "eps", None)
    weight = getattr(module, "weight", None)
    if not isinstance(eps, float) or not isinstance(weight, torch.Tensor):
        raise RelpUnsupportedError(
            f"RMSNorm {type(module).__name__} exposes no eps/weight to "
            "reconstruct the LN-rule factor from"
        )
    xf = x.detach().float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    wf = weight.detach().float()
    offset = getattr(module, _NORM_OFFSET_FLAG, None)
    if offset is None:
        yf = y.detach().float()
        matches = [
            candidate for candidate in (0.0, 1.0)
            if torch.allclose(
                xf * ((candidate + wf) * scale), yf,
                rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL,
            )
        ]
        if len(matches) != 1:
            raise RelpUnsupportedError(
                f"{type(module).__name__} output matches "
                f"{'neither' if not matches else 'both'} of the plain-weight "
                "and zero-centered (1+w) RMSNorm shapes — cannot reconstruct "
                "its LN-rule factor"
            )
        offset = matches[0]
        setattr(module, _NORM_OFFSET_FLAG, offset)
    return (offset + wf) * scale


_SILU_NAMES = ("silu", "swish")
_GELU_TANH_NAMES = ("gelutanh", "newgelu", "pytorchgelutanh", "fastgelu")


def _act_backward_factor(act: Any, x: torch.Tensor) -> torch.Tensor:
    """The identity-rule factor ``act(x)/x`` computed division-free, fp32."""
    xf = x.detach().float()
    name = type(act).__name__.casefold().replace("activation", "")
    if any(tag in name for tag in _SILU_NAMES):
        return torch.sigmoid(xf)
    approximate = getattr(act, "approximate", None)
    if any(tag in name for tag in _GELU_TANH_NAMES) or approximate == "tanh":
        return 0.5 * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf.pow(3))
            )
        )
    if "gelu" in name:
        return 0.5 * (1.0 + torch.erf(xf / math.sqrt(2.0)))
    raise RelpUnsupportedError(
        f"no identity-rule factor for activation {type(act).__name__}"
    )


def _verify_once(
    module: nn.Module,
    reconstructed: torch.Tensor,
    actual: torch.Tensor,
    what: str,
) -> None:
    if getattr(module, _VERIFIED_FLAG, False):
        return
    if not torch.allclose(
        reconstructed, actual.float(), rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL,
    ):
        raise RelpUnsupportedError(
            f"{type(module).__name__} {what} does not match its reconstructed "
            "RelP factor — the module's real forward is not the shape the "
            "rules assume"
        )
    setattr(module, _VERIFIED_FLAG, True)


def _relp_norm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # no_grad: the value input must reach the Function graph-free — the
    # module's params require grad, and only the x-path gradient is real.
    with torch.no_grad():
        y = type(self).forward(self, x.detach())
    # Detection doubles as verification: the returned factor reconstructed
    # this module's own output within tolerance on its first forward.
    factor = _norm_backward_factor(self, x, y)
    return _DetachedFactorProduct.apply(x, y, factor)


def _relp_gated_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate_proj = cast(nn.Module, getattr(self, "gate_proj"))
    up_proj = cast(nn.Module, getattr(self, "up_proj"))
    down_proj = cast(nn.Module, getattr(self, "down_proj"))
    act_fn = cast(Callable[[torch.Tensor], torch.Tensor], getattr(self, "act_fn"))
    gate_pre = cast(torch.Tensor, gate_proj(x))
    act_value = act_fn(gate_pre.detach())
    factor = _act_backward_factor(act_fn, gate_pre)
    _verify_once(
        self, gate_pre.detach().float() * factor, act_value, "activation",
    )
    gate = _DetachedFactorProduct.apply(gate_pre, act_value, factor)
    up = cast(torch.Tensor, up_proj(x))
    result = cast(torch.Tensor, down_proj(_HalfProduct.apply(gate, up)))
    if not getattr(self, _MLP_VERIFIED_FLAG, False):
        # A module can duck-type the gated shape while its real forward adds
        # scaling, dropout, or routing; the rewrite must reproduce the true
        # forward bit-for-bit or refuse to fit.
        with torch.no_grad():
            expected = type(self).forward(self, x.detach())
        if not torch.equal(result.detach(), expected):
            raise RelpUnsupportedError(
                f"{type(self).__name__} forward does not match the canonical "
                "gated-MLP shape the RelP rules rewrite"
            )
        setattr(self, _MLP_VERIFIED_FLAG, True)
    return result


def _is_gated_mlp(module: nn.Module) -> bool:
    return all(
        hasattr(module, attr)
        for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")
    )


def _is_residual_norm(module: nn.Module) -> bool:
    return "rmsnorm" in type(module).__name__.casefold()


def _instance_patch(
    module: nn.Module,
    fn: Any,
    patched: list[nn.Module],
) -> None:
    if "forward" in module.__dict__:
        raise RelpUnsupportedError(
            f"{type(module).__name__} carries an instance-level forward "
            "override; RelP patching would silently bypass it"
        )
    patched.append(module)
    object.__setattr__(module, "forward", MethodType(fn, module))


@contextmanager
def relp_backward_rules(layer_modules: Sequence[nn.Module]) -> Iterator[None]:
    """Apply the dense RelP rules to every decoder layer for the duration.

    ``layer_modules`` is the eager transformer block list the Jacobian fit
    already resolves.  Every block must expose exactly one standard gated
    MLP (``gate_proj``/``up_proj``/``down_proj``/``act_fn``) and at least
    one residual-stream RMSNorm as direct children; anything else — MoE
    experts, fused gate-up projections, non-RMS norms — raises
    :class:`RelpUnsupportedError` before any forward runs.  Each patched
    module additionally self-verifies on its first forward.
    """
    patched: list[nn.Module] = []
    try:
        for index, layer in enumerate(layer_modules):
            norms = [
                child for _, child in layer.named_children()
                if _is_residual_norm(child)
            ]
            mlps = [
                child for _, child in layer.named_children()
                if _is_gated_mlp(child)
            ]
            if len(mlps) != 1 or not norms:
                raise RelpUnsupportedError(
                    f"RelP backward rules need one gated MLP and RMSNorm "
                    f"children per decoder layer; layer {index} has "
                    f"{len(mlps)} gated MLPs and {len(norms)} RMSNorms "
                    "(MoE and fused-projection architectures are unsupported)"
                )
            for norm in norms:
                _instance_patch(norm, _relp_norm_forward, patched)
            _instance_patch(mlps[0], _relp_gated_mlp_forward, patched)
        yield
    finally:
        for module in reversed(patched):
            if "forward" in module.__dict__:
                object.__delattr__(module, "forward")
            for flag in (_VERIFIED_FLAG, _MLP_VERIFIED_FLAG, _NORM_OFFSET_FLAG):
                if hasattr(module, flag):
                    delattr(module, flag)
