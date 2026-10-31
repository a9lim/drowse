"""Device, dtype, and memory helpers plus the MPS-only torch workarounds."""

from __future__ import annotations

from typing import Any

import torch

_ORIG_HISTC = None
_ORIG_LDEXP = None


def detect_device(requested: str = "auto") -> str:
    """Pick the best available device.

    'auto' probes in order: cuda > mps > cpu.
    An explicit value ('cuda', 'mps', 'cpu') is returned as-is.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """Best default dtype for a device. bf16 on CUDA and MPS, fp32 on CPU.

    bf16 on MPS matters for models whose residual stream exceeds fp16 range
    (e.g. Gemma-3-4b-it hits ~1e5 by the final layer, well past fp16's 65504
    max). Modern PyTorch MPS handles bf16 natively.
    """
    if device in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def resolve_dtype(dtype: torch.dtype | str | None, device: str) -> torch.dtype:
    if dtype is None:
        return pick_dtype(device)
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def memory_gb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024**3, 2)
    if device.startswith("mps") and torch.backends.mps.is_available():
        # `torch.mps` exists on every torch build but the underlying call
        # raises RuntimeError ("Cannot execute getCurrentAllocatedMemory()
        # without MPS backend") when the backend isn't actually present
        # (e.g. Linux CI with `device='mps'` requested by tests).  Gating
        # on `is_available()` mirrors the cuda branch and keeps the older
        # `AttributeError` swallow as a belt-and-suspenders fallback for
        # very old torch builds.
        try:
            return round(torch.mps.current_allocated_memory() / 1024**3, 2)
        except (AttributeError, RuntimeError):
            return 0.0
    return 0.0


def _histc_mps_safe(input: torch.Tensor, bins: int = 100, min: float = 0, max: float = 0, *, out: torch.Tensor | None = None) -> torch.Tensor:
    assert _ORIG_HISTC is not None
    if input.device.type == "mps" and not input.is_floating_point():
        input = input.float()
    return _ORIG_HISTC(input, bins=bins, min=min, max=max, out=out)


def _ldexp_mps_safe(input: Any, other: Any, *, out: torch.Tensor | None = None) -> Any:
    assert _ORIG_LDEXP is not None
    if hasattr(input, "device") and input.device.type == "mps":
        in_cpu = input.cpu()
        other_cpu = other.cpu() if hasattr(other, "device") else other
        res = _ORIG_LDEXP(in_cpu, other_cpu)
        if out is not None:
            out.copy_(res.to(out.device))
            return out
        return res.to(input.device)
    return _ORIG_LDEXP(input, other, out=out) if out is not None \
        else _ORIG_LDEXP(input, other)


def patch_torch_for_mps() -> bool:
    """Install MPS-only torch workarounds lazily and idempotently.

    The patches are process-global because PyTorch exposes these as module
    functions, but drowse now installs them only when an MPS model load is
    actually requested.
    """
    global _ORIG_HISTC, _ORIG_LDEXP
    installed = False
    if getattr(torch.histc, "_drowse_mps_safe", False) is False:
        _ORIG_HISTC = torch.histc
        _histc_mps_safe._drowse_mps_safe = True  # pyright: ignore[reportFunctionMemberAccess]  # FunctionType stub forbids dynamic attrs
        torch.histc = _histc_mps_safe
        installed = True
    if getattr(torch.ldexp, "_drowse_mps_safe", False) is False:
        _ORIG_LDEXP = torch.ldexp
        _ldexp_mps_safe._drowse_mps_safe = True  # pyright: ignore[reportFunctionMemberAccess]  # FunctionType stub forbids dynamic attrs
        torch.ldexp = _ldexp_mps_safe
        installed = True
    return installed
