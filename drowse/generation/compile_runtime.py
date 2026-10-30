"""Compile, StaticCache and CUDA-graph routing for one session's model.

:class:`CompiledRuntime` is decided once, when the session's model is loaded,
and read on every generation:

- :meth:`CompiledRuntime.prepare` (``from_pretrained``) probes CUDA-graph
  support, picks the ``torch.compile`` mode, installs the persistent
  branchless offset + capture hooks *before* compiling (so they ride inside
  the traced graph), and compiles with a load-time probe.  If compilation
  falls back to eager it removes those hooks again.
- :meth:`CompiledRuntime.for_model` covers a model handed to the session
  directly: a model the caller already compiled still gets the StaticCache
  route; nothing is compiled or hooked here.

On the hot path the decode loop asks :meth:`graph_thread_safe` (CUDA graph
trees are thread-local) and :meth:`acquire_static_cache` (one identity-stable
StaticCache reused across generations, so CUDA graphs and Dynamo guards keyed
on the K/V tensors don't recompile per request).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import torch

from drowse.model import static_cache as _static_cache
from drowse.model import compile as compile_mod
from drowse.model.arch import get_layers
from drowse.readout import capture_buffer
from drowse.steering import hooks

_log = logging.getLogger(__name__)

#: Device types ``torch.compile`` is applied on (CPU skips compilation).
COMPILE_DEVICES = frozenset({"cuda", "mps"})


def _hidden_size(model: Any) -> int | None:
    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and hasattr(config, "get_text_config"):
        hidden_size = config.get_text_config().hidden_size
    return None if hidden_size is None else int(hidden_size)


class CompiledRuntime:
    """The compile / StaticCache / CUDA-graph decisions for one model.

    ``compiled`` — ``torch.compile`` stuck (the model is an ``OptimizedModule``
    wrapping ``_orig_mod``).  ``cuda_graphs_active`` — CUDA-graph capture was
    requested and the architecture supports it.  ``static_cache_active`` —
    generations may use a preallocated StaticCache (either of the above, and
    the architecture's StaticCache constructs).  ``capture_buffers`` are the
    persistent per-layer last-token capture buffers a compiled session reads
    instead of registering transient capture hooks.
    """

    def __init__(
        self,
        *,
        compiled: bool = False,
        cuda_graphs_active: bool = False,
        static_cache_active: bool = False,
        capture_buffers: dict[int, torch.Tensor] | None = None,
        capture_handles: list[Any] | None = None,
        offset_buffers: dict[int, torch.Tensor] | None = None,
        offset_handles: list[Any] | None = None,
    ) -> None:
        self.compiled = compiled
        self.cuda_graphs_active = cuda_graphs_active
        self.static_cache_active = static_cache_active
        # CUDA graph trees are thread-local in PyTorch.  ``prepare`` warms them
        # on the constructing thread; a synchronous generation can replay them
        # there, while a streaming worker thread must route to eager.
        self.owner_thread = threading.get_ident()
        self.capture_buffers: dict[int, torch.Tensor] = capture_buffers or {}
        self._capture_handles: list[Any] = capture_handles or []
        self._offset_buffers: dict[int, torch.Tensor] = offset_buffers or {}
        self._offset_handles: list[Any] = offset_handles or []
        self._static_cache: object | None = None
        self._static_cache_len = 0

    # ------------------------------------------------------------- building

    @classmethod
    def for_model(
        cls,
        model: Any,
        device: torch.device,
        *,
        cuda_graphs: bool = False,
    ) -> "CompiledRuntime":
        """Route an already-loaded model (compiled by the caller or not)."""
        cuda_graphs_active = False
        if cuda_graphs:
            supported, reason = _static_cache.is_cuda_graphs_supported(model, device)
            if supported:
                cuda_graphs_active = True
            elif reason is not None:
                _static_cache.warn_once(model, reason)
        compiled = hasattr(model, "_orig_mod")
        static_cache_active = False
        if compiled or cuda_graphs_active:
            supported, reason = _static_cache.is_static_cache_supported(model, device)
            if supported:
                static_cache_active = True
            elif reason is not None:
                _static_cache.warn_once(model, reason)
        return cls(
            compiled=compiled,
            cuda_graphs_active=cuda_graphs_active,
            static_cache_active=static_cache_active,
        )

    @classmethod
    def prepare(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        compile: bool,
        compile_mode: str | None,
        cuda_graphs: bool,
        persistent_capture: bool,
        on_progress: Callable[[str], None] | None = None,
    ) -> "tuple[Any, CompiledRuntime]":
        """Compile ``model`` if asked and return ``(model, runtime)``.

        The returned model is the compiled ``OptimizedModule`` when compile
        stuck, else the eager model.  ``compile_mode=None`` picks
        ``"reduce-overhead"`` (CUDA-graph capture, paired with StaticCache)
        when CUDA graphs are supported and ``"default"`` (kernel fusion only)
        otherwise; an explicit mode wins regardless.  The CUDA-graph probe
        runs against the bare module first, so an architecture that can't
        build a StaticCache never gets a graph-capture compile mode.
        ``persistent_capture`` installs the persistent capture hooks (a
        session built with ``probes=[]`` takes the no-capture compiled mode;
        later ad-hoc probes fall back to transient capture).
        """
        device = next(model.parameters()).device
        cg_supported = False
        if cuda_graphs:
            cg_supported, _reason = _static_cache.is_cuda_graphs_supported(
                model, device,
            )
        mode = compile_mode
        if mode is None:
            mode = "reduce-overhead" if cg_supported else "default"

        offset_buffers: dict[int, torch.Tensor] = {}
        offset_handles: list[Any] = []
        capture_buffers: dict[int, torch.Tensor] = {}
        capture_handles: list[Any] = []
        if compile and device.type in COMPILE_DEVICES:
            # Attach the persistent hooks BEFORE compile so they ride inside
            # the captured graph (post-compile hook changes are not retraced).
            # The offset hooks carry static-affine steering by updating their
            # buffers in place per generation; the capture hooks ``copy_`` the
            # last-token slice per layer.  MPS compiles too (inductor fuses
            # the per-layer kernels); graph capture stays CUDA-only, so MPS
            # always lands on ``"default"``.
            hidden_size = _hidden_size(model)
            if hidden_size is not None:
                dtype = next(model.parameters()).dtype
                layers = get_layers(model)
                offset_buffers, offset_handles = hooks.install_persistent_offset_hooks(
                    layers, hidden_size, device, dtype,
                )
                if persistent_capture:
                    capture_buffers, capture_handles = (
                        capture_buffer.install_persistent_capture_hooks(
                            layers, hidden_size, device, dtype,
                        )
                    )
            if on_progress is not None:
                on_progress(f"Compiling model with torch.compile(mode={mode!r})...")
            # The probe turns an architecture-specific inductor / Triton bug
            # into a caught warning + eager fallback at load time.
            model = compile_mod.compile_with_probe(model, tokenizer, device, mode=mode)
        elif compile:
            message = (
                f"compile=True but device={device.type} — skipping "
                "torch.compile (supported only on CUDA / MPS)"
            )
            _log.info(message)
            if on_progress is not None:
                on_progress(message)

        runtime = cls.for_model(model, device, cuda_graphs=cuda_graphs)
        if runtime.compiled:
            runtime.capture_buffers = capture_buffers
            runtime._capture_handles = capture_handles
            runtime._offset_buffers = offset_buffers
            runtime._offset_handles = offset_handles
        else:
            # Compile fell back to eager: the persistent hooks would be pure
            # overhead with no compiled graph to ride.
            for handle in (*offset_handles, *capture_handles):
                handle.remove()
        return model, runtime

    def take_offset_hooks(
        self,
    ) -> "tuple[dict[int, torch.Tensor], list[Any]] | None":
        """Hand the persistent offset buffers + handles to the steering manager.

        Returns ``None`` when there are none (an eager session).  The runtime
        keeps no reference: the steering manager owns them from here on.
        """
        if not self._offset_handles:
            return None
        taken = (self._offset_buffers, self._offset_handles)
        self._offset_buffers, self._offset_handles = {}, []
        return taken

    # ------------------------------------------------------------- routing

    def graph_thread_safe(self) -> bool:
        """Whether this thread may replay the warmed CUDA graph tree."""
        return (
            not self.cuda_graphs_active
            or threading.get_ident() == self.owner_thread
        )

    def compiled_on(self, device: torch.device) -> bool:
        """The compiled module is the routing default on ``device``."""
        return self.compiled and device.type in COMPILE_DEVICES

    def compiled_clean_capture(self, device: torch.device, *, want_hidden: bool) -> bool:
        """A generation's capture may ride the persistent buffers.

        The compiled graph and StaticCache are live, the persistent capture
        buffers were adopted, and the caller didn't ask for the full per-step
        hidden stack (``return_hidden`` keeps transient full-retention capture).
        """
        return bool(
            self.compiled_on(device)
            and self.static_cache_active
            and self.capture_buffers
            and not want_hidden
        )

    def acquire_static_cache(
        self,
        model: Any,
        *,
        max_cache_len: int,
        device: torch.device,
    ) -> object:
        """A reset, identity-stable StaticCache with at least ``max_cache_len``.

        CUDA graphs guard on the K/V tensors' identities, so a fresh cache per
        request would force a full Dynamo recompile even with every shape
        unchanged.  The cache grows only when a request needs more room, and
        is reset in place otherwise; the session lock serializes its users.
        """
        cached = self._static_cache
        if cached is not None and self._static_cache_len >= max_cache_len:
            reset = getattr(cached, "reset", None)
            if callable(reset):
                reset()
                return cached
            self._static_cache = None
            self._static_cache_len = 0
        cache = _static_cache.make_static_cache(
            model,
            max_cache_len=max_cache_len,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        self._static_cache = cache
        self._static_cache_len = max_cache_len
        return cache

    # ------------------------------------------------------------ teardown

    def detach_persistent_capture(self) -> None:
        """Remove the persistent capture hooks and drop their buffers (idempotent)."""
        for handle in self._capture_handles:
            handle.remove()
        self._capture_handles = []
        self.capture_buffers = {}

    def close(self) -> None:
        """Release every device buffer this runtime holds."""
        self.detach_persistent_capture()
        for handle in self._offset_handles:
            handle.remove()
        self._offset_handles = []
        self._offset_buffers = {}
        self._static_cache = None
        self._static_cache_len = 0


__all__ = ["COMPILE_DEVICES", "CompiledRuntime"]
