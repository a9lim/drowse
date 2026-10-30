"""Hidden-state capture buffers: the read side of the per-layer forward hooks.

:class:`HiddenCapture` records the last-position hidden state at each hooked
layer on every forward; :func:`install_persistent_capture_hooks` is its
compile-clean source (always-on ``copy_`` hooks traced into the compiled
graph).  Steering hooks live in :mod:`drowse.steering.hooks`; nothing here
writes the residual stream.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

StepSink = Callable[[int, dict[int, torch.Tensor]], None]


class HiddenCapture:
    """Accumulates the last-position hidden state at each hooked layer.

    Paired with a KV-cached generation loop, one capture per forward gives N
    captures for N generated tokens: capture ``k`` is the state that produced
    generated token ``k`` (step 0 is the last prompt position).

    Hot-path discipline: the hook does device work only — a ``(D,)`` slice
    ``detach()``-ed and either ``copy_``-ed into a preallocated per-layer
    buffer or cloned into a per-layer list.  Stacking, fp32 casting and every
    host read happen after the forward.

    Retention (:meth:`set_retention`, armed after :meth:`attach`) is the one
    policy, applied by :meth:`_retain` from both the transient hook and
    :meth:`ingest_persistent`:

    - unarmed (what :meth:`attach` leaves): full retention, a distinct clone
      per step, so :meth:`stacked` returns ``[T, D]`` (the ``return_hidden``
      path);
    - ``depth == 1``: one preallocated ``(D,)`` buffer per layer, overwritten
      in place — zero steady-state allocation, ``bucket[-1]`` is the latest;
    - ``depth > 1``: a ring of the last ``depth`` slices, deep enough for
      finalize to walk back past trailing special tokens
      (:meth:`tail_slice_at`).

    A ``step_sink`` (optional) receives ``(step_id, latest_per_layer)`` once
    per forward via :meth:`fire_step_sink` / :meth:`ingest_persistent`,
    *after* the forward returns, so its host-side read never drains the device
    pipeline mid-forward.  Without one the decode loop pays zero per-token
    scoring.
    """

    def __init__(self) -> None:
        self._per_layer: dict[int, list[torch.Tensor]] = {}
        self._handles: list[Any] = []
        self._incremental: bool = False
        self._step_sink: StepSink | None = None
        # Highest hooked layer: it fires last in a forward, so by the time it
        # stores its slice every hooked layer holds this step's value.  It
        # advances ``_forward_count``, which maps a generated-token index to
        # its tail-ring slot.
        self._max_layer: int | None = None
        self._tail_depth: int = 1
        self._tail_layers: frozenset[int] | None = None
        self._forward_count: int = 0
        # Persistent compile-clean source: layer → the always-on buffer the
        # pre-compile ``copy_`` hook writes.  Empty ⇒ transient-hook path.
        self._persistent_buffers: dict[int, torch.Tensor] = {}
        # Batched ``[B, D]`` tail ring for the ``generate_batch`` fast path.
        self._batch_per_layer: dict[int, list[torch.Tensor]] = {}
        self._batch_forward_count: int = 0
        # Selected rows from the first multi-token prefill forward (visible
        # authored tokens a loom-attached generation also measures).
        self._prompt_positions: tuple[int, ...] = ()
        self._prompt_per_layer: dict[int, torch.Tensor] = {}

    def _reset_retention(self) -> None:
        self._incremental = False
        self._step_sink = None
        self._max_layer = None
        self._tail_depth = 1
        self._tail_layers = None
        self._forward_count = 0

    def _retain(
        self, bucket: list[torch.Tensor], src: torch.Tensor, layer: int,
    ) -> None:
        """Store this forward's ``(D,)`` slice under the armed retention."""
        if not self._incremental:
            bucket.append(src.clone())
            return
        deep = self._tail_depth > 1 and (
            self._tail_layers is None or layer in self._tail_layers
        )
        if deep:
            bucket.append(src.clone())
            if len(bucket) > self._tail_depth:
                bucket.pop(0)
        elif bucket:
            # Safe to overwrite: every consumer (step sink, gate callback,
            # token tap) reads the slice synchronously before the next forward.
            bucket[0].copy_(src)
        else:
            bucket.append(src.clone())

    def attach(
        self, layers: "torch.nn.ModuleList", layer_indices: list[int]
    ) -> None:
        """Register one transient capture hook per layer (full retention)."""
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        self._reset_retention()
        self._persistent_buffers = {}
        self._prompt_positions = ()
        self._prompt_per_layer = {}
        for idx in layer_indices:
            bucket = self._per_layer[idx]

            def _make(bucket_ref: list[torch.Tensor], layer_idx: int) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    if (
                        self._prompt_positions
                        and layer_idx not in self._prompt_per_layer
                        and int(h.shape[1]) > self._prompt_positions[-1]
                    ):
                        positions = torch.tensor(
                            self._prompt_positions,
                            device=h.device,
                            dtype=torch.long,
                        )
                        self._prompt_per_layer[layer_idx] = (
                            h[0].index_select(0, positions).detach().clone()
                        )
                    self._retain(bucket_ref, h[0, -1, :].detach(), layer_idx)
                    if self._incremental and layer_idx == self._max_layer:
                        self._forward_count += 1
                return _hook

            self._handles.append(
                layers[idx].register_forward_hook(_make(bucket, idx)),
            )

    def attach_batch_tail(
        self,
        layers: "torch.nn.ModuleList",
        layer_indices: list[int],
        *,
        depth: int,
    ) -> None:
        """Capture batched ``[B, D]`` last-position slices in a bounded ring.

        Used by the ``generate_batch`` fast path for stateless aggregate probe
        reads.  No per-token sinks, hidden-state returns, or persistent
        buffers — those stay on the single-row loop.
        """
        self.clear()
        self._batch_per_layer = {idx: [] for idx in layer_indices}
        tail_depth = max(1, int(depth))
        for idx in layer_indices:
            bucket = self._batch_per_layer[idx]

            def _make(bucket_ref: list[torch.Tensor], layer_idx: int) -> Any:
                def _hook(module: Any, input: Any, output: Any) -> None:
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    bucket_ref.append(h[:, -1, :].detach().clone())
                    if len(bucket_ref) > tail_depth:
                        bucket_ref.pop(0)
                    if layer_idx == layer_indices[-1]:
                        self._batch_forward_count += 1
                return _hook

            self._handles.append(
                layers[idx].register_forward_hook(_make(bucket, idx)),
            )

    def set_retention(
        self,
        *,
        depth: int = 1,
        step_sink: StepSink | None = None,
        tail_layers: "set[int] | frozenset[int] | None" = None,
    ) -> None:
        """Arm incremental capture: how much to retain, and who reads it.

        Must follow :meth:`attach` / :meth:`attach_persistent`.  ``depth`` is
        the per-layer ring depth (``1`` = one in-place buffer); ``step_sink``
        is the per-token reader or ``None``; ``tail_layers`` restricts the
        deep ring to the layers a final aggregate consumes (the rest stay
        length-1 and are omitted from :meth:`tail_slice_at`).
        """
        self._incremental = True
        self._tail_depth = max(1, int(depth))
        self._tail_layers = (
            frozenset(int(layer) for layer in tail_layers)
            if tail_layers is not None
            else None
        )
        self._step_sink = step_sink
        self._max_layer = max(self._per_layer) if self._per_layer else None

    def set_incremental(self, step_sink: StepSink) -> None:
        """Length-1 buffers + per-token scoring (the INCREMENTAL mode)."""
        self.set_retention(depth=1, step_sink=step_sink)

    def set_aggregate_tail(self, depth: int) -> None:
        """Bounded tail ring, no per-token scoring (the AGGREGATE_ONLY mode)."""
        self.set_retention(depth=depth)

    def set_tail_with_sink(
        self,
        depth: int,
        step_sink: StepSink,
        *,
        tail_layers: set[int] | frozenset[int] | None = None,
    ) -> None:
        """Tail ring plus a per-token sink (GATING_SUBSET / LEAN_INCREMENTAL)."""
        self.set_retention(
            depth=depth, step_sink=step_sink, tail_layers=tail_layers,
        )

    def attach_persistent(
        self, layer_indices: list[int], buffers: dict[int, torch.Tensor],
    ) -> None:
        """Compile-clean capture: accumulate from persistent buffers, no hooks.

        The always-on pre-compile hooks
        (:func:`install_persistent_capture_hooks`) ``copy_`` each layer's
        latest slice into ``buffers[L]`` inside the compiled graph; the decode
        loop calls :meth:`ingest_persistent` post-forward to run the same
        retention + step sink the transient hook runs.
        """
        self._per_layer = {idx: [] for idx in layer_indices}
        self._handles = []
        self._reset_retention()
        self._persistent_buffers = {
            idx: buffers[idx] for idx in layer_indices if idx in buffers
        }

    def ingest_persistent(self, step_id: int) -> None:
        """Post-forward accumulation from the persistent buffers.

        Wired as the decode loop's ``step_callback`` when capture is
        persistent, so the bucket shapes and the forward → ingest → gate
        scoring order match the transient path exactly.  No-op unless attached
        via :meth:`attach_persistent`.
        """
        if not self._persistent_buffers:
            return
        for idx, bucket in self._per_layer.items():
            src = self._persistent_buffers.get(idx)
            if src is not None:
                self._retain(bucket, src, idx)
        self._forward_count += 1
        if self._step_sink is not None:
            self._step_sink(step_id, self.latest_per_layer())

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        # A later transient capture (or a no-probe gen) must not read stale
        # slices through ``ingest_persistent``.
        self._persistent_buffers = {}
        self._prompt_positions = ()
        self._prompt_per_layer = {}

    def clear(self) -> None:
        self._per_layer = {}
        self._batch_per_layer = {}
        self._handles = []
        self._reset_retention()
        self._batch_forward_count = 0
        self._persistent_buffers = {}
        self._prompt_positions = ()
        self._prompt_per_layer = {}

    def set_prompt_positions(self, positions: list[int]) -> None:
        """Retain selected rows from the first multi-token prefill forward.

        Positions are local to the ``input_ids`` passed into that forward (a
        caller using a KV prefix subtracts the cached prefix length first).
        Persistent latest-slice buffers cannot provide prompt rows; the
        session forces this mode onto transient hooks.
        """
        normalized = tuple(sorted({int(pos) for pos in positions if pos >= 0}))
        self._prompt_positions = normalized
        self._prompt_per_layer = {}

    def prompt_stacked(self) -> dict[int, torch.Tensor]:
        """Return selected prefill rows as ``{layer: [positions, D]}``."""
        return dict(self._prompt_per_layer)

    def tail_slice_at(self, forward_index: int) -> dict[int, torch.Tensor]:
        """Per-layer ``[D]`` slice for decode ``forward_index`` from the ring.

        Generated token ``k`` was produced by forward ``k``; the ring holds the
        last ``len(bucket)`` forwards ending at ``_forward_count - 1``.  A
        walk-back past the ring clamps to the oldest retained slice.
        """
        out: dict[int, torch.Tensor] = {}
        F = self._forward_count
        for idx, bucket in self._per_layer.items():
            if not bucket:
                continue
            if self._tail_layers is not None and idx not in self._tail_layers:
                continue
            start = F - len(bucket)
            pos = max(0, min(forward_index - start, len(bucket) - 1))
            out[idx] = bucket[pos]
        return out

    def batch_tail_slice_at(
        self,
        row_index: int,
        forward_index: int,
    ) -> dict[int, torch.Tensor]:
        """Per-layer ``[D]`` slice for one row from the batched tail ring."""
        out: dict[int, torch.Tensor] = {}
        F = self._batch_forward_count
        for idx, bucket in self._batch_per_layer.items():
            if not bucket:
                continue
            start = F - len(bucket)
            pos = max(0, min(forward_index - start, len(bucket) - 1))
            batch_slice = bucket[pos]
            if row_index < 0 or row_index >= int(batch_slice.shape[0]):
                continue
            out[idx] = batch_slice[row_index]
        return out

    def stacked(self) -> dict[int, torch.Tensor]:
        """Per-layer ``(n_captures, D)`` tensors in the capture dtype."""
        out: dict[int, torch.Tensor] = {}
        for idx, bucket in self._per_layer.items():
            if bucket:
                out[idx] = torch.stack(bucket)
        return out

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        """The most recent ``[D]`` capture per layer (empty layers omitted)."""
        out: dict[int, torch.Tensor] = {}
        for idx, bucket in self._per_layer.items():
            if bucket:
                out[idx] = bucket[-1]
        return out

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        """The raw per-layer buckets, for callers that select ``[-1]`` themselves."""
        return self._per_layer

    def is_transient(self) -> bool:
        """True iff transient per-gen forward hooks are registered.

        The compiled-clean routing gate: the compiled graph was traced with
        only the persistent hooks present, so it stays valid exactly when no
        transient hook is installed.
        """
        return bool(self._handles)

    def fire_step_sink(self, step_id: int) -> None:
        """Run the per-token step sink once, after the model forward."""
        if self._step_sink is not None:
            self._step_sink(step_id, self.latest_per_layer())


def install_persistent_capture_hooks(
    layers: "torch.nn.ModuleList",
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> "tuple[dict[int, torch.Tensor], list[Any]]":
    """Attach one branchless ``buf.copy_(hidden[0, -1, :])`` hook per layer.

    Call **before** ``torch.compile`` so the ``copy_`` is fused into the traced
    graph; a probed generation then reads the latest slice per forward via
    :meth:`HiddenCapture.ingest_persistent`.  Hooks cover every layer (the
    probe roster isn't known at compile time) and are pure writes, so they are
    safe to leave installed.  Returns ``(buffers, handles)``.
    """
    buffers: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def _make(buf: torch.Tensor) -> Any:
        def _hook(module: Any, inp: Any, out: Any) -> Any:
            h = out if isinstance(out, torch.Tensor) else out[0]
            buf.copy_(h[0, -1, :])
            return out
        return _hook

    for idx in range(len(layers)):
        buf = torch.zeros(hidden_size, device=device, dtype=dtype)
        buffers[idx] = buf
        handles.append(layers[idx].register_forward_hook(_make(buf)))
    return buffers, handles


__all__ = ["HiddenCapture", "StepSink", "install_persistent_capture_hooks"]
