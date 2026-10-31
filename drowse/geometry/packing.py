"""Upload an artifact's many small tensors as a few device allocations.

A fitted manifold or an attached probe holds thousands of per-layer tensors
of a few KB each.  Uploaded one at a time to an MPS device, they scatter
long-lived blocks through the heap allocator's small pool.  Every later
allocation there then pays to coalesce the free space around them, including
each decode step's KV-cache ``cat``: about 15% of unsteered tok/s on
SmolLM2-360M with the 17 default probes attached.  :func:`pack_to_device`
moves a freshly built, unshared object tree to the device as one buffer per
dtype and rebinds its tensors to views of that buffer.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import torch

# Byte alignment of each tensor's segment inside a packed buffer.
_ALIGN = 64


def _tensor_slots(
    obj: Any, skip: frozenset[str], out: list[tuple[Any, Any, bool]],
) -> None:
    """Collect ``(container, key, is_attr)`` for each tensor under ``obj``.

    Walks dataclass fields (minus ``skip``) and dict values.  Lists and
    tuples are not entered.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name in skip:
                continue
            value = getattr(obj, f.name)
            if isinstance(value, torch.Tensor):
                out.append((obj, f.name, True))
            else:
                _tensor_slots(value, skip, out)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                out.append((obj, key, False))
            else:
                _tensor_slots(value, skip, out)


def pack_to_device(
    obj: Any, device: torch.device, *, skip: Iterable[str] = (),
) -> None:
    """Move every tensor under ``obj`` to ``device`` in place, packed.

    ``obj`` must be a fresh tree that nothing else shares, because its
    containers are rebound in place.  Tensors are grouped by dtype; each
    group uploads as one buffer, and each tensor becomes a contiguous view
    of it with the same shape.  A tensor reached twice stays one view.  On
    the CPU the tensors are moved without packing.
    """
    device = torch.device(device)
    slots: list[tuple[Any, Any, bool]] = []
    _tensor_slots(obj, frozenset(skip), slots)

    def get(slot: tuple[Any, Any, bool]) -> torch.Tensor:
        container, key, is_attr = slot
        return getattr(container, key) if is_attr else container[key]

    def put(slot: tuple[Any, Any, bool], value: torch.Tensor) -> None:
        container, key, is_attr = slot
        if is_attr:
            setattr(container, key, value)
        else:
            container[key] = value

    if device.type == "cpu":
        for slot in slots:
            put(slot, get(slot).detach().to(device))
        return

    groups: dict[torch.dtype, list[torch.Tensor]] = {}
    seen: set[int] = set()
    for slot in slots:
        tensor = get(slot)
        if id(tensor) not in seen:
            seen.add(id(tensor))
            groups.setdefault(tensor.dtype, []).append(tensor)

    views: dict[int, torch.Tensor] = {}
    for dtype, tensors in groups.items():
        step = max(1, _ALIGN // torch.empty((), dtype=dtype).element_size())
        offsets = []
        total = 0
        for tensor in tensors:
            offsets.append(total)
            total += -(-tensor.numel() // step) * step
        host = torch.empty(total, dtype=dtype)
        for tensor, offset in zip(tensors, offsets):
            host[offset:offset + tensor.numel()].copy_(tensor.detach().reshape(-1))
        buffer = host.to(device)
        for tensor, offset in zip(tensors, offsets):
            views[id(tensor)] = buffer[offset:offset + tensor.numel()].view(tensor.shape)

    for slot in slots:
        put(slot, views[id(get(slot))])
