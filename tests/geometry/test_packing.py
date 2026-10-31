"""Packed device uploads: one buffer per dtype, views with the original shapes."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from drowse.geometry.domains import CustomDomain
from drowse.geometry.manifold import Manifold
from drowse.geometry.packing import pack_to_device
from drowse.geometry.subspace import LayerSubspace

_ACCEL = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else None
)
needs_accel = pytest.mark.skipif(_ACCEL is None, reason="needs an MPS or CUDA device")


@dataclass
class _Leaf:
    a: torch.Tensor
    b: torch.Tensor | None = None
    ids: torch.Tensor | None = None
    note: str = "kept"


@dataclass
class _Tree:
    layers: dict[int, _Leaf] = field(default_factory=dict)
    skipped: torch.Tensor | None = None


def _tree() -> _Tree:
    shared = torch.arange(6.0).reshape(2, 3)
    return _Tree(
        layers={
            0: _Leaf(a=shared, b=torch.ones(5), ids=torch.tensor([1, 2, 3])),
            1: _Leaf(a=shared),
        },
        skipped=torch.zeros(2),
    )


def _manifold() -> Manifold:
    torch.manual_seed(0)
    layers = {}
    for layer in range(3):
        q, _ = torch.linalg.qr(torch.randn(8, 2))
        layers[layer] = LayerSubspace.affine(
            torch.randn(8), q.T.contiguous(), node_coords=torch.randn(3, 2),
        )
    return Manifold(
        name="packed", domain=CustomDomain(2), node_labels=["a", "b", "c"],
        node_coords=torch.randn(3, 2), layers=layers,
        mahalanobis_share={0: 1.0, 1: 1.0, 2: 1.0},
    )


def _layer_tensors(m: Manifold) -> list[torch.Tensor]:
    out = [m.node_coords]
    for sub in m.layers.values():
        out += [t for t in vars(sub).values() if isinstance(t, torch.Tensor)]
    return out


def test_packing_rebinds_every_tensor_with_its_shape_and_dtype() -> None:
    tree = _tree()
    pack_to_device(tree, torch.device("meta"), skip=("skipped",))
    leaf0, leaf1 = tree.layers[0], tree.layers[1]
    assert leaf0.a.device.type == "meta" and leaf0.a.shape == (2, 3)
    assert leaf0.b is not None and leaf0.b.shape == (5,)
    assert leaf0.ids is not None and leaf0.ids.dtype == torch.int64
    assert leaf1.a is leaf0.a and leaf1.b is None and leaf0.note == "kept"
    assert tree.skipped is not None and tree.skipped.device.type == "cpu"


def test_packing_onto_the_cpu_only_moves() -> None:
    tree = _tree()
    pack_to_device(tree, torch.device("cpu"))
    assert torch.equal(tree.layers[0].a, torch.arange(6.0).reshape(2, 3))
    assert tree.layers[0].b is not None
    assert tree.layers[0].a.untyped_storage().data_ptr() != (
        tree.layers[0].b.untyped_storage().data_ptr()
    )


def test_a_cpu_manifold_copy_keeps_separate_tensors() -> None:
    copy = _manifold().to(device=torch.device("cpu"), dtype=torch.float32)
    storages = {t.untyped_storage().data_ptr() for t in _layer_tensors(copy)}
    assert len(storages) == len(_layer_tensors(copy))


@needs_accel
def test_packed_values_survive_in_one_aligned_buffer_per_dtype() -> None:
    assert _ACCEL is not None
    tree = _tree()
    pack_to_device(tree, _ACCEL)
    leaf0 = tree.layers[0]
    assert leaf0.b is not None and leaf0.ids is not None
    assert torch.equal(leaf0.a.cpu(), torch.arange(6.0).reshape(2, 3))
    assert torch.equal(leaf0.ids.cpu(), torch.tensor([1, 2, 3]))
    assert tree.skipped is not None
    floats = {t.untyped_storage().data_ptr() for t in (leaf0.a, leaf0.b, tree.skipped)}
    assert len(floats) == 1
    for t in (leaf0.a, leaf0.b, leaf0.ids):
        assert t.is_contiguous()
        assert t.storage_offset() * t.element_size() % 64 == 0


@needs_accel
def test_a_device_manifold_is_one_upload() -> None:
    assert _ACCEL is not None
    m = _manifold()
    on_device = m.to(device=_ACCEL, dtype=torch.float32)
    tensors = _layer_tensors(on_device)
    assert {t.device.type for t in tensors} == {_ACCEL.type}
    assert len({t.untyped_storage().data_ptr() for t in tensors}) == 1
    for got, want in zip(tensors, _layer_tensors(m)):
        assert torch.equal(got.cpu(), want)
