"""The installed steering is a function of the steering stack.

CPU tests over a model-free session skeleton whose layers carry the persistent
compiled-offset hooks a compiled CUDA/MPS session adopts, so the offset path
is exercised without a compiled model:

- an exited scope leaves no steering behind (offsets zeroed, flag cleared);
- a generation scope nested in a user scope restores the outer steering on
  exit, and the generation teardown does not erase it;
- a push that fails to compose leaves the outer scope installed;
- repeated pushes of the same expression reuse one composed plan.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from drowse.errors import SteeringCompositionError
from drowse.events import EventBus
from drowse.session.session import DrowseSession, GenState
from drowse.steering.composer import SteeringComposer
from drowse.steering.grammar import ManifoldTerm
from drowse.steering.hooks import SteeringManager, install_persistent_offset_hooks
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from drowse.store import selectors as _sel
from tests._whitener import isotropic_whitener
from tests.test_manifold_steering import _manifold

_D = 8
_N_LAYERS = 3


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


def _unit(i: int) -> torch.Tensor:
    v = torch.zeros(_D)
    v[i] = 1.0
    return v


def _session(*, compiled: bool = True) -> tuple[DrowseSession, dict[int, torch.Tensor]]:
    session = DrowseSession.__new__(DrowseSession)
    s = cast(Any, session)
    s._model = None
    s._tokenizer = None
    s._model_info = {"model_id": "stub/model"}
    s._layers = nn.ModuleList([nn.Identity() for _ in range(_N_LAYERS)])
    s._device = torch.device("cpu")
    s._dtype = torch.float32
    s._profiles = {"a": {1: _unit(0)}, "b": {1: _unit(1)}}
    s._manifolds = {}
    s._layer_means = {1: torch.zeros(_D)}
    s._whitener = isotropic_whitener([1], _D)
    s._steering = SteeringManager()
    buffers: dict[int, torch.Tensor] = {}
    if compiled:
        buffers, handles = install_persistent_offset_hooks(
            s._layers, _D, torch.device("cpu"), torch.float32,
        )
        s._steering.adopt_compiled_offsets(buffers, handles)
    s._steering_composer = SteeringComposer(session)
    s._gen_lock = threading.RLock()
    s._gen_phase = GenState.IDLE
    s._internal_steering_pop = False
    s._compiled = compiled
    s._compiled_clean_eligible = False
    s._monitor = SimpleNamespace(probe_names=[], set_subspace_coords=lambda _on: None)
    s._end_capture = lambda: None
    s._close_instrument_runs = lambda: None
    s._prefix_cache = None
    s._active_role = None
    s.events = EventBus()
    return session, buffers


def _offsets(buffers: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {L: buf.clone() for L, buf in buffers.items()}


def _zero(buffers: dict[int, torch.Tensor]) -> bool:
    return all(not bool(buf.any()) for buf in buffers.values())


def test_exited_scope_zeroes_the_compiled_offsets() -> None:
    """A scope exited outside any generation must not keep steering.

    The empty-stack rebuild used to detach transient hooks only, leaving the
    offset buffers live, so the next unsteered generation on the compiled
    route was steered by the exited scope.
    """
    session, buffers = _session()
    manager = session._steering
    with session.steering("0.5 a"):
        assert manager.uses_compiled_offsets
        assert not manager.hooks
        assert not _zero(buffers)
    assert _zero(buffers)
    assert not manager.uses_compiled_offsets
    assert manager.plan is None


def test_generation_scope_restores_outer_offsets_through_teardown() -> None:
    """An inner generation scope must leave the outer scope's offsets in place.

    The generation teardown used to zero the buffers after the inner pop had
    re-installed the outer scope, so every later generation in the outer
    scope ran unsteered.
    """
    session, buffers = _session()
    manager = session._steering
    with session.steering("0.5 a"):
        outer = _offsets(buffers)
        with session._generation_transaction() as txn:
            txn.steering_cm = session.steering("0.3 b")
            txn.steering_cm.__enter__()
            assert not torch.equal(buffers[1], outer[1])
        assert manager.uses_compiled_offsets
        for L, buf in buffers.items():
            assert torch.equal(buf, outer[L])
        # A generation with no inner steering keeps them too.
        with session._generation_transaction():
            pass
        for L, buf in buffers.items():
            assert torch.equal(buf, outer[L])
    assert _zero(buffers)


def test_failed_push_keeps_the_outer_scope_installed() -> None:
    """A push whose composition raises must not tear down the outer scope.

    Composition used to clear the manager before composing, so a failure left
    the outer entries on the stack with nothing installed.
    """
    session, _buffers = _session(compiled=False)
    session._manifolds["mood"] = _manifold(layers=(2,))
    manager = session._steering
    bad = Steering(alphas={
        "mood%0.5": ManifoldTerm(
            along=-0.5, onto=0.0, trigger=Trigger.BOTH,
            manifold="mood", position=(0.5,),
        ),
    })
    with session.steering("0.5 a"):
        outer_plan = manager.plan
        outer_hooks = dict(manager.hooks)
        assert outer_hooks
        with pytest.raises(SteeringCompositionError):
            with session.steering(bad):
                pass
        assert manager.plan is outer_plan
        assert manager.hooks == outer_hooks
        hidden = torch.zeros(1, 1, _D)
        session._layers[1](hidden)
        assert bool(hidden.any())  # the outer push still fires
    assert not manager.hooks


def test_repeated_push_reuses_the_composed_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _buffers = _session(compiled=False)
    composer = session._steering_composer
    calls: list[tuple[str, ...]] = []
    original = composer.compose

    def counting(entries: Any, deps: Any = None) -> Any:
        calls.append(tuple(entries))
        return original(entries, deps)

    monkeypatch.setattr(composer, "compose", counting)
    for _ in range(3):
        with session.steering("0.5 a"):
            plan = session._steering.plan
    assert calls == [("a",)]
    with session.steering("0.5 a"):
        assert session._steering.plan is plan

    # Re-registering the profile invalidates the cached plan.
    session._profiles["a"] = {1: 2.0 * _unit(0)}
    with session.steering("0.5 a"):
        assert session._steering.plan is not plan
    assert calls == [("a",), ("a",)]


def test_unknown_vector_raises_before_the_scope_is_entered() -> None:
    from drowse.session.session import ProfileNotRegisteredError

    session, _buffers = _session(compiled=False)
    with pytest.raises(ProfileNotRegisteredError):
        session.steering("0.5 missing")
    assert session._steering_composer._stack == []
