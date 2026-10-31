"""The installed steering is a function of the steering stack.

CPU tests over a model-free steering composer whose layers carry the persistent
compiled-offset hooks a compiled CUDA/MPS session adopts, so the offset path
is exercised without a compiled model:

- an exited scope leaves no steering behind (offsets zeroed, flag cleared);
- a generation scope nested in a user scope restores the outer steering on
  exit, and the generation teardown does not erase it;
- a push that fails to compose leaves the outer scope installed;
- repeated pushes of the same expression reuse one composed plan.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from drowse.errors import SteeringCompositionError
from drowse.generation.compile_runtime import CompiledRuntime
from drowse.steering.grammar import ManifoldTerm
from drowse.steering.hooks import SteeringManager, install_persistent_offset_hooks
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from tests.support.whitener import isotropic_whitener
from tests.support.manifolds import loop_manifold as _manifold
from tests.support.composer import SteeringRig

_D = 8
_N_LAYERS = 3


def _unit(i: int) -> torch.Tensor:
    v = torch.zeros(_D)
    v[i] = 1.0
    return v


def _session(*, compiled: bool = True) -> tuple[SteeringRig, dict[int, torch.Tensor]]:
    """A steering composer over identity layers, with the persistent
    compiled-offset hooks installed (``compiled=True``) as a compiled session
    adopts them."""
    layers = nn.ModuleList([nn.Identity() for _ in range(_N_LAYERS)])
    manager = SteeringManager()
    buffers: dict[int, torch.Tensor] = {}
    if compiled:
        buffers, handles = install_persistent_offset_hooks(
            layers, _D, torch.device("cpu"), torch.float32,
        )
        manager.adopt_compiled_offsets(buffers, handles)
    rig = SteeringRig(
        layers=layers,
        manager=manager,
        profiles={"a": {1: _unit(0)}, "b": {1: _unit(1)}},
        layer_means={1: torch.zeros(_D)},
        whitener=isotropic_whitener([1], _D),
    )
    rig.attach_engine(CompiledRuntime(compiled=compiled))
    return rig, buffers


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
    manager = session.manager
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
    manager = session.manager
    with session.steering("0.5 a"):
        outer = _offsets(buffers)
        with session.engine.transaction() as txn:
            txn.steering_cm = session.steering("0.3 b")
            txn.steering_cm.__enter__()
            assert not torch.equal(buffers[1], outer[1])
        assert manager.uses_compiled_offsets
        for L, buf in buffers.items():
            assert torch.equal(buf, outer[L])
        # A generation with no inner steering keeps them too.
        with session.engine.transaction():
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
    session.registry.put_manifold("mood", _manifold(layers=(2,)))
    manager = session.manager
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
        session.layers[1](hidden)
        assert bool(hidden.any())  # the outer push still fires
    assert not manager.hooks


def test_inverted_regen_flips_affine_manifold_terms_and_refuses_curved() -> None:
    """The loom ``inverted`` regen negates every ``%`` term's ``along``.

    On an affine fit that pushes away from the position; a curved position has
    no opposite, so the inverted curved term refuses at composition (it used
    to clamp silently to a no-op, so "inverted" regenerated unsteered text).
    """
    from drowse import Recipe
    from drowse.geometry.fold import fold_directions_to_subspace

    session, _buffers = _session(compiled=False)
    ray = fold_directions_to_subspace(
        "ray", {1: _unit(0)}, session.neutral.means,
        whitener=session.neutral.resident_whitener, label="pole",
    )
    session.registry.put_manifold("ray", ray)
    session.registry.put_manifold("mood", _manifold(layers=(2,)))
    manager = session.manager

    def pushed(expr: str) -> torch.Tensor:
        with session.steering(expr):
            hidden = torch.zeros(1, 1, _D)
            session.layers[1](hidden)
        return hidden[0, 0]

    inverted = Recipe(steering="0.5 ray%pole").invert_steering().steering
    assert inverted is not None
    forward, backward = pushed("0.5 ray%pole"), pushed(inverted)
    assert forward[0] > 0
    assert torch.allclose(backward, -forward)

    inverted = Recipe(steering="0.5 mood%n1").invert_steering().steering
    assert inverted is not None
    with pytest.raises(SteeringCompositionError, match="no opposite"):
        session.steering(inverted).__enter__()
    assert manager.plan is None
    assert session.composer._stack == []


@pytest.mark.parametrize("compiled", [True, False])
def test_failed_pop_rebuild_uninstalls_the_exited_scope(compiled: bool) -> None:
    """A pop whose outer head no longer composes must not keep the inner plan.

    The rebuild after ``pop`` composes the outer head; when a profile it names
    was unregistered inside the inner scope that raises, and the exited inner
    plan used to stay installed (offsets live, flag set) under a stack that
    said it was gone.
    """
    from drowse.errors import ProfileNotRegisteredError

    session, buffers = _session(compiled=compiled)
    manager = session.manager
    composer = session.composer
    outer = session.steering("0.5 a")
    outer.__enter__()
    inner = session.steering("0.3 b")
    inner.__enter__()
    session.registry.pop_profile("a")
    with pytest.raises(ProfileNotRegisteredError):
        inner.__exit__(None, None, None)
    assert [list(entry) for entry in composer._stack] == [["a"]]
    assert manager.plan is None
    assert not manager.hooks
    assert not manager.uses_compiled_offsets
    assert _zero(buffers)
    hidden = torch.zeros(1, 1, _D)
    for layer in session.layers:
        hidden = layer(hidden)
    assert not bool(hidden.any())
    # Restoring the profile lets the outer scope exit cleanly.
    session.registry.put_profile("a", {1: _unit(0)})
    outer.__exit__(None, None, None)
    assert composer._stack == []


def test_repeated_push_reuses_the_composed_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _buffers = _session(compiled=False)
    composer = session.composer
    calls: list[tuple[str, ...]] = []
    original = composer.compose

    def counting(entries: Any, deps: Any = None) -> Any:
        calls.append(tuple(entries))
        return original(entries, deps)

    monkeypatch.setattr(composer, "compose", counting)
    for _ in range(3):
        with session.steering("0.5 a"):
            plan = session.manager.plan
    assert calls == [("a",)]
    with session.steering("0.5 a"):
        assert session.manager.plan is plan

    # Re-registering the profile invalidates the cached plan.
    session.registry.put_profile("a", {1: 2.0 * _unit(0)})
    with session.steering("0.5 a"):
        assert session.manager.plan is not plan
    assert calls == [("a",), ("a",)]


def test_bare_variant_manifold_term_concretizes_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``emotions:role%pirate`` used to reach the loader unconcretized and fail."""
    from drowse.steering.grammar import parse_expr

    session, _buffers = _session(compiled=False)
    composer = session.composer
    registry = session.registry
    monkeypatch.setattr(
        registry, "concretize_key",
        lambda key: key.replace(":role", ":role-pirate"),
    )
    loaded: list[str] = []
    monkeypatch.setattr(registry, "ensure_manifold_loaded", loaded.append)
    entries = composer.resolve(parse_expr("0.5 emotions:role%pirate"))
    (term,) = entries.values()
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "emotions:role-pirate"
    assert loaded == ["emotions:role-pirate"]


def test_unknown_vector_raises_before_the_scope_is_entered() -> None:
    from drowse.errors import ProfileNotRegisteredError

    session, _buffers = _session(compiled=False)
    with pytest.raises(ProfileNotRegisteredError):
        session.steering("0.5 missing")
    assert session.composer._stack == []
