"""SteeringHook per-trigger conditional apply and the shared AffineOp.

Every affine injection lowers to one :class:`AffineOp`; a group fires only on
the decode steps where its :class:`Trigger` is active, and ``ctx`` mutation
between forwards gates the apply.  A ``nn.Identity``-equivalent module is
enough to drive the hook.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from drowse.steering.hooks import (
    AffineGroup,
    AffineOp,
    LoweredAffine,
    SteeringHook,
    SteeringManager,
    SteeringPlan,
)
from drowse.geometry.manifold import (
    CustomDomain,
    LayerSubspace,
    subspace_inject,
    synthesize_subspace,
)
from drowse.steering.triggers import Trigger, TriggerContext
from tests._whitener import isotropic_whitener


_DIM = 16


def _unit(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _affine_group(layer: int, trigger: Trigger) -> AffineGroup:
    """A rank-1 merged affine subspace from one push term, lowered."""
    u = _unit(_DIM)
    neutral = 20.0 + torch.randn(_DIM, generator=torch.Generator().manual_seed(1))
    synth = synthesize_subspace(
        push=[({layer: u.reshape(1, -1)}, {layer: torch.tensor([1.0])}, 0.8)],
        ablate=[], neutral_means={layer: neutral},
        whitener=isotropic_whitener([layer], _DIM),
    )
    low = LoweredAffine(
        subspace=synth.layers[layer], target=synth.target_coord[layer],
        eff_along=1.0, kappa=synth.kappa[layer],
    )
    return AffineGroup(
        trigger,
        AffineOp.from_lowered(low, device=torch.device("cpu"), dtype=torch.float32),
    )


def _mixed_op(kappa: torch.Tensor, along: float, dtype: torch.dtype = torch.float32):
    basis = torch.eye(2, _DIM)
    mean = torch.linspace(-0.2, 0.3, _DIM)
    sub = LayerSubspace.affine(mean, basis)
    target = torch.tensor([0.7, -0.25])
    low = LoweredAffine(subspace=sub, target=target, eff_along=along, kappa=kappa)
    op = AffineOp.from_lowered(low, device=torch.device("cpu"), dtype=dtype)
    return op, sub, target, mean, basis


# ----------------------------------------------------------------- grouping ---

def test_recompose_stamps_group_and_cold_feet():
    hook = SteeringHook()
    ctx = TriggerContext()
    hook.recompose([_affine_group(0, Trigger.AFTER_THINKING)], ctx)
    assert len(hook.groups) == 1
    assert hook.groups[0].trigger == Trigger.AFTER_THINKING
    assert hook._feet == [None]
    assert not hook.is_static  # a gated group consults ctx


def test_single_always_active_affine_group_is_static():
    hook = SteeringHook()
    hook.recompose([_affine_group(0, Trigger.BOTH)], TriggerContext())
    assert hook.is_static


# ------------------------------------------------------- conditional apply ---

def test_both_trigger_always_applies():
    hook = SteeringHook()
    ctx = TriggerContext()
    hook.recompose([_affine_group(0, Trigger.BOTH)], ctx)
    hidden = (20.0 + torch.randn(1, 3, _DIM)).to(torch.float32)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)


@pytest.mark.parametrize(
    "kappa",
    [torch.tensor([0.0, 1.0]), torch.tensor([0.4, 1.0]), torch.tensor([0.0, 0.0])],
)
def test_affine_op_matches_unified_kernel(kappa: torch.Tensor) -> None:
    """The split push + compact ablation is algebraically the affine kernel."""
    along = 0.65
    op, sub, target, mean, basis = _mixed_op(kappa, along)
    hidden = torch.randn(1, 3, _DIM)
    expected, _ = subspace_inject(
        hidden, sub, CustomDomain(2), target, torch.zeros(2), along, 0.0,
        mean_proj=mean @ basis.T, kappa=kappa,
    )
    actual = hidden.clone()
    op.apply(actual)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rank_one_ablation_matches_unified_kernel() -> None:
    """The rank-1 dot + ``addcmul_`` path is the same projection."""
    basis = _unit(_DIM, seed=3).reshape(1, -1)
    mean = torch.linspace(-0.2, 0.3, _DIM)
    sub = LayerSubspace.affine(mean, basis)
    target = torch.tensor([0.3])
    kappa = torch.tensor([0.8])
    low = LoweredAffine(subspace=sub, target=target, eff_along=1.5, kappa=kappa)
    op = AffineOp.from_lowered(low, device=torch.device("cpu"), dtype=torch.float32)
    assert op.scaled_row is not None
    hidden = torch.randn(2, 4, _DIM)
    expected, _ = subspace_inject(
        hidden, sub, CustomDomain(1), target, torch.zeros(1), 1.5, 0.0, kappa=kappa,
    )
    actual = hidden.clone()
    op.apply(actual)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_affine_op_recast_is_memoized_and_exact() -> None:
    """A block output narrower than the lowering dtype recasts once.

    A wrapper model can emit a narrower block output than its first
    parameter's dtype.  The recast lands back on the op so the per-token path
    allocates nothing after the first fire, and the result still matches the
    unified kernel.
    """
    kappa = torch.tensor([0.0, 1.0])
    along = 0.65
    op, sub, target, mean, basis = _mixed_op(kappa, along)
    assert op.rows is not None and op.rows.dtype is torch.float32

    hidden = torch.randn(1, 3, _DIM).to(torch.float16)
    expected, _ = subspace_inject(
        hidden.to(torch.float32), sub, CustomDomain(2), target, torch.zeros(2),
        along, 0.0, mean_proj=mean @ basis.T, kappa=kappa,
    )
    actual = hidden.clone()
    op.apply(actual)
    assert torch.allclose(actual.to(torch.float32), expected, atol=2e-3, rtol=2e-3)

    rows = op.rows
    assert rows is not None and rows.dtype is torch.float16
    assert op.const is not None and op.const.dtype is torch.float16
    second = hidden.clone()
    op.apply(second)
    assert op.rows is rows
    assert torch.allclose(second, actual, atol=0, rtol=0)


def test_non_both_skips_when_inactive():
    hook = SteeringHook()
    ctx = TriggerContext()
    ctx.thinking = True  # inside the thinking section ⇒ AFTER_THINKING inactive
    hook.recompose([_affine_group(0, Trigger.AFTER_THINKING)], ctx)
    hidden = (20.0 + torch.randn(1, 3, _DIM)).to(torch.float32)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert torch.allclose(hidden, before, atol=1e-6)


def test_ctx_mutation_between_forwards_gates_apply():
    hook = SteeringHook()
    ctx = TriggerContext()
    ctx.thinking = True  # start inside thinking → inactive
    hook.recompose([_affine_group(0, Trigger.AFTER_THINKING)], ctx)

    hidden = (20.0 + torch.randn(1, 1, _DIM)).to(torch.float32)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)  # inactive → no-op
    assert torch.allclose(hidden, before, atol=1e-6)

    # Leave the thinking section; the same hook now fires on response tokens.
    ctx.thinking = False
    hidden2 = (20.0 + torch.randn(1, 1, _DIM)).to(torch.float32)
    before2 = hidden2.clone()
    hook.hook_fn(None, None, hidden2)
    assert not torch.allclose(hidden2, before2, atol=1e-3)


def test_manager_threads_ctx_into_hooks():
    mgr = SteeringManager()
    synth = synthesize_subspace(
        push=[({0: _unit(_DIM).reshape(1, -1)}, {0: torch.tensor([1.0])}, 0.5)],
        ablate=[], neutral_means={0: 20.0 + torch.randn(_DIM)},
        whitener=isotropic_whitener([0], _DIM),
    )
    plan = SteeringPlan()
    plan.add_subspace("__affine__", synth)
    layers = nn.ModuleList([nn.Identity() for _ in range(2)])
    mgr.install(plan, layers, torch.device("cpu"), torch.float32)
    # Every attached hook shares the manager's mutable context.
    for hook in mgr.hooks.values():
        assert hook._ctx is mgr.ctx
