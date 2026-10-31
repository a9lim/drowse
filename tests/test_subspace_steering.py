"""Plan-level lowering of the dispatch-synthesized merged affine subspace.

Vectors / poles / ``~``/``|`` / ``!`` / affine-``%`` all reach the model as one
``synthesize_subspace`` → ``SteeringPlan.add_subspace`` → :class:`AffineOp`
path.  These tests pin the lowering — ``eff_along_L = share_L · gain``
(``share_L`` mean-1 normalized; no lever, no ``[0, 1]`` clamp), the κ gain
compensation, the installed ops, and compiled/eager offset parity.  The kernel
math lives in ``test_manifold_math.py`` / ``test_synthesize_subspace.py``.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from drowse.steering.hooks import (
    _SUBSPACE_GAIN,
    AffineGroup,
    LoweredAffine,
    SteeringManager,
    SteeringPlan,
    _lower_affine_subspaces,
    install_persistent_offset_hooks,
)
from drowse.geometry.whitener import LayerWhitener
from drowse.geometry.inject import (
    SynthesizedSubspace,
    synthesize_subspace as _synthesize_subspace,
)
from drowse.steering.triggers import Trigger
from tests.support.whitener import isotropic_whitener


_DIM = 8


def synthesize_subspace(
    *args: Any,
    neutral_means: dict[int, torch.Tensor],
    whitener: LayerWhitener | None = None,
    **kwargs: Any,
) -> SynthesizedSubspace:
    if whitener is None:
        whitener = isotropic_whitener(neutral_means, _DIM)
    return _synthesize_subspace(
        *args, neutral_means=neutral_means, whitener=whitener, **kwargs,
    )


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v)


def _row(v: torch.Tensor) -> torch.Tensor:
    return _unit(v).reshape(1, -1)


def _model_layers(n: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Identity() for _ in range(n)])


def _group(mgr: SteeringManager, layer: int, gi: int = 0) -> AffineGroup:
    group = mgr.hooks[layer].groups[gi]
    assert isinstance(group, AffineGroup)
    return group


def _low(synth: SynthesizedSubspace, layer: int) -> LoweredAffine:
    return _lower_affine_subspaces(synth)[layer]


def _installed(
    synth: SynthesizedSubspace,
    n_layers: int,
    *,
    trigger: Trigger = Trigger.BOTH,
) -> SteeringManager:
    plan = SteeringPlan()
    plan.add_subspace("__affine__", synth, trigger=trigger)
    mgr = SteeringManager()
    mgr.install(plan, _model_layers(n_layers), torch.device("cpu"), torch.float32)
    return mgr


def _single_layer_synth(
    layer: int = 0, *, coeff: float = 0.5, coord: float = 1.0,
) -> SynthesizedSubspace:
    u = _unit(torch.randn(_DIM))
    neutral = 20.0 + torch.randn(_DIM)
    return synthesize_subspace(
        push=[({layer: _row(u)}, {layer: torch.tensor([coord])}, coeff)],
        ablate=[], neutral_means={layer: neutral},
    )


def _equal_share_synth(
    layers: tuple[int, ...] = (0, 1, 2), *, coeff: float = 0.5,
) -> SynthesizedSubspace:
    """One push term spanning ``layers`` with identical per-layer magnitude.

    Every layer gets the same ``share_L = coeff·1.0``, so the **mean-1**
    normalization (``Σ share = n_layers``) leaves each at ``1.0`` — the clean
    substrate for the share-weight arithmetic.
    """
    basis_dirs = {L: _row(torch.randn(_DIM)) for L in layers}
    coord_dirs = {L: torch.tensor([1.0]) for L in layers}
    neutral_means = {L: 20.0 + torch.randn(_DIM) for L in layers}
    return synthesize_subspace(
        push=[(basis_dirs, coord_dirs, coeff)],
        ablate=[], neutral_means=neutral_means,
    )


def _unequal_share_synth(
    coords: tuple[float, ...] = (3.0, 1.0), *, coeff: float = 0.5,
) -> SynthesizedSubspace:
    """One push term with **unequal** per-layer magnitudes ``coeff·|coord|``.

    ``coords = (3, 1)`` ⇒ raw shares ``(1.5, 0.5)``, mean-1 normalized to
    ``(1.5, 0.5)`` (sum already = n_layers) — so the high-signal layer's
    ``eff_along`` exceeds 1 and must NOT be clamped (Step 8).
    """
    layers = tuple(range(len(coords)))
    basis_dirs = {L: _row(torch.randn(_DIM)) for L in layers}
    coord_dirs = {L: torch.tensor([float(c)]) for L, c in zip(layers, coords)}
    neutral_means = {L: 20.0 + torch.randn(_DIM) for L in layers}
    return synthesize_subspace(
        push=[(basis_dirs, coord_dirs, coeff)],
        ablate=[], neutral_means=neutral_means,
    )


# ----------------------------------------------------------------- lowering ---

def test_add_subspace_lowers_to_one_affine_op_per_layer():
    synth = _single_layer_synth(0)
    mgr = _installed(synth, 2)
    assert list(mgr.hooks) == [0]
    group = _group(mgr, 0)
    assert group.trigger is Trigger.BOTH
    low = _low(synth, 0)
    assert low.subspace.is_affine
    # The synth target rides through verbatim into the push constant.
    assert torch.allclose(low.target, synth.target_coord[0].to(torch.float32), atol=1e-6)
    assert group.op.const is not None
    assert torch.equal(group.op.const, low.offset())
    assert group.op.rows is None  # pure push: no ablation rows


def test_subspace_forces_slow_path():
    mgr = _installed(_single_layer_synth(0), 1)
    assert mgr.all_fast_path() is False


def test_static_steerable_for_affine_both_subspace():
    # An always-active affine subspace is a single static AffineOp on every
    # covered layer, so the gen stays StaticCache / graph-capture eligible.
    mgr = _installed(_equal_share_synth((0, 1, 2)), 3)
    assert mgr.all_fast_path() is False        # it IS steered
    assert mgr.static_steerable() is True       # but statically so
    assert all(h.is_static for h in mgr.hooks.values())


def test_static_steerable_false_when_unsteered():
    # Unsteered is the cheaper ``all_fast_path`` case, not ``static_steerable``.
    mgr = SteeringManager()
    assert mgr.static_steerable() is False
    assert mgr.all_fast_path() is True


def _offset_manager(n_layers: int) -> tuple[SteeringManager, dict[int, torch.Tensor], nn.ModuleList]:
    layers = _model_layers(n_layers)
    buffers, handles = install_persistent_offset_hooks(
        layers, _DIM, torch.device("cpu"), torch.float32,
    )
    mgr = SteeringManager()
    mgr.adopt_compiled_offsets(buffers, handles)
    return mgr, buffers, layers


def test_static_plan_rides_the_compiled_offsets():
    # ``all_fast_path`` reports "no transient hook", not "unsteered": a static
    # plan on a session with offset buffers writes them and attaches nothing.
    synth = _single_layer_synth(0)
    mgr, buffers, layers = _offset_manager(1)
    plan = SteeringPlan()
    plan.add_subspace("__affine__", synth)
    mgr.install(plan, layers, torch.device("cpu"), torch.float32)
    assert mgr.uses_compiled_offsets
    assert mgr.all_fast_path() is True       # no transient hook...
    assert mgr.static_steerable() is False   # ...so nothing to classify
    assert mgr.plan is plan                  # ...but the steering is live
    assert torch.equal(buffers[0], _low(synth, 0).offset())


def test_static_steerable_false_for_gated_subspace():
    # A phase-gated trigger keeps the hook on the ctx-consulting path.
    mgr = _installed(_single_layer_synth(0), 1, trigger=Trigger.AFTER_THINKING)
    assert mgr.static_steerable() is False


def test_custom_trigger_rides_through():
    mgr = _installed(_single_layer_synth(0), 1, trigger=Trigger.AFTER_THINKING)
    assert _group(mgr, 0).trigger == Trigger.AFTER_THINKING


def test_equal_but_distinct_both_trigger_is_interned():
    # A trigger equal to BOTH but built elsewhere must still take the static
    # path (and qualify for the compiled offsets).
    mgr = _installed(_single_layer_synth(0), 1, trigger=Trigger())
    assert _group(mgr, 0).trigger is Trigger.BOTH
    assert mgr.static_steerable() is True


# ------------------------------------------------------------------- gain ---

def test_single_layer_share_one_gives_base_gain():
    # One covered layer ⇒ mean-1 share == 1; no lever, no clamp ⇒
    # eff_along = 1.0 · _SUBSPACE_GAIN.
    assert _low(_single_layer_synth(0), 0).eff_along == pytest.approx(
        _SUBSPACE_GAIN, abs=1e-6,
    )


def test_share_weight_mean_one_across_layers():
    # Equal per-layer magnitude ⇒ each mean-1 share = 1.0 (not 1/3) ⇒
    # eff_along = _SUBSPACE_GAIN at every layer — n_layers-invariant.
    synth = _equal_share_synth((0, 1, 2))
    for L in (0, 1, 2):
        # mean-1 normalization divides then re-scales in fp32 (~gain·eps).
        assert _low(synth, L).eff_along == pytest.approx(_SUBSPACE_GAIN, rel=1e-5)


def test_n_layers_invariance():
    # A single covered layer and a 4-layer equal-share fit both put ≈ gain on
    # each contributing layer.
    for n in (1, 2, 4):
        synth = _equal_share_synth(tuple(range(n)))
        for L in range(n):
            assert _low(synth, L).eff_along == pytest.approx(_SUBSPACE_GAIN, abs=1e-6)


def test_high_share_layer_unclamped_share_weighting():
    # Unequal magnitudes ⇒ the high-signal layer's mean-1 share > 1 and its
    # eff_along is NOT clamped.  share (1.5, 0.5) ⇒ eff_along ratio 3:1.
    synth = _unequal_share_synth((3.0, 1.0))
    a0, a1 = _low(synth, 0).eff_along, _low(synth, 1).eff_along
    assert a0 == pytest.approx(1.5 * _SUBSPACE_GAIN, abs=1e-6)
    assert a1 == pytest.approx(0.5 * _SUBSPACE_GAIN, abs=1e-6)
    assert a0 / a1 == pytest.approx(3.0, abs=1e-6)


# --------------------------------------------------------------- hot path ---

def test_hot_path_translates_in_subspace_component_by_target():
    # A pure-push affine group translates the in-subspace coord by the fixed
    # offset ``eff·target`` (κ=0 on every push axis): ``coord = q + eff·target``
    # — it preserves ``q`` (the per-token spread) rather than collapsing onto
    # the target.  Base-robust: exact at any gain.
    u = _unit(torch.randn(_DIM))
    neutral = 20.0 + torch.randn(_DIM)
    synth = synthesize_subspace(
        push=[({0: _row(u)}, {0: torch.tensor([1.0])}, 0.7)],
        ablate=[], neutral_means={0: neutral},
    )
    plan = SteeringPlan()
    plan.add_subspace("__affine__", synth)
    layers = _model_layers(1)
    SteeringManager().install(plan, layers, torch.device("cpu"), torch.float32)
    low = _low(synth, 0)
    eff = low.eff_along
    assert eff == pytest.approx(_SUBSPACE_GAIN, abs=1e-6)  # share == 1
    assert low.is_pure_push  # κ=0 on the push axis

    sub = synth.layers[0]
    target = synth.target_coord[0]
    # Start near neutral so the displacement stays well under the norm_cap.
    hidden = (sub.mean + 0.3 * torch.randn(1, 4, _DIM)).to(torch.float32)
    q_coord = (hidden - sub.mean) @ sub.basis.T
    before = hidden.clone()
    layers[0](hidden)  # Identity forwards → hook mutates in place
    assert not torch.allclose(hidden, before, atol=1e-3)
    coord = (hidden - sub.mean) @ sub.basis.T
    expected = q_coord + eff * target.expand_as(q_coord)
    assert torch.allclose(coord, expected, atol=1e-4)


# -------------------------------------------- compiled / eager push parity ---
#
# The persistent compiled offsets (``SteeringPlan.static_offsets``) and the
# transient :class:`AffineOp` constant are two consumers of ONE lowering,
# ``_lower_affine_subspaces``, so ``compile=True`` and ``compile=False``
# generations steer identically.

def _ablation_synth(layer: int = 0, *, coeff: float = 0.4):
    """A push term plus an orthogonal ablation term at one layer (κ ≠ 0)."""
    push_dir = torch.zeros(_DIM)
    push_dir[0] = 1.0
    abl_dir = torch.zeros(_DIM)
    abl_dir[1] = 1.0
    neutral = 20.0 + torch.randn(_DIM)
    return synthesize_subspace(
        push=[({layer: push_dir.reshape(1, -1)},
               {layer: torch.tensor([1.0])}, 0.5)],
        ablate=[({layer: abl_dir.reshape(1, -1)}, coeff)],
        neutral_means={layer: neutral},
    )


def _plan(*synths: SynthesizedSubspace) -> SteeringPlan:
    plan = SteeringPlan()
    for i, synth in enumerate(synths):
        plan.add_subspace(f"__affine__{i}", synth)
    return plan


def test_static_offsets_match_transient_constant_add():
    # The compiled offset for each layer is the transient op's constant.
    plan = _plan(_unequal_share_synth((3.0, 1.0)))
    mgr = SteeringManager()
    mgr.install(plan, _model_layers(2), torch.device("cpu"), torch.float32)
    offsets = plan.static_offsets()
    assert offsets is not None
    assert set(offsets) == set(mgr.hooks)
    for L, hook in mgr.hooks.items():
        assert hook.is_static
        assert torch.equal(offsets[L], _group(mgr, L).op.const)


def test_static_offsets_sum_multiple_subspaces_like_the_hook_groups():
    # Two always-active affine subspaces sharing a layer: the compiled path
    # sums their offsets into one buffer, the eager path adds one per group.
    plan = _plan(_single_layer_synth(0, coeff=0.5), _single_layer_synth(0, coeff=0.3))
    mgr = SteeringManager()
    mgr.install(plan, _model_layers(1), torch.device("cpu"), torch.float32)
    offsets = plan.static_offsets()
    assert offsets is not None
    hook = mgr.hooks[0]
    assert not hook.is_static  # two groups
    consts = [_group(mgr, 0, gi).op.const for gi in range(2)]
    assert all(c is not None for c in consts)
    assert torch.allclose(offsets[0], consts[0] + consts[1], atol=1e-6)


def test_static_offsets_use_the_shared_lowering():
    synth = _unequal_share_synth((3.0, 1.0))
    offsets = _plan(synth).static_offsets()
    assert offsets is not None
    lowered = _lower_affine_subspaces(synth)
    assert set(lowered) == set(offsets)
    for L, low in lowered.items():
        assert low.eff_along == pytest.approx(
            _SUBSPACE_GAIN * (1.5 if L == 0 else 0.5), abs=1e-5,
        )
        assert low.is_pure_push
        assert torch.equal(offsets[L], low.offset())


def test_installed_op_is_built_from_the_lowering():
    # The ablation rows / scale / mean projection of the installed op are the
    # lowered subspace's nonzero-κ rows, scaled to the user coefficient.
    synth = _ablation_synth(0)
    mgr = SteeringManager()
    mgr.install(_plan(synth), _model_layers(1), torch.device("cpu"), torch.float32)
    low = _lower_affine_subspaces(synth)[0]
    op = _group(mgr, 0).op
    active = low.kappa != 0
    assert op.rows is not None and op.scale is not None and op.mean_proj is not None
    assert torch.allclose(op.rows, low.subspace.basis[active], atol=1e-6)
    assert torch.allclose(op.scale, low.eff_along * low.kappa[active], atol=1e-6)
    assert torch.allclose(
        op.mean_proj, low.subspace.mean @ low.subspace.basis[active].T, atol=1e-5,
    )
    assert op.const is not None and torch.equal(op.const, low.offset())


def test_ablation_disqualifies_the_compiled_offsets():
    # An ``!`` ablation makes the injection depend on ``h``: the compiled
    # offsets decline and the transient op carries ablation rows.
    synth = _ablation_synth(0)
    plan = _plan(synth)
    mgr = SteeringManager()
    mgr.install(plan, _model_layers(1), torch.device("cpu"), torch.float32)
    assert plan.static_offsets() is None
    assert _group(mgr, 0).op.rows is not None
    assert not _lower_affine_subspaces(synth)[0].is_pure_push


def test_gain_compensated_kappa_recovers_the_user_coefficient():
    # The lowering divides the requested κ by ``eff_along`` so the kernel's
    # ``along · κ`` product is exactly the user's ablation fraction.
    synth = _ablation_synth(0, coeff=0.4)
    low = _lower_affine_subspaces(synth)[0]
    product = low.eff_along * low.kappa
    assert float(product.abs().max()) == pytest.approx(0.4, abs=1e-5)


def test_clear_all_drops_the_plan():
    mgr = _installed(_single_layer_synth(0), 1)
    mgr.clear_all()
    assert mgr.plan is None
    assert mgr.hooks == {}


# ----------------------------------------------------------------- gain bound ---

def test_affine_steer_coeff_half_displaces_in_sane_whitened_band():
    """A coefficient-0.5 affine steer on a synthetic whitened subspace displaces
    the in-subspace foot by a whitened amount in ``[0.2, 3.0]`` — bounding both the
    "did nothing" and "blew up" failure modes WITHOUT pinning the exact calibration.

    The test constructs a geometry where the whitened displacement is knowable:
    one layer, one direction, isotropic whitener (Σ ≈ I) so the whitened norm ≈
    the Euclidean norm.  The actual displacement per step is
    ``eff_along * target_coord = share * _SUBSPACE_GAIN * target_coord``.

    For a single-layer synth: ``share = 1.0`` (mean-1 of one layer), and
    ``target_coord = coeff * 1.0 = 0.5`` (whitened-unit target scaled by α).
    So the expected per-layer displacement is ``_SUBSPACE_GAIN * 0.5``.

    The bound ``[1.0, 25.0]`` catches a 10× error in either direction:
    - < 1.0 would mean _SUBSPACE_GAIN was accidentally set to ≤ 2.0, which
      would make ``α≈0.5`` steer too weakly to be coherent on any model.
    - > 25.0 would mean _SUBSPACE_GAIN was accidentally set to ≥ 50.0, which
      is wildly beyond the live-calibrated value and would shatter outputs.
    """
    # Single direction in DIM-space; whitened target is coeff * unit whitened dir.
    u = _unit(torch.randn(_DIM))
    neutral = torch.zeros(_DIM)   # zero neutral so decompose is clean
    synth = synthesize_subspace(
        push=[({0: _row(u)}, {0: torch.tensor([1.0])}, 0.5)],
        ablate=[], neutral_means={0: neutral},
    )

    # Displacement = eff_along * target_coord.
    # eff_along = share * _SUBSPACE_GAIN = 1.0 * _SUBSPACE_GAIN.
    # target_coord = synthesize puts the whitened-unit direction scaled by coeff.
    # Without a whitener the fallback is Euclidean: target_coord = 0.5 * 1.0 = 0.5.
    eff_along = _low(synth, 0).eff_along
    target_coord = synth.target_coord[0]

    # In-subspace displacement = |eff_along * target_coord| per axis.
    displacement = float((eff_along * target_coord).abs().sum())

    assert 1.0 <= displacement <= 25.0, (
        f"Affine steer with coeff=0.5 produced displacement={displacement:.4f}, "
        f"outside sane band [1.0, 25.0].  "
        f"(eff_along={eff_along:.4f}, target_coord={target_coord.tolist()}, "
        f"_SUBSPACE_GAIN={_SUBSPACE_GAIN}).  "
        "This catches a >10x miscalibration of _SUBSPACE_GAIN."
    )
