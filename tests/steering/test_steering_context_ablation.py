"""Ablation dispatch through a steering scope and the composer rebuild.

Runs on a real steering composer over three no-op layers (``SteeringRig``):
no model is loaded, and only the steering-stack and hook-manager wiring is
exercised.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from drowse.errors import ProfileNotRegisteredError
from drowse.steering.grammar import AblationTerm
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from tests.support.whitener import isotropic_whitener
from tests.support.composer import SteeringRig
from tests.support.sessions import forward_logits, register_vectors


class _NoopModule(torch.nn.Module):
    def forward(self, x):  # pyright: ignore[reportMissingParameterType]  # noop stub, intentionally untyped
        return (x,)


def _rig() -> SteeringRig:
    return SteeringRig(
        layers=torch.nn.ModuleList([_NoopModule(), _NoopModule(), _NoopModule()]),
        layer_means={1: torch.zeros(3)},
        whitener=isotropic_whitener([1], 3),
    )


def test_session_steering_dispatches_ablation_to_manager():
    session = _rig()
    session.registry.put_profile("refusal", {1: torch.tensor([1.0, 0.0, 0.0])})
    session.neutral.means[1] = torch.tensor([0.5, 0.0, 0.0])

    steering = Steering(alphas={
        "!refusal": AblationTerm(coeff=1.0, trigger=Trigger.BOTH, target="refusal"),
    })
    with session.steering(steering):
        # 4.0: ``!`` lowers to an ablation axis (target 0) inside the merged
        # affine subspace — one ``add_subspace`` entry per trigger group, an
        # ``subspace_inject`` group on the covered layer.
        plan = session.manager.plan
        assert plan
        synth = plan.affine[0].synth
        assert 1 in synth.layers
        assert 1 in session.manager.hooks
        assert session.manager.hooks[1].groups

    # Post-exit: steering cleared, hook detached.
    assert session.manager.plan is None
    assert not session.manager.hooks


def test_session_steering_ablation_missing_profile_raises():
    session = _rig()
    steering = Steering(alphas={
        "!nonexistent": AblationTerm(
            coeff=1.0, trigger=Trigger.BOTH, target="nonexistent",
        ),
    })
    with pytest.raises(ProfileNotRegisteredError):
        with session.steering(steering):
            pass


def test_session_steering_string_with_ablation_end_to_end():
    """session.steering('!refusal') string form parses through to the manager."""
    session = _rig()
    session.registry.put_profile("refusal", {1: torch.tensor([1.0, 0.0, 0.0])})
    session.neutral.means[1] = torch.tensor([0.5, 0.0, 0.0])

    with session.steering("!refusal"):
        plan = session.manager.plan
        assert plan
        synth = plan.affine[0].synth
        assert 1 in synth.layers
        assert session.manager.hooks[1].groups

    assert session.manager.plan is None


@pytest.mark.parametrize("coeff", [0.15, 1.0, -0.3])
def test_ablation_coefficient_survives_affine_gain(coeff: float) -> None:
    """The shared affine gain must not amplify mean-ablation coefficients."""
    session = _rig()
    session.registry.put_profile("refusal", {1: torch.tensor([1.0, 0.0, 0.0])})

    steering = Steering(alphas={
        "!refusal": AblationTerm(
            coeff=coeff, trigger=Trigger.BOTH, target="refusal",
        ),
    })
    with session.steering(steering):
        hook = session.manager.hooks[1]
        assert len(hook.groups) == 1
        op = hook.groups[0].op
        assert op.scale is not None
        assert float(op.scale[0]) == pytest.approx(coeff)


def test_zero_ablation_coefficient_is_a_true_noop() -> None:
    session = _rig()
    session.registry.put_profile("refusal", {1: torch.tensor([1.0, 0.0, 0.0])})

    steering = Steering(alphas={
        "!refusal": AblationTerm(
            coeff=0.0, trigger=Trigger.BOTH, target="refusal",
        ),
    })
    with session.steering(steering):
        assert not session.manager.plan
        assert not session.manager.hooks


def test_mean_ablation_pins_the_concept_coordinate_to_neutral(tiny_session: Any) -> None:
    """``!a`` at alpha 1 replaces every position's component along the
    concept direction with the neutral mean's: ``h'.d = mu.d`` (the
    Euclidean operator ``h' = h - (h.d - mu.d) d``), read from the layer
    outputs of a real forward."""
    profile = register_vectors(tiny_session, "a")["a"]
    means = tiny_session.layer_means
    seen: dict[int, torch.Tensor] = {}

    def _record(layer: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            seen[layer] = hidden[0].detach().float().clone()
        return hook

    with tiny_session.steering("!a"):
        handles = [
            module.register_forward_hook(_record(layer))
            for layer, module in enumerate(tiny_session.layers)
        ]
        try:
            forward_logits(tiny_session)
        finally:
            for handle in handles:
                handle.remove()

    assert sorted(seen) == list(range(len(tiny_session.layers)))
    for layer, hidden in seen.items():
        direction = profile[layer] / profile[layer].norm()
        neutral = float(means[layer].float() @ direction)
        assert torch.allclose(
            hidden @ direction, torch.full((hidden.shape[0],), neutral), atol=1e-5,
        )
