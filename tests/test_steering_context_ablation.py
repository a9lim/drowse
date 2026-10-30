"""Ablation dispatch through a steering scope and the composer rebuild.

Runs on a real steering composer over three no-op layers (``SteeringRig``):
no model is loaded, and only the steering-stack and hook-manager wiring is
exercised.
"""
from __future__ import annotations

import pytest
import torch

from drowse.errors import ProfileNotRegisteredError
from drowse.steering.grammar import AblationTerm
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from tests._whitener import isotropic_whitener
from tests.support.composer import SteeringRig


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
