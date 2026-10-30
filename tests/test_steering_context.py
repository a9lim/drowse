"""``session.steering()`` scope semantics on a real session: flattening, events,
hook attachment, failure rollback, and role-label unanimity.

Every scope is observed through public surfaces only: ``SteeringApplied`` /
``SteeringCleared`` events, the model's own forward (are steering hooks
acting?), and the prompts a scripted model actually receives (which assistant
role label did the scope select?).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse import SamplingConfig
from drowse.errors import RoleBaselineMismatchWarning
from drowse.events import SteeringApplied, SteeringCleared
from drowse.session.session import ConcurrentExtractionError, ProfileNotRegisteredError
from drowse.steering.composer import SteeringComposer
from drowse.steering.grammar import SteeringExprError
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from tests.support.sessions import TinySessionFactory, forward_logits, register_vectors

pytestmark = pytest.mark.usefixtures("isolated_home")


@pytest.fixture
def session(tiny_session: Any) -> Any:
    register_vectors(tiny_session, "a", "b", "angry.calm", "known")
    return tiny_session


def _record(session: Any) -> list[Any]:
    events: list[Any] = []
    session.events.subscribe(events.append)
    return events


def _applied(events: list[Any]) -> list[SteeringApplied]:
    return [event for event in events if isinstance(event, SteeringApplied)]


def _steering_events(events: list[Any]) -> list[Any]:
    return [e for e in events if isinstance(e, (SteeringApplied, SteeringCleared))]


def test_single_scope_attaches_hooks_and_detaches_on_exit(session: Any) -> None:
    events = _record(session)
    baseline = forward_logits(session)
    with session.steering("0.5 angry.calm"):
        steered = forward_logits(session)
        assert not torch.allclose(steered, baseline)
    assert torch.equal(forward_logits(session), baseline)
    assert [type(e).__name__ for e in _steering_events(events)] == [
        "SteeringApplied", "SteeringCleared",
    ]
    assert _applied(events)[0].alphas == {"angry.calm": 0.5}


def test_plain_materialization_does_not_wake_whitener() -> None:
    """Only ``~``/``|`` terms need the projection-time whitener read."""

    class _NoWhitener:
        _profiles: dict[str, object] = {}

        @property
        def whitener(self) -> None:
            raise AssertionError("plain steering touched the lazy whitener")

    composer = SteeringComposer(_NoWhitener())  # type: ignore[arg-type]
    assert composer.materialize_projections(Steering(alphas={"plain": 0.2})) == {}


def test_generation_preflight_builds_the_lazy_whitener(
    make_tiny_session: TinySessionFactory,
) -> None:
    """A ``~`` projection on a cold session needs the whitener mid-generation.

    The public ``whitener`` getter refuses while a generation is in flight, so
    the lock-owning generation preflight must publish it first.
    """
    session = make_tiny_session(warm=False)
    register_vectors(session, "a", "b")
    result = session.generate(
        "hi", steering="0.3 a~b", sampling=SamplingConfig(max_tokens=2, seed=0),
    ).first
    assert result.token_count == 2
    assert session.whitener is not None


def test_fit_refuses_active_steering_before_touching_artifacts(
    session: Any, tmp_path: Path,
) -> None:
    with session.steering("0.3 a"):
        with pytest.raises(ConcurrentExtractionError, match="active steering"):
            session.fit(tmp_path / "does-not-exist")


def test_nested_scopes_flatten_inner_wins(session: Any) -> None:
    events = _record(session)
    with session.steering("0.3 a"):
        with session.steering("0.5 a + 0.1 b"):
            pass
    assert [e.alphas for e in _applied(events)] == [
        {"a": 0.3}, {"a": 0.5, "b": 0.1}, {"a": 0.3},
    ]
    assert isinstance(events[-1], SteeringCleared)


def test_nested_scopes_union_distinct_names(session: Any) -> None:
    events = _record(session)
    with session.steering("0.3 a"):
        with session.steering("0.1 b"):
            pass
    assert [e.alphas for e in _applied(events)] == [
        {"a": 0.3}, {"a": 0.3, "b": 0.1}, {"a": 0.3},
    ]
    assert sum(isinstance(e, SteeringCleared) for e in events) == 1


def test_steering_accepts_steering_instance(session: Any) -> None:
    events = _record(session)
    with session.steering(Steering(alphas={"a": 0.2})):
        pass
    assert _applied(events)[0].alphas == {"a": 0.2}


def test_unknown_vector_raises_on_enter_and_rolls_back(session: Any) -> None:
    events = _record(session)
    baseline = forward_logits(session)
    with pytest.raises(ProfileNotRegisteredError):
        with session.steering("0.5 unknown"):
            pass
    assert events == []
    assert torch.equal(forward_logits(session), baseline)
    # The failed entry left nothing on the stack for the next scope to inherit.
    with session.steering("0.5 known"):
        pass
    assert _applied(events)[0].alphas == {"known": 0.5}


def test_failed_enter_under_outer_scope_preserves_outer_entry(session: Any) -> None:
    """An inner failed enter must not pop the outer scope's entry."""
    events = _record(session)
    with session.steering("0.3 a"):
        with pytest.raises(ProfileNotRegisteredError):
            with session.steering("0.5 unknown"):
                pass
        with session.steering("0.1 b"):
            pass
    assert [e.alphas for e in _applied(events)] == [
        {"a": 0.3}, {"a": 0.3, "b": 0.1}, {"a": 0.3},
    ]


def test_failed_enter_under_outer_scope_keeps_outer_hooks_acting(session: Any) -> None:
    """A failed inner enter leaves the outer scope's hooks steering.

    The composition-failure case inside ``SteeringComposer.push`` is pinned
    at the manager level by ``test_steering_install``.
    """
    with session.steering("0.3 a"):
        outer = forward_logits(session)
        with pytest.raises(ProfileNotRegisteredError):
            with session.steering("0.5 unknown"):
                pass
        assert torch.equal(forward_logits(session), outer)


def test_global_trigger_flows_to_entries(session: Any) -> None:
    events = _record(session)
    with session.steering(Steering(alphas={"a": 0.3}, trigger=Trigger.AFTER_THINKING)):
        pass
    assert _applied(events)[0].entries == {"a": (0.3, Trigger.AFTER_THINKING)}


def test_per_entry_trigger_flows_to_entries(session: Any) -> None:
    events = _record(session)
    with session.steering("0.3 a + 0.4 b@thinking"):
        pass
    assert _applied(events)[0].entries == {
        "a": (0.3, Trigger.BOTH), "b": (0.4, Trigger.THINKING_ONLY),
    }


def test_nested_trigger_regimes_compose(session: Any) -> None:
    events = _record(session)
    with session.steering(Steering(alphas={"a": 0.3}, trigger=Trigger.BOTH)):
        with session.steering(Steering(alphas={"b": 0.5}, trigger=Trigger.AFTER_THINKING)):
            pass
    assert [e.entries for e in _applied(events)] == [
        {"a": (0.3, Trigger.BOTH)},
        {"a": (0.3, Trigger.BOTH), "b": (0.5, Trigger.AFTER_THINKING)},
        {"a": (0.3, Trigger.BOTH)},
    ]


def test_steering_applied_event_carries_entries(session: Any) -> None:
    events = _record(session)
    with session.steering("0.3 a + 0.5 b@after"):
        applied = _applied(events)[-1]
    assert applied.alphas == {"a": 0.3, "b": 0.5}
    assert applied.entries == {
        "a": (0.3, Trigger.BOTH), "b": (0.5, Trigger.AFTER_THINKING),
    }


def test_pole_alias_resolves_to_manifold_term_with_trigger() -> None:
    """A bare pole (``wolf``) resolves through the manifold tier to a
    label-form ``ManifoldTerm`` at that node, keeping its trigger."""
    from drowse.steering.grammar import ManifoldTerm, parse_expr
    from drowse.store import selectors
    from drowse.store.manifold_authoring import create_discover_manifold_folder

    create_discover_manifold_folder(
        "default", "deer.wolf", "x", fit_mode="pca",
        node_corpora={"deer": ["a statement."], "wolf": ["b statement."]},
        hyperparams={"max_dim": 1},
    )
    selectors.invalidate()

    term = parse_expr("0.4 wolf@after").alphas["default/deer.wolf%wolf"]
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "default/deer.wolf"
    assert term.position == "wolf"
    assert term.along == 0.4
    assert term.trigger == Trigger.AFTER_THINKING


# ---------------------------------------------------------------------------
# Role-augmented steering: one assistant-role label per scope
# ---------------------------------------------------------------------------

_PIRATE = "honest.deceptive:role-pirate"
_SAGE = "angry.calm:role-sage"


class TestRoleUnanimity:
    """Every role-tagged term in one scope must agree on a single role; the
    agreed role relabels the assistant header of the generation prompt."""

    @pytest.fixture
    def scripted(self, make_tiny_session: TinySessionFactory) -> tuple[Any, Any]:
        session, script = make_tiny_session.scripted("ok")
        register_vectors(
            session, _PIRATE, "honest.deceptive:role-sage", _SAGE,
            "angry.calm:role-pirate", "honest.deceptive",
        )
        return session, script

    @staticmethod
    def _assistant_label(session: Any, script: Any) -> str:
        session.generate("hi", stateless=True, sampling=SamplingConfig(max_tokens=4))
        prompt = script.prompts[-1]
        tail = prompt.rsplit("<|start_header_id|>", 1)[1]
        return tail.split("<|end_header_id|>", 1)[0]

    def test_conflicting_roles_raise_naming_both(self, scripted: tuple[Any, Any]) -> None:
        session, _script = scripted
        with pytest.raises(SteeringExprError) as info:
            with session.steering(f"0.5 {_PIRATE} + 0.3 honest.deceptive:role-sage"):
                pass
        assert "pirate" in str(info.value)
        assert "sage" in str(info.value)

    def test_same_role_relabels_the_assistant_turn(self, scripted: tuple[Any, Any]) -> None:
        session, script = scripted
        events = _record(session)
        with session.steering(f"0.5 {_PIRATE} + 0.3 angry.calm:role-pirate"):
            assert self._assistant_label(session, script) == "pirate"
        assert set(_applied(events)[0].alphas) == {_PIRATE, "angry.calm:role-pirate"}
        assert self._assistant_label(session, script) == "assistant"

    def test_plain_and_role_mixing_warns(self, scripted: tuple[Any, Any]) -> None:
        session, script = scripted
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with session.steering(f"0.5 honest.deceptive + 0.3 {_SAGE}"):
                assert self._assistant_label(session, script) == "sage"
        mismatch = [w for w in caught if issubclass(w.category, RoleBaselineMismatchWarning)]
        assert mismatch and "sage" in str(mismatch[0].message)
        assert self._assistant_label(session, script) == "assistant"

    def test_role_only_expression_does_not_warn(self, scripted: tuple[Any, Any]) -> None:
        session, script = scripted
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with session.steering(f"0.5 {_PIRATE}"):
                assert self._assistant_label(session, script) == "pirate"
        assert not [w for w in caught if issubclass(w.category, RoleBaselineMismatchWarning)]

    def test_nested_role_scopes_inner_wins(self, scripted: tuple[Any, Any]) -> None:
        session, script = scripted
        with session.steering(f"0.5 {_PIRATE}"):
            assert self._assistant_label(session, script) == "pirate"
            with session.steering(f"0.3 {_SAGE}"):
                assert self._assistant_label(session, script) == "sage"
            assert self._assistant_label(session, script) == "pirate"
        assert self._assistant_label(session, script) == "assistant"
