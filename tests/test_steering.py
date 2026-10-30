"""Steering dataclass unit tests — from_value coercion + trigger entries."""

from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger


def test_construct():
    s = Steering(alphas={"foo": 0.5})
    assert s.alphas == {"foo": 0.5}
    assert s.thinking is None
    assert s.trigger is Trigger.BOTH


def test_thinking_field():
    s = Steering(alphas={"foo": 0.1}, thinking=True)
    assert s.thinking is True


def test_from_value_string_parses_expr():
    s = Steering.from_value("0.5 foo")
    assert isinstance(s, Steering)
    assert s.alphas == {"foo": 0.5}


def test_from_value_none_returns_none():
    assert Steering.from_value(None) is None


def test_from_value_passthrough():
    s = Steering(alphas={"x": 1.0})
    assert Steering.from_value(s) is s


def test_from_value_rejects_dict_input():
    import pytest
    with pytest.raises(TypeError) as ei:
        Steering.from_value({"foo": 0.5})  # pyright: ignore[reportArgumentType]  # intentional wrong-type test
    assert "str | Steering | None" in str(ei.value)


def test_default_trigger_applies_to_bare_floats():
    s = Steering(alphas={"foo": 0.5}, trigger=Trigger.AFTER_THINKING)
    entries = s.entries()
    assert entries == {"foo": (0.5, Trigger.AFTER_THINKING)}


def test_tuple_entry_overrides_default_trigger():
    s = Steering(
        alphas={"foo": (0.5, Trigger.THINKING_ONLY)},
        trigger=Trigger.AFTER_THINKING,
    )
    entries = s.entries()
    assert entries == {"foo": (0.5, Trigger.THINKING_ONLY)}


def test_mixed_entries_normalize_correctly():
    s = Steering(
        alphas={
            "bare":  0.3,
            "tuple": (0.4, Trigger.AFTER_THINKING),
        },
    )
    entries = s.entries()
    assert entries["bare"] == (0.3, Trigger.BOTH)
    assert entries["tuple"] == (0.4, Trigger.AFTER_THINKING)


def test_expression_preserves_trigger_entries():
    # The expression grammar is the new input shape; ``@after`` etc. map
    # onto the same ``(alpha, Trigger)`` tuple form that direct-construction
    # accepts.
    s = Steering.from_value("0.5 foo + 0.3 bar@response")
    assert s is not None  # non-None string input always yields a Steering
    entries = s.entries()
    assert entries["foo"] == (0.5, Trigger.BOTH)
    assert entries["bar"] == (0.3, Trigger.GENERATED_ONLY)


def test_entries_coerces_int_alpha_to_float():
    s = Steering(alphas={"foo": 1})
    entries = s.entries()
    assert entries == {"foo": (1.0, Trigger.BOTH)}
    assert isinstance(entries["foo"][0], float)


# ---------------------------------------------------------------------------
# entries(): one shape-blind (coeff, trigger) view over every term kind.
# ---------------------------------------------------------------------------


def _mixed_steering() -> Steering:
    from drowse.steering.grammar import AblationTerm, ManifoldTerm, ProjectedTerm

    return Steering(alphas={
        "bare": 0.3,
        "tuple": (0.4, Trigger.AFTER_THINKING),
        "a~b": ProjectedTerm(
            coeff=0.5, trigger=Trigger.GENERATED_ONLY,
            operator="~", base="a", onto="b",
        ),
        "!gone": AblationTerm(
            coeff=1.0, trigger=Trigger.BOTH, target="gone",
        ),
        "personas%hacker": ManifoldTerm(
            along=0.6, onto=0.2, trigger=Trigger.BOTH,
            manifold="personas", position="hacker",
        ),
    })


def test_entries_cover_every_term_kind():
    assert _mixed_steering().entries() == {
        "bare": (0.3, Trigger.BOTH),
        "tuple": (0.4, Trigger.AFTER_THINKING),
        "a~b": (0.5, Trigger.GENERATED_ONLY),
        "!gone": (1.0, Trigger.BOTH),
        # A manifold term's scalar is its ``along`` half.
        "personas%hacker": (0.6, Trigger.BOTH),
    }


# ---------------------------------------------------------------------------
# triggers() + the entry readers: the shape-blind views over an entry.
# ---------------------------------------------------------------------------


def test_triggers_covers_every_entry_kind():
    assert _mixed_steering().triggers() == {
        Trigger.BOTH, Trigger.AFTER_THINKING, Trigger.GENERATED_ONLY,
    }


def test_triggers_gives_bare_floats_the_steering_default():
    steering = Steering(
        alphas={"bare": 0.3}, trigger=Trigger.GENERATED_ONLY,
    )
    assert steering.triggers() == {Trigger.GENERATED_ONLY}


def test_entry_readers_are_shape_blind():
    from drowse.steering.terms import entry_coeff, entry_trigger
    from drowse.steering.grammar import AblationTerm, ManifoldTerm

    tuple_entry = (0.4, Trigger.AFTER_THINKING)
    ablation = AblationTerm(coeff=0.25, trigger=Trigger.BOTH, target="gone")
    manifold = ManifoldTerm(
        along=0.6, onto=0.2, trigger=Trigger.GENERATED_ONLY,
        manifold="personas", position="hacker",
    )

    assert entry_trigger(tuple_entry) is Trigger.AFTER_THINKING
    assert entry_trigger(ablation) is Trigger.BOTH
    assert entry_trigger(manifold) is Trigger.GENERATED_ONLY

    assert entry_coeff(tuple_entry) == 0.4
    assert entry_coeff(ablation) == 0.25
    # A manifold term's scalar strength is its ``along`` half — what the flat
    # ``{name: alpha}`` views report.
    assert entry_coeff(manifold) == 0.6
