"""Steering dataclass unit tests — from_value coercion + trigger entries."""

from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from drowse.steering.grammar import parse_expr


def test_construct():
    s = Steering(alphas={"foo": 0.5})
    assert s.alphas == {"foo": 0.5}
    assert s.thinking is None
    assert s.trigger is Trigger.BOTH


def test_thinking_field():
    s = Steering(alphas={"foo": 0.1}, thinking=True)
    assert s.thinking is True


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


# ---------------------------------------------------------------------------
# Gated terms in Steering
# ---------------------------------------------------------------------------


class TestSteeringIntegration:
    def test_entries_propagate_gate(self):
        # ``myprobe.other`` is a synthetic, non-bundled name so the steering
        # term stays a plain vector (a bundled name like ``angry.calm`` now
        # routes to a 2-node pca ManifoldTerm).  The gate probe stays
        # ``angry.calm`` — gate probes are stored verbatim, never resolved,
        # so the gate-parsing intent is unaffected.
        s = parse_expr("0.5 myprobe.other@when:angry.calm>0.3")
        out = s.entries()
        assert "myprobe.other" in out
        _alpha, trig = out["myprobe.other"]
        assert trig.gate is not None

    def test_steering_str_round_trips_gate(self):
        s = parse_expr("0.3 myprobe.other@when:angry.calm>=0.5")
        # ``Steering.__str__`` delegates to ``format_expr``.
        assert str(s) == "0.3 myprobe.other@when:angry.calm>=0.5"

    def test_steering_from_value_accepts_gate_string(self):
        s = Steering.from_value("0.5 myprobe.other@when:angry.calm>0.4")
        assert s is not None
        assert "myprobe.other" in s.alphas
