"""Trigger.active() semantics + preset coverage.

Pure CPU dataclass tests — no model, no hooks. Covers every preset against
every lifecycle ctx state (prefill / decode-thinking / decode-response)
plus the first_n / after_n window modifiers.
"""

from __future__ import annotations

from typing import Any

import pytest

from drowse.steering.grammar import parse_expr
from drowse.steering.terms import Steering
from drowse.steering.triggers import ProbeGate, Trigger, TriggerContext

# Gate tests parse expressions; bare-name resolution reads a fresh home.
pytestmark = pytest.mark.usefixtures("isolated_home")


def _ctx(*, prefill: bool = False, thinking: bool = False, gen_step: int = 0) -> TriggerContext:
    return TriggerContext(is_prefill=prefill, thinking=thinking, gen_step=gen_step)


def test_default_trigger_fires_everywhere():
    t = Trigger.BOTH
    assert t.active(_ctx(prefill=True))
    assert t.active(_ctx(thinking=True))
    assert t.active(_ctx(thinking=False))
    assert t.active(_ctx(gen_step=99))


def test_generated_only_skips_prefill():
    t = Trigger.GENERATED_ONLY
    assert not t.active(_ctx(prefill=True))
    assert t.active(_ctx(thinking=True))
    assert t.active(_ctx(thinking=False))


def test_prompt_only_skips_decode():
    t = Trigger.PROMPT_ONLY
    assert t.active(_ctx(prefill=True))
    assert not t.active(_ctx(thinking=True))
    assert not t.active(_ctx(thinking=False))


def test_after_thinking_fires_only_during_response():
    t = Trigger.AFTER_THINKING
    assert not t.active(_ctx(prefill=True))
    assert not t.active(_ctx(thinking=True))
    assert t.active(_ctx(thinking=False))


def test_thinking_only_fires_only_during_thinking():
    t = Trigger.THINKING_ONLY
    assert not t.active(_ctx(prefill=True))
    assert t.active(_ctx(thinking=True))
    assert not t.active(_ctx(thinking=False))


def test_first_n_window():
    t = Trigger.first(5)
    for step in range(5):
        assert t.active(_ctx(gen_step=step)), f"step={step}"
    assert not t.active(_ctx(gen_step=5))
    assert not t.active(_ctx(gen_step=100))
    # first() also turns off prefill (wait-for-generation semantics)
    assert not t.active(_ctx(prefill=True))


def test_after_n_window():
    t = Trigger.after(3)
    for step in range(3):
        assert not t.active(_ctx(gen_step=step)), f"step={step}"
    assert t.active(_ctx(gen_step=3))
    assert t.active(_ctx(gen_step=100))
    assert not t.active(_ctx(prefill=True))


def test_custom_trigger_compose():
    # Only during thinking, only after step 2.
    t = Trigger(prompt=False, thinking=True, response=False, after_n=2)
    assert not t.active(_ctx(thinking=True, gen_step=0))
    assert not t.active(_ctx(thinking=True, gen_step=1))
    assert t.active(_ctx(thinking=True, gen_step=2))
    assert not t.active(_ctx(thinking=False, gen_step=2))


def test_trigger_is_hashable_and_frozen():
    """Frozen dataclass means equal triggers share identity and can key dicts."""
    a = Trigger.BOTH
    b = Trigger()
    assert a == b
    assert hash(a) == hash(b)
    # Verify frozenness.
    with pytest.raises(Exception):
        a.prompt = False  # pyright: ignore[reportAttributeAccessIssue]  # frozen dataclass; assignment is intentional to test frozenness


def test_presets_are_trigger_instances():
    for t in (Trigger.BOTH, Trigger.GENERATED_ONLY, Trigger.PROMPT_ONLY,
              Trigger.AFTER_THINKING, Trigger.THINKING_ONLY):
        assert isinstance(t, Trigger)


def test_context_reset():
    ctx = TriggerContext(is_prefill=True, thinking=True, gen_step=42)
    ctx.reset()
    assert ctx.is_prefill is False
    assert ctx.thinking is False
    assert ctx.gen_step == 0


# ---------------------------------------------------------------------------
# Manifold-probe gates: fuzzy channels and composition
# ---------------------------------------------------------------------------


def _trigger_of(steering: Steering, key: str) -> Any:
    """Pull the :class:`Trigger` attached to ``key`` in ``steering.alphas``."""
    val = steering.alphas[key]
    if isinstance(val, tuple):
        _, trig = val
        return trig
    return steering.trigger

def _scores_ctx(scores: dict[str, float]) -> TriggerContext:
    """Build a decode-phase TriggerContext with the given probe scores."""
    ctx = TriggerContext()
    ctx.is_prefill = False
    ctx.probe_scores = dict(scores)
    return ctx

class TestFuzzyChannelGates:
    def test_membership_and_assignment_gates_fire(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:membership > 0.6")
        trig = _trigger_of(s, "happy.sad")
        assert trig.active(_scores_ctx({"circumplex:membership": 0.7}))
        assert not trig.active(_scores_ctx({"circumplex:membership": 0.5}))
        s2 = parse_expr("0.3 happy.sad @when:circumplex~elated > 0.5")
        trig2 = _trigger_of(s2, "happy.sad")
        assert trig2.active(_scores_ctx({"circumplex~elated": 0.8}))
        assert not trig2.active(_scores_ctx({"circumplex~elated": 0.3}))

class TestVectorAndManifoldGatesCompose:
    """The session merges vector and manifold scalars into one dict.

    A multi-term expression where one term gates on a vector probe and
    another on a manifold scalar must read both keys out of the same
    ``ctx.probe_scores`` without interference.  This is the cross-probe
    convergence point the Phase 1 ``flat_scalars`` adapter exists for.
    """

    def test_both_gates_fire_when_both_keys_above(self):
        text = (
            "0.3 happy.sad@when:angry.calm>0.4 "
            "+ 0.2 calm@when:circumplex:fraction>0.5"
        )
        s = parse_expr(text)
        ctx = _scores_ctx({
            "angry.calm": 0.6,
            "circumplex:fraction": 0.7,
        })
        assert _trigger_of(s, "happy.sad").active(ctx)
        assert _trigger_of(s, "calm").active(ctx)

    def test_only_vector_gate_fires_when_manifold_below(self):
        text = (
            "0.3 happy.sad@when:angry.calm>0.4 "
            "+ 0.2 calm@when:circumplex:fraction>0.5"
        )
        s = parse_expr(text)
        ctx = _scores_ctx({
            "angry.calm": 0.6,
            "circumplex:fraction": 0.3,
        })
        assert _trigger_of(s, "happy.sad").active(ctx)
        assert not _trigger_of(s, "calm").active(ctx)

    def test_only_manifold_gate_fires_when_vector_below(self):
        text = (
            "0.3 happy.sad@when:angry.calm>0.4 "
            "+ 0.2 calm@when:circumplex:fraction>0.5"
        )
        s = parse_expr(text)
        ctx = _scores_ctx({
            "angry.calm": 0.2,
            "circumplex:fraction": 0.7,
        })
        assert not _trigger_of(s, "happy.sad").active(ctx)
        assert _trigger_of(s, "calm").active(ctx)

    def test_mixed_fraction_and_label_against_one_manifold(self):
        # Two gates against different channels of the same manifold —
        # the namespaced-key trick keeps them independent in
        # ``probe_scores`` and so independent in the trigger logic.
        text = (
            "0.3 happy.sad@when:circumplex:fraction>=0.5 "
            "+ 0.2 calm@when:circumplex@elated<-0.2"
        )
        s = parse_expr(text)
        ctx = _scores_ctx({
            "circumplex:fraction": 0.6,
            "circumplex@elated": -0.5,
        })
        assert _trigger_of(s, "happy.sad").active(ctx)
        assert _trigger_of(s, "calm").active(ctx)


# ---------------------------------------------------------------------------
# ProbeGate and Trigger.when
# ---------------------------------------------------------------------------


class TestProbeGate:
    def test_gt_strict(self):
        g = ProbeGate(probe="x", op=">", threshold=0.4)
        assert g.evaluate(0.5)
        assert not g.evaluate(0.4)
        assert not g.evaluate(0.0)

    def test_gte(self):
        g = ProbeGate(probe="x", op=">=", threshold=0.4)
        assert g.evaluate(0.5)
        assert g.evaluate(0.4)
        assert not g.evaluate(0.39)

    def test_lt_strict(self):
        g = ProbeGate(probe="x", op="<", threshold=-0.2)
        assert g.evaluate(-0.3)
        assert not g.evaluate(-0.2)
        assert not g.evaluate(0.5)

    def test_lte(self):
        g = ProbeGate(probe="x", op="<=", threshold=0.0)
        assert g.evaluate(-0.1)
        assert g.evaluate(0.0)
        assert not g.evaluate(0.0001)

    def test_frozen_dataclass_equality(self):
        a = ProbeGate(probe="angry.calm", op=">", threshold=0.4)
        b = ProbeGate(probe="angry.calm", op=">", threshold=0.4)
        c = ProbeGate(probe="angry.calm", op=">", threshold=0.5)
        assert a == b
        assert a != c
        # Hashable — required for ``Trigger`` (frozen=True) + dict keys.
        assert hash(a) == hash(b)

class TestTriggerWhen:
    def test_factory_builds_gate_only_trigger(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        assert t.gate == ProbeGate(probe="angry.calm", op=">", threshold=0.4)
        # ``when`` triggers turn off prefill (no probe reading available)
        # and leave the other windows wide open so the gate is the only
        # discriminator on decode steps.
        assert t.prompt is False
        assert t.generated is True
        assert t.thinking is True
        assert t.response is True
        assert t.first_n is None
        assert t.after_n is None

    def test_active_inactive_during_prefill(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = True
        # Even with a passing score, prefill keeps the gate inactive —
        # probe scores aren't refreshed until *after* the first forward.
        ctx.probe_scores = {"angry.calm": 0.8}
        assert not t.active(ctx)

    def test_active_when_score_above_threshold(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": 0.5}
        assert t.active(ctx)

    def test_inactive_when_score_below(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": 0.2}
        assert not t.active(ctx)

    def test_inactive_when_probe_missing(self):
        # Gate names a probe that the monitor doesn't carry.  Should
        # report inactive instead of raising — gating on a non-existent
        # probe is a degenerate ask, but a hard error mid-generation
        # would be worse than silently doing nothing.
        t = Trigger.when("nonexistent", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": 0.9}
        assert not t.active(ctx)

    def test_inactive_when_probe_scores_empty(self):
        # Pre-first-forward state: probe_scores is empty.
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        assert ctx.probe_scores == {}
        assert not t.active(ctx)

    def test_negative_threshold(self):
        t = Trigger.when("angry.calm", "<", -0.2)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": -0.5}
        assert t.active(ctx)
        ctx.probe_scores = {"angry.calm": -0.1}
        assert not t.active(ctx)

class TestTriggerContextProbeScores:
    def test_default_empty_dict(self):
        ctx = TriggerContext()
        assert ctx.probe_scores == {}

    def test_reset_clears_scores(self):
        ctx = TriggerContext()
        ctx.probe_scores = {"angry.calm": 0.7}
        ctx.gen_step = 5
        ctx.reset()
        assert ctx.probe_scores == {}
        assert ctx.gen_step == 0

    def test_independent_instances_dont_share_state(self):
        a = TriggerContext()
        b = TriggerContext()
        a.probe_scores["x"] = 0.3
        # Mutable default with field(default_factory=dict) — each
        # instance must get its own dict, not a shared module-level one.
        assert b.probe_scores == {}
