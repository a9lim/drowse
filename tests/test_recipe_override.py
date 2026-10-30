"""Tests for the recipe-override regen mechanism (v2.3 phase 5)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from drowse import Recipe, SamplingConfig
from drowse.core.loom import INHERIT_SYSTEM_PROMPT, InheritSystemPrompt, LoomTree
from drowse.core.session import DrowseSession
from drowse.core.steering_expr import ManifoldTerm, parse_expr


def test_recipe_prompt_capture_distinguishes_legacy_null_empty_and_text():
    legacy = Recipe.from_dict(Recipe().to_dict())
    assert "system_prompt" not in legacy.to_dict()
    assert legacy.resolved_system_prompt("current") == "current"
    for prompt in (None, "", "Speak like a pirate"):
        captured = Recipe.from_dict(Recipe(system_prompt=prompt).to_dict())
        assert captured.resolved_system_prompt("different current instruction") == prompt
        assert captured.overlay(Recipe(seed=7)).system_prompt == prompt
    assert Recipe(system_prompt="pirate").overlay(Recipe(system_prompt=None)).system_prompt is None


@pytest.mark.parametrize("prompt", [INHERIT_SYSTEM_PROMPT, None, "", "Speak like a pirate"])
@pytest.mark.parametrize("raw", [False, True])
def test_finalized_generation_recipe_keeps_prompt_through_probe_hashing(
    prompt: str | None | InheritSystemPrompt, raw: bool,
):
    tree = LoomTree()
    parent = tree.add_user_turn("Describe a voyage")
    session = cast(Any, SimpleNamespace(
        tree=tree,
        config=SimpleNamespace(system_prompt="Current default instruction"),
        _monitor=SimpleNamespace(probe_names=["calm", "missing"]),
        _probe_hash=lambda name: "calm-sha256" if name == "calm" else None,
    ))
    node_id = DrowseSession._start_loom_assistant(
        session, None, stateless=False, raw=raw, parent_node_id=parent,
        sampling=SamplingConfig(seed=7), steering_obj=None,
        use_thinking_req=False, system_prompt=prompt,
    )
    assert node_id is not None
    tree.finalize_assistant(node_id, text="A voyage", finish_reason="stop", raw_token_ids=[1, 2])
    saved_recipe = tree.get(node_id).to_dict()["recipe"]
    expected = None if raw else "Current default instruction" if isinstance(prompt, InheritSystemPrompt) else prompt
    assert saved_recipe["system_prompt"] == expected
    assert saved_recipe["probe_hashes"] == {"calm": "calm-sha256"}
    restored = LoomTree.from_dict(tree.to_dict()).get(node_id)
    assert restored.recipe is not None
    assert restored.recipe.resolved_system_prompt("Changed after generation") == expected


# ---------------------------------------------------------------------------
# Recipe.overlay
# ---------------------------------------------------------------------------


def test_overlay_none_fields_fall_through():
    base = Recipe(steering="0.3 honest.deceptive", seed=42, thinking=False)
    override = Recipe(seed=99)
    out = base.overlay(override)
    assert out.steering == "0.3 honest.deceptive"
    assert out.seed == 99
    assert out.thinking is False


def test_overlay_none_override_returns_self():
    base = Recipe(steering="0.3 honest", seed=42)
    assert base.overlay(None) is base


def test_overlay_preserves_probes_and_hashes():
    base = Recipe(
        steering="0.3 honest", probes=["angry.calm"],
        probe_hashes={"angry.calm": "abc"},
    )
    override = Recipe(steering="0.5 warm")
    out = base.overlay(override)
    # probes / probe_hashes are not overrideable.
    assert out.probes == ["angry.calm"]
    assert out.probe_hashes == {"angry.calm": "abc"}


def test_overlay_steering_replaces_when_set():
    base = Recipe(steering="0.3 honest.deceptive")
    override = Recipe(steering="")  # explicit empty = unsteered
    out = base.overlay(override)
    assert out.steering == ""


# ---------------------------------------------------------------------------
# Recipe.invert_steering
# ---------------------------------------------------------------------------


def test_invert_simple_term():
    r = Recipe(steering="0.3 honest.deceptive")
    inv = r.invert_steering()
    assert inv.steering is not None  # invert_steering always returns a str
    assert "-0.3" in inv.steering


def test_invert_compound_expression():
    r = Recipe(steering="0.3 honest.deceptive + 0.5 warm.clinical@after")
    inv = r.invert_steering()
    # Both signs flipped — first term renders ``-0.3`` directly; the
    # second renders as a ``- 0.5`` separator-and-magnitude pair through
    # ``format_expr``.  Either way both coefficients are negated.
    assert inv.steering is not None  # invert_steering always returns a str
    assert "-0.3" in inv.steering
    assert "- 0.5" in inv.steering or "-0.5" in inv.steering
    # Trigger preserved.
    assert "after" in inv.steering


def test_invert_empty_steering():
    r = Recipe(steering=None)
    inv = r.invert_steering()
    assert inv.steering == ""


def test_invert_manifold_term():
    # Regression: a recipe whose steering carries a manifold ``%`` term must
    # invert without error — the directional ``along`` flips sign, ``onto``
    # (a collapse fraction) carries through unchanged.
    r = Recipe(steering="0.6,0.3 circumplex%happy")
    inv = r.invert_steering()
    assert inv.steering is not None
    reparsed = parse_expr(inv.steering)
    (term,) = reparsed.alphas.values()
    assert isinstance(term, ManifoldTerm)
    assert term.along == pytest.approx(-0.6)
    assert term.onto == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Recipe.compose_modifier
# ---------------------------------------------------------------------------


def test_compose_unsteered():
    r = Recipe(steering="0.3 honest", seed=42)
    mod = r.compose_modifier("unsteered")
    assert mod.steering == ""


def test_compose_inverted_flips_signs():
    r = Recipe(steering="0.4 warm.clinical")
    mod = r.compose_modifier("inverted")
    assert mod.steering is not None  # compose_modifier("inverted") always returns a str
    assert "-0.4" in mod.steering


def test_compose_reseed_gives_fresh_seed():
    r = Recipe(seed=42)
    mod = r.compose_modifier("reseed")
    assert mod.seed is not None
    assert mod.seed != 42  # nonzero entropy chance of collision, but vanishingly small


def test_compose_cool():
    r = Recipe()
    mod = r.compose_modifier("cool")
    assert mod.sampling is not None  # compose_modifier("cool") always sets sampling
    assert mod.sampling.temperature == pytest.approx(0.3)


def test_compose_hot():
    r = Recipe()
    mod = r.compose_modifier("hot")
    assert mod.sampling is not None  # compose_modifier("hot") always sets sampling
    assert mod.sampling.temperature == pytest.approx(1.2)


def test_compose_unknown_raises():
    with pytest.raises(ValueError, match="unknown recipe-override mode"):
        Recipe().compose_modifier("foo")


def test_compose_custom_string_recipe():
    mod = Recipe().compose_modifier("seed=42, temperature=3, thinking=false")
    assert mod.seed == 42
    assert mod.thinking is False
    assert mod.sampling is not None
    assert mod.sampling.temperature == pytest.approx(3)


def test_compose_custom_string_preserves_steering_commas():
    mod = Recipe().compose_modifier(
        "steering=0.6,0.3 circumplex%0.2,0.8, temperature=0.4"
    )
    assert mod.steering == "0.6,0.3 circumplex%0.2,0.8"
    assert mod.sampling is not None
    assert mod.sampling.temperature == pytest.approx(0.4)


@pytest.mark.parametrize(
    "modifier",
    [
        "temperature=fast",
        "temperature=-1",
        "thinking=yes",
        "seed=1, seed=2",
        "unknown=1",
    ],
)
def test_compose_custom_string_rejects_invalid_fields(modifier: str):
    with pytest.raises(ValueError):
        Recipe().compose_modifier(modifier)


def test_compose_custom_recipe_passthrough():
    """``compose_modifier`` with a Recipe arg returns it unchanged.

    Programmatic callers may hand a Recipe in directly; ``compose_modifier``
    should not reinterpret it.
    """
    partial = Recipe(steering="0.5 calm", sampling=SamplingConfig(temperature=0.4))
    out = Recipe(steering="0.3 honest", seed=42).compose_modifier(partial)
    assert out is partial  # passes through, no copy


def test_compose_custom_via_overlay():
    """Custom mode = pass a Recipe partial directly, no compose_modifier."""
    base = Recipe(steering="0.3 honest", seed=42)
    partial = Recipe(sampling=SamplingConfig(temperature=0.5), seed=99)
    out = base.overlay(partial)
    assert out.steering == "0.3 honest"
    assert out.sampling is not None  # overlay with explicit sampling always sets it
    assert out.sampling.temperature == pytest.approx(0.5)
    assert out.seed == 99


def test_compose_modifier_routes_through_overlay_in_regen_with_modifier():
    """End-to-end: regen_with_modifier accepts a Recipe partial as ``mode``.

    Verifies the full custom-mode wiring: a caller parses ``custom: <expr>`` into
    a Recipe and passes it to ``session.regen_with_modifier(... mode=<Recipe>)``,
    which routes through ``compose_modifier(Recipe) -> Recipe`` and overlays
    onto the parent recipe.  No model load — uses _resolve_recipe_override
    which is the pure-Python overlay path.
    """
    from drowse.core.session import DrowseSession
    from drowse import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()
            uid = self.tree.add_user_turn("hi")
            recipe = Recipe(steering="0.3 honest", seed=42)
            aid = self.tree.begin_assistant(uid, recipe=recipe)
            self.tree.finalize_assistant(aid, text="hello")
            self.aid = aid

    stub = _StubSession()
    stub_any: Any = stub
    stub_any._resolve_anchor_recipe = DrowseSession._resolve_anchor_recipe.__get__(
        stub, _StubSession
    )
    resolve = DrowseSession._resolve_recipe_override.__get__(stub, _StubSession)
    custom = Recipe(sampling=SamplingConfig(temperature=0.4))
    new_steering, new_sampling, new_thinking = resolve(
        custom,
        parent_node_id=stub.aid,
        steering=None,
        sampling=None,
        thinking=None,
    )
    assert new_steering == "0.3 honest"
    assert new_sampling is not None
    assert new_sampling.temperature == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# session._resolve_recipe_override engine integration (without model load)
# ---------------------------------------------------------------------------


def test_resolve_override_with_string_mode():
    """The session helper applies overlay onto the parent recipe + returns the kwargs."""
    from drowse.core.session import DrowseSession
    from drowse import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()
            uid = self.tree.add_user_turn("hi")
            recipe = Recipe(steering="0.3 honest", seed=42)
            aid = self.tree.begin_assistant(uid, recipe=recipe)
            self.tree.finalize_assistant(aid, text="hello")
            self.aid = aid

    stub = _StubSession()
    stub_any: Any = stub
    stub_any._resolve_anchor_recipe = DrowseSession._resolve_anchor_recipe.__get__(
        stub, _StubSession
    )
    resolve = DrowseSession._resolve_recipe_override.__get__(stub, _StubSession)
    new_steering, new_sampling, new_thinking = resolve(
        "unsteered",
        parent_node_id=stub.aid,
        steering=None,
        sampling=None,
        thinking=None,
    )
    # Unsteered modifier wipes steering.
    assert new_steering == ""


def test_resolve_override_passthrough_when_none():
    from drowse.core.session import DrowseSession
    from drowse import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()

    stub = _StubSession()
    resolve = DrowseSession._resolve_recipe_override.__get__(stub, _StubSession)
    out = resolve(
        None, parent_node_id=None,
        steering="0.5 warm", sampling=None, thinking=None,
    )
    assert out == ("0.5 warm", None, None)


def test_resolve_override_with_custom_recipe():
    from drowse.core.session import DrowseSession
    from drowse import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()

    stub = _StubSession()
    stub_any: Any = stub
    stub_any._resolve_anchor_recipe = DrowseSession._resolve_anchor_recipe.__get__(
        stub, _StubSession
    )
    resolve = DrowseSession._resolve_recipe_override.__get__(stub, _StubSession)
    partial = Recipe(sampling=SamplingConfig(temperature=0.9), seed=7)
    new_steering, new_sampling, new_thinking = resolve(
        partial, parent_node_id=None,
        steering="0.3 honest", sampling=None, thinking=None,
    )
    # Steering inherits the caller's explicit kwarg; sampling carries
    # the override's seed and temperature.
    assert new_steering == "0.3 honest"
    assert new_sampling.temperature == pytest.approx(0.9)
    assert new_sampling.seed == 7
