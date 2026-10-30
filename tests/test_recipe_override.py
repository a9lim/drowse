"""Recipe composition: regen overrides, the seed's single home, modifiers."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse import Recipe, SamplingConfig
from drowse.loom.errors import LoomTreeError
from drowse.generation.render import PromptRenderer
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, InheritSystemPrompt
from drowse.loom.recipe import compose_call, derive_seed_schedule
from drowse.loom.tree import LoomTree
from drowse.session.session import DrowseSession
from drowse.steering.grammar import ManifoldTerm, parse_expr
from tests.support.engine import attach_engine


def test_recipe_prompt_capture_distinguishes_legacy_null_empty_and_text():
    legacy = Recipe.from_dict(Recipe().to_dict())
    assert "system_prompt" not in legacy.to_dict()
    assert legacy.resolved_system_prompt("current") == "current"
    for prompt in (None, "", "Speak like a pirate"):
        captured = Recipe.from_dict(Recipe(system_prompt=prompt).to_dict())
        assert captured.resolved_system_prompt("different current instruction") == prompt
        # A modifier that sets no prompt keeps the anchor's captured one.
        composed = compose_call(anchor=captured, modifier=Recipe(sampling=SamplingConfig(seed=7)))
        assert composed.system_prompt == prompt
    composed = compose_call(
        anchor=Recipe(system_prompt="pirate"), modifier=Recipe(system_prompt=None),
    )
    assert composed.system_prompt is None


def _renderer(tree: LoomTree, default_prompt: str | None) -> PromptRenderer:
    return PromptRenderer(
        tokenizer=None,
        model=None,
        traits=cast(Any, None),
        device=torch.device("cpu"),
        history=lambda: tree,
        default_system_prompt=lambda: default_prompt,
        steering_role=lambda: None,
    )


@pytest.mark.parametrize("prompt", [INHERIT_SYSTEM_PROMPT, None, "", "Speak like a pirate"])
@pytest.mark.parametrize("raw", [False, True])
def test_finalized_generation_recipe_keeps_prompt_through_probe_hashing(
    prompt: str | None | InheritSystemPrompt, raw: bool,
):
    tree = LoomTree()
    parent = tree.add_user_turn("Describe a voyage")
    session = cast(Any, SimpleNamespace(
        tree=tree,
        _renderer=_renderer(tree, "Current default instruction"),
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
    # The seed has one in-memory home; a stamped sampling block persists
    # in full with the top-level seed mirroring it.
    assert saved_recipe["seed"] == 7
    assert isinstance(saved_recipe["sampling"], dict)
    assert saved_recipe["sampling"]["seed"] == 7
    restored = LoomTree.from_dict(tree.to_dict()).get(node_id)
    assert restored.recipe is not None
    assert restored.recipe.resolved_system_prompt("Changed after generation") == expected
    assert restored.recipe.sampling == SamplingConfig(seed=7)


# ---------------------------------------------------------------------------
# The seed lives in sampling.seed only
# ---------------------------------------------------------------------------


def test_recipe_has_no_separate_seed_field():
    with pytest.raises(TypeError):
        Recipe(seed=7)  # type: ignore[call-arg]
    assert not hasattr(Recipe(), "seed")


def test_seed_carrying_sampling_block_serializes_in_full():
    """Regression: a block holding only the seed (every fan sibling
    generated with ``sampling=None``) collapsed to ``"sampling": null``,
    which the browser runtime refuses to fork and the TS port never writes."""
    data = Recipe(steering="0.25 calm", sampling=SamplingConfig(seed=7)).to_dict()
    assert isinstance(data["sampling"], dict)
    assert data["sampling"]["seed"] == 7
    assert data["seed"] == 7
    restored = Recipe.from_dict(data)
    assert restored.sampling == SamplingConfig(seed=7)
    assert restored.to_dict() == data


def test_seed_only_recipe_keeps_the_null_sampling_shape():
    recipe = Recipe.from_parts(steering="0.25 calm", seed=7)
    assert recipe.seed_only
    assert recipe.sampling == SamplingConfig(seed=7)
    data = recipe.to_dict()
    assert data["sampling"] is None
    assert data["seed"] == 7
    restored = Recipe.from_dict(data)
    assert restored.seed_only
    assert restored.to_dict() == data
    # Folding a seed into an existing block is not seed-only.
    folded = Recipe.from_parts(sampling=SamplingConfig(temperature=0.5), seed=7)
    assert not folded.seed_only
    assert folded.sampling == SamplingConfig(temperature=0.5, seed=7)


@pytest.mark.parametrize("sampling", [
    None, SamplingConfig(), SamplingConfig(temperature=0.5, seed=7),
])
def test_seed_only_marker_requires_a_bare_seed_block(sampling: SamplingConfig | None):
    with pytest.raises(ValueError):
        Recipe(sampling=sampling, seed_only=True)


def test_fan_siblings_round_trip_with_their_sampling_blocks():
    """The serial fan stamps ``SamplingConfig(seed=s)`` on each sibling when
    the call passed no sampling; each persists a full block with its seed."""
    tree = LoomTree()
    parent = tree.add_user_turn("Describe a voyage")
    session = cast(Any, SimpleNamespace(
        tree=tree,
        _renderer=_renderer(tree, None),
        _monitor=SimpleNamespace(probe_names=[]),
        _probe_hash=lambda _name: None,
    ))
    schedule = derive_seed_schedule(None, 3)
    node_ids = []
    for seed in schedule:
        node_id = DrowseSession._start_loom_assistant(
            session, None, stateless=False, raw=False, parent_node_id=parent,
            sampling=SamplingConfig(seed=seed), steering_obj=None,
            use_thinking_req=False, system_prompt=INHERIT_SYSTEM_PROMPT,
        )
        assert node_id is not None
        tree.finalize_assistant(node_id, text="A voyage", finish_reason="stop")
        node_ids.append(node_id)
    saved = tree.to_dict()
    rows = {row["id"]: row for row in saved["nodes"]}
    for node_id, seed in zip(node_ids, schedule):
        recipe = rows[node_id]["recipe"]
        assert isinstance(recipe["sampling"], dict)
        assert recipe["sampling"]["seed"] == seed
        assert recipe["seed"] == seed
    assert LoomTree.from_dict(saved).to_dict() == saved


def test_full_sampling_round_trips_with_its_seed():
    recipe = Recipe(sampling=SamplingConfig(temperature=0.5, seed=3, stop=("\n",)))
    data = recipe.to_dict()
    assert data["seed"] == 3
    assert data["sampling"]["seed"] == 3
    assert data["sampling"]["stop"] == ["\n"]
    assert Recipe.from_dict(data) == recipe


def test_all_default_sampling_stays_an_explicit_block():
    data = Recipe(sampling=SamplingConfig()).to_dict()
    assert data["seed"] is None
    assert isinstance(data["sampling"], dict)
    assert Recipe.from_dict(data).sampling == SamplingConfig()


def test_top_level_seed_folds_into_sampling():
    data = Recipe(sampling=SamplingConfig(temperature=0.5)).to_dict()
    data["seed"] = 9
    assert Recipe.from_dict(data).sampling == SamplingConfig(temperature=0.5, seed=9)
    data["sampling"]["seed"] = 4  # disagreement: the top-level seed wins
    sampling = Recipe.from_dict(data).sampling
    assert sampling is not None and sampling.seed == 9


@pytest.mark.parametrize("field, value", [
    ("seed", "7"), ("seed", True), ("steering", 3), ("thinking", "yes"),
    ("probes", "calm"), ("probe_hashes", {"calm": 1}), ("sampling", []),
])
def test_recipe_reader_rejects_malformed_fields(field: str, value: Any):
    data = Recipe().to_dict()
    data[field] = value
    with pytest.raises(LoomTreeError):
        Recipe.from_dict(data)


# ---------------------------------------------------------------------------
# compose_call — the one composition rule
# ---------------------------------------------------------------------------

_ANCHOR = Recipe(
    steering="0.3 honest.deceptive",
    sampling=SamplingConfig(
        temperature=0.7, max_tokens=64, stop=("\n",), top_p=0.5, seed=11,
    ),
    thinking=False,
)
_CALLER = SamplingConfig(return_top_k=5, assistant_role="pirate", max_tokens=200)


def _override(mode: str | Recipe, sampling: SamplingConfig | None = _CALLER, **kwargs: Any):
    return compose_call(
        sampling=sampling, anchor=_ANCHOR,
        modifier=_ANCHOR.compose_modifier(mode), **kwargs,
    )


@pytest.mark.parametrize("mode", ["cool", "hot", "reseed", "temperature=1.5", "unsteered"])
def test_override_keeps_anchor_sampling_and_caller_fields(mode: str):
    """Regression: an override used to replace the caller's whole sampling."""
    out = _override(mode)
    assert out.sampling is not None
    # The anchor's knobs survive a modifier that doesn't touch them …
    assert out.sampling.top_p == 0.5
    assert out.sampling.stop == ("\n",)
    assert out.sampling.max_tokens == 64
    # … and the caller's per-call fields survive the anchor.
    assert out.sampling.return_top_k == 5
    assert out.sampling.assistant_role == "pirate"


def test_override_modifier_fields_win():
    assert _override("cool").sampling.temperature == pytest.approx(0.3)
    assert _override("hot").sampling.temperature == pytest.approx(1.2)
    assert _override("reseed").sampling.temperature == pytest.approx(0.7)
    custom = _override("temperature=1.5", sampling=SamplingConfig(temperature=0.7))
    assert custom.sampling.temperature == pytest.approx(1.5)


def test_override_anchor_steering_and_thinking_beat_the_caller():
    out = _override("cool", steering="", thinking=True)
    assert out.steering == "0.3 honest.deceptive"
    assert out.thinking is False
    assert _override("unsteered", steering="0.5 warm").steering == ""


@pytest.mark.parametrize("mode", ["unsteered", "cool", "inverted", "hot"])
def test_fanned_override_keeps_per_sibling_seeds(mode: str):
    """Regression: every n>1 override sibling collapsed onto the anchor seed."""
    schedule = derive_seed_schedule(None, 3)
    seeds = [
        _override(mode, sampling=SamplingConfig(seed=seed)).sampling.seed
        for seed in schedule
    ]
    assert seeds == schedule
    assert len(set(seeds)) == 3


def test_unseeded_one_shot_regen_reproduces_the_anchor_seed():
    assert _override("cool").sampling.seed == 11
    assert _override("unsteered", sampling=None).sampling.seed == 11


def test_modifier_seed_beats_the_caller_seed():
    pinned = _override("seed=42", sampling=SamplingConfig(seed=7))
    assert pinned.sampling.seed == 42
    reseeded = _override("reseed", sampling=SamplingConfig(seed=7))
    assert reseeded.sampling.seed not in (7, 11)


def test_observation_fields_come_from_the_call_only():
    anchor = Recipe(sampling=SamplingConfig(
        return_top_k=8, persist_per_layer_scores=True, logprobs=3,
        return_probe_readings=False,
    ))
    out = compose_call(
        sampling=SamplingConfig(temperature=0.2),
        anchor=anchor, modifier=anchor.compose_modifier("cool"),
    )
    assert out.sampling == SamplingConfig(temperature=0.3)


def test_compose_without_tiers_is_the_identity():
    sampling = SamplingConfig(temperature=0.4, seed=3)
    assert compose_call(
        steering="0.5 warm", sampling=sampling, thinking=True,
    ) == ("0.5 warm", sampling, True, INHERIT_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Engine hook: compose_call is one call into loom's compose_call
# ---------------------------------------------------------------------------


def _anchored_tree() -> tuple[LoomTree, str, str]:
    tree = LoomTree()
    uid = tree.add_user_turn("hi")
    aid = tree.begin_assistant(uid, recipe=Recipe(
        steering="0.3 honest", sampling=SamplingConfig(seed=42),
        system_prompt="Anchor prompt",
    ))
    tree.finalize_assistant(aid, text="hello")
    return tree, uid, aid


def _resolve(tree: LoomTree, mode: Any, **kwargs: Any):
    engine = attach_engine(SimpleNamespace(tree=tree))
    call: dict[str, Any] = {
        "steering": None, "sampling": None, "thinking": None,
        "system_prompt": INHERIT_SYSTEM_PROMPT, "raw": False,
        "gen_seat": "assistant",
    }
    call.update(kwargs)
    return engine.compose_call(recipe_override=mode, **call)


def test_resolve_override_composes_onto_the_anchor_once():
    tree, _uid, aid = _anchored_tree()
    out = _resolve(tree, Recipe(sampling=SamplingConfig(temperature=0.4)), parent_node_id=aid)
    assert out.steering == "0.3 honest"
    assert out.sampling == SamplingConfig(temperature=0.4, seed=42)
    # The captured system prompt comes back with the rest — no second pass.
    assert out.system_prompt == "Anchor prompt"


def test_resolve_override_from_a_user_anchor_walks_to_the_generated_ancestor():
    tree, uid, aid = _anchored_tree()
    child = tree.add_user_turn("again", parent_id=aid)
    assert _resolve(tree, "unsteered", parent_node_id=child).steering == ""
    assert _resolve(tree, "unsteered", parent_node_id=uid).sampling is None


def test_resolve_override_passthrough_when_none():
    tree, _uid, _aid = _anchored_tree()
    out = _resolve(tree, None, parent_node_id=None, steering="0.5 warm")
    assert out == ("0.5 warm", None, None, INHERIT_SYSTEM_PROMPT)


def test_resolve_override_without_anchor_layers_the_partial_on_the_call():
    tree = LoomTree()
    partial = Recipe(sampling=SamplingConfig(temperature=0.9, seed=7))
    out = _resolve(tree, partial, parent_node_id=None, steering="0.3 honest")
    assert out.steering == "0.3 honest"
    assert out.sampling == SamplingConfig(temperature=0.9, seed=7)


@pytest.mark.parametrize("call", [
    {"recipe_override": "unsteered"},
    {"recipe_override": "inverted"},   # the anchor below is unsteered
    {"steering": ""},
    {"steering": "   "},
])
def test_composed_blank_steering_reaches_generation_as_no_steering(
    call: dict, tiny_session: Any,
):
    """Regression: the ``unsteered`` / ``inverted`` modes (and an explicit
    ``steering=""``) composed ``""``, which the grammar rejects, so every
    such regen failed with "empty steering expression" on a real session."""
    tree = tiny_session.tree
    user = tree.add_user_turn("hi")
    anchor = tree.begin_assistant(user, recipe=Recipe(sampling=SamplingConfig(seed=3)))
    tree.finalize_assistant(anchor, text="hello")

    spec = tiny_session._engine.resolve_call(None, parent_node_id=user, **call)
    assert spec.steering_obj is None
    result = tiny_session.generate(
        None, parent_node_id=user, sampling=SamplingConfig(max_tokens=1),
        append_same_role=False, **call,
    ).first
    assert result.applied_steering is None


def test_anchor_recipe_resolution():
    tree, uid, aid = _anchored_tree()
    assert tree.anchor_recipe(aid) is tree.get(aid).recipe
    assert tree.anchor_recipe(tree.add_user_turn("more", parent_id=aid)) is tree.get(aid).recipe
    assert tree.anchor_recipe(uid) is None
    assert tree.anchor_recipe(None) is None
    assert tree.anchor_recipe("missing") is None


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


def test_inverted_regen_of_a_curved_term_is_refused_at_composition():
    """The ``inverted`` modifier negates a ``%`` term's ``along``; on a
    curved fit that has no opposite, so composing the inverted recipe raises
    (422) instead of silently clamping to a no-op.  The loom can't know the
    curvature (it is per-model), so the refusal is the composer's: pin the
    hand-off between the two."""
    from drowse.errors import SteeringCompositionError
    from drowse.geometry.domains import BoxAxis, BoxDomain
    from drowse.geometry.manifold import Manifold
    from drowse.steering.hooks import SteeringPlan

    anchor = Recipe(steering="0.5 months%january + 0.3 calm")
    inverted = anchor.compose_modifier("inverted")
    assert inverted.steering is not None
    terms = parse_expr(inverted.steering).alphas
    (term,) = [t for t in terms.values() if isinstance(t, ManifoldTerm)]
    assert term.along == pytest.approx(-0.5)

    loop = Manifold(
        name="months",
        domain=BoxDomain([BoxAxis("t", periodic=True, period=12.0)]),
        node_labels=["january", "february", "march"],
        node_coords=torch.tensor([[0.0], [1.0], [2.0]]),
        layers={},
        feature_space="raw",
    )
    plan = SteeringPlan()
    with pytest.raises(SteeringCompositionError, match="cannot be negated or inverted"):
        plan.add_manifold(
            "months", loop, term.position, along=term.along, onto=term.onto,
        )
    assert not plan


# ---------------------------------------------------------------------------
# Recipe.compose_modifier
# ---------------------------------------------------------------------------


def test_compose_unsteered():
    r = Recipe(steering="0.3 honest", sampling=SamplingConfig(seed=42))
    mod = r.compose_modifier("unsteered")
    assert mod.steering == ""


def test_compose_inverted_flips_signs():
    r = Recipe(steering="0.4 warm.clinical")
    mod = r.compose_modifier("inverted")
    assert mod.steering is not None  # compose_modifier("inverted") always returns a str
    assert "-0.4" in mod.steering


def test_compose_reseed_gives_fresh_seed():
    r = Recipe(sampling=SamplingConfig(seed=42))
    mod = r.compose_modifier("reseed")
    assert mod.sampling is not None and mod.sampling.seed is not None
    assert mod.sampling.seed != 42  # vanishingly small collision chance
    assert mod.sampling == SamplingConfig(seed=mod.sampling.seed)


def test_compose_cool():
    mod = Recipe().compose_modifier("cool")
    assert mod.sampling == SamplingConfig(temperature=0.3)


def test_compose_hot():
    mod = Recipe().compose_modifier("hot")
    assert mod.sampling == SamplingConfig(temperature=1.2)


def test_compose_unknown_raises():
    with pytest.raises(ValueError, match="unknown recipe-override mode"):
        Recipe().compose_modifier("foo")


def test_compose_custom_string_recipe():
    mod = Recipe().compose_modifier("seed=42, temperature=3, thinking=false")
    assert mod.thinking is False
    assert mod.sampling == SamplingConfig(temperature=3, seed=42)


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
    """``compose_modifier`` with a Recipe arg returns it unchanged."""
    partial = Recipe(steering="0.5 calm", sampling=SamplingConfig(temperature=0.4))
    out = Recipe(steering="0.3 honest").compose_modifier(partial)
    assert out is partial  # passes through, no copy


def test_with_probe_hashes_drops_unhashable_probes():
    recipe = Recipe(probes=["calm", "missing"])
    stamped = recipe.with_probe_hashes(lambda name: "h" if name == "calm" else None)
    assert stamped.probe_hashes == {"calm": "h"}
    assert recipe.probe_hashes == {}
