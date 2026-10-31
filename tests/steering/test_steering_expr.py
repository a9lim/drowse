"""Grammar corpus for drowse.steering.grammar.

Bare names that don't match an installed pack resolve to themselves with
sign +1 (``resolve_pole`` fallthrough), so most of these tests run in an
isolated DROWSE_HOME with no packs installed.
"""
from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest

from drowse.store import selectors as sel
from drowse.store.manifold_authoring import create_discover_manifold_folder
from drowse.steering.terms import Steering
from drowse.steering.grammar import (
    ProjectedTerm,
    SteeringExprError,
    format_expr,
    parse_expr,
    referenced_selectors,
)
from drowse.steering.triggers import ProbeGateOp, Trigger


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[None, None, None]:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    sel.invalidate()
    yield
    sel.invalidate()


def _mk(tmp_path: Path, ns: str, name: str, tags: list[str] | None = None) -> Path:
    """Author an installed concept as a 2-node ``pca`` manifold (4.0).

    A bipolar composite name (``deer.wolf``) becomes a manifold whose two
    node labels are the poles (``deer``/``wolf``), so a bare pole resolves
    through the manifold tier (``resolve_manifold_label`` /
    ``resolve_manifold_name``); a plain name gets generic ``pos``/``neg``
    nodes.  ``tags`` are patched into ``manifold.json``.
    """
    if "." in name:
        pos, neg = name.split(".", 1)
        node_corpora = {pos: ["a statement."], neg: ["b statement."]}
    else:
        node_corpora = {"pos": ["a statement."], "neg": ["b statement."]}
    folder = create_discover_manifold_folder(
        ns, name, "x", fit_mode="pca",
        node_corpora=node_corpora, hyperparams={"max_dim": 1},
    )
    if tags:
        import json
        mpath = folder / "manifold.json"
        data = json.loads(mpath.read_text())
        data["tags"] = list(tags)
        mpath.write_text(json.dumps(data))
    return folder


# -------------------------------------------------------------- basic ---

def test_single_term():
    s = parse_expr("0.5 honest")
    assert s.alphas == {"honest": 0.5}


def test_implicit_coefficient():
    from drowse.steering.grammar import DEFAULT_COEFF
    s = parse_expr("honest")
    assert s.alphas == {"honest": DEFAULT_COEFF}
    assert DEFAULT_COEFF == 0.5  # contract — documented default.


def test_star_form():
    s = parse_expr("0.5*honest")
    assert s.alphas == {"honest": 0.5}


def test_integer_coefficient():
    s = parse_expr("2 honest")
    assert s.alphas == {"honest": 2.0}


def test_leading_sign_negates():
    s = parse_expr("-0.5 honest")
    assert s.alphas == {"honest": -0.5}


def test_negated_bare():
    from drowse.steering.grammar import DEFAULT_COEFF
    s = parse_expr("-honest")
    assert s.alphas == {"honest": -DEFAULT_COEFF}


def test_leading_plus():
    s = parse_expr("+0.5 honest")
    assert s.alphas == {"honest": 0.5}


def test_addition():
    s = parse_expr("0.5 honest + 0.3 warm")
    assert s.alphas == {"honest": 0.5, "warm": 0.3}


def test_subtraction():
    s = parse_expr("0.5 honest - 0.2 manipulative")
    assert s.alphas == {"honest": 0.5, "manipulative": -0.2}


def test_three_terms():
    s = parse_expr("0.5 honest + 0.3 warm - 0.2 manipulative")
    assert s.alphas == {"honest": 0.5, "warm": 0.3, "manipulative": -0.2}


def test_scientific_notation():
    s = parse_expr("1e-2 honest")
    assert s.alphas == pytest.approx({"honest": 0.01})


def test_decimal_shorthand():
    s = parse_expr(".25 honest")
    assert s.alphas == {"honest": 0.25}


# ------------------------------------------------------------ bipolar ---

def test_bipolar_dotted():
    s = parse_expr("0.5 angry.calm")
    assert s.alphas == {"angry.calm": 0.5}


def test_underscore_segments_survive():
    s = parse_expr("0.5 high_context.low_context")
    assert s.alphas == {"high_context.low_context": 0.5}


# ---------------------------------------------------------- namespace ---

def test_namespace_prefix_preserved_in_key():
    # User-typed namespace prefixes must survive into the alphas key so
    # downstream lookups can disambiguate ``alice/foo`` from ``bob/foo``
    # when both packs are installed.  Bare references (no slash) keep
    # the canonical name verbatim.
    s = parse_expr("0.5 bob/foo")
    assert s.alphas == {"bob/foo": 0.5}


def test_namespace_with_bipolar():
    s = parse_expr("0.3 bob/deer.wolf")
    assert s.alphas == {"bob/deer.wolf": 0.3}


def test_namespace_disambiguates_collision(tmp_path: Path) -> None:
    # Two concepts sharing a name across namespaces — the parser must keep
    # the user's explicit ``alice/`` / ``bob/`` prefix so each lands at a
    # distinct registry slot.  4.0: a 2-node ``pca`` manifold name resolves
    # through the composite-name tier to its node-0 (+) pole, so each term
    # becomes a ``ManifoldTerm`` keyed ``<ns>/shared%pos`` — still distinct
    # per namespace, which is the behavior under test.
    from drowse.steering.grammar import ManifoldTerm
    _mk(tmp_path, "alice", "shared", tags=[])
    _mk(tmp_path, "bob", "shared", tags=[])
    sel.invalidate()
    s = parse_expr("0.3 alice/shared + 0.4 bob/shared")
    assert set(s.alphas) == {"alice/shared%pos", "bob/shared%pos"}
    a = s.alphas["alice/shared%pos"]
    b = s.alphas["bob/shared%pos"]
    assert isinstance(a, ManifoldTerm) and isinstance(b, ManifoldTerm)
    assert a.manifold == "alice/shared" and a.along == 0.3
    assert b.manifold == "bob/shared" and b.along == 0.4


def test_namespace_round_trips_through_format():
    # Namespace-qualified keys must render back through ``format_expr``
    # to a string the parser accepts unchanged.
    s = parse_expr("0.3 alice/foo + 0.5 bob/honest:sae")
    assert format_expr(s) == "0.3 alice/foo + 0.5 bob/honest:sae"


# ----------------------------------------------------------- variant ---

def test_sae_variant_suffix_preserved():
    s = parse_expr("0.5 honest:sae")
    assert s.alphas == {"honest:sae": 0.5}


def test_sae_release_variant():
    s = parse_expr("0.5 honest:sae-gemma-scope")
    assert s.alphas == {"honest:sae-gemma-scope": 0.5}


def test_raw_variant_elided():
    # ``:raw`` is the default, so the key drops the suffix.
    s = parse_expr("0.5 honest:raw")
    assert s.alphas == {"honest": 0.5}


def test_sae_release_with_digits():
    s = parse_expr("0.5 honest:sae-gemma-scope-2b-pt-res-canonical")
    assert s.alphas == {"honest:sae-gemma-scope-2b-pt-res-canonical": 0.5}


# ---------------------------------------------------------- triggers ---

def test_trigger_after():
    s = parse_expr("0.3 warm@after")
    assert s.alphas == {"warm": (0.3, Trigger.AFTER_THINKING)}


def test_trigger_before():
    s = parse_expr("0.3 warm@before")
    assert s.alphas == {"warm": (0.3, Trigger.PROMPT_ONLY)}


def test_trigger_both():
    s = parse_expr("0.3 warm@both")
    assert s.alphas == {"warm": (0.3, Trigger.BOTH)}


def test_trigger_thinking():
    s = parse_expr("0.3 warm@thinking")
    assert s.alphas == {"warm": (0.3, Trigger.THINKING_ONLY)}


def test_trigger_response():
    s = parse_expr("0.3 warm@response")
    assert s.alphas == {"warm": (0.3, Trigger.GENERATED_ONLY)}


def test_trigger_prompt_alias_for_before():
    s = parse_expr("0.3 warm@prompt")
    assert s.alphas == {"warm": (0.3, Trigger.PROMPT_ONLY)}


def test_trigger_generated_alias_for_response():
    s = parse_expr("0.3 warm@generated")
    assert s.alphas == {"warm": (0.3, Trigger.GENERATED_ONLY)}


def test_mixed_triggers_per_term():
    s = parse_expr("0.5 honest + 0.3 warm@after")
    assert s.alphas == {
        "honest": 0.5,
        "warm": (0.3, Trigger.AFTER_THINKING),
    }


def test_unknown_trigger_raises():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.3 warm@splatty")
    assert "unknown trigger" in str(ei.value)


def test_unknown_trigger_mentions_hf_revision_hint():
    # HF revisions would land as a trigger-shaped token — the error should
    # steer users away from that mistake.
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.3 bob/honest@abc1234")
    assert "HF revisions" in str(ei.value)


# --------------------------------------------------------- projection ---

def test_projection_orthogonal():
    s = parse_expr("0.5 honest|sycophantic")
    key = "honest|sycophantic"
    assert key in s.alphas
    v = s.alphas[key]
    assert isinstance(v, ProjectedTerm)
    assert v.coeff == 0.5
    assert v.operator == "|"
    assert v.base == "honest"
    assert v.onto == "sycophantic"
    assert v.trigger == Trigger.BOTH


def test_projection_onto():
    s = parse_expr("0.5 honest~sycophantic")
    key = "honest~sycophantic"
    assert key in s.alphas
    v = s.alphas[key]
    assert isinstance(v, ProjectedTerm)
    assert v.operator == "~"


def test_projection_chained_rejected():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.5 a~b~c")
    assert "chained projection" in str(ei.value).lower()


def test_projection_with_trigger():
    s = parse_expr("0.5 honest|sycophantic@after")
    v = next(iter(s.alphas.values()))
    assert isinstance(v, ProjectedTerm)
    assert v.trigger == Trigger.AFTER_THINKING


def test_projection_with_variant():
    s = parse_expr("0.5 honest:sae|sycophantic")
    key = "honest:sae|sycophantic"
    assert key in s.alphas


def test_projection_and_plain_coexist():
    s = parse_expr("0.5 warm + 0.3 warm|cold")
    assert "warm" in s.alphas
    assert "warm|cold" in s.alphas


# --------------------------------------------------------- summation ---

def test_same_name_sums_no_trigger():
    s = parse_expr("0.3 warm + 0.2 warm")
    assert s.alphas == {"warm": pytest.approx(0.5)}


def test_same_name_same_trigger_sums():
    s = parse_expr("0.3 warm@after + 0.2 warm@after")
    entry = s.alphas["warm"]
    assert isinstance(entry, tuple)
    assert entry[0] == pytest.approx(0.5)
    assert entry[1] == Trigger.AFTER_THINKING


def test_conflicting_triggers_reject():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.3 warm@before + 0.2 warm@after")
    assert "conflicting" in str(ei.value).lower()


def test_bare_and_triggered_same_name_reject():
    with pytest.raises(SteeringExprError):
        parse_expr("0.3 warm + 0.2 warm@after")


def test_plain_vs_projected_same_key_reject():
    # ``warm|cold`` is a valid projection key; a plain entry under the
    # same synthetic string can't coexist.  Parser constructs the key
    # internally — users don't type ``warm|cold`` as a plain name.
    # Direct-construction covers this; parser paths always route
    # projections to ProjectedTerm values.
    pass  # placeholder — no parser-level way to trigger


# ------------------------------------------------------------ errors ---

def test_empty_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("")


def test_whitespace_only_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("   \t  ")


def test_trailing_operator_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest +")


def test_bad_character_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest !")


def test_quoted_identifier_hints_underscore():
    # User's mental model: ``"human" . "artificial intelligence"``.
    # Grammar has no quoting — the error must steer them to the slug form
    # rather than just saying "unexpected character '\"'".
    with pytest.raises(SteeringExprError) as ei:
        parse_expr('0.5 "artificial intelligence"')
    msg = str(ei.value)
    assert "quoted" in msg.lower()
    assert "underscore" in msg.lower()
    assert "artificial_intelligence" in msg


def test_trailing_ident_hints_underscore():
    # ``artificial intelligence`` (no quotes): first atom parses fine,
    # the second IDENT is stranded — error should suggest underscores.
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.5 artificial intelligence")
    msg = str(ei.value)
    assert "underscore" in msg.lower()
    assert "artificial_intelligence" in msg


def test_missing_selector_after_coeff():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 ")


def test_colon_without_variant():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest:")


def test_at_without_trigger():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest@")


def test_slash_without_name():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 bob/")


def test_dot_without_second_pole():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 angry.")


# ------------------------------------------------------ pole aliasing ---

def test_pole_alias_resolves_through_manifold(tmp_path: Path) -> None:
    # 4.0: a bipolar concept is a 2-node ``pca`` manifold (``deer.wolf`` with
    # nodes ``deer``/``wolf``).  A bare pole resolves through the manifold
    # *label* tier — ``0.5 wolf`` synthesizes a label-form ``ManifoldTerm`` at
    # the ``wolf`` node (``default/deer.wolf%wolf``) rather than the old
    # signed-vector ``deer.wolf @ -0.5``.
    from drowse.steering.grammar import ManifoldTerm
    _mk(tmp_path, "default", "deer.wolf")
    sel.invalidate()
    s = parse_expr("0.5 wolf")
    assert set(s.alphas) == {"default/deer.wolf%wolf"}
    term = s.alphas["default/deer.wolf%wolf"]
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "default/deer.wolf"
    assert term.position == "wolf"
    assert term.along == 0.5


def test_pole_positive_pole_resolves_through_manifold(tmp_path: Path) -> None:
    from drowse.steering.grammar import ManifoldTerm
    _mk(tmp_path, "default", "deer.wolf")
    sel.invalidate()
    s = parse_expr("0.5 deer")
    assert set(s.alphas) == {"default/deer.wolf%deer"}
    term = s.alphas["default/deer.wolf%deer"]
    assert isinstance(term, ManifoldTerm)
    assert term.position == "deer"
    assert term.along == 0.5


def test_pole_composite_name_resolves_to_node0(tmp_path: Path) -> None:
    # The composite-name tier: a 2-node ``pca`` manifold *name* (``deer.wolf``,
    # whose ``.`` skips the bare-label tier) steers toward node 0 — the
    # ``orient_to=0`` (+) pole, ``deer``.
    from drowse.steering.grammar import ManifoldTerm
    _mk(tmp_path, "default", "deer.wolf")
    sel.invalidate()
    s = parse_expr("0.5 deer.wolf")
    assert set(s.alphas) == {"default/deer.wolf%deer"}
    term = s.alphas["default/deer.wolf%deer"]
    assert isinstance(term, ManifoldTerm)
    assert term.position == "deer"
    assert term.along == 0.5


def test_registered_profile_can_shadow_installed_manifold(tmp_path: Path) -> None:
    _mk(tmp_path, "default", "deer.wolf")
    _mk(tmp_path, "local", "deer.wolf")
    sel.invalidate()
    s = parse_expr("0.5 deer.wolf", profile_names={"deer.wolf"})
    assert s.alphas == {"deer.wolf": 0.5}


# -------------------------------------------------------- format/round-trip ---

def test_format_single():
    s = parse_expr("0.5 honest")
    assert format_expr(s) == "0.5 honest"


def test_format_add_subtract():
    s = parse_expr("0.5 honest - 0.2 manipulative")
    assert format_expr(s) == "0.5 honest - 0.2 manipulative"


def test_format_three_terms():
    s = parse_expr("0.5 honest + 0.3 warm - 0.2 manipulative")
    assert format_expr(s) == "0.5 honest + 0.3 warm - 0.2 manipulative"


def test_format_trigger_emitted():
    s = parse_expr("0.3 warm@after")
    assert format_expr(s) == "0.3 warm@after"


def test_format_both_trigger_elided():
    s = parse_expr("0.3 warm@both")
    assert format_expr(s) == "0.3 warm"


def test_format_projection_orthogonal():
    s = parse_expr("0.5 honest|sycophantic")
    assert format_expr(s) == "0.5 honest|sycophantic"


def test_format_projection_onto():
    s = parse_expr("0.5 honest~sycophantic")
    assert format_expr(s) == "0.5 honest~sycophantic"


def test_format_projection_with_trigger():
    s = parse_expr("0.5 honest|sycophantic@after")
    assert format_expr(s) == "0.5 honest|sycophantic@after"


def test_format_leading_negative():
    s = parse_expr("-0.5 honest + 0.3 warm")
    assert format_expr(s) == "-0.5 honest + 0.3 warm"


def test_format_variant_suffix():
    s = parse_expr("0.5 honest:sae-gemma-scope")
    assert format_expr(s) == "0.5 honest:sae-gemma-scope"


def test_round_trip_corpus():
    # Each round-trips through format -> parse -> format and lands stable.
    corpus = [
        "0.5 honest",
        "0.5 honest + 0.3 warm",
        "0.5 honest - 0.2 manipulative",
        "0.3 warm@after",
        "0.5 honest|sycophantic",
        "0.3 bob/deer.wolf",
        "0.5 honest:sae",
    ]
    for text in corpus:
        s1 = parse_expr(text)
        rendered = format_expr(s1)
        s2 = parse_expr(rendered)
        assert format_expr(s2) == rendered, (text, rendered)


class TestRoundTripGolden:
    """Golden corpus: parse -> format -> parse produces identical IRs.

    Formats are canonicalized on the first render (coefficient leading,
    ``@trigger`` only when non-BOTH, ``:raw`` elided) — subsequent
    renders are bit-identical.
    """

    @pytest.mark.parametrize("text,canonical", [
        ("honest", "0.5 honest"),
        ("0.5 honest", "0.5 honest"),
        ("0.5*honest", "0.5 honest"),
        ("-0.5 honest", "-0.5 honest"),
        ("-honest", "-0.5 honest"),
        ("+0.5 honest", "0.5 honest"),
        ("0.5 honest + 0.3 warm", "0.5 honest + 0.3 warm"),
        ("0.5 honest - 0.2 manipulative", "0.5 honest - 0.2 manipulative"),
        ("-0.5 honest + 0.3 warm", "-0.5 honest + 0.3 warm"),
        ("0.3 warm@after", "0.3 warm@after"),
        ("0.3 warm@both", "0.3 warm"),
        ("0.3 warm@before", "0.3 warm@before"),
        ("0.3 warm@prompt", "0.3 warm@before"),
        ("0.3 warm@generated", "0.3 warm@response"),
        ("0.3 warm@thinking", "0.3 warm@thinking"),
        ("0.5 honest|sycophantic", "0.5 honest|sycophantic"),
        ("0.5 honest~sycophantic", "0.5 honest~sycophantic"),
        ("0.5 honest|sycophantic@after", "0.5 honest|sycophantic@after"),
        ("0.5 honest:sae", "0.5 honest:sae"),
        ("0.5 honest:raw", "0.5 honest"),
        (
            "0.5 honest:sae-gemma-scope-2b-pt-res-canonical",
            "0.5 honest:sae-gemma-scope-2b-pt-res-canonical",
        ),
        ("1e-2 honest", "0.01 honest"),
        (".25 honest", "0.25 honest"),
        ("2 honest", "2 honest"),
    ])
    def test_canonical_form(self, text: str, canonical: str) -> None:
        s = parse_expr(text)
        assert format_expr(s) == canonical

    @pytest.mark.parametrize("text", [
        "0.5 honest",
        "0.5 honest + 0.3 warm",
        "0.5 honest - 0.2 manipulative",
        "-0.5 honest + 0.3 warm - 0.2 manipulative",
        "0.3 warm@after",
        "0.3 warm@thinking + 0.5 honest@response",
        "0.5 honest|sycophantic",
        "0.5 honest~sycophantic",
        "0.5 honest:sae",
        "0.5 honest:sae-gemma-scope",
        "0.5 honest:sae|sycophantic",
    ])
    def test_format_parse_format_is_stable(self, text: str) -> None:
        """Render -> re-parse -> render produces the same string."""
        s1 = parse_expr(text)
        r1 = format_expr(s1)
        s2 = parse_expr(r1)
        r2 = format_expr(s2)
        assert r1 == r2
        assert s1.alphas == s2.alphas

    def test_str_dunder_is_formatter(self):
        s = parse_expr("0.5 honest + 0.3 warm@after")
        assert str(s) == format_expr(s)

    def test_empty_steering_renders_empty(self):
        s = Steering(alphas={})
        assert format_expr(s) == ""

    def test_direct_construction_round_trips(self):
        """Steering built directly (not via parser) also stringifies back."""
        s = Steering(alphas={"honest": 0.5, "warm": (0.3, Trigger.AFTER_THINKING)})
        rendered = str(s)
        reparsed = parse_expr(rendered)
        assert reparsed.alphas["honest"] == 0.5
        assert reparsed.alphas["warm"] == (0.3, Trigger.AFTER_THINKING)


def test_steering_str_uses_formatter():
    s = parse_expr("0.5 honest + 0.3 warm")
    assert str(s) == "0.5 honest + 0.3 warm"


# --------------------------------------------------------- from_value ---

def test_from_value_string():
    s = Steering.from_value("0.5 honest")
    assert s is not None
    assert s.alphas == {"honest": 0.5}


def test_from_value_none():
    assert Steering.from_value(None) is None


def test_from_value_steering_passthrough():
    s = Steering(alphas={"honest": 0.5})
    assert Steering.from_value(s) is s


def test_from_value_rejects_dict():
    with pytest.raises(TypeError) as ei:
        Steering.from_value({"honest": 0.5})  # pyright: ignore[reportArgumentType]  # intentional bad-type test
    assert "str | Steering | None" in str(ei.value)


def test_from_value_rejects_list():
    with pytest.raises(TypeError):
        Steering.from_value([("honest", 0.5)])  # pyright: ignore[reportArgumentType]  # intentional bad-type test


# -------------------------------------------------------------- referenced_selectors ---
# Install-time hook: the CLI walks the expression AST to decide which
# packs to fetch.  Must survive the parser without running through
# ``resolve_pole`` so namespace prefixes are preserved.


def test_referenced_selectors_single_term():
    assert referenced_selectors("0.5 honest") == [(None, "honest", "raw")]


def test_referenced_selectors_namespace_preserved():
    # Namespace must NOT be folded into the concept name — the CLI needs
    # the raw (ns, concept, variant) triple to resolve packs.
    assert referenced_selectors("0.5 bob/deer.wolf") == [
        ("bob", "deer.wolf", "raw"),
    ]


def test_referenced_selectors_variant_preserved():
    assert referenced_selectors("0.5 honest:sae-gemma-scope") == [
        (None, "honest", "sae-gemma-scope"),
    ]


def test_referenced_selectors_sum_of_terms():
    out = referenced_selectors("0.5 honest + 0.3 alice/warm")
    assert out == [(None, "honest", "raw"), ("alice", "warm", "raw")]


def test_referenced_selectors_projection_contributes_two_atoms():
    # Projection terms yield both base and onto so install-time code can
    # fetch both packs.
    assert referenced_selectors("0.5 honest~sycophantic") == [
        (None, "honest", "raw"),
        (None, "sycophantic", "raw"),
    ]


def test_referenced_selectors_empty_or_whitespace_returns_empty():
    assert referenced_selectors("") == []
    assert referenced_selectors("   \t  ") == []


# -------------------------------------------------------------- ablation ---

def test_bare_ablation_defaults_to_coeff_one():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("!honest")
    assert set(s.alphas.keys()) == {"!honest"}
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == 1.0
    assert term.target == "honest"
    assert term.trigger == Trigger.BOTH


def test_ablation_term_is_frozen():
    from dataclasses import FrozenInstanceError
    from drowse.steering.grammar import AblationTerm
    t = AblationTerm(coeff=1.0, trigger=Trigger.BOTH, target="x")
    with pytest.raises(FrozenInstanceError):
        t.coeff = 0.5  # pyright: ignore[reportAttributeAccessIssue]  # frozen dataclass — assignment expected to raise FrozenInstanceError


def test_ablation_explicit_coefficient():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("0.5 !honest")
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == 0.5


def test_ablation_negative_sign():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("-!honest")
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == -1.0


def test_ablation_star_form():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("0.7 * !honest")
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == 0.7


def test_ablation_signed_explicit():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("-0.3 !honest")
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == -0.3


def test_ablation_with_namespace(tmp_path: Path) -> None:
    from drowse.steering.grammar import AblationTerm
    _mk(tmp_path, "bob", "custom", tags=[])
    s = parse_expr("!bob/custom")
    # Namespace prefix is preserved through to the registry key,
    # matching plain-term behavior (see test_namespace_prefix_preserved_in_key)
    # — the ablation key is ``!<namespace>/<concept>`` so two packs sharing
    # a concept name across namespaces ablate independently.
    term = s.alphas["!bob/custom"]
    assert isinstance(term, AblationTerm)
    assert term.target == "bob/custom"


def test_ablation_with_trigger():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("!refusal@response")
    term = s.alphas["!refusal"]
    assert isinstance(term, AblationTerm)
    assert term.target == "refusal"
    assert term.trigger == Trigger.GENERATED_ONLY
    assert term.coeff == 1.0


def test_ablation_with_sae_variant():
    from drowse.steering.grammar import AblationTerm
    # Variant is baked into the canonical name for plain terms (see
    # test_sae_variant_suffix_preserved), and ablation stores the full
    # variant-suffixed target verbatim.
    s = parse_expr("!honest:sae")
    term = s.alphas["!honest:sae"]
    assert isinstance(term, AblationTerm)
    assert term.target == "honest:sae"


# Ablation + projection is a grammar error — ablation ("project the
# component onto d̂ then land it on the mean") and projection ("decompose
# direction d̂ along onto") are different operations on the same atom.
# Composing them in one term yields ambiguous hook math; the spec opts
# for a hard error and tells the user to split the term.


def test_ablation_with_orthogonal_projection_rejected():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("!honest|sycophantic")
    assert "ablation" in str(ei.value).lower()
    assert "projection" in str(ei.value).lower()


def test_ablation_with_onto_projection_rejected():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("!honest~sycophantic")
    assert "ablation" in str(ei.value).lower()
    assert "projection" in str(ei.value).lower()


# ---------------------------------------------------- ablation format/round-trip ---

def test_format_ablation_bare():
    s = parse_expr("!honest")
    assert format_expr(s) == "!honest"


def test_format_ablation_explicit_coeff():
    s = parse_expr("0.5 !honest")
    assert format_expr(s) == "0.5 !honest"


def test_format_ablation_negative_bare():
    s = parse_expr("-!honest")
    assert format_expr(s) == "-!honest"


def test_format_ablation_trigger_emitted():
    s = parse_expr("!refusal@response")
    assert format_expr(s) == "!refusal@response"


def test_format_ablation_trigger_both_elided():
    s = parse_expr("!refusal@both")
    assert format_expr(s) == "!refusal"


def test_format_ablation_composed_with_additive():
    s = parse_expr("0.3 honest + !sycophantic")
    assert format_expr(s) == "0.3 honest + !sycophantic"


def test_format_ablation_separator_when_negative():
    # A negative ablation following an additive term should render through
    # the ``" - "`` separator, mirroring how ``- 0.2 foo`` is formatted.
    s = parse_expr("0.3 honest - !sycophantic")
    assert format_expr(s) == "0.3 honest - !sycophantic"


def test_format_ablation_variant_preserved():
    s = parse_expr("!honest:sae-gemma-scope")
    assert format_expr(s) == "!honest:sae-gemma-scope"


@pytest.mark.parametrize("text", [
    "!honest",
    "0.5 !honest",
    "-!honest",
    "-0.3 !honest",
    "!refusal@response",
    "!honest:sae",
    "!honest:sae-gemma-scope",
    "0.3 honest + !sycophantic",
    "0.3 honest - !sycophantic",
    "!refusal + !sycophantic",
])
def test_ablation_format_parse_format_is_stable(text: str) -> None:
    """Ablation expressions round-trip through parse -> format stably."""
    s1 = parse_expr(text)
    r1 = format_expr(s1)
    s2 = parse_expr(r1)
    r2 = format_expr(s2)
    assert r1 == r2
    assert s1.alphas == s2.alphas


# -------------------------------------------------- ablation IR integration ---

# Ablation and additive terms on the same concept are different operations
# (clean the residual stream vs push a direction); they live under distinct
# keys so a single Steering can express both.


def test_ablation_and_plain_same_concept_coexist():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("0.3 honest + !honest")
    assert "honest" in s.alphas
    assert "!honest" in s.alphas
    assert s.alphas["honest"] == pytest.approx(0.3)
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == 1.0


def test_repeated_ablation_sums_coefficients():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("0.3 !honest + 0.2 !honest")
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == pytest.approx(0.5)
    assert term.trigger == Trigger.BOTH


def test_repeated_ablation_with_matching_trigger_sums():
    from drowse.steering.grammar import AblationTerm
    s = parse_expr("0.3 !honest@response + 0.2 !honest@response")
    term = s.alphas["!honest"]
    assert isinstance(term, AblationTerm)
    assert term.coeff == pytest.approx(0.5)
    assert term.trigger == Trigger.GENERATED_ONLY


def test_repeated_ablation_with_conflicting_triggers_rejected():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.3 !honest@before + 0.2 !honest@after")
    msg = str(ei.value).lower()
    assert "conflicting" in msg
    assert "ablation" in msg or "trigger" in msg


def test_entries_report_ablation_alongside_additive_terms():
    s = parse_expr("0.3 honest + !sycophantic + 0.4 warm@after")
    entries = s.entries()
    assert set(entries) == {"honest", "!sycophantic", "warm"}
    assert entries["honest"] == (0.3, Trigger.BOTH)
    assert entries["!sycophantic"] == (1.0, Trigger.BOTH)
    assert entries["warm"] == (0.4, Trigger.AFTER_THINKING)


def test_referenced_selectors_includes_ablation_target():
    # Install-time hook must fetch the ablation target's pack the same
    # way it fetches additive terms — the atom lives in the AST whether
    # or not ``!`` prefixes it.
    assert referenced_selectors("!refusal") == [(None, "refusal", "raw")]


def test_referenced_selectors_ablation_with_namespace_and_variant():
    assert referenced_selectors("!bob/honest:sae-gemma-scope") == [
        ("bob", "honest", "sae-gemma-scope"),
    ]


def test_referenced_selectors_mixed_additive_and_ablation():
    out = referenced_selectors("0.3 honest + !sycophantic")
    assert out == [(None, "honest", "raw"), (None, "sycophantic", "raw")]


def test_steering_str_emits_ablation():
    s = parse_expr("0.3 honest + !sycophantic")
    assert str(s) == "0.3 honest + !sycophantic"


def test_direct_ablation_construction_round_trips():
    """Steering built directly with an AblationTerm round-trips through str."""
    from drowse.steering.grammar import AblationTerm
    s = Steering(alphas={
        "honest": 0.3,
        "!sycophantic": AblationTerm(
            coeff=1.0, trigger=Trigger.BOTH, target="sycophantic",
        ),
    })
    rendered = str(s)
    reparsed = parse_expr(rendered)
    assert reparsed.alphas["honest"] == 0.3
    term = reparsed.alphas["!sycophantic"]
    assert isinstance(term, AblationTerm)
    assert term.target == "sycophantic"
    assert term.coeff == 1.0


# ------------------------------------------------------------- probe gates ---
#
# ``@when:<probe><op><num>``: the full namespaced probe string is stored
# verbatim as ``ProbeGate.probe``, matching the keys ``Monitor.flat_scalars``
# (and the lens/SAE instruments) emit.


def _gate_of(steering: Steering, key: str):
    """The gate on ``key``'s trigger."""
    val = steering.alphas[key]
    trig = val[1] if isinstance(val, tuple) else steering.trigger
    assert trig.gate is not None
    return trig.gate


# Every probe-gate shape the grammar accepts: (input, probe, op, threshold,
# canonical form).  Inputs may carry spaces around the operator; the
# canonical form has none and round-trips byte-for-byte.
_GATES = [
    # vector probes: bipolar, monopolar, coordinate axis
    ("0.3 angry.calm@when:angry.calm>0.4", "angry.calm", ">", 0.4, None),
    ("0.3 happy.sad @when:angry.calm > 0.4", "angry.calm", ">", 0.4,
     "0.3 happy.sad@when:angry.calm>0.4"),
    ("0.5 angry.calm@when:manipulative>0.3", "manipulative", ">", 0.3, None),
    ("0.5 angry.calm@when:angry.calm<-0.5", "angry.calm", "<", -0.5, None),
    ("0.5 angry.calm@when:angry.calm<=-0.25", "angry.calm", "<=", -0.25, None),
    # manifold subspace fraction, every operator
    ("0.3 happy.sad @when:circumplex:fraction > 0.5", "circumplex:fraction", ">", 0.5,
     "0.3 happy.sad@when:circumplex:fraction>0.5"),
    ("0.3 happy.sad@when:circumplex:fraction>=0.25", "circumplex:fraction", ">=", 0.25, None),
    ("0.3 happy.sad@when:circumplex:fraction<0.25", "circumplex:fraction", "<", 0.25, None),
    ("0.3 happy.sad@when:circumplex:fraction<=0.25", "circumplex:fraction", "<=", 0.25, None),
    # label similarity (negated distance: thresholds are naturally negative)
    ("0.3 happy.sad @when:circumplex@elated > 0.7", "circumplex@elated", ">", 0.7,
     "0.3 happy.sad@when:circumplex@elated>0.7"),
    ("0.3 happy.sad@when:circumplex@elated>-0.1", "circumplex@elated", ">", -0.1, None),
    ("0.3 happy.sad@when:circumplex@elated>=-0.1", "circumplex@elated", ">=", -0.1, None),
    ("0.3 happy.sad@when:circumplex@elated<-0.5", "circumplex@elated", "<", -0.5, None),
    ("0.3 happy.sad@when:circumplex@elated<=-0.1", "circumplex@elated", "<=", -0.1, None),
    # fuzzy channels: tube membership, soft assignment
    ("0.3 happy.sad@when:circumplex:membership>0.6", "circumplex:membership", ">", 0.6, None),
    ("0.3 happy.sad@when:circumplex~elated>0.5", "circumplex~elated", ">", 0.5, None),
]


@pytest.mark.parametrize(("text", "probe", "op", "threshold", "canonical"), _GATES)
def test_probe_gate_shapes_parse_and_round_trip(
    text: str, probe: str, op: ProbeGateOp, threshold: float, canonical: str | None,
) -> None:
    from drowse.steering.triggers import ProbeGate

    steering = parse_expr(text)
    (key,) = steering.alphas
    assert _gate_of(steering, key) == ProbeGate(probe=probe, op=op, threshold=threshold)
    entry = steering.alphas[key]
    assert isinstance(entry, tuple)
    assert entry[1].prompt is False  # a gate never fires during prefill
    canonical = canonical or text
    assert format_expr(steering) == canonical
    assert format_expr(parse_expr(canonical)) == canonical
    assert dict(parse_expr(canonical).alphas) == dict(steering.alphas)


@pytest.mark.parametrize(("text", "match"), [
    ("0.5 angry.calm@when", "when"),
    ("0.5 angry.calm@when:angry.calm 0.4", None),
    ("0.5 angry.calm@when:angry.calm>", None),
    ("0.5 angry.calm@bogus", "when"),
    ("0.3 happy.sad @when:circumplex:frac > 0.5", "fraction"),
    ("0.3 happy.sad @when:circumplex:bogus > 0.5", "membership"),
])
def test_malformed_probe_gates_are_rejected(text: str, match: str | None) -> None:
    with pytest.raises(SteeringExprError, match=match):
        parse_expr(text)


def test_gated_and_ungated_terms_compose_without_interaction():
    text = (
        "0.3 happy.sad@when:angry.calm>0.4 "
        "+ 0.2 calm@when:circumplex:fraction>=0.5 "
        "+ 0.1 honest@when:circumplex@elated<-0.2 + 0.2 warm.clinical"
    )
    s = parse_expr(text)
    assert _gate_of(s, "happy.sad").probe == "angry.calm"
    assert _gate_of(s, "calm").probe == "circumplex:fraction"
    assert _gate_of(s, "honest").probe == "circumplex@elated"
    # A bare-float entry inherits the steering default (BOTH, no gate).
    assert isinstance(s.alphas["warm.clinical"], float)


# ---------------------------------------------------------------------------
# Trigger grammar totality: every publicly-constructible Trigger has a form.
#
# format_expr used to render an unnameable Trigger as the sentinel "@custom",
# which parse_expr rejects — so a Steering built from Trigger.first/after (or
# a gate narrowed to a phase) stamped an unparseable string onto
# Recipe.steering / GenerationResult.applied_steering, and any regen /
# transcript import / loom replay that re-parses it raised.
# ---------------------------------------------------------------------------


def _public_triggers() -> list[Trigger]:
    """One Trigger per public factory, plus the documented gate compounds."""
    gate = Trigger.when("angry.calm", ">", 0.4)
    return [
        # Presets.
        Trigger.BOTH,
        Trigger.GENERATED_ONLY,
        Trigger.PROMPT_ONLY,
        Trigger.AFTER_THINKING,
        Trigger.THINKING_ONLY,
        # Counted decode windows.
        Trigger.first(0),
        Trigger.first(5),
        Trigger.after(3),
        Trigger.after(128),
        # Probe gates: a negative threshold (label-similarity channels read
        # negative) and every comparison operator.
        gate,
        Trigger.when("angry.calm", ">=", 0.0),
        Trigger.when("circumplex:fraction", "<", 0.25),
        Trigger.when("circumplex@elated", "<=", -0.5),
        # Gate narrowed to a phase — the compound shape Trigger.when's own
        # docstring points callers at.
        replace(gate, thinking=False),
        replace(gate, response=False),
        replace(gate, generated=False, prompt=True),
        replace(Trigger.BOTH, gate=gate.gate),
        # Gate on a counted window.
        replace(Trigger.first(4), gate=gate.gate),
        replace(Trigger.after(2), gate=gate.gate),
    ]


@pytest.mark.parametrize("trig", _public_triggers())
def test_every_public_trigger_round_trips(trig: Trigger) -> None:
    """``parse_expr(format_expr(s)) == s`` over every public Trigger factory."""
    steering = Steering(alphas={"alice/probe": (0.5, trig)})
    text = format_expr(steering)
    back = parse_expr(text)
    entry = back.alphas["alice/probe"]
    # Trigger.BOTH renders without an "@" tag, so it lands as a bare float
    # inheriting the Steering-level default.
    got = entry[1] if isinstance(entry, tuple) else back.trigger
    assert got == trig, text
    # Re-rendering is a fixed point, so a stamped recipe string is stable.
    assert format_expr(back) == text


def _trigger_of(text: str, key: str = "alice/x") -> Trigger:
    """The per-entry Trigger a one-term expression parses to."""
    entry = parse_expr(text).alphas[key]
    assert isinstance(entry, tuple)
    return entry[1]


def test_counted_window_forms_parse() -> None:
    assert _trigger_of("0.5 alice/x@first:5") == Trigger.first(5)
    assert _trigger_of("0.5 alice/x@after:3") == Trigger.after(3)
    # The bare preset keeps its post-thinking meaning; only the ":N" form is
    # the token window.
    assert _trigger_of("0.5 alice/x@after") == Trigger.AFTER_THINKING


def test_compound_trigger_keeps_phase_and_gate() -> None:
    trig = _trigger_of("0.5 alice/x@after&when:angry.calm>0.4")
    assert trig.thinking is False
    assert trig.gate is not None
    assert trig.gate.probe == "angry.calm"
    assert trig.gate.threshold == pytest.approx(0.4)


@pytest.mark.parametrize("text", [
    "0.5 alice/x@first",             # counted form needs ":N"
    "0.5 alice/x@first:2.5",         # non-integer count
    "0.5 alice/x@first:-1",          # negative count
    "0.5 alice/x@after&both",        # compound's second clause must be a gate
    "0.5 alice/x@invented",          # unknown keyword
])
def test_malformed_trigger_forms_rejected(text: str) -> None:
    with pytest.raises(SteeringExprError):
        parse_expr(text)


def test_unknown_trigger_error_lists_the_new_forms() -> None:
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.5 alice/x@invented")
    msg = str(ei.value)
    assert "first:<n>" in msg
    assert "after:<n>" in msg
    assert "&when:" in msg


def test_unrepresentable_trigger_raises_instead_of_emitting_a_sentinel() -> None:
    """A hand-built Trigger with no grammar form fails loudly at render.

    Both counts at once is not something any public factory produces; the
    alternative — emitting a sentinel — is what poisoned recipes before.
    """
    trig = Trigger(prompt=False, first_n=5, after_n=2)
    with pytest.raises(SteeringExprError, match="programmatic-only"):
        format_expr(Steering(alphas={"alice/x": (0.5, trig)}))
