"""CPU tests for the jlens grammar tier: gates, lazy resolution, ns guard."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.steering.grammar import (
    SteeringExprError,
    format_expr,
    parse_expr,
    referenced_selectors,
)
from drowse.cli.verbs.lens import lens_fit_is_noop
from drowse.lens.lifecycle import fit_covers_layers
from drowse.store.manifold_schema import ManifoldFormatError
from drowse.store.manifold_authoring import create_discover_manifold_folder
from tests.support.lens import LENS_PROMPTS as _PROMPTS
from tests.support.lens import LensHarness


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


# ------------------------------------------------------------- gate grammar --


def _sole_trigger(steering: Any) -> Any:
    (entry,) = steering.alphas.values()
    assert isinstance(entry, tuple)
    return entry[1]


def test_when_gate_accepts_namespaced_probe() -> None:
    trigger = _sole_trigger(parse_expr("0.3 formal.casual@when:jlens/fake>0.4"))
    assert trigger.gate is not None
    assert trigger.gate.probe == "jlens/fake"
    assert trigger.gate.op == ">"
    assert trigger.gate.threshold == 0.4


def test_when_gate_namespaced_probe_round_trips() -> None:
    text = "0.3 formal.casual@when:jlens/fake>0.4"
    assert format_expr(parse_expr(text)) == format_expr(parse_expr(format_expr(parse_expr(text))))
    assert "jlens/fake" in format_expr(parse_expr(text))


def test_when_gate_namespaced_manifold_channels() -> None:
    for text, probe in [
        ("0.5 warm.clinical@when:default/emotions@happy>-0.5", "default/emotions@happy"),
        ("0.5 warm.clinical@when:default/emotions:fraction>0.5", "default/emotions:fraction"),
        ("0.5 warm.clinical@when:default/personas~hacker>0.5", "default/personas~hacker"),
        ("0.5 warm.clinical@when:default/personas[3]>0.4", "default/personas[3]"),
    ]:
        trigger = _sole_trigger(parse_expr(text))
        assert trigger.gate is not None
        assert trigger.gate.probe == probe, text


def test_jlens_steering_atom_round_trips() -> None:
    steering = parse_expr("0.5 jlens/orange")
    assert "jlens/orange" in steering.alphas
    round_tripped = parse_expr(format_expr(steering))
    assert "jlens/orange" in round_tripped.alphas

    ablation = parse_expr("!jlens/fake")
    assert "jlens/fake" in format_expr(ablation)


def test_reserved_namespace_atoms_preserve_case() -> None:
    """``jlens/Paris`` must name the same token as ``@when:jlens/Paris``.

    The atom used to slug-canonicalize to ``jlens/paris`` (a different
    vocabulary entry) while the gate kept ``jlens/Paris`` verbatim.
    """
    steering = parse_expr("0.3 jlens/Paris@when:jlens/Paris>0.01")
    (key,) = steering.alphas
    assert key == "jlens/Paris"
    assert _sole_trigger(steering).gate.probe == key
    assert format_expr(steering) == "0.3 jlens/Paris@when:jlens/Paris>0.01"
    assert "neuron/L8n344" in parse_expr("0.2 neuron/L8n344").alphas
    assert "!jlens/Fake" in parse_expr("!jlens/Fake").alphas


@pytest.mark.parametrize("namespace", ["jlens", "slens"])
def test_token_id_atom_and_probe(namespace: str) -> None:
    text = f"0.3 {namespace}/#1234 + 0.2 x@when:{namespace}/#7>0.01"
    steering = parse_expr(text)
    assert f"{namespace}/#1234" in steering.alphas
    gated = steering.alphas["x"]
    assert isinstance(gated, tuple) and gated[1].gate is not None
    assert gated[1].gate.probe == f"{namespace}/#7"
    assert format_expr(steering) == text
    assert parse_expr(format_expr(steering)) == steering
    assert referenced_selectors(f"!{namespace}/#12") == [(namespace, "#12", "raw")]


@pytest.mark.parametrize(
    "text",
    [
        "0.3 #12",                # no namespace
        "0.3 sae/#12",            # sae takes a bare feature id
        "0.3 alice/#12",          # ordinary namespaces take names
        "0.3 jlens/#1.5",         # ids are integers
        "0.3 jlens/#",            # id required
        "0.3 jlens/fake:sae",     # instrument atoms take no variant
        "x@when:sae/#3>0",
    ],
)
def test_token_id_and_reserved_atom_rejections(text: str) -> None:
    with pytest.raises(SteeringExprError):
        parse_expr(text)


def test_reserved_but_unsteerable_namespaces_refuse_to_resolve(
    make_tiny_session: Any,
) -> None:
    from drowse.session.session import ProfileNotRegisteredError

    session = make_tiny_session(warm=False)
    for key in ("slens/owl", "neuron/L8n344", "head/L20h17"):
        with pytest.raises(ProfileNotRegisteredError, match="reserved"):
            session.ensure_profile_registered(key)


# ---------------------------------------------------------- lazy resolution --


#: Byte-tokenized prompts long enough for the estimator on the tiny session.
_TINY_PROMPTS = [
    "a first prompt that is long enough for the tiny model",
    "the second prompt, also long enough for the tiny one",
    "and a third one to round out the tiny model's corpus",
]


def test_profile_registration_resolves_jlens_atom(make_tiny_session: Any) -> None:
    session = make_tiny_session(warm=False)
    lens = session.lens.fit(_TINY_PROMPTS)

    dirs = session.ensure_profile_registered("jlens/g")
    assert "jlens/g" in session.profiles
    assert set(dirs) == set(lens.source_layers)
    assert all(isinstance(v, torch.Tensor) for v in dirs.values())


def test_token_id_atom_steers_and_probes_the_vocab_id(make_tiny_session: Any) -> None:
    """``jlens/#<id>`` addresses vocab id ``<id>`` directly, never tokenizing.

    The name used to reach ``resolve_word_token`` as the literal text
    ``"#<id>"``: a misleading multi-token error, or on a tokenizer with a
    single ``#<id>`` piece, the wrong token.  The word and the id atoms for
    the same vocabulary entry must agree.
    """
    from drowse.lens.jlens import resolve_word_token
    from drowse.model.readout import get_unembedding

    session = make_tiny_session()  # warm: a cold scope trips C2 (scope lane)
    lens = session.lens.fit(_TINY_PROMPTS)
    gid = resolve_word_token(session.tokenizer, "g")
    (key,) = parse_expr(f"0.3 jlens/#{gid}").alphas
    by_id = session.ensure_profile_registered(key)
    expected = lens.token_direction(gid, get_unembedding(session.model))
    assert set(by_id) == set(expected)
    for layer, vec in by_id.items():
        assert torch.allclose(vec, expected[layer])
    by_word = session.ensure_profile_registered("jlens/g")
    for layer, vec in by_id.items():
        assert torch.allclose(vec, by_word[layer])

    name = session.add_probe(f"jlens/#{gid}")
    assert name == f"jlens/#{gid}"
    spec = session.lens.probes[name]
    assert spec["token_id"] == gid
    assert spec["word"] == f"#{gid}"
    with session.steering(f"0.3 jlens/g@when:jlens/#{gid}>0.01"):
        assert session._steering_composer.gated_lens_probe_keys() == {f"jlens/#{gid}"}


def test_token_id_atom_rejects_ids_outside_the_vocab(make_tiny_session: Any) -> None:
    from drowse.lens.jlens import InvalidTokenIdError, resolve_token_atom
    from drowse.model.readout import get_unembedding

    session = make_tiny_session(warm=False)
    session.lens.fit(_TINY_PROMPTS)
    vocab = int(get_unembedding(session.model).shape[0])
    with pytest.raises(InvalidTokenIdError, match="outside this vocabulary"):
        session.ensure_profile_registered(f"jlens/#{vocab}")
    with pytest.raises(InvalidTokenIdError):
        session.add_probe(f"jlens/#{vocab}")
    assert not session.lens.probes
    for bad in ("#", "#-1", "#1.5", "#x"):
        with pytest.raises(InvalidTokenIdError):
            resolve_token_atom(session.tokenizer, bad, vocab_size=vocab)


def test_profile_registration_jlens_requires_fitted_lens(make_tiny_session: Any) -> None:
    from drowse.lens.jlens import LensNotFittedError

    session = make_tiny_session(warm=False)
    with pytest.raises(LensNotFittedError, match="drowse lens fit"):
        session.ensure_profile_registered("jlens/g")


def test_add_probe_routes_jlens_to_lens_registry() -> None:
    """A ``jlens/<word>`` probe is a READOUT-channel probe (probability
    under the lens softmax), not a linear probe — ``add_probe``
    routes it to the session lens-probe registry, never folding the
    ``W_U[v] @ J_l`` direction and never falling through to ``extract()``."""
    session = LensHarness()
    session.lens.fit(_PROMPTS)

    name = session.lens.attach("jlens/g")
    assert name == "jlens/g"
    spec = session.lens.probes["jlens/g"]
    assert spec["word"] == "g"
    assert spec["layers"] == session.lens.resident().source_layers
    # No direction fold, no profile registration — the readout channel is
    # computed from lens logits, not a whitened coordinate.
    assert "jlens/g" not in session.profiles


def test_lens_probe_scores_strength_channel() -> None:
    """Lens-probe readings carry ONE channel — ``coords = (strength,)``,
    the mean fitted-layer probability (the gate channel + the workspace card's
    number; apples-to-apples across tokens and layers) — plus the
    per-layer ``(p_l,)`` trace, matching ``token_readout_stats`` on the
    same logits."""
    from drowse.lens.jlens import token_readout_stats

    session = LensHarness()
    session.lens.fit(_PROMPTS)
    session.lens.attach("jlens/g")
    layers = [int(l) for l in session.lens.resident().source_layers]
    d_model = next(iter(session.lens.resident().jacobians.values())).shape[0]
    hidden = {
        l: torch.randn(d_model, generator=torch.Generator().manual_seed(l))
        for l in layers
    }
    readings = session.lens.score_probes(hidden)
    reading = readings["jlens/g"]
    strength = reading.value
    assert reading.unit == "mean_token_probability"
    assert 0.0 <= strength <= 1.0
    assert sorted(reading.per_layer) == layers
    # Cross-check against the pure math on the same logits.
    logits = session.lens.runtime.logits_rows(
        session.lens.resident(), [(l, hidden[l]) for l in layers],
    )
    depths = [l / (len(session.layers) - 1) for l in layers]
    token_id = session.lens.probes["jlens/g"]["token_id"]
    ((exp_str, exp_com, exp_spread, per_layer),) = token_readout_stats(
        logits.float(), depths, [token_id],
    )
    assert strength == pytest.approx(exp_str)
    # strength is the mean of the per-layer probabilities — the objective
    # apples-to-apples identity a within-layer normalization can't satisfy.
    assert strength == pytest.approx(sum(per_layer) / len(per_layer))
    assert reading.depth is not None
    assert reading.depth.center == pytest.approx((exp_com,))
    assert reading.depth.spread == pytest.approx((exp_spread,))
    # The depth mass names its own basis — ``depth_com`` means three
    # unrelated things across families, so a bare float invites a
    # meaningless cross-family comparison.
    assert reading.depth.basis == "readout_probability_mass"
    for l, p_l in zip(layers, per_layer):
        assert reading.per_layer[l] == pytest.approx(p_l)
        assert 0.0 <= p_l <= 1.0
    # There are no geometry fields to default: a readout probe has no
    # subspace behind it, so the native reading simply doesn't carry
    # fraction / residual / nearest / assignment / membership.  They
    # reappear only at the explicit compatibility projection.
    for absent in ("fraction", "residual", "nearest", "membership", "coords"):
        assert not hasattr(reading, absent)
    compat = reading.to_probe_reading()
    assert compat.coords == (strength,)
    assert compat.fraction == 0.0
    assert compat.membership == 1.0


def test_gated_lens_probe_keys_and_gate_scalars() -> None:
    """The gate scalar key space: ``jlens/<word>`` = strength (the one
    channel), emitted by ``scalar_gate_keys`` over the native readings — no
    ``:fraction`` / ``:membership`` constants."""
    session = LensHarness()
    session.lens.fit(_PROMPTS)
    session.lens.attach("jlens/g")

    class _FlatCapture:
        def __init__(self, latest: dict[int, torch.Tensor]) -> None:
            self._latest = latest

        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return self._latest

    layers = [int(l) for l in session.lens.resident().source_layers]
    d_model = next(iter(session.lens.resident().jacobians.values())).shape[0]
    session.capture = _FlatCapture(
        {l: torch.randn(d_model, generator=torch.Generator().manual_seed(l))
         for l in layers}
    )
    scalars = session.lens.gate_scalars(step_id=7)
    assert "jlens/g" in scalars
    assert "jlens/g[0]" in scalars
    assert "jlens/g[1]" not in scalars  # one channel — strength only
    assert scalars["jlens/g"] == scalars["jlens/g[0]"]
    # The stash is armed for the display step to reuse this forward's logits —
    # keyed by the forward's step id (step identity replaced ``fresh``).
    assert session.lens.step_stash is not None
    assert session.lens.step_stash["step"] == 7
    assert session.lens.step_stash["layers"] == tuple(layers)


def test_composer_detects_gates_on_pinned_lens_probes(make_tiny_session: Any) -> None:
    """A gate on a pinned lens probe is recognized from the steering stack
    without any monitor probe attached."""
    session = make_tiny_session()  # warm: a cold scope trips C2 (scope lane)
    session.lens.fit(_TINY_PROMPTS)
    session.add_probe("jlens/g")
    with session.steering("0.3 jlens/g@when:jlens/g>0.4"):
        assert session._steering_composer.gated_lens_probe_keys() == {"jlens/g"}


def test_lens_gate_scalar_scores_only_referenced_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.jlens as jlens_module

    session = LensHarness()
    session.lens.fit(_PROMPTS)
    session.lens.attach("jlens/g")
    session.lens.attach("jlens/a")
    session.lens.enable_live(layers=[1])
    session.lens.active_for_generation = False

    class _FlatCapture:
        def __init__(self, latest: dict[int, torch.Tensor]) -> None:
            self._latest = latest

        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return self._latest

    layers = [int(l) for l in session.lens.resident().source_layers]
    d_model = next(iter(session.lens.resident().jacobians.values())).shape[0]
    session.capture = _FlatCapture(
        {l: torch.randn(d_model, generator=torch.Generator().manual_seed(l + 10))
         for l in layers}
    )

    def _fail_full_probabilities(_logits: torch.Tensor) -> torch.Tensor:
        raise AssertionError("gate-only lens scalar scoring should be column-only")

    monkeypatch.setattr(
        jlens_module, "readout_probabilities", _fail_full_probabilities,
    )

    scalars = session.lens.gate_scalars({"jlens/g"})

    assert "jlens/g" in scalars
    assert "jlens/g[0]" in scalars
    assert "jlens/a" not in scalars
    assert session.lens.step_stash is not None
    assert "probabilities" not in session.lens.step_stash


# -------------------------------------------------------------- ns reservation


@pytest.mark.parametrize("namespace", ["jlens", "slens", "sae", "neuron", "head"])
def test_manifold_authoring_rejects_reserved_runtime_namespace(
    namespace: str,
) -> None:
    with pytest.raises(ManifoldFormatError, match="reserved"):
        create_discover_manifold_folder(
            namespace, "fake", "should not exist",
            node_corpora={"a": ["x"], "b": ["y"]},
            fit_mode="pca",
        )


def test_lens_preflight_rejects_contiguous_prefix_for_all_layers() -> None:
    sidecar: dict[str, object] = {
        "sites": {"out": {"layers": list(range(5))}},
        "model_layer_count": 12,
    }
    assert not fit_covers_layers(sidecar, "all")


def test_lens_preflight_resolves_workspace_from_model_depth() -> None:
    sidecar: dict[str, object] = {
        "sites": {"out": {"layers": [4, 5, 6, 7, 8, 9]}},
        "model_layer_count": 11,
    }
    assert fit_covers_layers(sidecar, "workspace")
    sidecar["sites"] = {"out": {"layers": [4, 5, 6]}}
    assert not fit_covers_layers(sidecar, "workspace")


def test_lens_noop_preflight_requires_exact_model_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens.corpus import HfLensCorpus
    from drowse.lens.store import lens_manifest_path

    corpus = HfLensCorpus("repo", "config", "a" * 40)
    expected_spec = corpus.spec
    sidecar_path = lens_manifest_path("toy/model", "j")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = sidecar_path.parent / "out.layer-0.gen-test.safetensors"
    shard_path.write_bytes(b"lens")
    sidecar: dict[str, object] = {
        "format_version": 7,
        "kind": "J",
        "sites": {"out": {
            "layers": [0],
            "tensor_files": {"0": shard_path.name},
            "tensor_sha256": {"0": hashlib.sha256(b"lens").hexdigest()},
        }},
        "model_layer_count": 2,
        "model_source_fingerprint": "source-a",
        "seq_len": 128,
        "corpus_spec": expected_spec,
        "raw_prompt_count": 100,
        "usable_prompt_count": 100,
        "n_prompts": 100,
        "d_model": 6,
    }
    from drowse.lens.store import lens_estimator_policy

    sidecar["estimator_policy"] = lens_estimator_policy("J")
    monkeypatch.setattr(
        "drowse.lens.store.load_local_lens_sidecar",
        lambda _model, _name="j": sidecar,
    )
    monkeypatch.setattr(
        "drowse.model.fingerprint.model_source_fingerprint",
        lambda *_args, **_kwargs: "source-a",
    )
    args = argparse.Namespace(
        force=False, model="toy/model", quantize=None, device="cpu",
        seq_len=None, corpus=None, prompts=100, kind="j",
    )
    assert lens_fit_is_noop(args, [0], hf_corpus=corpus)
    shard_path.write_bytes(b"corrupt")
    assert not lens_fit_is_noop(args, [0], hf_corpus=corpus)
    shard_path.write_bytes(b"lens")
    sidecar["model_source_fingerprint"] = "source-b"
    assert not lens_fit_is_noop(args, [0], hf_corpus=corpus)


def test_lens_noop_preflight_rejects_changed_dataset_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens.corpus import HfLensCorpus
    from drowse.lens.store import lens_manifest_path

    old = HfLensCorpus("repo", "config", "0" * 40)
    new = HfLensCorpus("repo", "config", "1" * 40)

    sidecar_path = lens_manifest_path("toy/model", "j")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = sidecar_path.parent / "out.layer-0.gen-test.safetensors"
    shard_path.write_bytes(b"lens")
    sidecar: dict[str, object] = {
        "format_version": 7, "kind": "J",
        "sites": {"out": {
            "layers": [0],
            "tensor_files": {"0": shard_path.name},
            "tensor_sha256": {"0": hashlib.sha256(b"lens").hexdigest()},
        }},
        "model_layer_count": 2,
        "model_source_fingerprint": "source-a",
        "seq_len": 128, "corpus_spec": old.spec,
        "raw_prompt_count": 100, "usable_prompt_count": 100,
        "n_prompts": 100, "d_model": 6,
    }
    from drowse.lens.store import lens_estimator_policy

    sidecar["estimator_policy"] = lens_estimator_policy("J")
    monkeypatch.setattr(
        "drowse.lens.store.load_local_lens_sidecar",
        lambda _model, _name="j": sidecar,
    )
    monkeypatch.setattr(
        "drowse.model.fingerprint.model_source_fingerprint",
        lambda *_args, **_kwargs: "source-a",
    )
    args = argparse.Namespace(
        force=False, model="toy/model", quantize=None, device="cpu",
        seq_len=None, corpus=None, prompts=100, kind="j",
    )
    assert lens_fit_is_noop(args, [0], hf_corpus=old)
    assert not lens_fit_is_noop(args, [0], hf_corpus=new)
    # Another dataset, or the same one read from another column, is another
    # corpus even at the recorded commit.
    assert not lens_fit_is_noop(
        args, [0], hf_corpus=HfLensCorpus("other/repo", "config", "0" * 40),
    )
    assert not lens_fit_is_noop(
        args, [0],
        hf_corpus=HfLensCorpus("repo", "config", "0" * 40, "content"),
    )


def test_lens_noop_preflight_holds_fit_lock_during_sidecar_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    import drowse.lens.store as lens_io

    held = False

    @contextmanager
    def fake_lock(_model: str, _name: str = "j"):
        nonlocal held
        held = True
        try:
            yield
        finally:
            held = False

    def read_sidecar(_model: str, _name: str = "j") -> None:
        assert held
        return None

    monkeypatch.setattr(lens_io, "lens_fit_lock", fake_lock)
    monkeypatch.setattr(lens_io, "load_local_lens_sidecar", read_sidecar)
    args = argparse.Namespace(
        force=False, model="toy/model", quantize=None, device="cpu",
        seq_len=None, corpus=None, prompts=100, kind="j",
    )

    assert not lens_fit_is_noop(args, [0])
    assert not held
