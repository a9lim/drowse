"""The loom-anchored token replay, end to end on a tiny session.

A token replay rebuilds the forward that *produced* a loom token and reads it
with one instrument family.  The contract every family shares is that the
replay is the producing forward: on a generation that ran with live per-token
readouts, ``session.<family>.token_readout(node, raw_index)`` reports what the
live readout reported for that token (up to KV-cache-vs-full-forward rounding),
under the node's recorded steering.  ``steered=False`` reads the unsteered
counterfactual instead.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from drowse.generation.sampling import SamplingConfig
from drowse.generation.token_callback import TokenConsumer, TokenConsumerOptions
from tests.support.sessions import TinySessionFactory, register_vectors

_LENS_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "A small corpus for the tiny lens fit.",
    "Every row of the batch reads its own tail.",
]

_LIVE = TokenConsumer(
    on_event=lambda _event: None,
    options=TokenConsumerOptions(
        live_scores=True, per_layer_scores=True,
        lens_readout=True, sae_readout=True,
    ),
)

_STEERING = "0.4 mood"


def _all_families_session(
    factory: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """A scripted tiny session with a geometry probe, a fitted J-lens and a
    resident mock SAE, every family's live readout on."""
    from drowse.sae import runtime as sae_runtime
    from drowse.sae import sources as sae_sources
    from drowse.sae.backend import ResolvedSae
    from tests.support.sae import MockSaeBackend

    session, _script = factory.scripted("Hi there, friend.")
    register_vectors(session, "mood")
    session.add_probe("mood")
    session.lens.fit(_LENS_CORPUS)
    width = int(session.model.config.hidden_size)
    backend = MockSaeBackend(layers=frozenset({1}), d_model=width)
    monkeypatch.setattr(
        sae_runtime, "resolve_sae_backend",
        lambda *_a, **_k: ResolvedSae(backend, 1, width),
    )
    monkeypatch.setattr(sae_sources, "load_sae_feature_meta", lambda *_a, **_k: {})
    monkeypatch.setattr(sae_sources, "save_sae_metadata", lambda *_a, **_k: None)
    session.sae.load("mock-release", layer=1)
    session.lens.set_live(True)
    session.sae.set_live(True)
    return session


def _live_rows(session: Any) -> tuple[Any, dict[int, dict[str, Any]]]:
    session.generate(
        "Say hi.", steering=_STEERING,
        sampling=SamplingConfig(max_tokens=5), on_token=_LIVE,
    )
    node = session.tree.get(session.tree.active_node_id)
    rows = {int(row["raw_index"]): row["measurements"] for row in node.tokens}
    assert len(rows) == len(node.raw_token_ids) == 5
    return node, rows


def _replay(session: Any, family: str, node_id: str, k: int, **kw: Any) -> Any:
    envelope = session.instruments[family].token_readout(node_id, k, **kw)
    measurements = envelope["measurements"]
    assert measurements["scope"] == "replay"
    assert measurements["provenance"] == "replayed"
    return measurements["instruments"][family]


def test_every_family_replays_the_producing_forward(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    node, rows = _live_rows(session)

    for k, live in rows.items():
        live = live["instruments"]

        geometry = _replay(session, "geometry", node.id, k)
        assert geometry["binding"] == {"source": None, "steering": _STEERING}
        got, want = geometry["readings"]["mood"], live["geometry"]["readings"]["mood"]
        assert got["coords"] == pytest.approx(want["coords"], rel=1e-4, abs=1e-5)
        assert got["fraction"] == pytest.approx(want["fraction"], rel=1e-4, abs=1e-5)
        for layer, coords in want["coords_per_layer"].items():
            assert got["coords_per_layer"][layer] == pytest.approx(
                coords, rel=1e-4, abs=1e-5,
            )

        sae = _replay(session, "sae", node.id, k)
        assert sae["binding"] == live["sae"]["binding"]
        live_features = {f["id"]: f["activation"] for f in live["sae"]["readout"]["features"]}
        replay_features = {f["id"]: f["activation"] for f in sae["readout"]["features"]}
        assert set(replay_features) == set(live_features)
        for fid, activation in live_features.items():
            assert replay_features[fid] == pytest.approx(activation, rel=1e-4, abs=1e-6)

        # Replay wide enough that every live top token is in the replayed
        # set, then compare probabilities by token id (the tiny lens is
        # nearly flat, so the ranking itself can swap near-ties).
        lens = _replay(session, "lens", node.id, k, top_k=256)
        assert lens["binding"] == live["lens"]["binding"]
        replay_layers = {
            row["layer"]: {t["id"]: t["logprob"] for t in row["tokens"]}
            for row in lens["readout"]["layers"]
        }
        for row in live["lens"]["readout"]["layers"]:
            replayed = replay_layers[row["layer"]]
            for token in row["tokens"]:
                assert math.exp(replayed[token["id"]]) == pytest.approx(
                    math.exp(token["logprob"]), rel=1e-4, abs=1e-7,
                )


def test_unsteered_replay_reads_the_counterfactual(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    node, rows = _live_rows(session)
    k = 2
    live = rows[k]["instruments"]

    geometry = _replay(session, "geometry", node.id, k, apply_steering=False)
    assert geometry["binding"] == {"source": None, "steering": None}
    assert geometry["readings"]["mood"]["coords"] != pytest.approx(
        live["geometry"]["readings"]["mood"]["coords"], rel=1e-3,
    )
    for family in ("lens", "sae"):
        replayed = _replay(session, family, node.id, k, apply_steering=False)
        assert replayed["binding"]["steering"] is None
        assert replayed["binding"]["source"] == live[family]["binding"]["source"]
    # Re-reading the steered replay afterwards still lands on the live
    # reading: the counterfactual left no steering behind.
    again = _replay(session, "geometry", node.id, k)
    assert again["readings"]["mood"]["coords"] == pytest.approx(
        live["geometry"]["readings"]["mood"]["coords"], rel=1e-4, abs=1e-5,
    )
