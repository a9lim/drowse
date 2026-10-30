"""The loom-anchored token replay (:class:`drowse.loom.replay.TokenReplayer`).

A token replay rebuilds the forward that *produced* a loom token and reads it
with one instrument family.  The contract every family shares is that the
replay is the producing forward: on a generation that ran with live per-token
readouts, ``session.<family>.token_readout(node, raw_index)`` reports what the
live readout reported for that token (up to KV-cache-vs-full-forward rounding),
under the node's recorded steering.  ``steered=False`` reads the unsteered
counterfactual instead.  The first tests pin that end to end on a tiny
session; the rest pin the replayer's rebuild on a toy model.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, SamplingConfig
from drowse.generation.token_callback import TokenConsumer, TokenConsumerOptions
from drowse.loom.errors import InvalidNodeOperationError, UnknownNodeError
from drowse.loom.recipe import Recipe
from tests.support.replay import ReplayRig
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


# ---------------------------------------------------------------------------
# The replayer itself (toy model, real loom tree, recorded collaborators)
# ---------------------------------------------------------------------------


def _hidden_keys(hidden: dict[int, Any]) -> list[int]:
    return sorted(hidden)


def test_replay_rebuilds_the_producing_forward() -> None:
    rig = ReplayRig()
    raw_ids = rig.tokenizer.encode("abcdefg")
    node_id = rig.turn(
        raw_ids, Recipe(system_prompt="Saved pirate instruction"),
        role_label="guide",
    )

    token, read = rig.replay(
        node_id, 3, layers=[0, 2], read=_hidden_keys, label="test replay",
    )

    # Prompt + raw[:3], pooled at the final position: the forward that
    # PRODUCED the token, never including the token itself.
    assert rig.captures == [(rig.prompt_len + 3, rig.prompt_len + 2, (0, 2))]
    assert read == [0, 2]
    assert (token.node_id, token.raw_index) == (node_id, 3)
    assert token.token_id == raw_ids[3]
    assert token.token_text == rig.tokenizer.decode([raw_ids[3]])
    assert token.steering is None
    # Continue-mode rebuild: no text resend — the history walk to the
    # node's parent carries every prior turn; the gen header opens the
    # node's own seat under its stamped label.
    assert rig.prepare_calls == [{
        "input": None, "raw": False, "thinking": False,
        "system_prompt": "Saved pirate instruction",
        "parent_node_id": rig.tree.get(node_id).parent_id,
        "user_role": None, "assistant_role": "guide",
        "gen_seat": "assistant",
    }]


def test_replay_at_index_zero_reads_the_prompt_only() -> None:
    rig = ReplayRig()
    node_id = rig.turn(rig.tokenizer.encode("abc"))
    rig.replay(node_id, 0, layers=[1], read=_hidden_keys, label="test replay")
    assert rig.captures == [(rig.prompt_len, rig.prompt_len - 1, (1,))]


def test_replay_of_a_user_seat_turn_opens_the_user_seat() -> None:
    rig = ReplayRig()
    node_id = rig.turn(
        rig.tokenizer.encode("abc"), seat="user", role_label="narrator",
    )
    rig.replay(node_id, 1, layers=[0], read=_hidden_keys, label="test replay")
    call = rig.prepare_calls[0]
    assert call["gen_seat"] == "user"
    assert call["user_role"] == "narrator"
    assert call["assistant_role"] is None
    # No recipe: the system prompt is inherited, not cleared.
    assert call["system_prompt"] is INHERIT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("supports", "recipe_thinking", "expected"),
    [
        (True, None, True),     # unset: the family default
        (False, None, False),
        (True, False, False),   # the recipe's choice
        (False, True, False),   # a family that cannot think never does
    ],
)
def test_replay_resolves_the_recipe_thinking(
    supports: bool, recipe_thinking: bool | None, expected: bool,
) -> None:
    rig = ReplayRig(supports_thinking=supports)
    node_id = rig.turn(rig.tokenizer.encode("abc"), Recipe(thinking=recipe_thinking))
    rig.replay(node_id, 1, layers=[0], read=_hidden_keys, label="test replay")
    assert rig.prepare_calls[0]["thinking"] is expected


def test_replay_raw_mode_renders_the_flat_buffer() -> None:
    rig = ReplayRig()
    node_id = rig.turn(rig.tokenizer.encode("abcd"))
    rig.replay(
        node_id, 1, layers=[0], read=_hidden_keys, raw=True, label="test replay",
    )
    call = rig.prepare_calls[0]
    assert (call["input"], call["raw"]) == ("", True)
    assert call["parent_node_id"] == rig.tree.get(node_id).parent_id


def test_replay_reopens_the_recipe_steering_outside_the_exclusive_section() -> None:
    rig = ReplayRig()
    node_id = rig.turn(
        rig.tokenizer.encode("abcd"), Recipe(steering="0.3 formal.casual"),
    )

    def read(hidden: dict[int, Any]) -> None:
        rig.events.append("read")

    def layers() -> list[int]:
        rig.events.append("layers")
        return [0]

    token, _ = rig.replay(node_id, 2, layers=layers, read=read, label="test replay")
    assert token.steering == "0.3 formal.casual"
    assert rig.steering_scopes == ["0.3 formal.casual"]
    # The scope wraps the render (the steering role shapes it) and the
    # exclusive section; the composer's push/pop would deadlock inside it.
    # The layer set and the read both resolve inside the exclusive section.
    assert rig.events == [
        "scope-open", "render", "exclusive-enter", "layers", "capture",
        "read", "exclusive-exit", "scope-close",
    ]
    assert rig.exclusive_messages == [(
        "test replay called while another model use is in flight",
        "test replay called while a generation is in flight",
    )]

    # The unsteered counterfactual opens no scope and reports no steering.
    rig.events.clear()
    token, _ = rig.replay(
        node_id, 2, layers=[0], read=read, apply_steering=False,
        label="test replay",
    )
    assert token.steering is None
    assert rig.steering_scopes == ["0.3 formal.casual"]
    assert "scope-open" not in rig.events


def test_replay_refuses_a_bad_anchor_before_any_work() -> None:
    rig = ReplayRig()
    node_id = rig.turn(rig.tokenizer.encode("abc"))
    user_id = rig.tree.get(node_id).parent_id
    assert user_id is not None and rig.tree.root_id is not None

    def replay(target: str, index: int) -> None:
        rig.replay(target, index, layers=[0], read=_hidden_keys, label="test replay")

    with pytest.raises(UnknownNodeError):
        replay("nope", 0)
    # A committed user turn is a valid *shape* under the cast model (user-
    # seat generated nodes are forkable/readable), but this one has no
    # decode record, so it fails on that instead of its role.
    with pytest.raises(
        InvalidNodeOperationError, match="test replay: .*no raw token record",
    ):
        replay(user_id, 0)
    # The system root stays out of bounds: not a turn at all.
    with pytest.raises(InvalidNodeOperationError, match="only a turn"):
        replay(rig.tree.root_id, 0)
    for index in (3, -1):
        with pytest.raises(InvalidNodeOperationError, match="out of range"):
            replay(node_id, index)
    # A finalized turn without raw ids (transcript import).
    bare = rig.tree.begin_assistant(user_id)
    rig.tree.finalize_assistant(bare, text="no raw record", finish_reason="stop")
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        replay(bare, 0)
    assert rig.events == []
