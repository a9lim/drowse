"""Conversation verbs on a real tiny session: fork, prefill, append, restore,
and the generation reservation the tree now owns."""

from __future__ import annotations

from typing import Any

import pytest

from drowse import LoomMutated, SamplingConfig
from drowse.loom.errors import LoomTreeError, MutationDuringGenerationError
from tests.support.sessions import TinySessionFactory


def _ids(session: Any, text: str) -> list[int]:
    return list(session.tokenizer.encode(text, add_special_tokens=False))


def test_fork_from_token_lands_a_forced_sibling(
    make_tiny_session: TinySessionFactory,
) -> None:
    session, _script = make_tiny_session.scripted("Hello there.")
    source = session.generate("Say hi.").first
    tree = session.tree
    node_id = tree.active_node_id
    assert source.text == "Hello there."
    raw = tree.nodes[node_id].raw_token_ids
    assert raw

    forked = session.fork_from_token(node_id, 1, replacement_text="a")

    sibling_id = tree.active_node_id
    assert sibling_id != node_id
    assert tree.nodes[sibling_id].parent_id == tree.nodes[node_id].parent_id
    assert tree.nodes[sibling_id].raw_token_ids[:2] == raw[:1] + _ids(session, "a")
    assert forked.text.startswith("Ha")
    assert tree.reservation is None


def test_prefill_assistant_opens_the_reply_with_the_text(
    make_tiny_session: TinySessionFactory,
) -> None:
    session, _script = make_tiny_session.scripted("Hello there.")
    user_id = session.append_turn(None, "Say hi.", role="user")

    result = session.prefill_assistant(
        user_id, "Hey", sampling=SamplingConfig(max_tokens=4),
    )

    reply_id = session.tree.active_node_id
    reply = session.tree.nodes[reply_id]
    assert reply.parent_id == user_id  # the user turn was re-used, not duplicated
    assert reply.role == "assistant"
    assert result.text.startswith("Hey")
    assert reply.raw_token_ids[:3] == _ids(session, "Hey")


def test_append_turn_coalesces_and_validates_one_way(tiny_session: Any) -> None:
    user = tiny_session.append_turn(None, "one", role="user")
    assert tiny_session.append_turn(user, " two", role="user") == user
    reply = tiny_session.append_turn(user, "three", role="assistant")
    node = tiny_session.tree.nodes[reply]
    assert node.recipe is None and node.tokens is None
    assert node.raw_token_ids == _ids(tiny_session, "three")
    with pytest.raises(ValueError, match="non-empty"):
        tiny_session.append_turn(reply, "", role="user")


def test_restore_announces_after_the_new_tree_is_installed(tiny_session: Any) -> None:
    tiny_session.append_turn(None, "kept", role="user")
    saved = tiny_session.tree.to_dict()
    tiny_session.tree.reset()
    before = tiny_session.tree
    seen: list[tuple[str, bool, int]] = []

    def _observe(event: object) -> None:
        if isinstance(event, LoomMutated) and event.op == "restore":
            # A WS subscriber serializes ``session.tree`` on this event.
            seen.append((
                event.op,
                tiny_session.tree is not before,
                len(event.added),
            ))

    unsubscribe = tiny_session.events.subscribe(_observe)
    try:
        restored = tiny_session.restore_tree(saved)
    finally:
        unsubscribe()

    assert tiny_session.tree is restored
    assert seen == [("restore", True, 2)]  # the saved root and its user turn
    assert restored.rev > before.rev
    assert restored.session_id == before.session_id

    foreign = dict(saved, model_id="someone/else")
    with pytest.raises(LoomTreeError, match="does not match loaded model"):
        tiny_session.restore_tree(foreign)
    assert tiny_session.tree is restored


def test_a_generation_reserves_its_subtree_until_it_ends(
    make_tiny_session: TinySessionFactory,
) -> None:
    session, _script = make_tiny_session.scripted("Hello there.")
    user = session.append_turn(None, "Say hi.", role="user")
    refused: list[str] = []

    def _on_token(*_args: Any, **_kwargs: Any) -> None:
        if refused:
            return
        try:
            session.tree.edit(user, "changed mid-generation")
        except MutationDuringGenerationError as exc:
            refused.append(str(exc))

    session.generate(None, on_token=_on_token, sampling=SamplingConfig(max_tokens=3))

    assert refused and "reservation" in refused[0]
    assert session.tree.reservation is None
    session.tree.edit(user, "free again")


def test_authored_turns_gain_rows_only_when_a_channel_measures_them(
    make_tiny_session: TinySessionFactory,
) -> None:
    """End to end: an authored user turn stays uncaptured on a probe-less
    generation (F11 skips it), and gains measured token rows the first time
    a generation with a live geometry probe re-reads it."""
    from tests.support.sessions import register_vectors

    session, _script = make_tiny_session.scripted("ok")
    user = session.append_turn(None, "plain words", role="user")
    session.generate(None, on_token=lambda *_a, **_k: None)
    assert session.tree.nodes[user].tokens is None

    register_vectors(session, "probe_axis")
    session.add_probe("probe_axis")
    session.tree.navigate(user)
    session.generate(None, on_token=lambda *_a, **_k: None)

    rows = session.tree.nodes[user].tokens
    assert rows, "the authored turn was not captured"
    assert "".join(row["text"] for row in rows) == "plain words"
    assert all("measurements" in row for row in rows)
