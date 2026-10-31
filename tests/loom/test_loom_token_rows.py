"""Loom token rows are complete at creation and snapshot-safe.

Regression for the row-aliasing race: ``LoomNode.to_dict`` handed out the
live token lists, and ``raw_index`` was stamped onto rows at finalize time,
after the WS forwarder could already be serializing them.  Rows now carry
``raw_index`` from the moment the token tap creates them and are never
mutated afterwards; serialization copies the lists.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from drowse.generation.decode import GenerationState
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree
from drowse.session.engine import ConsumerPlan, TokenTap


def test_to_dict_snapshots_token_lists() -> None:
    tree = LoomTree()
    user = tree.add_user_turn("hi")
    node = tree.begin_assistant(user, recipe=Recipe())
    tree.append_token(node, {"token_id": 1, "text": "a", "raw_index": 0})
    snapshot = tree.node_to_dict(node, include_tokens=True)
    tree.append_token(node, {"token_id": 2, "text": "b", "raw_index": 1})
    tree.append_token(node, {"token_id": 3, "text": "c", "raw_index": 2}, thinking=True)
    assert [row["text"] for row in snapshot["tokens"]] == ["a"]
    assert snapshot["thinking_tokens"] == []
    assert snapshot["tokens"] is not tree.get(node).tokens
    full = tree.to_dict(include_tokens=True)
    (row,) = [n for n in full["nodes"] if n["id"] == node]
    assert row["tokens"] is not tree.get(node).tokens


def test_token_rows_carry_raw_index_when_created() -> None:
    tree = LoomTree()
    user = tree.add_user_turn("prompt")
    node_id = tree.begin_assistant(user, recipe=Recipe())
    session = SimpleNamespace(
        tree=tree,
        lens=SimpleNamespace(live=None, last_step_readings=None),
        sae=SimpleNamespace(live=None, last_step_readings=None),
    )
    state = GenerationState()
    tap = TokenTap(
        cast(Any, session),
        reads=cast(Any, SimpleNamespace(geometry_step_readings=None)),
        state=state,
        consumer=lambda *_args: None,
        plan=ConsumerPlan(
            live_scores_on=True, has_monitor_probes=False,
            wants_live_token_scores=False, persists_layer_scores=False,
            persists_subspace_coords=False, persists_probe_row=False,
            lens=False, sae=False, perplexity_requested=False, need_tap=True,
        ),
        assistant_node_id=node_id,
        collect_logprobs=False,
        readout_top_k=8,
        recipe_steering=None,
    )
    rows_at_emit: list[dict[str, Any]] = []
    # Decode order with a suppressed delimiter (raw 1) between a thinking
    # token (raw 0) and two response tokens (raw 2, raw 3): the emitted-token
    # and raw-step coordinates diverge.
    for raw_index, text, thinking in ((0, "t", True), (2, "a", False), (3, "b", False)):
        state.emit_map.append((raw_index, thinking))
        tap(text, thinking, 10 + raw_index, None, None, None, raw_index)
        node = tree.get(node_id)
        emitted = node.thinking_tokens if thinking else node.tokens
        assert emitted is not None
        rows_at_emit.append(dict(emitted[-1]))

    assert [row["raw_index"] for row in rows_at_emit] == [0, 2, 3]
    node = tree.get(node_id)
    assert [row["raw_index"] for row in node.thinking_tokens or []] == [0]
    assert [row["raw_index"] for row in node.tokens or []] == [2, 3]
