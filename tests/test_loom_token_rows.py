"""Loom token rows are complete at creation and snapshot-safe.

Regression for the row-aliasing race: ``LoomNode.to_dict`` handed out the
live token lists, and ``raw_index`` was stamped onto rows at finalize time,
after the WS forwarder could already be serializing them.  Rows now carry
``raw_index`` from the moment the token tap creates them and are never
mutated afterwards; serialization copies the lists.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import torch

from drowse.generation.decode import GenerationConfig, GenerationState
from drowse.generation.results import GenerationResult
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree
from drowse.session.session import CaptureState, DrowseSession, GenState, PreparedCall
from drowse.steering.triggers import TriggerContext
from tests.test_generation import _CurrentSessionStub


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


def _stateful_tap_session(tree: LoomTree, node_id: str) -> Any:
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session.tree = tree
    session._gen_lock = threading.Lock()
    session._gen_phase = GenState.IDLE
    session._gen_state = GenerationState()
    session._prepare_generation_call = lambda *_args: PreparedCall(
        steering_obj=None,
        use_thinking_req=False,
        gen_config=GenerationConfig(
            max_new_tokens=4, temperature=0.0, top_p=1.0, top_k=None,
        ),
        lp_count=None,
        seed=None,
        stop_list=None,
        logit_bias=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logprobs_list=None,
    )
    session._whitener = None
    session._seat_stop_augmentation = lambda stop_list, **_kwargs: stop_list
    session._monitor = SimpleNamespace(
        probe_names=[],
        set_subspace_coords=lambda _enabled: None,
        begin_live=lambda: None,
        end_live=lambda: None,
    )
    session._steering_composer = SimpleNamespace(
        _stack=[],
        active=False,
        steering_needs_probe_gating=lambda: False,
        gated_probe_keys=lambda: set(),
        gated_lens_probe_keys=lambda: set(),
        gated_sae_probe_keys=lambda: set(),
        gated_probe_names=lambda: set(),
    )
    session._start_loom_assistant = lambda *_args, **_kwargs: node_id
    session._pending_authored_prompt_targets = lambda *_args, **_kwargs: []
    session._snapshot_steering_alphas = lambda: {}
    session._generation_preamble = lambda *_args, **_kwargs: (
        torch.tensor([[1]], dtype=torch.long),
        False,
        1,
    )
    session.events = SimpleNamespace(emit=lambda _event: None)
    session._begin_capture = lambda *_args, **_kwargs: False
    session._steering = SimpleNamespace(
        ctx=TriggerContext(),
        reset_manifold_feet=lambda: None,
        has_compiled_offsets=lambda: False,
        zero_compiled_offsets=lambda: None,
    )
    session._capture_state = CaptureState()
    session._compiled = False
    session._static_cache_active = False
    session._capture_buffers = {}
    session._capture = SimpleNamespace(per_layer_buckets=lambda: {})
    session._incremental_readings = []
    session._incremental_gate_scores = []
    session._lens_instrument.last_step_readings = None
    session._sae_instrument.last_step_readings = None
    session._last_token_probe_payload = None
    session._end_capture = lambda: None
    session._active_gen_reservation = None
    session._finalize_generation = lambda *_args, **_kwargs: GenerationResult(
        text="ab", tokens=[1], token_count=1, tok_per_sec=10.0, elapsed=0.1,
    )
    return session


def test_token_rows_carry_raw_index_when_created() -> None:
    tree = LoomTree()
    user = tree.add_user_turn("prompt")
    node_id = tree.begin_assistant(user, recipe=Recipe())
    session = _stateful_tap_session(tree, node_id)
    rows_at_emit: list[dict[str, Any]] = []

    def _run_generation_loop(_input_ids: Any, _gen_config: Any, *, effective_tap: Any, **_kwargs: Any):
        assert effective_tap is not None
        state = session._gen_state
        # Decode order with a suppressed delimiter (raw 1) between a
        # thinking token (raw 0) and two response tokens (raw 2, raw 3) —
        # the emitted-token and raw-step coordinates diverge.
        for raw_index, text, thinking in ((0, "t", True), (2, "a", False), (3, "b", False)):
            state.emit_map.append((raw_index, thinking))
            effective_tap(text, thinking, 10 + raw_index, None, None, None)
            node = tree.get(node_id)
            rows_at_emit.append(dict((node.thinking_tokens if thinking else node.tokens)[-1]))
        return [5, 6, 7, 8], 0.1

    session._run_generation_loop = _run_generation_loop
    DrowseSession._generate_core(
        session, "prompt", stateless=False, on_token=lambda *_args: None,
    )

    assert [row["raw_index"] for row in rows_at_emit] == [0, 2, 3]
    node = tree.get(node_id)
    assert [row["raw_index"] for row in node.thinking_tokens or []] == [0]
    assert [row["raw_index"] for row in node.tokens or []] == [2, 3]
