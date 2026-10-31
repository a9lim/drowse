"""Authored-prompt capture: plan (F11), tokenize, match, persist."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse.readout.types import ProbeReading
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree
from drowse.session.authored_capture import (
    AuthoredPromptCapture,
    AuthoredPromptMatch,
    AuthoredPromptTarget,
    match_targets,
    pending_targets,
    plan_authored_capture,
    tokenize_authored_text,
)


class _OffsetEchoTokenizer:
    """Character tokenizer with exact fast-tokenizer-style offsets.

    Counts calls so tests can pin how often authored text is tokenized.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.encodes = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encodes += 1
        return [ord(char) for char in text]

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def decode(self, ids: Any, **_kwargs: Any) -> str:
        return "".join(chr(int(tid)) for tid in ids)


class _EncodeOnlyTokenizer:
    """A slow tokenizer: no offsets, one-token decodes."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("return_offsets_mapping")

    def decode(self, ids: Any, **_kwargs: Any) -> str:
        return "".join(chr(int(tid)) for tid in ids)


def _authored_path(*texts: str) -> tuple[LoomTree, str]:
    """A path of authored user turns, then a generated assistant node."""
    tree = LoomTree()
    parent: str | None = None
    for text in texts:
        parent = tree.add_user_turn(text, parent_id=parent)
        tree.nodes[parent].tokens = None
    assert parent is not None
    return tree, tree.begin_assistant(parent, recipe=Recipe(steering="0.2 formal"))


def _plan(tree: LoomTree, tok: Any, node: str, **overrides: Any) -> Any:
    ends: list[int] = []

    def content_end() -> int:
        ends.append(1)
        return 99

    kwargs: dict[str, Any] = {
        "raw": False,
        "monitor": SimpleNamespace(),
        "lens": SimpleNamespace(live=None),
        "sae": SimpleNamespace(live=None),
        "monitor_active": False,
        "lens_active": False,
        "sae_active": False,
        "readout_top_k": 8,
        "persist_per_layer_scores": False,
        "content_end": content_end,
    }
    kwargs.update(overrides)
    return plan_authored_capture(tree, tok, node, **kwargs), ends


def test_no_measuring_channel_means_no_tokenization() -> None:
    """Regression (F11): with no monitor probe, lens or SAE consumer, a loom
    generation used to tokenize every uncaptured authored turn on its path
    (twice each) on every send, then throw the targets away."""
    tok = _OffsetEchoTokenizer()
    tree, node = _authored_path("first turn", "second turn")

    plan, ends = _plan(tree, tok, node)

    assert plan is None
    assert tok.calls == 0 and tok.encodes == 0
    assert ends == []


def test_an_active_channel_tokenizes_each_target_once() -> None:
    tok = _OffsetEchoTokenizer()
    tree, node = _authored_path("first turn", "second turn")

    plan, ends = _plan(tree, tok, node, lens_active=True)

    assert isinstance(plan, AuthoredPromptCapture)
    assert [t.token_texts for t in plan.targets] == [
        tuple("first turn"), tuple("second turn"),
    ]
    assert tok.calls == 2 and tok.encodes == 0  # one offsets call per target
    assert plan.search_end == 99 and ends == [1]
    assert plan.steering == "0.2 formal"
    assert (plan.monitor_active, plan.lens_active, plan.sae_active) == (False, True, False)


def test_raw_and_targetless_plans_skip_the_content_render() -> None:
    tok = _OffsetEchoTokenizer()
    tree, node = _authored_path("flat text")
    plan, ends = _plan(tree, tok, node, raw=True, monitor_active=True)
    assert plan is not None and plan.search_end is None and ends == []

    generated = LoomTree()
    user = generated.add_user_turn("typed")
    reply = generated.begin_assistant(user, recipe=Recipe())
    generated.finalize_assistant(reply, text="said")
    next_node = generated.begin_assistant(user, recipe=Recipe())
    generated.nodes[user].tokens = [{"token_id": 1, "text": "typed"}]  # captured
    plan, ends = _plan(generated, tok, next_node, monitor_active=True)
    assert plan is None and ends == []


def test_slow_tokenizer_falls_back_to_exact_decodes() -> None:
    target = tokenize_authored_text(_EncodeOnlyTokenizer(), "n", "ab", thinking=True)
    assert target == AuthoredPromptTarget("n", True, (97, 98), ("a", "b"))


def test_targets_map_to_producer_rows_after_prefix_cache() -> None:
    tree = LoomTree()
    user_id = tree.add_user_turn(" ab ")
    assistant_id = tree.begin_assistant(user_id, recipe=Recipe())
    tok = _OffsetEchoTokenizer()

    targets = pending_targets(tree, tok, assistant_id, raw=False)
    assert len(targets) == 1
    assert "".join(targets[0].token_texts) == " ab "

    prompt = torch.tensor([[1, *targets[0].token_ids, 2]], dtype=torch.long)
    matches, producer_positions = match_targets(
        tok, targets, prompt, cache_position_offset=2,
    )

    assert producer_positions == [0, 1]
    # The first authored token and its producer are already in the cached
    # prefix. Later tokens map to suffix-local producer rows.
    assert matches[0].capture_columns == (None, None, 0, 1)


def test_match_excludes_generation_role_and_preserves_trim() -> None:
    tok = _OffsetEchoTokenizer()
    target = tokenize_authored_text(tok, "node", " assistant ", thinking=False)
    assert target is not None
    content_ids = tok.encode("assistant")
    # The same word appears again in the synthetic generation header. The
    # no-generation render length bounds matching to visible message content.
    prompt = torch.tensor(
        [[1, *content_ids, 2, *content_ids, 3]], dtype=torch.long,
    )
    matches, _positions = match_targets(
        tok, [target], prompt,
        cache_position_offset=0, search_end=2 + len(content_ids),
    )

    assert matches[0].token_positions[0] == 1
    assert "".join(matches[0].target.token_texts) == " assistant "


def _reading(value: float) -> ProbeReading:
    return ProbeReading(
        fraction=0.0, nearest=[], coords=(value,), coords_per_layer={0: (value,)},
    )


@pytest.mark.parametrize("matched", [True, False])
def test_persist_writes_all_measurement_channels(matched: bool) -> None:
    tree = LoomTree()
    user_id = tree.add_user_turn("x")
    tree.begin_assistant(user_id, recipe=Recipe(steering="0.2 formal"))
    seen_top_k: dict[str, int] = {}

    def _lens_capture(_hidden: Any, *, top_k: int) -> Any:
        seen_top_k["lens"] = top_k
        return (
            {0: [("yes", 0.8)]},
            [("yes", 0.8, 0.5, 0.1)],
            {0: [9]},
            {"jlens/yes": _reading(0.8)},
        )

    def _sae_capture(_hidden: Any, *, top_k: int) -> Any:
        seen_top_k["sae"] = top_k
        return ([(12, 1.5, "feature", 3.0)], {"sae/12": _reading(0.5)})

    capture = AuthoredPromptCapture(
        tree=tree,
        tokenizer=_OffsetEchoTokenizer(),
        targets=[AuthoredPromptTarget(user_id, False, (120,), ("x",))],
        monitor=cast(Any, SimpleNamespace(
            score_single_token=lambda _hidden: {"formal": _reading(0.25)},
        )),
        lens=cast(Any, SimpleNamespace(
            live={"source": "local:default"}, authored_capture=_lens_capture,
        )),
        sae=cast(Any, SimpleNamespace(
            live={"source": "saelens:test", "layer": 0},
            authored_capture=_sae_capture,
        )),
        monitor_active=True,
        lens_active=True,
        sae_active=True,
        readout_top_k=8,
        persist_per_layer_scores=True,
        steering="0.2 formal",
    )
    if matched:
        assert capture.match(
            torch.tensor([[1, 120, 2]]), cache_position_offset=0,
        ) == [0]
    else:
        capture.matches = []

    capture.persist({0: torch.tensor([[3.0, 4.0]])})

    tokens = tree.nodes[user_id].tokens
    if not matched:
        assert tokens is None  # an authored turn stays uncaptured
        return
    assert tokens is not None
    row = tokens[0]
    assert row["text"] == "x"
    assert row["probes"] == {"formal": 0.25, "jlens/yes": 0.8, "sae/12": 0.5}
    instruments = row["measurements"]["instruments"]
    assert set(instruments) == {"geometry", "lens", "sae"}
    assert instruments["geometry"]["readings"]["formal"]["coords"] == [0.25]
    assert instruments["lens"]["readout"]["layers"][0]["tokens"][0]["id"] == 9
    assert instruments["sae"]["readout"]["features"][0]["id"] == 12
    assert seen_top_k == {"lens": 8, "sae": 8}
    assert isinstance(capture.matches[0], AuthoredPromptMatch)
