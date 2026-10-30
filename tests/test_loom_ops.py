"""drowse.loom.ops: fork/prefill plans and the generation turn opener (CPU, no model)."""

from __future__ import annotations

from typing import Any

import pytest

from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, SamplingConfig
from drowse.loom.errors import InvalidNodeOperationError, MutationDuringGenerationError
from drowse.loom.ops import begin_generation_turn, plan_fork, plan_prefill
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree


class _Tok:
    """Maps known phrases to fixed ids; anything else to one id per char."""

    def __init__(self, table: dict[str, list[int]] | None = None) -> None:
        self.table = table or {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.table.get(text, [ord(c) for c in text]))


def _generated(
    tree: LoomTree,
    *,
    seat: str = "assistant",
    label: str | None = "guide",
    recipe: Recipe | None = None,
) -> tuple[str, str]:
    parent = tree.add_user_turn("hi")
    node = tree.begin_assistant(
        parent,
        recipe=recipe if recipe is not None else Recipe(
            steering="0.2 calm",
            sampling=SamplingConfig(max_tokens=4, seed=9),
            thinking=False,
            system_prompt="be terse",
        ),
        role_label=label,
        seat=seat,
    )
    tree.finalize_assistant(node, text="abc", raw_token_ids=[10, 11, 12])
    return parent, node


def test_fork_forces_a_multitoken_replacement_under_the_source_recipe() -> None:
    tree = LoomTree()
    parent, node = _generated(tree)
    tok = _Tok({"replacement phrase": [41, 42]})

    turn = plan_fork(
        tree, tok, node, 1, replacement_text="replacement phrase",
        default_max_tokens=16,
    )
    assert turn.input is None
    assert turn.forced_prefix == [10, 41, 42]
    assert turn.parent_node_id == parent
    assert turn.gen_seat == "assistant"
    assert turn.steering == "0.2 calm"
    assert turn.thinking is False
    assert turn.system_prompt == "be terse"
    assert turn.sampling.max_tokens == 7  # forced span + the recipe's 4
    assert turn.sampling.seed == 9
    assert turn.sampling.assistant_role == "guide"

    reseeded = plan_fork(tree, tok, node, 1, 11, seed=123, default_max_tokens=16)
    assert reseeded.forced_prefix == [10, 11]
    assert reseeded.sampling.seed == 123
    assert reseeded.generate_kwargs()["forced_prefix"] == [10, 11]


def test_fork_of_a_user_seat_node_stays_in_the_user_seat() -> None:
    tree = LoomTree()
    _parent, node = _generated(
        tree, seat="user", label="narrator", recipe=Recipe(),
    )
    turn = plan_fork(tree, _Tok(), node, 0, 7, default_max_tokens=16)
    assert turn.gen_seat == "user"
    assert turn.sampling.user_role == "narrator"
    assert turn.sampling.max_tokens == 17  # no recipe budget: session default
    assert turn.system_prompt is INHERIT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"raw_index": 3, "alt_token_id": 1}, "out of range"),
        ({"raw_index": 0}, "exactly one"),
        ({"raw_index": 0, "alt_token_id": 1, "replacement_text": "x"}, "exactly one"),
        ({"raw_index": 0, "replacement_text": "gone"}, "empty"),
    ],
)
def test_fork_refuses_bad_targets(kwargs: dict[str, Any], match: str) -> None:
    tree = LoomTree()
    _parent, node = _generated(tree)
    with pytest.raises(InvalidNodeOperationError, match=match):
        plan_fork(
            tree, _Tok({"gone": []}), node, kwargs.pop("raw_index"),
            default_max_tokens=8, **kwargs,
        )


def test_fork_refuses_a_node_without_raw_ids() -> None:
    tree = LoomTree()
    user = tree.add_user_turn("transcript-loaded")
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        plan_fork(tree, _Tok(), user, 0, 1, default_max_tokens=8)


def test_prefill_opens_the_reply_with_the_text_and_skips_thinking() -> None:
    tree = LoomTree()
    root = tree.root_id
    user = tree.add_user_turn("question")
    turn = plan_prefill(
        tree, _Tok({"Sure,": [5, 6]}), user, "Sure,",
        steering="0.3 calm", sampling=SamplingConfig(max_tokens=10, seed=1),
        default_max_tokens=64,
    )
    assert turn.input == "question"
    assert turn.parent_node_id == root  # dedup re-uses the user turn
    assert turn.forced_prefix == [5, 6]
    assert turn.thinking is False
    assert turn.steering == "0.3 calm"
    assert turn.sampling == SamplingConfig(max_tokens=12, seed=1)
    assert turn.gen_seat == "assistant"
    assert turn.system_prompt is INHERIT_SYSTEM_PROMPT

    default_budget = plan_prefill(
        tree, _Tok({"Sure,": [5, 6]}), user, "Sure,", default_max_tokens=64,
    )
    assert default_budget.sampling.max_tokens == 66


def test_prefill_refuses_non_user_nodes_and_empty_text() -> None:
    tree = LoomTree()
    _parent, node = _generated(tree)
    with pytest.raises(InvalidNodeOperationError, match="not a user node"):
        plan_prefill(tree, _Tok(), node, "x", default_max_tokens=8)
    user = tree.add_user_turn("q")
    with pytest.raises(InvalidNodeOperationError, match="empty"):
        plan_prefill(tree, _Tok({"x": []}), user, "x", default_max_tokens=8)


def test_generation_turn_commits_the_user_turn_and_reserves_it() -> None:
    tree = LoomTree()
    node = begin_generation_turn(
        tree, "hello", raw=False, parent_node_id=None,
        sampling=SamplingConfig(user_role="captain", assistant_role="deer"),
        recipe=Recipe(),
    )
    user = tree.nodes[node].parent_id
    assert user is not None
    assert tree.nodes[user].role_label == "captain"
    assert tree.nodes[node].role_label == "deer"
    assert tree.reservation == user
    with pytest.raises(MutationDuringGenerationError):
        tree.edit(user, "changed mid-generation")


def test_generation_turn_refuses_a_user_under_user_send() -> None:
    tree = LoomTree()
    tree.add_user_turn("waiting")
    with pytest.raises(InvalidNodeOperationError, match="cannot send"):
        begin_generation_turn(
            tree, "again", raw=False, parent_node_id=None, sampling=None,
            recipe=Recipe(),
        )
    assert tree.reservation is None


def test_generation_turn_user_seat_and_flat_paths() -> None:
    tree = LoomTree()
    user = tree.add_user_turn("waiting")
    # A user-seat generation may hang under a user node (scene seating).
    node = begin_generation_turn(
        tree, None, raw=False, parent_node_id=user,
        sampling=SamplingConfig(user_role="narrator"), recipe=Recipe(),
        gen_seat="user",
    )
    assert tree.nodes[node].role == "user"
    assert tree.nodes[node].role_label == "narrator"

    # Flat mode: no labels, an empty span just extends the parent.
    flat = LoomTree()
    span = flat.add_user_turn("once")
    node = begin_generation_turn(
        flat, "", raw=True, parent_node_id=span,
        sampling=SamplingConfig(assistant_role="deer"), recipe=Recipe(),
    )
    assert flat.nodes[node].parent_id == span
    assert flat.nodes[node].role_label is None
    assert flat.reservation == span


def test_generation_turn_continues_an_existing_message() -> None:
    tree = LoomTree()
    parent, node = _generated(tree)
    recipe = Recipe(steering="0.1 warm")
    returned = begin_generation_turn(
        tree, None, raw=False, parent_node_id=node, sampling=None,
        recipe=recipe, continuation_node_id=node,
    )
    assert returned == node
    assert tree.nodes[node].recipe == recipe
    assert tree.nodes[node].text == ""
    assert tree.reservation == node
    assert tree.nodes[node].parent_id == parent
