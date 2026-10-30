"""Conversation operations over a :class:`~drowse.loom.tree.LoomTree`.

The tree owns the data model and its mutation primitives; this module owns
the conversation rules that need a tokenizer or the chat family's scene
grammar on top of it:

- the one authored-append rule (:func:`append_turn`): concatenate onto a
  same-seat, same-label parent, otherwise add a new turn, with one
  validation for both seats;
- the seating checks a legacy (non-scene) chat template imposes
  (:func:`check_user_send_target`) and the committed-thinking check
  (:func:`check_thinking_commit`);
- the forced-prefix plans behind a token fork and an answer prefill
  (:func:`plan_fork`, :func:`plan_prefill`), which the session executes;
- opening the turn a stateful generation streams into
  (:func:`begin_generation_turn`), which also reserves its subtree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from drowse.generation.sampling import (
    INHERIT_SYSTEM_PROMPT,
    InheritSystemPrompt,
    SamplingConfig,
)
from drowse.loom.errors import InvalidNodeOperationError
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree

if TYPE_CHECKING:
    from drowse.model.scene import TurnGrammar

_SEATS = ("user", "assistant")


def check_user_send_target(tree: LoomTree, parent_node_id: str | None) -> None:
    """Refuse sending a new user turn from a user-role node.

    The resolved parent (``parent_node_id``, else the active node) must not
    itself be a user node: that node is already waiting for an assistant,
    and a user-under-user shape cannot render through a legacy chat
    template.  Regen paths pass the grandparent explicitly so
    ``add_user_turn``'s dedup re-uses the existing user sibling.

    Raises :class:`InvalidNodeOperationError` (HTTP 400).
    """
    target_parent_id = (
        parent_node_id if parent_node_id is not None else tree.active_node_id
    )
    parent = tree.nodes.get(target_parent_id)
    if parent is not None and parent.role == "user":
        raise InvalidNodeOperationError(
            f"cannot send a new user turn from a user node "
            f"({target_parent_id}): the active turn is already "
            f"waiting for an assistant.  Use /regen to redo the "
            f"assistant, or navigate away first."
        )


def check_thinking_commit(grammar: "TurnGrammar | None", thinking: str | None) -> None:
    """Refuse a committed thinking block the renderer can't carry.

    Committed thinking renders through the scene stitcher's family think
    delimiters.  A family without them (gemma), a channel-thinking family
    (gpt-oss) or a template-fallback family (GLM) would raise at the *next*
    render, long after the commit; fail at commit time instead, with the
    renderer's own error.
    """
    if thinking is None:
        return
    if grammar is None or grammar.think_open is None:
        from drowse.model.scene import SceneThinkingUnsupportedError

        raise SceneThinkingUnsupportedError(
            "this model family has no scene-mode think delimiters — "
            "a committed thinking block could not be rendered"
        )


def _encode_nonempty(tokenizer: Any, text: str) -> list[int]:
    ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise InvalidNodeOperationError(
            f"append_turn: {text!r} tokenized to an empty sequence — "
            f"nothing to append"
        )
    return ids


def append_turn(
    tree: LoomTree,
    tokenizer: Any,
    grammar: "TurnGrammar | None",
    parent_node_id: str | None,
    text: str,
    *,
    seat: str,
    flat: bool = False,
    role_label: str | None = None,
    thinking: str | None = None,
) -> str:
    """Commit one authored span in ``seat`` without generating.

    When the parent (``parent_node_id``, else the active node) already sits
    in ``seat`` under the same ``role_label``, ``text`` is concatenated
    exactly onto it (``thinking`` onto its thinking block) and the node
    becomes an authored message.  Otherwise a new turn lands under the
    parent: a user turn through ``add_user_turn`` (which re-uses an
    identical sibling), an assistant turn as a finished authored node that
    keeps its token ids so it stays forkable.

    Seating is free in scene mode (``grammar`` present) and in ``flat``
    (base-model) mode, where the seat is authorship provenance.  A legacy
    chat template only renders alternating turns, so there a new user turn
    may not hang under a user node and a new assistant turn must hang under
    one.

    Raises :class:`InvalidNodeOperationError` for an unknown seat, empty
    ``text``, text that tokenizes to nothing, or a seating violation, and
    :class:`~drowse.model.scene.SceneThinkingUnsupportedError` when the
    family cannot render committed thinking.  Returns the node id; the
    active node advances to it.
    """
    if seat not in _SEATS:
        raise InvalidNodeOperationError(
            f"append_turn: seat must be 'user' or 'assistant', got {seat!r}"
        )
    if text == "":
        raise InvalidNodeOperationError("append_turn: text must be non-empty")
    check_thinking_commit(grammar, thinking)
    parent_id = parent_node_id or tree.active_node_id
    parent = tree.get(parent_id)
    if parent.role == seat and parent.role_label == role_label:
        return tree.append_authored(
            parent_id,
            text,
            thinking_text=thinking,
            raw_token_ids=_encode_nonempty(tokenizer, parent.text + text),
        )
    if not flat and grammar is None:
        if seat == "user":
            check_user_send_target(tree, parent_id)
        elif parent.role != "user":
            raise InvalidNodeOperationError(
                f"append_turn: {parent_id!r} is a {parent.role} node, not a "
                f"user node — an authored assistant turn hangs off a user "
                f"turn (free seating requires scene mode)"
            )
    raw_token_ids = _encode_nonempty(tokenizer, text)
    if seat == "user":
        return tree.add_user_turn(
            text, parent_id=parent_id, role_label=role_label,
            thinking_text=thinking,
        )
    return tree.add_assistant_turn(
        parent_id,
        text,
        role_label=role_label,
        thinking_text=thinking,
        raw_token_ids=raw_token_ids,
    )


@dataclass(frozen=True)
class ForcedTurn:
    """A generation whose first decode steps replay ``forced_prefix``.

    The plan a token fork or an answer prefill hands to the session's
    generation core: ``input`` is the new user text (``None`` continues the
    path under ``parent_node_id``) and the token budget already covers the
    forced span.
    """

    input: str | None
    parent_node_id: str | None
    forced_prefix: list[int]
    sampling: SamplingConfig
    steering: Any = None  # str | Steering | None
    thinking: bool | None = None
    gen_seat: str = "assistant"
    system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT

    def generate_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the generation core (all but ``input``)."""
        return {
            "steering": self.steering,
            "sampling": self.sampling,
            "thinking": self.thinking,
            "parent_node_id": self.parent_node_id,
            "forced_prefix": self.forced_prefix,
            "gen_seat": self.gen_seat,
            "system_prompt": self.system_prompt,
        }


def plan_fork(
    tree: LoomTree,
    tokenizer: Any,
    node_id: str,
    raw_index: int,
    alt_token_id: int | None = None,
    *,
    replacement_text: str | None = None,
    seed: int | None = None,
    default_max_tokens: int,
) -> ForcedTurn:
    """Plan regenerating ``node_id`` as a sibling with one token replaced.

    The fork replays the node's raw decode sequence up to ``raw_index``,
    forces either ``alt_token_id`` or the tokenization of
    ``replacement_text`` there, then samples freely.  It reuses the node's
    steering, sampling, thinking and system-prompt recipe; by default the
    original seed re-applies and the sampler still draws every forced step,
    keeping the RNG stream aligned through the fork point.  ``seed`` asks
    for a new ending instead.  The sibling lands under the same parent, in
    the same seat and under the same cast label as the source node.

    Raises :class:`InvalidNodeOperationError` when the node has no raw token
    record (a transcript-loaded node), ``raw_index`` is out of range, the
    node has no parent, or not exactly one replacement is given.
    """
    node = tree.get(node_id)
    raw = node.raw_token_ids
    if not raw:
        raise InvalidNodeOperationError(
            f"fork_from_token: {node_id!r} has no raw token record "
            f"(transcript-loaded node — not forkable)"
        )
    if not 0 <= raw_index < len(raw):
        raise InvalidNodeOperationError(
            f"fork_from_token: raw_index {raw_index} out of range "
            f"[0, {len(raw)}) for {node_id!r}"
        )
    if node.parent_id is None:
        raise InvalidNodeOperationError(
            f"fork_from_token: {node_id!r} has no parent to anchor "
            f"the forked sibling under"
        )
    if (alt_token_id is None) == (replacement_text is None):
        raise InvalidNodeOperationError(
            "fork_from_token: provide exactly one alternative token id "
            "or replacement text"
        )
    if replacement_text is not None:
        replacement_ids = [
            int(t) for t in tokenizer.encode(
                replacement_text, add_special_tokens=False,
            )
        ]
        if not replacement_ids:
            raise InvalidNodeOperationError(
                "fork_from_token: replacement text tokenized to an empty "
                "sequence"
            )
    else:
        assert alt_token_id is not None
        replacement_ids = [int(alt_token_id)]
    forced_prefix = [int(t) for t in raw[:raw_index]] + replacement_ids

    recipe = node.recipe
    base = (
        recipe.sampling
        if recipe is not None and recipe.sampling is not None
        else SamplingConfig()
    )
    # The forced span burns ``len(forced_prefix)`` decode steps before the
    # continuation starts; extend the budget so a deep fork keeps the
    # original continuation headroom.
    sampling = replace(
        base,
        max_tokens=len(forced_prefix) + (base.max_tokens or default_max_tokens),
        **({"seed": seed} if seed is not None else {}),
    )
    # The sibling carries the source node's label on its own seat; the
    # tree's stamped label wins over whatever the recipe's sampling held.
    if node.role == "user":
        sampling = replace(sampling, user_role=node.role_label)
    else:
        sampling = replace(sampling, assistant_role=node.role_label)
    return ForcedTurn(
        input=None,
        parent_node_id=node.parent_id,
        forced_prefix=forced_prefix,
        sampling=sampling,
        steering=recipe.steering if recipe is not None else None,
        thinking=recipe.thinking if recipe is not None else None,
        gen_seat=node.role,
        system_prompt=(
            recipe.system_prompt if recipe is not None else INHERIT_SYSTEM_PROMPT
        ),
    )


def plan_prefill(
    tree: LoomTree,
    tokenizer: Any,
    node_id: str,
    text: str,
    *,
    steering: Any = None,
    sampling: SamplingConfig | None = None,
    default_max_tokens: int,
) -> ForcedTurn:
    """Plan an answer prefill: an assistant reply to ``node_id`` opening with ``text``.

    ``text`` is the start of the response, so the thinking channel is
    skipped.  The plan re-sends the user turn under its own parent, where
    ``add_user_turn``'s dedup re-uses it, so the reply lands as a sibling
    assistant node under ``node_id``.  ``steering``/``sampling`` apply as in
    an ordinary generation, with the budget extended by the forced span.

    Raises :class:`InvalidNodeOperationError` when ``node_id`` is not a user
    node or ``text`` tokenizes to nothing.
    """
    node = tree.get(node_id)
    if node.role != "user":
        raise InvalidNodeOperationError(
            f"prefill_assistant: {node_id!r} is a {node.role} node, "
            f"not a user node — prefill seeds the assistant reply "
            f"that hangs off a user turn"
        )
    forced_prefix = list(tokenizer.encode(text, add_special_tokens=False))
    if not forced_prefix:
        raise InvalidNodeOperationError(
            f"prefill_assistant: {text!r} tokenized to an empty "
            f"sequence — nothing to prefill"
        )
    base = sampling if sampling is not None else SamplingConfig()
    return ForcedTurn(
        input=node.text,
        parent_node_id=node.parent_id,
        forced_prefix=forced_prefix,
        sampling=replace(
            base,
            max_tokens=len(forced_prefix) + (base.max_tokens or default_max_tokens),
        ),
        steering=steering,
        thinking=False,
    )


def begin_generation_turn(
    tree: LoomTree,
    input: Any,
    *,
    raw: bool,
    parent_node_id: str | None,
    sampling: SamplingConfig | None,
    recipe: Recipe,
    gen_seat: str = "assistant",
    continuation_node_id: str | None = None,
) -> str:
    """Open the node a stateful generation streams into; reserve its subtree.

    A ``str`` input commits the new user turn first (under ``sampling``'s
    ``user_role`` label); in flat (``raw``) mode a non-empty span lands as
    its own node and an empty one just extends the parent, with no seating
    check.  ``None`` or a message list anchors on the parent directly.  The
    generated node takes ``gen_seat`` and that seat's label from
    ``sampling``; ``continuation_node_id`` reuses an existing same-seat
    message whose text the caller replays as the forced decode prefix.
    """
    # Per-message labels ride this send's sampling box and are stamped at
    # creation, so a turn keeps its label regardless of later box changes.
    # Flat mode has no chat template, so labels don't apply there.
    user_role = sampling.user_role if sampling is not None and not raw else None
    assistant_role = (
        sampling.assistant_role if sampling is not None and not raw else None
    )
    if isinstance(input, str) and raw:
        anchor_id = (
            tree.add_user_turn(input, parent_id=parent_node_id) if input != ""
            else parent_node_id or tree.active_node_id
        )
    elif isinstance(input, str):
        if gen_seat == "assistant":
            check_user_send_target(tree, parent_node_id)
        anchor_id = tree.add_user_turn(
            input, parent_id=parent_node_id, role_label=user_role,
        )
    else:
        anchor_id = parent_node_id or tree.active_node_id
    tree.reserve(anchor_id)
    if continuation_node_id is not None:
        return tree.begin_continuation(continuation_node_id, recipe=recipe)
    return tree.begin_assistant(
        anchor_id,
        recipe=recipe,
        role_label=user_role if gen_seat == "user" else assistant_role,
        seat=gen_seat,
    )


__all__ = [
    "ForcedTurn",
    "append_turn",
    "begin_generation_turn",
    "check_thinking_commit",
    "check_user_send_target",
    "plan_fork",
    "plan_prefill",
]
