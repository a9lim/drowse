"""Loom-anchored token replay: rebuild the forward that produced a token.

:class:`TokenReplayer` is the one replay workflow behind every read family's
``token_readout`` (geometry, lens, SAE).  It resolves the ``(node_id,
raw_index)`` anchor, reopens the node's recorded steering, rebuilds the
node's render (continue-mode, or the flat raw buffer) plus the raw decode
prefix, and runs one capture forward pooled at the producing position.  The
family's own read runs inside the exclusive-GPU section through ``read``, so
device work never races a generation.  A token fork addresses tokens through
the same anchor (:func:`drowse.loom.ops.resolve_token_anchor`).

The instruments take a replayer as their :class:`~drowse.readout.replay.TokenReplay`
dependency; the session builds one over its tree, renderer, steering
composer, exclusive guard and capture forward.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT
from drowse.loom.ops import TokenAnchor, resolve_token_anchor
from drowse.readout.replay import ReplayedToken, ReplayLayers

if TYPE_CHECKING:
    from drowse.generation.render import PromptRenderer
    from drowse.loom.tree import LoomTree
    from drowse.model.chat import ChatTraits

T = TypeVar("T")

#: ``exclusive(busy_msg, *, phase_msg=...)``: the session's exclusive-GPU guard.
ExclusiveGuard = Callable[..., AbstractContextManager[Any]]


class TokenReplayer:
    """Replays loom tokens' producing forwards for the read instruments.

    Dependencies:

    - ``tree``: the live loom tree (a callable: the session swaps its tree
      on restore);
    - ``renderer``: rebuilds the node's prompt render;
    - ``traits``: the chat facts (whether the family can think);
    - ``tokenizer``: decodes the replayed token's text;
    - ``open_scope``: opens a steering scope for an expression;
    - ``exclusive``: the exclusive-GPU guard;
    - ``capture``: ``capture(ids, *, layer_indices, pool_index)`` — one
      capture forward returning ``{layer: (hidden,) fp32}``.
    """

    def __init__(
        self,
        *,
        tree: Callable[[], "LoomTree"],
        renderer: "PromptRenderer",
        traits: "ChatTraits",
        tokenizer: Any,
        open_scope: Callable[[str], AbstractContextManager[Any]],
        exclusive: ExclusiveGuard,
        capture: Callable[..., dict[int, torch.Tensor]],
    ) -> None:
        self._tree = tree
        self._renderer = renderer
        self._traits = traits
        self._tokenizer = tokenizer
        self._open_scope = open_scope
        self._exclusive = exclusive
        self._capture = capture

    def __call__(
        self,
        node_id: str,
        raw_index: int,
        *,
        layers: ReplayLayers,
        read: Callable[[dict[int, torch.Tensor]], T],
        apply_steering: bool = True,
        raw: bool = False,
        label: str,
    ) -> tuple[ReplayedToken, T]:
        """Replay the forward that produced ``node_id``'s ``raw_index``-th token.

        ``read`` receives ``{layer: (hidden,) fp32}`` pooled at the producing
        position and runs inside the exclusive-GPU section; ``layers`` may be
        a callable, resolved there too.  ``apply_steering`` reopens the node
        recipe's steering expression (a steered generation reads its steered
        state): an always-active affine term reproduces the decode
        injections exactly, and phase-split or probe-gated terms fire as the
        trigger context's rest state decides.  ``raw`` selects the flat
        render, which the node does not record.  Raises ``UnknownNodeError``
        / ``InvalidNodeOperationError`` on a bad anchor and
        ``ConcurrentExtractionError`` while another model op is in flight;
        ``label`` prefixes those messages.
        """
        anchor = resolve_token_anchor(self._tree(), node_id, raw_index, label=label)
        recipe = anchor.node.recipe
        steering = (
            recipe.steering if (apply_steering and recipe is not None) else None
        )
        # The steering scope opens OUTSIDE the exclusive-GPU section: the
        # composer's push/pop take the generation lock blocking, so nesting
        # the scope inside the exclusive guard would self-deadlock.
        scope = self._open_scope(steering) if steering else nullcontext()
        with scope:
            ids = self._producing_ids(anchor, raw=raw)
            with self._exclusive(
                f"{label} called while another model use is in flight",
                phase_msg=f"{label} called while a generation is in flight",
            ):
                layer_indices = list(layers() if callable(layers) else layers)
                hidden = self._capture(
                    ids, layer_indices=layer_indices, pool_index=ids.shape[1] - 1,
                )
                result = read(hidden)
        return (
            ReplayedToken(
                node_id=node_id,
                raw_index=anchor.raw_index,
                token_id=anchor.token_id,
                token_text=str(self._tokenizer.decode([anchor.token_id])),
                steering=steering,
            ),
            result,
        )

    def _producing_ids(self, anchor: TokenAnchor, *, raw: bool) -> torch.Tensor:
        """The token stream the producing forward saw: the node's prompt
        render plus the raw decode prefix before the token."""
        node = anchor.node
        if raw:
            prompt_ids = self._renderer.prepare_input(
                "", raw=True, parent_node_id=node.parent_id,
            )
        else:
            # Continue-mode rebuild: the history walk to the parent carries
            # every prior turn with its stamped label, so the render is
            # byte-identical to the one that produced the node, for any seat.
            recipe = node.recipe
            thinking = recipe.thinking if recipe is not None else None
            supports_thinking = self._traits.supports_thinking
            prompt_ids = self._renderer.prepare_input(
                None,
                thinking=(
                    supports_thinking if thinking is None
                    else bool(thinking) and supports_thinking
                ),
                parent_node_id=node.parent_id,
                system_prompt=(
                    recipe.system_prompt if recipe is not None
                    else INHERIT_SYSTEM_PROMPT
                ),
                user_role=node.role_label if node.role == "user" else None,
                assistant_role=(
                    node.role_label if node.role == "assistant" else None
                ),
                gen_seat=node.role,
            )
        if anchor.raw_index == 0:
            return prompt_ids
        prefix = torch.tensor(
            [anchor.prefix], dtype=prompt_ids.dtype, device=prompt_ids.device,
        )
        return torch.cat([prompt_ids, prefix], dim=1)


__all__ = ["TokenReplayer"]
