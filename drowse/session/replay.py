"""Loom-anchored token replay: rebuild the forward that produced a token.

The family-neutral half of an instrument's ``token_readout``.  It validates
the ``(node_id, raw_index)`` anchor, reopens the node's recorded steering,
rebuilds the node's render (continue-mode, or the flat raw buffer) plus the
raw decode prefix, and runs one capture forward pooled at the producing
position.  The family's own read runs inside the exclusive-GPU section
through ``read``, so device work never races a generation.

The SAE instrument reads through this (the session wires it in as an explicit
dependency).  The lens replay (``LensInstrument._replay_readout``) and the
geometry replay (``DrowseSession.geometry_token_readout``) still carry their
own copies until the read plane's replays are unified.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from drowse.fit.capture import _capture_all_hidden_states
from drowse.loom.errors import InvalidNodeOperationError
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from drowse.session.session import DrowseSession

T = TypeVar("T")


@dataclass(frozen=True)
class ReplayedToken:
    """The token a replay rebuilt, and the steering it ran under."""

    node_id: str
    raw_index: int
    token_id: int
    token_text: str
    steering: str | None


#: ``replay(node_id, raw_index, *, layers, read, apply_steering, raw, label)``
#: -> ``(ReplayedToken, read(hidden))`` — the instrument-facing signature.
ReplayFn = Callable[..., "tuple[ReplayedToken, Any]"]


def replay_token(
    session: "DrowseSession",
    node_id: str,
    raw_index: int,
    *,
    layers: Sequence[int],
    read: Callable[[dict[int, torch.Tensor]], T],
    apply_steering: bool = True,
    raw: bool = False,
    label: str = "token_readout",
) -> tuple[ReplayedToken, T]:
    """Replay the forward that produced ``node_id``'s ``raw_index``-th token.

    ``read`` receives ``{layer: (hidden,) fp32}`` pooled at the producing
    position and runs inside the exclusive-GPU section.  ``apply_steering``
    reopens the node recipe's steering expression (a steered generation reads
    its steered state); ``raw`` selects the flat render, which the node does
    not record.  Raises ``UnknownNodeError`` / :class:`InvalidNodeOperationError`
    on a bad anchor; ``label`` prefixes those messages.
    """
    node = session.tree.get(node_id)
    if node.role not in ("assistant", "user"):
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} is a {node.role} node — "
            f"only a turn with a decode record can be read out"
        )
    raw_ids = node.raw_token_ids
    if not raw_ids:
        raise InvalidNodeOperationError(
            f"{label}: {node_id!r} has no raw token record"
        )
    if not 0 <= raw_index < len(raw_ids):
        raise InvalidNodeOperationError(
            f"{label}: raw_index {raw_index} out of range [0, {len(raw_ids)})"
        )
    recipe = node.recipe
    steering_expr = (
        recipe.steering if (apply_steering and recipe is not None) else None
    )
    # The steering scope opens OUTSIDE the exclusive-GPU section: the
    # composer's push/pop acquire ``_gen_lock`` blocking (non-reentrant), so
    # nesting the scope inside ``_model_exclusive`` would self-deadlock.
    scope = session.steering(steering_expr) if steering_expr else nullcontext()
    with scope:
        traits = session.chat_traits
        thinking_req = recipe.thinking if recipe is not None else None
        use_thinking = (
            traits.supports_thinking
            if thinking_req is None
            else bool(thinking_req) and traits.supports_thinking
        )
        if raw:
            prompt_ids = session.renderer.prepare_input(
                "", raw=True, parent_node_id=node.parent_id,
            )
        else:
            # Continue-mode rebuild: the history walk to ``node.parent_id``
            # carries every prior turn with its stamped label, so the render
            # is byte-identical to the one that produced the node, for any
            # seat.
            prompt_ids = session.renderer.prepare_input(
                None,
                thinking=use_thinking,
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
        if raw_index > 0:
            prefix = torch.tensor(
                [[int(token) for token in raw_ids[:raw_index]]],
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            ids = torch.cat([prompt_ids, prefix], dim=1)
        else:
            ids = prompt_ids
        with session._model_exclusive(
            f"{label} called while another model use is in flight",
            phase_msg=f"{label} called while a generation is in flight",
        ):
            hidden = _capture_all_hidden_states(
                session.model, session.layers, ids,
                layer_indices=list(layers), pool_index=ids.shape[1] - 1,
            )
            result = read(hidden)
    token_id = int(raw_ids[raw_index])
    return (
        ReplayedToken(
            node_id=node_id,
            raw_index=int(raw_index),
            token_id=token_id,
            token_text=str(session.tokenizer.decode([token_id])),
            steering=steering_expr,
        ),
        result,
    )


__all__ = ["ReplayFn", "ReplayedToken", "replay_token"]
