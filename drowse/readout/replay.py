"""The token-replay contract the read instruments are built against.

A token replay rebuilds the forward that produced one loom token and hands
the captured hidden states to a family read.  The workflow itself (the loom
anchor, the node's render, the recipe steering, the exclusive capture) sits
above the instruments in the layer order, in
:class:`drowse.loom.replay.TokenReplayer`; each instrument takes it as an
explicit :class:`TokenReplay` dependency and supplies only its family read,
so the three families share one rebuild and differ only in what they read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch

T = TypeVar("T")

#: The layers a replay captures: a fixed set, or a zero-argument callable
#: resolved inside the exclusive section (a roster read there sees the same
#: roster the family read scores).
ReplayLayers = Iterable[int] | Callable[[], Iterable[int]]


@dataclass(frozen=True)
class ReplayedToken:
    """The token a replay rebuilt, and the steering it ran under."""

    node_id: str
    raw_index: int
    token_id: int
    token_text: str
    #: The recipe steering expression the replay reopened, or ``None`` for an
    #: unsteered replay (``apply_steering=False``, or a node with none).
    steering: str | None


class TokenReplay(Protocol):
    """Replay the forward that produced ``node_id``'s ``raw_index``-th token.

    ``read`` receives the hidden states captured at ``layers``, pooled at the
    producing position, inside the exclusive-GPU section, and its result is
    returned beside the :class:`ReplayedToken`.  ``apply_steering`` reopens
    the node recipe's steering; ``raw`` selects the flat render.  ``label``
    names the caller in error messages.
    """

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
    ) -> tuple[ReplayedToken, T]: ...


__all__ = ["ReplayLayers", "ReplayedToken", "TokenReplay"]
