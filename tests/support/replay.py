"""Token-replay test support.

- :func:`unwired_replay` — the ``replay`` dependency for an instrument a test
  never replays through (it fails loudly if one does).
- :class:`ReplayRig` — a real :class:`~drowse.session.replay.TokenReplayer`
  over a toy model and a real loom tree, with the prompt render, the steering
  scopes, the exclusive section and the capture forward recorded.  Instrument
  tests hand ``rig.replay`` to the instrument under test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch

from tests._jlens_toys import CharTokenizer, frozen_toy

__all__ = ["PROMPT_RENDER", "ReplayRig", "unwired_replay"]

#: What the rig's renderer returns for every prompt render.
PROMPT_RENDER = "the prompt render, chat shaped."


def unwired_replay(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("no token replay wired for this instrument")


class ReplayRig:
    """A real token replayer with its collaborators recorded.

    - ``tree``: the live loom tree (add turns with :meth:`turn`).
    - ``replay``: the :class:`TokenReplayer` under test.
    - ``prepare_calls``: every prompt render's arguments.
    - ``steering_scopes``: every steering expression a scope opened for.
    - ``captures``: ``(n_ids, pool_index, layer_indices)`` per capture forward.
    - ``events``: the ordered scope / render / exclusive / capture events.
    - ``on_exclusive``: run on entering the exclusive section (to stage a
      race the section must win).
    """

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        n_layers: int = 3,
        supports_thinking: bool | None = None,
    ) -> None:
        from drowse.loom.tree import LoomTree
        from drowse.model.chat import ChatTraits
        from drowse.session.replay import TokenReplayer

        self.model = frozen_toy(n_layers=n_layers) if model is None else model
        self.tokenizer = CharTokenizer() if tokenizer is None else tokenizer
        self.layers = self.model.model.layers
        self.tree = LoomTree(model_id="toy/jlens-model")
        # The replayer reads one chat fact: whether the family can think.
        self.traits: Any = (
            ChatTraits(self.tokenizer) if supports_thinking is None
            else SimpleNamespace(supports_thinking=supports_thinking)
        )
        self.prepare_calls: list[dict[str, Any]] = []
        self.steering_scopes: list[str] = []
        self.captures: list[tuple[int, int | None, tuple[int, ...]]] = []
        self.exclusive_messages: list[tuple[str, str | None]] = []
        self.events: list[str] = []
        self.on_exclusive: Callable[[], None] | None = None
        self.replay = TokenReplayer(
            tree=lambda: self.tree,
            renderer=self,
            traits=self.traits,
            tokenizer=self.tokenizer,
            open_scope=self._open_scope,
            exclusive=self._exclusive,
            capture=self._capture,
        )

    @property
    def prompt_len(self) -> int:
        return len(self.tokenizer.encode(PROMPT_RENDER))

    def turn(
        self,
        raw_ids: list[int] | None,
        recipe: Any = None,
        *,
        seat: str = "assistant",
        role_label: str | None = None,
    ) -> str:
        """A finished generated turn under a fresh user turn; returns its id."""
        user_id = self.tree.add_user_turn("a user turn")
        node_id = self.tree.begin_assistant(
            user_id, recipe=recipe, seat=seat, role_label=role_label,
        )
        self.tree.finalize_assistant(
            node_id, text="a generated turn", finish_reason="stop",
            raw_token_ids=raw_ids,
        )
        return node_id

    # -- the replayer's collaborators -----------------------------------------

    def prepare_input(
        self,
        input: Any,
        raw: bool = False,
        thinking: bool = False,
        stateless: bool = False,
        parent_node_id: str | None = None,
        user_role: str | None = None,
        assistant_role: str | None = None,
        to_device: bool = True,
        gen_seat: str = "assistant",
        system_prompt: Any = None,
    ) -> torch.Tensor:
        del stateless, to_device
        self.events.append("render")
        self.prepare_calls.append({
            "input": input, "raw": raw, "thinking": thinking,
            "system_prompt": system_prompt, "parent_node_id": parent_node_id,
            "user_role": user_role, "assistant_role": assistant_role,
            "gen_seat": gen_seat,
        })
        return torch.tensor([self.tokenizer.encode(PROMPT_RENDER)], dtype=torch.long)

    @contextmanager
    def _open_scope(self, expression: str) -> Iterator[None]:
        self.steering_scopes.append(expression)
        self.events.append("scope-open")
        yield
        self.events.append("scope-close")

    @contextmanager
    def _exclusive(self, busy_msg: str, *, phase_msg: str | None = None) -> Iterator[None]:
        self.exclusive_messages.append((busy_msg, phase_msg))
        self.events.append("exclusive-enter")
        if self.on_exclusive is not None:
            self.on_exclusive()
        yield
        self.events.append("exclusive-exit")

    def _capture(self, ids: torch.Tensor, **kwargs: Any) -> dict[int, torch.Tensor]:
        from drowse.fit.capture import _capture_all_hidden_states

        pool = kwargs.get("pool_index")
        self.events.append("capture")
        self.captures.append((
            int(ids.shape[1]),
            int(pool) if pool is not None else None,
            tuple(kwargs.get("layer_indices") or ()),
        ))
        return _capture_all_hidden_states(self.model, self.layers, ids, **kwargs)
