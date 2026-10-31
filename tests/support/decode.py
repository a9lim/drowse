"""Fake decode-loop models and tokenizers (no checkpoint, no session).

- :class:`FakeLogitsModel` — an HF-causal-LM-shaped stub whose forward
  returns ``.logits`` from a callable; :func:`make_logits_model` builds one.
- :func:`scripted_model` — a decode stub forcing a fixed token sequence
  (:class:`NoCacheModel` withholds its KV cache).
- :class:`StopTokenizer` — a four-piece tokenizer spelling ``"Hello STOP"``.
- :class:`CharTokenizer` — code-point ids for scoring and naturalness tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import torch

__all__ = [
    "CharTokenizer",
    "FakeLogitsModel",
    "NoCacheModel",
    "StopTokenizer",
    "make_logits_model",
    "scripted_logits",
    "scripted_model",
]


class FakeLogitsModel:
    """Minimal HF-causal-LM-like stub returning ``.logits`` from a callable.

    The forward (``__call__``) accepts the kwargs the engine passes —
    ``input_ids`` and ``use_cache`` (both ignored beyond ``input_ids``,
    which is handed to ``logits_fn``) — and wraps ``logits_fn``'s output in
    a ``SimpleNamespace`` so consumers can read ``out.logits``.  ``logits_fn``
    is the single point of variation: a constant log-uniform distribution, a
    seeded-random one, a scripted argmax, etc.

    Args:
        logits_fn: maps an ``input_ids`` tensor ``(1, T)`` to a logits tensor;
            return ``(1, T, V)`` for per-position consumers (trajectory /
            joint-logprob reads) or ``(1, 1, V)`` for single-step decode.
        with_parameters: when ``True`` (the default) expose ``parameters()``
            yielding one throwaway CPU param so
            ``next(model.parameters()).device`` works.  Set ``False`` to model
            a stub whose consumer never inspects the device.
        with_past_key_values: when ``True`` set ``out.past_key_values`` to a
            sentinel object (the decode loop only checks it for ``None``); when
            ``False`` leave it absent.
        config / generation_config: optional ``SimpleNamespace`` attributes some
            consumers read (e.g. ``config.vocab_size``,
            ``generation_config.eos_token_id``).
    """

    def __init__(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        with_parameters: bool = True,
        with_past_key_values: bool = False,
        config: SimpleNamespace | None = None,
        generation_config: SimpleNamespace | None = None,
    ) -> None:
        self._logits_fn = logits_fn
        self._with_past_key_values = with_past_key_values
        # One throwaway parameter so ``next(model.parameters()).device`` works;
        # the value is irrelevant, only its (CPU) device matters.
        self._param = torch.zeros(1, requires_grad=False) if with_parameters else None
        if config is not None:
            self.config = config
        if generation_config is not None:
            self.generation_config = generation_config

    def parameters(self) -> Any:
        if self._param is None:
            raise AttributeError("FakeLogitsModel built with with_parameters=False")
        yield self._param

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        use_cache: bool | None = None,
        **_kwargs: Any,
    ) -> Any:
        del use_cache  # stub is stateless; cache flag is ignored
        assert input_ids is not None, "FakeLogitsModel needs input_ids"
        logits = self._logits_fn(input_ids)
        out = SimpleNamespace(logits=logits)
        if self._with_past_key_values:
            out.past_key_values = object()
        return out


def make_logits_model(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    **kwargs: Any,
) -> FakeLogitsModel:
    """Thin factory for :class:`FakeLogitsModel` (reads well at call sites)."""
    return FakeLogitsModel(logits_fn, **kwargs)


class CharTokenizer:
    """Deterministic char-code tokenizer — encode text to its code points.

    The lightest tokenizer-ish helper the suite reaches for: a string becomes
    a ``(1, T)`` long tensor of ``ord(c) % mod`` ids (offset by 1 so id 0 stays
    free for a BOS fallback), and ``decode`` is best-effort.  Callable form
    matches the HF ``tokenizer(text, return_tensors="pt")`` surface; ``encode``
    is the bare-list/tensor sibling.

    Args:
        mod: modulus bounding the id range (so a small ``vocab`` model stays in
            range).  Defaults to a value comfortably above ASCII.
    """

    bos_token_id = 0

    def __init__(self, mod: int = 256) -> None:
        self._mod = mod

    def _ids(self, text: str) -> list[int]:
        return [(ord(c) % self._mod) + 1 for c in text] or [self.bos_token_id]

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens  # always returns a pt tensor
        return {"input_ids": torch.tensor([self._ids(text)], dtype=torch.long)}

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        ids = self._ids(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


class StopTokenizer:
    name_or_path = "stop-tokenizer"
    vocab_size = 4
    eos_token_id = 3
    added_tokens_encoder = {}
    all_special_ids = [3]

    _pieces = {
        0: "Hello",
        1: " STOP",
        2: " ignored",
        3: "",
    }

    def batch_decode(self, ids: Any) -> list[str]:
        return [self._pieces[row[0]] for row in ids]

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        pieces = []
        for tid in ids:
            if skip_special_tokens and tid in self.all_special_ids:
                continue
            pieces.append(self._pieces[int(tid)])
        return "".join(pieces)


def scripted_logits(
    tokens: list[int], vocab_size: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A scripted-argmax ``logits_fn``: forward *i* forces ``tokens[i]``
    (clamped at the last entry), ignoring ``input_ids`` entirely."""
    step = 0

    def _logits(_input_ids: torch.Tensor) -> torch.Tensor:
        nonlocal step
        tid = tokens[min(step, len(tokens) - 1)]
        step += 1
        out = torch.full((1, 1, vocab_size), -100.0)
        out[0, 0, tid] = 100.0
        return out

    return _logits


class NoCacheModel(FakeLogitsModel):
    """The scripted model with its KV cache withheld — the modeling files
    that ignore ``past_key_values`` and drive the O(N²) fallback."""

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        out = super().__call__(input_ids, use_cache=use_cache, **kwargs)
        out.past_key_values = None
        return out


def scripted_model(
    tokens: list[int],
    *,
    vocab_size: int = 4,
    eos_token_id: int = 3,
    model_cls: type[FakeLogitsModel] = FakeLogitsModel,
) -> Any:
    """A decode stub forcing a fixed token sequence, with a fresh KV-cache
    sentinel per forward.  ``model_cls=NoCacheModel`` withholds the cache."""
    return model_cls(
        scripted_logits(tokens, vocab_size),
        with_past_key_values=True,
        config=SimpleNamespace(vocab_size=vocab_size),
        generation_config=SimpleNamespace(eos_token_id=eos_token_id),
    )
