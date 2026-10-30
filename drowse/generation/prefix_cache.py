"""A reusable prompt-prefix KV cache: prefill once, reuse, roll back.

:meth:`PrefixCacheEntry.from_prefill` wraps the cache a prefix prefill just
produced, :meth:`PrefixCacheEntry.suffix_for` matches a generation's rendered
input against it, and :meth:`PrefixCacheEntry.rollback` rewinds the cache to
the prefix after the generation that reused it.  Rollback is per cache-layer
kind (transformers 5.x):

* **Full attention** (``DynamicLayer``): ``crop(-n)`` drops the decode tail.
* **Sliding window** and **linear-attention conv states** keep only their
  window, so rewinding past it needs the tail they normally discard.  A
  dynamic entry arms upstream ``activate_past_recording`` right after the
  prefill (never before it — recording a long prefill is the memory spike
  upstream warns about); ``crop(-n)`` then rewinds them and trims them back to
  their working size.
* **Linear-attention recurrent states** fold every token into one state no
  crop can rewind, so the entry snapshots them after the prefill and restores
  them in place.
* **StaticCache** layers write at their own ``cumulative_length`` counter;
  rewinding it to the prefix is the rollback (stale K/V past it is masked by
  the causal mask and overwritten by the next write).  A static entry never
  records past states — that would reassign the buffers whose addresses a
  compiled graph captured — so its linear-attention conv states are
  snapshotted alongside the recurrent ones.

Everything is applied to the one cache object the entry owns; no transformers
class is patched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# (layer, "conv_states" | "recurrent_states", state index, saved tensor)
_StateSnapshot = tuple[Any, str, int, torch.Tensor]


def _snapshot_states(cache: Any, kinds: tuple[str, ...]) -> tuple[_StateSnapshot, ...]:
    """Clone the named linear-attention states of every initialized layer."""
    saved: list[_StateSnapshot] = []
    for layer in getattr(cache, "layers", ()):
        for kind in kinds:
            states = getattr(layer, kind, None)
            initialized = getattr(layer, f"is_{kind}_initialized", None)
            if not isinstance(states, dict) or not isinstance(initialized, dict):
                continue
            for idx, ready in initialized.items():
                if ready and states.get(idx) is not None:
                    saved.append((layer, kind, idx, states[idx].detach().clone()))
    return tuple(saved)


def _restore_states(snapshot: tuple[_StateSnapshot, ...]) -> None:
    # The states were created inside the prefill's ``inference_mode``; an
    # in-place copy into an inference tensor is only legal inside it.
    with torch.inference_mode():
        for layer, kind, idx, saved in snapshot:
            getattr(layer, kind)[idx].copy_(saved)


def _rewind_static(cache: Any, prefix_len: int) -> bool:
    """Reset every static layer's write counter to ``prefix_len``."""
    with torch.inference_mode():
        for layer in getattr(cache, "layers", ()):
            counter = getattr(layer, "cumulative_length", None)
            if not isinstance(counter, torch.Tensor):
                continue
            filled = getattr(layer, "cumulative_length_int", None)
            if filled is not None:
                # A static sliding layer that filled its ring has rolled the
                # prefix out of the buffer; nothing can bring it back.
                if filled > int(getattr(layer, "max_cache_len", filled)):
                    return False
                layer.cumulative_length_int = prefix_len
            counter.fill_(prefix_len)
    return True


@dataclass(slots=True)
class PrefixCacheEntry:
    """A prefilled prefix cache plus what it takes to rewind it after reuse."""

    prefix_ids_cpu: torch.Tensor
    past_key_values: Any
    prefix_len: int
    static: bool = False
    max_cache_len: int | None = None
    state_snapshot: tuple[_StateSnapshot, ...] = ()

    @classmethod
    def from_prefill(
        cls,
        prefix_ids: torch.Tensor,
        past_key_values: Any,
        *,
        static: bool = False,
        max_cache_len: int | None = None,
    ) -> "PrefixCacheEntry":
        """Arm rollback on the cache the ``[1, T]`` prefix prefill just filled."""
        if static:
            snapshot = _snapshot_states(
                past_key_values, ("conv_states", "recurrent_states"),
            )
        else:
            activate = getattr(past_key_values, "activate_past_recording", None)
            if callable(activate):
                activate()
            snapshot = _snapshot_states(past_key_values, ("recurrent_states",))
        return cls(
            prefix_ids_cpu=prefix_ids[0].detach().to("cpu"),
            past_key_values=past_key_values,
            prefix_len=int(prefix_ids.shape[1]),
            static=static,
            max_cache_len=max_cache_len,
            state_snapshot=snapshot,
        )

    def suffix_for(
        self,
        input_ids: torch.Tensor,
        *,
        static_eligible: bool,
        required_max_new_tokens: int | None,
    ) -> torch.Tensor | None:
        """The ``[1, S]`` suffix to decode on top of this prefix, or ``None``.

        A hit needs the cached prefix tokens to match ``input_ids`` exactly and
        a non-empty suffix (the first decode step samples from the suffix's
        last logit).  A static entry additionally needs a StaticCache-eligible
        generation that fits the preallocated ``max_cache_len``.
        """
        prefix_len = self.prefix_len
        if input_ids.shape[1] <= prefix_len:
            return None
        head = input_ids[0, :prefix_len].detach().to("cpu")
        if not torch.equal(head, self.prefix_ids_cpu):
            return None
        suffix_ids = input_ids[:, prefix_len:].contiguous()
        if self.static:
            if not static_eligible:
                return None
            if self.max_cache_len is not None:
                required = (
                    prefix_len
                    + int(suffix_ids.shape[1])
                    + max(int(required_max_new_tokens or 0), 1)
                )
                if required > self.max_cache_len:
                    return None
        return suffix_ids

    def rollback(self) -> bool:
        """Rewind the cache to the prefix; ``False`` means drop the entry."""
        cache = self.past_key_values
        try:
            if self.static:
                if not _rewind_static(cache, self.prefix_len):
                    return False
            else:
                extra = int(cache.get_seq_length()) - self.prefix_len
                if extra < 0:
                    return False
                # ``crop(0)`` still trims recorded conv / sliding tails back to
                # their working size, so it runs even with nothing to drop.
                cache.crop(-extra)
            _restore_states(self.state_snapshot)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        return True
