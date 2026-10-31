"""A reusable prompt-prefix KV cache: prefill once, reuse, roll back.

:class:`PrefixCache` is the session's one slot: :meth:`PrefixCache.build`
prefills a prefix, :meth:`PrefixCache.hit` matches a generation's rendered
input against it, and :meth:`PrefixCache.release` rewinds it after the
generation that reused it.  Underneath, :meth:`PrefixCacheEntry.from_prefill`
wraps the cache a prefix prefill just produced, :meth:`PrefixCacheEntry.suffix_for`
holds the hit rules, and :meth:`PrefixCacheEntry.rollback` rewinds the cache to
the prefix.  Rollback is per cache-layer kind (transformers 5.x):

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

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

_log = logging.getLogger(__name__)

#: Floor on the shared token prefix a batch will KV-cache.  Below this the
#: saved prefill is in the noise and the double-render + cache-management
#: overhead isn't worth it; the prefix must also leave a non-empty suffix per
#: row (:meth:`PrefixCacheEntry.suffix_for` requires it), so the cached length
#: is capped at ``min_row_len - 1``.
PREFIX_CACHE_MIN_TOKENS = 16

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


class PrefixHit(NamedTuple):
    """A generation's view of a prefix-cache hit."""

    suffix_ids: torch.Tensor
    past_key_values: Any
    prefix_len: int
    static: bool


class PrefixCache:
    """The session's one prompt-prefix KV slot: build, hit, release.

    Opt-in and library-only: :meth:`build` prefills a prefix (``[1, T]``
    token ids) with no hooks of its own, a later generation whose rendered
    input starts with exactly those tokens decodes only the suffix on top of
    it, and :meth:`release` rewinds the cache afterwards.  The cached KV is a
    pure function of the prefix tokens (the builder refuses to run inside a
    steering scope and with capture attached), so the only invalidation that
    matters is steering that touches the prefill at consume time, which the
    caller checks per generation.
    """

    def __init__(self) -> None:
        self._entry: PrefixCacheEntry | None = None

    @property
    def entry(self) -> PrefixCacheEntry | None:
        return self._entry

    def __bool__(self) -> bool:
        return self._entry is not None

    def clear(self) -> None:
        """Drop the cached prefix (its KV is freed with the last reference)."""
        self._entry = None

    def build(
        self,
        model: Any,
        prefix_ids: torch.Tensor,
        *,
        static_max_cache_len: int | None = None,
        make_static: Any = None,
    ) -> int:
        """Prefill ``prefix_ids`` and keep the resulting cache; returns its length.

        ``static_max_cache_len`` asks for a StaticCache-backed entry sized to
        that many positions, built by ``make_static(max_cache_len)``; when the
        construction or the static prefill fails the prefix falls back to an
        ordinary DynamicCache entry.  Returns ``0`` (and keeps nothing) for an
        empty prefix or a model whose forward returns no cache.
        """
        self._entry = None
        prefix_len = int(prefix_ids.shape[1])
        if prefix_len == 0:
            return 0
        attention_mask = torch.ones_like(prefix_ids)
        static_cache: Any = None
        if static_max_cache_len is not None and make_static is not None:
            try:
                static_cache = make_static(static_max_cache_len)
            except Exception as exc:
                _log.debug("prefix cache: StaticCache skipped (%s)", exc)
        outputs: Any = None
        if static_cache is not None:
            try:
                with torch.inference_mode():
                    outputs = model(
                        input_ids=prefix_ids,
                        attention_mask=attention_mask,
                        past_key_values=static_cache,
                        use_cache=True,
                        cache_position=torch.arange(
                            prefix_len, device=prefix_ids.device, dtype=torch.long,
                        ),
                    )
            except Exception:
                _log.debug(
                    "prefix cache: StaticCache prefill failed; retrying DynamicCache",
                    exc_info=True,
                )
                static_cache = None
        if static_cache is not None:
            # StaticCache mutates in place.  Some modeling files return the
            # same object, others ``None``; keep the preallocated cache either way.
            returned = outputs.past_key_values
            past_key_values = returned if returned is not None else static_cache
        else:
            with torch.inference_mode():
                outputs = model(
                    input_ids=prefix_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
        if past_key_values is None:
            # A modeling file that ignores ``use_cache``: nothing to reuse.
            return 0
        self._entry = PrefixCacheEntry.from_prefill(
            prefix_ids,
            past_key_values,
            static=static_cache is not None,
            max_cache_len=static_max_cache_len if static_cache is not None else None,
        )
        return prefix_len

    def hit(
        self,
        input_ids: torch.Tensor,
        *,
        static_eligible: bool,
        required_max_new_tokens: int | None,
    ) -> PrefixHit | None:
        """The suffix to decode on top of the cached prefix, or ``None``.

        The hit rules live in :meth:`PrefixCacheEntry.suffix_for`.
        """
        entry = self._entry
        if entry is None:
            return None
        suffix_ids = entry.suffix_for(
            input_ids,
            static_eligible=static_eligible,
            required_max_new_tokens=required_max_new_tokens,
        )
        if suffix_ids is None:
            return None
        return PrefixHit(
            suffix_ids, entry.past_key_values, entry.prefix_len, entry.static,
        )

    def release(self, hit: PrefixHit) -> None:
        """Rewind the cache a generation reused; drop the entry if it can't be."""
        entry = self._entry
        if (
            entry is not None
            and entry.past_key_values is hit.past_key_values
            and not entry.rollback()
        ):
            self._entry = None


__all__ = [
    "PREFIX_CACHE_MIN_TOKENS",
    "PrefixCache",
    "PrefixCacheEntry",
    "PrefixHit",
]
