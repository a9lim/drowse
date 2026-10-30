"""A small bounded LRU map, the one cache helper for derived views.

The readout runtimes keep a few per-session caches of derived values (device
transport stacks, selector tensors, decoded tokens).  Each is bounded and
evicts its least recently used entry, and each is keyed by value, so a stale
entry is unreachable rather than wrong.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BoundedLRU(Generic[K, V]):
    """A mapping capped at ``maxsize`` entries, evicting least recently used."""

    __slots__ = ("maxsize", "_data")

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self.maxsize = int(maxsize)
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        """The cached value (marked most recently used), or ``None``."""
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> V:
        """Insert ``value`` as most recently used, evicting past ``maxsize``."""
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return value

    def get_or_build(self, key: K, build: Callable[[], V]) -> V:
        """The cached value, building and inserting it on a miss."""
        value = self.get(key)
        if value is not None:
            return value
        return self.put(key, build())

    def clear(self) -> None:
        self._data.clear()

    def keys(self) -> list[K]:
        """Keys from least to most recently used."""
        return list(self._data)

    def items(self) -> list[tuple[K, V]]:
        """Entries from least to most recently used (no recency update)."""
        return list(self._data.items())

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return iter(list(self._data))


__all__ = ["BoundedLRU"]
