"""The probe roster: one attach/detach surface over the three read families.

A selector routes by namespace: ``jlens/<word>`` to the J-lens instrument (a
readout-channel probe, never a whitened coordinate), ``sae/<id>`` to the SAE
instrument, and everything else to the geometry instrument (a manifold probe
on the unified Monitor).  The roster also owns the transcript probe-hash
caches, one per family, and the single place a roster change is reported:
:meth:`ProbeRoster.changed` drops the affected digests and fires the
``on_change`` callback (the session wires its analytics invalidation there).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

from drowse.readout.types import InstrumentFamily


class _Family(Protocol):
    """The registry surface every family shares."""

    @property
    def names(self) -> list[str]: ...

    def specs(self) -> dict[str, dict[str, Any]]: ...

    def try_detach(self, name: str) -> bool: ...

    def probe_hash(self, name: str) -> str | None: ...


class ReadoutFamily(_Family, Protocol):
    """A J-lens or SAE instrument: attach takes no geometry knobs."""

    def attach(self, selector: str, *, as_name: str | None = None) -> str: ...


class GeometryFamily(_Family, Protocol):
    """The geometry instrument: manifold probes on the Monitor."""

    def attach(
        self, selector: str, *, as_name: str | None = None, top_n: int = 3,
    ) -> str: ...

    def detach(self, name: str) -> None: ...

    def manifolds(self) -> dict[str, Any]: ...

    def refresh_swapped(self, swaps: Mapping[int, Any]) -> list[str]: ...


class ProbeRoster:
    """Routes probe attach/detach by namespace; caches probe digests.

    ``on_change`` runs after every roster mutation this object makes (and
    after :meth:`changed`), once the affected digests are dropped.
    """

    def __init__(
        self,
        *,
        geometry: GeometryFamily,
        lens: ReadoutFamily,
        sae: ReadoutFamily,
        on_change: Callable[[], None],
    ) -> None:
        self._geometry = geometry
        self._lens = lens
        self._sae = sae
        self._on_change = on_change
        # Transcript digests per family, filled lazily by ``probe_hash``.
        self._hashes: dict[InstrumentFamily, dict[str, str]] = {
            "lens": {}, "sae": {}, "geometry": {},
        }

    # -------------------------------------------------------------- routing

    def attach(
        self, selector: str, *, as_name: str | None = None, top_n: int = 3,
    ) -> str:
        """Attach a probe of any shape and return its name.

        ``jlens/`` and ``sae/`` selectors read their family's readout channel;
        everything else resolves to a manifold probe (``top_n`` is the
        nearest-node list length, geometry only).
        """
        if selector.startswith("jlens/"):
            name = self._lens.attach(selector, as_name=as_name)
        elif selector.startswith("sae/"):
            name = self._sae.attach(selector, as_name=as_name)
        else:
            name = self._geometry.attach(selector, as_name=as_name, top_n=top_n)
        self.changed([name])
        return name

    def detach(self, name: str) -> None:
        """Detach ``name`` from whichever family owns it.

        Lens/SAE detaches are atomic under their state locks and safe
        mid-generation (a bound run keeps its frozen roster); a geometry
        detach rejects while a generation runs.
        """
        if not (self._lens.try_detach(name) or self._sae.try_detach(name)):
            self._geometry.detach(name)
        self.changed([name])

    def specs(self) -> dict[str, dict[str, Any]]:
        """Every attached probe: ``{"manifold": …}`` / ``{"lens": spec}`` /
        ``{"sae": spec}``, from each family's locked snapshot."""
        out: dict[str, dict[str, Any]] = {
            name: {"manifold": manifold}
            for name, manifold in self._geometry.manifolds().items()
        }
        out.update({name: {"lens": spec} for name, spec in self._lens.specs().items()})
        out.update({name: {"sae": spec} for name, spec in self._sae.specs().items()})
        return out

    # ------------------------------------------------------------ digests

    def probe_hash(self, name: str) -> str | None:
        """The transcript digest of probe ``name``; ``None`` when unattached.

        Lens and SAE probes hash their readout-channel identity, geometry
        probes their baked tensor bytes.  Each family's digests are cached
        until :meth:`forget` drops them.
        """
        families: tuple[tuple[InstrumentFamily, _Family], ...] = (
            ("lens", self._lens), ("sae", self._sae), ("geometry", self._geometry),
        )
        for family, _ in families:
            cached = self._hashes[family].get(name)
            if cached is not None:
                return cached
        for family, instrument in families:
            digest = instrument.probe_hash(name)
            if digest is not None:
                self._hashes[family][name] = digest
                return digest
        return None

    def probe_hashes(self) -> dict[str, str]:
        """``{probe_name: digest}`` for every attached probe.

        Names come from the families' locked snapshots; a probe detached
        between enumeration and hashing is skipped.
        """
        out: dict[str, str] = {}
        for name in (*self._geometry.names, *self._lens.names, *self._sae.names):
            digest = self.probe_hash(name)
            if digest is not None:
                out[name] = digest
        return out

    # ------------------------------------------------------- invalidation

    def forget(self, names: Iterable[str]) -> None:
        """Drop the cached digests of ``names`` (every family)."""
        names = list(names)
        for cache in self._hashes.values():
            for name in names:
                cache.pop(name, None)

    def changed(self, names: Iterable[str]) -> None:
        """The probes ``names`` changed: drop their digests, run ``on_change``."""
        self.forget(names)
        self._on_change()

    def swap(self, swaps: Mapping[int, Any]) -> None:
        """Move geometry probes off replaced manifold objects (the artifact
        registry's ``on_swap`` payload) and drop their digests."""
        self.forget(self._geometry.refresh_swapped(swaps))

    def clear(self) -> None:
        """Drop every cached digest (session teardown)."""
        for cache in self._hashes.values():
            cache.clear()


__all__ = ["GeometryFamily", "ProbeRoster", "ReadoutFamily"]
