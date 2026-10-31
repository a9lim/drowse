"""Token-level probe payload shaping for generation callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

from drowse.readout.types import (
    InstrumentFamily,
    Reading,
    ScalarReading,
    as_probe_reading,
    reading_axis0,
    reading_per_layer_axis0,
)
from drowse.readout.measurements import build_measurements
from drowse.readout.types import ProbeReading

#: Per-family reading slot names, keyed by the ``family`` argument to
#: :meth:`TokenProbePayload.merge_readings`.  One slot per member of
#: ``InstrumentFamily`` — ``tests/readout/test_instruments_types.py`` pins the two
#: sets equal against ``session.instruments``, so a fourth read family
#: cannot land a slot-less payload.
_FAMILY_SLOTS: dict[InstrumentFamily, str] = {
    "geometry": "geometry_readings",
    "lens": "lens_readings",
    "sae": "sae_readings",
}


@dataclass(slots=True)
class TokenProbePayload:
    """Probe-derived payloads for one generated token.

    Readings are kept in per-family slots (``geometry_readings`` /
    ``lens_readings`` / ``sae_readings``) so the 5.x measurement envelope can
    split them into ``instruments.geometry`` / ``.lens`` / ``.sae`` while the
    flat ``scores`` / ``per_layer_scores`` cross-family views stay derived from
    all three.
    """

    scores: dict[str, float] | None = None
    per_layer_scores: dict[str, dict[str, float]] | None = None
    geometry_readings: dict[str, ProbeReading] | None = None
    lens_readings: dict[str, ScalarReading] | None = None
    sae_readings: dict[str, ScalarReading] | None = None

    @classmethod
    def from_geometry(
        cls, readings: dict[str, ProbeReading], *, per_layer: bool,
    ) -> "TokenProbePayload":
        """One token's monitor readings, shaped into the geometry slot.

        ``per_layer`` adds the per-layer axis-0 rows (the loom heatmap row).
        """
        return cls(
            scores=_axis0_scores(readings),
            per_layer_scores=_per_layer_axis0(readings) if per_layer else None,
            geometry_readings=readings,
        )

    @property
    def has_readings(self) -> bool:
        """Whether any family slot is populated — the cheap guard.

        Distinct from :attr:`all_readings` on purpose: the per-token payload
        gate must not pay for the cross-family ``ProbeReading`` projection
        just to answer "is there anything here".
        """
        return bool(
            self.geometry_readings or self.lens_readings or self.sae_readings
        )

    @property
    def all_readings(self) -> dict[str, ProbeReading]:
        """The three family slots merged into one ``ProbeReading`` dict.

        A COMPATIBILITY view, and the only one on this path: the library
        ``TokenEvent.probe_readings`` field keys probes across families by a
        single reading type, so the single-axis families' native
        ``ScalarReading``s are projected here.  The native WS path never
        touches it — ``to_token_payload`` serializes each family's own
        shape — so the projection does not run per token unless a caller
        asked for the cross-family dict.
        """
        merged: dict[str, ProbeReading] = {}
        if self.geometry_readings:
            merged.update(self.geometry_readings)
        for slot in (self.lens_readings, self.sae_readings):
            if slot:
                merged.update(
                    {n: as_probe_reading(r) for n, r in slot.items()}
                )
        return merged

    def to_token_payload(
        self,
        *,
        lens: Any = None,
        lens_aggregate: Any = None,
        lens_token_ids: Any = None,
        lens_source: str | None = None,
        sae: Any = None,
        sae_source: str | None = None,
        sae_layer: int | None = None,
        steering: str | None = None,
    ) -> dict[str, Any]:
        """Shape this token's payload as ``{"measurements": <envelope>}``.

        The measurement envelope is the whole wire record for a token — every
        cross-family scalar view is derived inside it, so there are no
        top-level aliases beside it.  ``lens`` is the native per-layer readout
        (``lens_readout`` in the envelope) and ``sae`` the native feature
        discovery (``sae_features``); the attached per-family readings come
        from the payload's own slots.
        """
        measurements = build_measurements(
            scope="token",
            geometry_readings=self.geometry_readings,
            lens_readings=self.lens_readings,
            sae_readings=self.sae_readings,
            per_layer_scores=self.per_layer_scores,
            lens_readout=lens,
            lens_aggregate=lens_aggregate,
            lens_token_ids=lens_token_ids,
            lens_source=lens_source,
            sae_features=sae,
            sae_source=sae_source,
            sae_layer=sae_layer,
            steering=steering,
        )
        return {"measurements": measurements}

    def merge_readings(
        self,
        extra: dict[str, Any],
        *,
        family: str,
        per_layer: bool = False,
    ) -> None:
        """Merge additional per-probe readings into a named family's slot.

        ``family`` is one of ``"geometry"`` / ``"lens"`` / ``"sae"`` — J-lens
        token probes and SAE feature probes score on their own paths rather
        than through the Monitor, so their readings land in their own slot.
        The flat ``scores`` (and, when ``per_layer``, ``per_layer_scores``)
        cross-family views are updated the same way regardless of family.
        """
        if not extra:
            return
        try:
            slot_name = _FAMILY_SLOTS[cast(InstrumentFamily, family)]
        except KeyError:
            raise ValueError(f"unknown reading family {family!r}") from None
        self.scores = {**(self.scores or {}), **_axis0_scores(extra)}
        current: dict[str, ProbeReading] | None = getattr(self, slot_name)
        setattr(self, slot_name, {**(current or {}), **extra})
        if per_layer:
            merged = dict(self.per_layer_scores or {})
            for layer, row in (_per_layer_axis0(extra) or {}).items():
                merged[layer] = {**merged.get(layer, {}), **row}
            self.per_layer_scores = merged or None


def _axis0_scores(readings: Mapping[str, Reading]) -> dict[str, float]:
    return {
        name: reading_axis0(reading) for name, reading in readings.items()
    }


def _per_layer_axis0(
    readings: Mapping[str, Reading],
) -> dict[str, dict[str, float]] | None:
    by_layer: dict[str, dict[str, float]] = {}
    for probe_name, reading in readings.items():
        for layer, value in reading_per_layer_axis0(reading).items():
            by_layer.setdefault(str(layer), {})[probe_name] = round(value, 6)
    return by_layer or None
