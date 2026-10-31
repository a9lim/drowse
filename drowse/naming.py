"""Concept naming rules — the on-disk slug, the bipolar-pole join, reserved namespaces.

Pure string rules, no model or IO, at the bottom of the layer order so the
store, the steering grammar and the session share one definition
(:func:`canonical_concept_name`, :data:`RESERVED_NAMESPACES`).
"""
from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")
BIPOLAR_SEP = "."

#: Instrument namespaces: atoms under them name instrument units (J-/S-lens
#: tokens, SAE features, neurons, heads), not fitted concepts.  Their names
#: skip slug canonicalization and take no variant, and no manifold may be
#: authored under them.
RESERVED_NAMESPACES = frozenset({"jlens", "slens", "sae", "neuron", "head"})


def _slug(s: str) -> str:
    """Normalize a single pole label to `[a-z0-9_]`.

    Collapses any non-alphanumeric run to `_`. Never produces the bipolar
    separator `.` — that is reserved for joining two slugged poles in
    `canonical_concept_name`.
    """
    return _SLUG_RE.sub("_", s.strip().lower()).strip("_")


def canonical_concept_name(concept: str, baseline: str | None = None) -> str:
    """Return the canonical on-disk name for a concept.

    Monopolar: `_slug(concept)`.
    Bipolar:   `f"{_slug(concept)}.{_slug(baseline)}"`.

    If `baseline` is None and `concept` already contains the bipolar
    separator `.`, the input is treated as a pre-composed bipolar name
    and each side is slugged independently. This makes `/steer happy.sad`
    and `/steer happy - sad` resolve to the same cache entry.
    """
    if baseline is None:
        if BIPOLAR_SEP in concept:
            pos, neg = concept.split(BIPOLAR_SEP, 1)
            return f"{_slug(pos)}{BIPOLAR_SEP}{_slug(neg)}"
        return _slug(concept)
    return f"{_slug(concept)}{BIPOLAR_SEP}{_slug(baseline)}"
