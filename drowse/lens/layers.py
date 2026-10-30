"""Layer selectors: the one parser and resolver for ``--layers`` / ``layers=``.

A selector is ``None`` or ``"all"`` (every available layer), ``"workspace"``
(the 40-90% depth band), ``"sample"`` (nine evenly spaced layers, a readout
convenience that fits refuse), or an explicit index list.  The CLI, the
server and the lens runtime all parse and resolve through here.

Torch-free at import: the CLI parses selectors before any model loads.
"""

from __future__ import annotations

from collections.abc import Sequence

#: The named selector modes.
LAYER_MODES = ("all", "workspace", "sample")
#: How many layers ``"sample"`` picks.
SAMPLE_COUNT = 9

LayerSelector = Sequence[int] | str | None


def parse_layer_selector(text: str | None) -> list[int] | str | None:
    """``"3,7,11"`` -> ``[3, 7, 11]``; a mode passes through lower-cased.

    ``None`` or blank text is ``None`` (the caller's default).  Raises
    ``ValueError`` on anything else.
    """
    if text is None or not text.strip():
        return None
    lowered = text.strip().lower()
    if lowered in LAYER_MODES:
        return lowered
    try:
        layers = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(
            f"malformed layers {text!r} (want a comma-separated index list, "
            "all, workspace, or sample)"
        ) from None
    if not layers:
        raise ValueError("layers must name at least one layer")
    return layers


def resolve_layer_selector(
    selector: LayerSelector,
    available: Sequence[int],
    *,
    n_layers: int,
    allow_sample: bool = True,
) -> list[int]:
    """Resolve ``selector`` against the ``available`` layers.

    ``available`` is what the selection draws from (a lens's fitted layers,
    or every source layer of a fit), ``n_layers`` the model depth that the
    workspace band is measured against.  An explicit list is returned as
    given, in order; callers that need a canonical set sort it themselves.
    """
    pool = [int(layer) for layer in available]
    if selector is None or selector == "all":
        return pool
    if isinstance(selector, str):
        mode = selector.lower()
        if mode == "workspace":
            from drowse.model.arch import workspace_layer_indices

            return workspace_layer_indices(pool, n_layers)
        if mode == "sample" and allow_sample:
            if len(pool) <= SAMPLE_COUNT:
                return pool
            step = (len(pool) - 1) / (SAMPLE_COUNT - 1)
            return sorted({pool[round(i * step)] for i in range(SAMPLE_COUNT)})
        allowed = ", ".join(m for m in LAYER_MODES if allow_sample or m != "sample")
        raise ValueError(
            f"unknown layer mode {selector!r} (want {allowed}, or an index list)"
        )
    return [int(layer) for layer in selector]


__all__ = [
    "LAYER_MODES",
    "LayerSelector",
    "SAMPLE_COUNT",
    "parse_layer_selector",
    "resolve_layer_selector",
]
