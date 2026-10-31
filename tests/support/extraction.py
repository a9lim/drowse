"""An :class:`~drowse.session.extraction.Extraction` wired to recording fakes.

Unit tests of the extraction glue (namespace resolution, the author-or-reuse
decision, fit/evict wiring) need no model: the pipeline is replaced by a fake
that records the folder it was asked to fit and returns a real folded vector
manifold, and every other dependency is a recorder.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import torch

__all__ = ["ExtractionRig", "folded_vector_manifold", "make_extraction"]


def folded_vector_manifold(name: str = "vector", direction: Any = None) -> Any:
    """A real rank-1 affine manifold (one layer), foldable to a profile."""
    from drowse.geometry.fold import fold_directions_to_subspace
    from tests.support.whitener import synthetic_whitener

    direction = torch.tensor([1.0, 0.0]) if direction is None else direction
    whitener = synthetic_whitener([0], int(direction.numel()))
    return fold_directions_to_subspace(
        name, {0: direction}, whitener.layer_means, label=name, whitener=whitener,
    )


@dataclass
class ExtractionRig:
    """The extraction under test plus what each fake dependency saw."""

    extraction: Any
    events: Any
    fitted: list[Path] = field(default_factory=list)
    fit_kwargs: list[dict[str, Any]] = field(default_factory=list)
    generated: list[list[str]] = field(default_factory=list)
    adopted: list[Path] = field(default_factory=list)
    evicted: list[tuple[str, str, str | None]] = field(default_factory=list)
    registered: list[str] = field(default_factory=list)


class _Corpus:
    def __init__(self, rig: ExtractionRig, on_generate: Callable[[], object] | None) -> None:
        self._rig = rig
        self._on_generate = on_generate

    def responses(self, concepts: list[str], *_args: Any, **_kwargs: Any) -> dict[str, list[str]]:
        self._rig.generated.append(list(concepts))
        if self._on_generate is not None:
            self._on_generate()
        return {concept: [f"{concept} reply"] for concept in concepts}


def make_extraction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fit: Callable[..., Any] | None = None,
    on_generate: Callable[[], object] | None = None,
    adopt: Callable[[Any, Any], None] | None = None,
    evict: Callable[..., None] | None = None,
    register_profile: Callable[[str, Any], None] | None = None,
    exclusive: Callable[..., Any] | None = None,
    handle: Any = None,
    dls: bool = True,
) -> ExtractionRig:
    """Build an ``Extraction`` over fakes; ``fit(folder, **kwargs)`` overrides the pipeline."""
    import drowse.session.extraction as extraction_mod
    from drowse.events import EventBus
    from drowse.session.extraction import Extraction

    events = EventBus()
    rig = ExtractionRig(extraction=None, events=events)
    manifold = folded_vector_manifold()

    class _Pipeline:
        def __init__(self, _handle: Any, _events: Any) -> None:
            pass

        def fit(self, folder: Any, **kwargs: Any) -> Any:
            rig.fitted.append(Path(folder))
            rig.fit_kwargs.append(kwargs)
            return manifold if fit is None else fit(folder, **kwargs)

    monkeypatch.setattr(extraction_mod, "ManifoldExtractionPipeline", _Pipeline)
    rig.extraction = Extraction(
        handle, events, cast(Any, _Corpus(rig, on_generate)),
        exclusive=exclusive or (lambda *_a, **_k: nullcontext()),
        assert_unsteered=lambda: None,
        adopt=adopt or (lambda folder, _m: rig.adopted.append(Path(folder))),
        evict=evict or (
            lambda namespace, name, *, feature: rig.evicted.append(
                (namespace, name, feature),
            )
        ),
        register_profile=register_profile or (
            lambda name, _profile: rig.registered.append(name)
        ),
        dls=dls,
    )
    return rig
