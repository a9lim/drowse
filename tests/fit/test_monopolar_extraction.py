"""Monopolar (concept-vs-neutral) extraction — CPU only, synthetic encoder.

A monopolar concept (``session.extract(c)`` with ``baseline=None``) authors a
**genuinely 1-node** ``pca`` manifold.  The pipeline recognizes the single-node
shape and folds ``concept − ν`` (ν = the model's neutral activation mean,
``layer_means``) into a 1-node neutral-anchored ray — neutral is the implicit
negative pole, sourced per-model at fit, never a stored corpus.

These tests exercise the pipeline on CPU with a deterministic stub encoder; the
end-to-end ``session.extract`` authoring is covered by the GPU smoke.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.events import EventBus
from drowse.fit.pipeline import ManifoldExtractionPipeline
from tests.support.sae import MockSaeBackend
from drowse.geometry.fold import folded_directions
from drowse.store.manifold_folder import ManifoldFolder
from drowse.store.manifold_authoring import create_discover_manifold_folder
from drowse.base.paths import manifold_dir
from tests.support.capture import DIM, N_LAYERS, StubHandle, batched, install_capture

# Per-layer neutral mean ν.  The concept centroid is ν + a push along dim 0, so
# the folded direction (concept − ν) is a clean unit vector along dim 0.
_NU = {L: torch.full((DIM,), float(L + 1)) for L in range(N_LAYERS)}


def _concept_encoder(
    model: Any, tokenizer: Any, prompt: str, response: str, layers: Any,
    device: Any, **_kwargs: Any,
) -> dict[int, torch.Tensor]:
    """Deterministic concept centroid: ν + a +2 push along dim 0, per layer."""
    out: dict[int, torch.Tensor] = {}
    for L in range(len(layers)):
        v = _NU[L].clone()
        v[0] += 2.0
        out[L] = v
    return out


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch.manual_seed(0)
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    install_capture(monkeypatch, batched(_concept_encoder))


def _author() -> Path:
    return create_discover_manifold_folder(
        "local", "agentic", "Monopolar axis: agentic (+) vs neutral (-).",
        fit_mode="pca",
        node_corpora={"agentic": [f"agentic statement {i}" for i in range(2)]},
        hyperparams={"max_dim": 1, "var_threshold": 0.7},
        node_kinds={"agentic": "abstract"},
    )


def test_monopolar_folder_is_genuinely_one_node() -> None:
    _author()
    mf = ManifoldFolder.load(manifold_dir("local", "agentic"))
    assert mf.node_labels == ["agentic"]
    assert mf.fit_mode == "pca"


def test_monopolar_fits_one_node_ray() -> None:
    folder = _author()
    pipe = ManifoldExtractionPipeline(StubHandle(layer_means=_NU), EventBus())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # the fold path runs no DLS
        manifold = pipe.fit(folder)

    # Genuinely one node — the concept; neutral is implicit (ν).
    assert manifold.node_labels == ["agentic"]
    # ``method`` is the persisted monopolar discriminator; the fit stamps no
    # separate flag, so what the sidecar records is what a reader can rely on.
    assert manifold.metadata.get("method") == "manifold_monopolar"
    assert manifold.metadata.get("fit_mode") == "pca"
    assert manifold.metadata.get("hyperparams") == {
        "max_dim": 1,
        "var_threshold": 0.7,
    }
    assert manifold.metadata.get("diagnostics") == {}
    # Every layer is an affine rank-1 ray, so it folds to a steering vector.
    dirs = folded_directions(manifold)
    assert sorted(dirs) == list(range(N_LAYERS))
    for vec in dirs.values():
        assert torch.isfinite(vec).all()
        # δ̂ = unit(concept − ν) points +along dim 0 (the only displaced axis).
        assert float(vec[0]) > 0.0
        assert torch.allclose(vec[1:], torch.zeros(DIM - 1), atol=1e-5)


def test_monopolar_sae_fit_keeps_single_node_branch() -> None:
    folder = _author()
    sae = MockSaeBackend(
        layers=frozenset(range(N_LAYERS)), d_model=DIM,
        release="mono-rel",
    )
    manifold = ManifoldExtractionPipeline(StubHandle(layer_means=_NU), EventBus()).fit(
        folder, sae=sae,
    )

    assert manifold.node_labels == ["agentic"]
    assert manifold.metadata.get("method") == "manifold_monopolar_sae"
    assert sorted(manifold.layers) == list(range(N_LAYERS))


def test_monopolar_requires_layer_means() -> None:
    # ν is the implicit negative pole — without it there is nothing to fold
    # against, so the fit fails fast with an actionable message.
    folder = _author()
    handle = StubHandle(layer_means=_NU)
    handle.layer_means = {}
    pipe = ManifoldExtractionPipeline(handle, EventBus())
    with pytest.raises(ValueError, match="neutral layer means"):
        pipe.fit(folder)


def test_monopolar_cache_hit_reloads_ray() -> None:
    folder = _author()
    pipe = ManifoldExtractionPipeline(StubHandle(layer_means=_NU), EventBus())
    first = pipe.fit(folder)
    # Second fit cache-hits on the unchanged 1-node corpus and reloads the ray.
    second = pipe.fit(folder)
    assert second.node_labels == first.node_labels == ["agentic"]
    d1 = folded_directions(first)
    d2 = folded_directions(second)
    assert sorted(d1) == sorted(d2)
    for L in d1:
        assert torch.allclose(d1[L], d2[L], atol=1e-5)
