"""Cold registry loading for integrity- and identity-bound manifold artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.session.registry import ArtifactRegistry, ModelBinding
from drowse.geometry.fold import fold_directions_to_subspace
from drowse.store.manifold_schema import ManifoldFormatError
from drowse.store.manifold_authoring import BakedTensor, publish_baked_manifold
from drowse.store.manifold_lifecycle import transfer_manifold
from drowse.base.paths import encode_release_id
from tests.support.whitener import synthetic_means, synthetic_whitener

_COMPONENTS = {"0:default/x": {"selector": "default/x", "alpha": 1.0, "tensor_sha256": "0" * 64}}


def _publish(manifold: Any, model_id: str, fingerprint: str) -> Path:
    return publish_baked_manifold(
        "local", "merged", "",
        [BakedTensor(model_id, manifold, fingerprint, components=_COMPONENTS)],
        method="merge",
    )


def _registry(model_id: str, model: torch.nn.Module) -> ArtifactRegistry:
    return ArtifactRegistry(ModelBinding(
        model=model, tokenizer=None, model_id=model_id,
        device=torch.device("cpu"), dtype=torch.float32,
    ))


def _fold(name: str, directions: dict[int, torch.Tensor], *, label: str) -> Any:
    layers = sorted(directions)
    dim = int(next(iter(directions.values())).numel())
    means = synthetic_means(layers, dim)
    whitener = synthetic_whitener(layers, dim, means=means)
    return fold_directions_to_subspace(
        name, directions, means, whitener=whitener, label=label,
    )


def test_baked_manifold_cold_loads_with_proven_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    model = torch.nn.Linear(4, 4, bias=False)
    model_id = "test/model"
    fingerprint = loaded_model_fingerprint(model, model_id)
    manifold = _fold(
        "merged", {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}, label="merged",
    )
    _publish(manifold, model_id, fingerprint)

    registry = _registry(model_id, model)
    loaded = registry.ensure_manifold_loaded("local/merged")
    assert registry.manifolds["local/merged"] is loaded


def test_cold_load_rejects_finite_tensor_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    model = torch.nn.Linear(4, 4, bias=False)
    model_id = "test/model"
    fingerprint = loaded_model_fingerprint(model, model_id)
    manifold = _fold("merged", {0: torch.ones(4)}, label="merged")
    folder = _publish(manifold, model_id, fingerprint)
    tensor = next(folder.glob("*.safetensors"))
    payload = bytearray(tensor.read_bytes())
    payload[-1] ^= 1
    tensor.write_bytes(payload)

    with pytest.raises(ManifoldFormatError, match="integrity"):
        _registry(model_id, model).ensure_manifold_loaded(
            "local/merged",
        )


def test_cold_load_rejects_untracked_fitted_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    model = torch.nn.Linear(4, 4, bias=False)
    model_id = "test/model"
    fingerprint = loaded_model_fingerprint(model, model_id)
    manifold = _fold("merged", {0: torch.ones(4)}, label="merged")
    folder = _publish(manifold, model_id, fingerprint)
    manifest_path = folder / "manifold.json"
    import json

    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError, match="no proof"):
        _registry(model_id, model).ensure_manifold_loaded(
            "local/merged",
        )


def test_transfer_variant_cold_loads_for_target_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    source_id = "src/model"
    target_id = "tgt/model"
    source_fp = "source-fingerprint"
    target_model = torch.nn.Linear(4, 4, bias=False)
    target_fp = loaded_model_fingerprint(target_model, target_id)
    manifold = _fold(
        "merged", {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}, label="merged",
    )
    folder = _publish(manifold, source_id, source_fp)
    from drowse.geometry.whitener import LayerWhitener
    from drowse.geometry.procrustes import LayerAlignment

    acts = {0: torch.randn(32, 4)}
    whitener = LayerWhitener.from_neutral_activations(
        acts, {0: acts[0].mean(dim=0)},
    )
    transfer_manifold(
        folder, from_model=source_id, to_model=target_id,
        alignment={0: LayerAlignment(torch.eye(4), torch.eye(4), torch.zeros(4))},
        whitener=whitener,
        target_layer_means={0: acts[0].mean(dim=0)},
        source_model_fingerprint=source_fp,
        target_model_fingerprint=target_fp,
    )

    key = f"local/merged:from-{encode_release_id(source_id)}"
    registry = _registry(target_id, target_model)
    registry.ensure_manifold_loaded(key)
    assert key in registry.manifolds
