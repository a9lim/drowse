"""``fit.transfer.load_or_fit_transfer_alignment``: the cross-model
Procrustes alignment behind ``manifold transfer`` — cache proofs, target
whitener reuse, single-flight model loads (threads and processes)."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import pytest


def _cross_process_alignment_worker(
    home: str,
    target: str,
    load_count: Any,
    start: Any,
    errors: Any,
) -> None:
    """Exercise the neutral-cache lock in a clean spawned interpreter."""
    import time

    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    patch = pytest.MonkeyPatch()
    patch.setenv("DROWSE_HOME", home)
    source_marker = Path(home) / "source-neutral-ready"
    rows = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "method": "neutral_activations", "format_version": 4,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {
                "0": {"shape": [3, 2], "dtype": "torch.float32"},
            },
            "n_prompts": 3,
        }

    def metadata(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        if model_id == "shared/source" and not source_marker.exists():
            raise FileNotFoundError(model_id)
        return sidecar(model_id)

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        del requested_layers
        if model_id == "shared/source" and not source_marker.exists():
            raise FileNotFoundError(model_id)
        return rows, sidecar(model_id)

    class Session:
        def __init__(self, model_id: str) -> None:
            assert model_id == "shared/source"
            with load_count.get_lock():
                load_count.value += 1
            self.model = object()
            self.tokenizer = object()
            self.layers = [object()]

        def __enter__(self) -> tuple[object, object, object]:
            return self.model, self.tokenizer, self.layers

        def __exit__(self, *_args: object) -> None:
            return None

    def capture(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert model_id == "shared/source" and force is False
        time.sleep(0.15)
        source_marker.write_text("ready")
        return rows, sidecar(model_id)

    patch.setattr(alignment_mod, "validate_neutral_cache_metadata", metadata)
    patch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    patch.setattr(
        alignment_mod, "_load_or_compute_neutral_activations_with_metadata_locked",
        capture,
    )
    patch.setattr(
        fp_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    patch.setattr(alignment_mod, "_loaded_model", lambda model_id, **_kwargs: Session(model_id))
    patch.setattr(
        alignment_mod, "load_alignment_map",
        lambda *_args, **_kwargs: (
            {0: torch.eye(2)}, {"quality_per_layer": {"0": 1.0}},
        ),
    )

    try:
        start.wait(timeout=120)
        alignment_mod.load_or_fit_transfer_alignment(
            "shared/source", target, force=False,
            requested_layers=[0],
        )
    except BaseException as exc:
        errors.put(repr(exc))
    finally:
        patch.undo()


def test_cold_alignment_reuses_loaded_target_neutrals_for_whitener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cold transfer neither revalidates identities nor reloads target rows."""
    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))

    class _SessionContext:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.model = object()
            self.tokenizer = object()
            self.layers = [object()]

        def __enter__(self) -> tuple[object, object, object]:
            return self.model, self.tokenizer, self.layers

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(alignment_mod, "_loaded_model", lambda model_id, **_kwargs: _SessionContext(model_id))
    loaded: list[str] = []
    src_acts = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])}
    tgt_acts = {0: torch.tensor([[2.0, 1.0], [1.0, 3.0], [4.0, 2.0]])}

    def load_with_metadata(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert force is False
        loaded.append(model_id)
        acts = src_acts if model_id == "src/model" else tgt_acts
        return acts, {
            "method": "neutral_activations", "format_version": 4,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {"0": {"shape": [3, 2], "dtype": "torch.float32"}},
            "n_prompts": 3,
        }

    monkeypatch.setattr(
        alignment_mod,
        "_load_or_compute_neutral_activations_with_metadata_locked",
        load_with_metadata,
    )
    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("cold cache")
        ),
    )
    monkeypatch.setattr(fp_mod, "model_source_fingerprint", lambda *_args: None)
    monkeypatch.setattr(
        alignment_mod, "fit_alignment",
        lambda _src, _tgt, **_kwargs: {0: torch.eye(2)},
    )
    monkeypatch.setattr(
        alignment_mod, "alignment_quality",
        lambda _maps, _src, _tgt: {0: 1.0},
    )
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **_kwargs: tmp_path / "alignment.safetensors",
    )
    result = alignment_mod.load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=True,
    )
    maps, quality = result.alignment, result.quality_per_layer
    whitener, layer_means = result.target_whitener, result.target_layer_means

    assert loaded == ["src/model", "tgt/model"]
    assert quality == {0: 1.0}
    assert torch.equal(maps[0], torch.eye(2))
    assert whitener.layers == {0}
    assert set(layer_means) == {0}


def test_cold_narrow_alignment_releases_full_seed_rosters_before_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import gc
    import weakref

    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    class _SessionContext:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.model = object()
            self.tokenizer = object()
            self.layers = [object() for _ in range(10)]

        def __enter__(self) -> tuple[object, object, object]:
            return self.model, self.tokenizer, self.layers

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(alignment_mod, "_loaded_model", lambda model_id, **_kwargs: _SessionContext(model_id))
    monkeypatch.setattr(fp_mod, "model_source_fingerprint", lambda *_args: None)
    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("cold cache")
        ),
    )

    unrequested_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def capture(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert force is False
        rows = {
            layer: torch.tensor(
                [[1.0 + layer, 0.0], [0.0, 1.0], [2.0, 1.0]],
            )
            for layer in range(10)
        }
        unrequested_refs.append(weakref.ref(rows[9]))
        return rows, {
            "method": "neutral_activations", "format_version": 4,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {str(layer): "0" * 64 for layer in range(10)},
            "tensor_files": {
                str(layer): f"neutral.layer-{layer}.safetensors"
                for layer in range(10)
            },
            "layers": list(range(10)),
            "tensor_schema": {
                str(layer): {
                    "shape": [3, 2], "dtype": "torch.float32",
                }
                for layer in range(10)
            },
            "n_prompts": 3,
        }

    monkeypatch.setattr(
        alignment_mod,
        "_load_or_compute_neutral_activations_with_metadata_locked",
        capture,
    )
    monkeypatch.setattr(alignment_mod, "load_alignment_map", lambda *_a, **_k: None)

    def fit(
        src: dict[int, torch.Tensor], tgt: dict[int, torch.Tensor], **kwargs: Any,
    ) -> dict[int, torch.Tensor]:
        gc.collect()
        assert len(unrequested_refs) == 2
        assert all(ref() is None for ref in unrequested_refs)
        assert set(src) == set(tgt) == {5}
        assert kwargs["requested_layers"] == {5}
        return {5: torch.eye(2)}

    monkeypatch.setattr(alignment_mod, "fit_alignment", fit)
    monkeypatch.setattr(
        alignment_mod, "alignment_quality", lambda *_args, **_kwargs: {5: 1.0},
    )
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **_kwargs: tmp_path / "alignment.safetensors",
    )

    result = alignment_mod.load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=True,
        requested_layers=[5],
    )

    assert set(result.alignment) == {5}
    assert result.target_whitener.layers == {5}


def test_cached_alignment_keeps_model_free_offline_whitener_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cached repeat loads target rows once and never loads model weights."""
    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "method": "neutral_activations", "format_version": 4,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {"0": {"shape": [3, 2], "dtype": "torch.float32"}},
            "n_prompts": 3,
        }

    metadata_calls: list[str] = []
    payload_calls: list[str] = []
    target_acts = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}

    def metadata_only(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        metadata_calls.append(model_id)
        return sidecar(model_id)

    def load_validated(
        model_id: str, **_kwargs: Any,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        payload_calls.append(model_id)
        assert model_id == "tgt/model"
        return target_acts, sidecar(model_id)

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata", metadata_only,
    )
    monkeypatch.setattr(
        alignment_mod, "load_validated_neutral_cache", load_validated,
    )
    monkeypatch.setattr(
        fp_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    monkeypatch.setattr(
        alignment_mod, "load_alignment_map",
        lambda *_args, **_kwargs: (
            {0: torch.eye(2)}, {"quality_per_layer": {"0": 0.75}},
        ),
    )
    monkeypatch.setattr(alignment_mod, "_loaded_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact cached repeat loaded a model")
        ))
    result = alignment_mod.load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=False,
    )

    assert result.quality_per_layer == {0: 0.75}
    assert result.target_whitener.layers == {0}
    assert metadata_calls == ["src/model", "tgt/model", "tgt/model"]
    assert payload_calls == ["tgt/model"]


def test_missing_cached_alignment_fits_offline_from_proven_neutral_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A map miss reuses both neutral caches and never loads either model."""
    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    src_acts = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])}
    tgt_acts = {0: torch.tensor([[2.0, 1.0], [1.0, 3.0], [4.0, 2.0]])}

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "method": "neutral_activations", "format_version": 4,
            "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {
                "0": {"shape": [3, 2], "dtype": "torch.float32"},
            },
            "n_prompts": 3,
        }

    metadata_calls: list[str] = []
    payload_calls: list[str] = []

    def metadata_only(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        metadata_calls.append(model_id)
        return sidecar(model_id)

    def load_validated(
        model_id: str, **_kwargs: Any,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        payload_calls.append(model_id)
        return (
            src_acts if model_id == "src/model" else tgt_acts,
            sidecar(model_id),
        )

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata", metadata_only,
    )
    monkeypatch.setattr(
        alignment_mod, "load_validated_neutral_cache", load_validated,
    )
    monkeypatch.setattr(
        fp_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    monkeypatch.setattr(
        alignment_mod, "load_alignment_map", lambda *_args, **_kwargs: None,
    )
    fitted: list[tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = []

    def fit_offline(
        src: dict[int, torch.Tensor], tgt: dict[int, torch.Tensor], **_kwargs: Any,
    ) -> dict[int, torch.Tensor]:
        fitted.append((src, tgt))
        return {0: torch.eye(2)}

    monkeypatch.setattr(alignment_mod, "fit_alignment", fit_offline)
    monkeypatch.setattr(
        alignment_mod, "alignment_quality",
        lambda _maps, _src, _tgt: {0: 0.9},
    )
    saved: list[dict[str, Any]] = []

    def save_offline(*_args: Any, **kwargs: Any) -> Path:
        saved.append(kwargs)
        return tmp_path / "alignment.safetensors"

    monkeypatch.setattr(alignment_mod, "save_alignment_map", save_offline)
    monkeypatch.setattr(alignment_mod, "_loaded_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline alignment fit loaded model weights")
        ))

    result = alignment_mod.load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=False,
    )

    assert metadata_calls == [
        "src/model", "tgt/model", "tgt/model", "src/model",
    ]
    assert payload_calls == ["tgt/model", "src/model"]
    assert len(fitted) == 1
    assert fitted[0][0] is src_acts
    assert set(fitted[0][1]) == {0}
    assert fitted[0][1][0] is tgt_acts[0]
    assert len(saved) == 1
    assert saved[0]["source_identity"] == result.source_identity
    assert saved[0]["target_identity"] == result.target_identity
    assert result.quality_per_layer == {0: 0.9}
    assert result.target_whitener.layers == {0}


def test_force_refits_only_requested_layer_without_loading_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "method": "neutral_activations", "format_version": 4, "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {str(layer): "0" * 64 for layer in range(10)},
            "tensor_files": {
                str(layer): f"neutral.layer-{layer}.safetensors"
                for layer in range(10)
            },
            "layers": list(range(10)),
            "tensor_schema": {
                str(layer): {"shape": [3, 2], "dtype": "torch.float32"}
                for layer in range(10)
            },
            "n_prompts": 3,
        }

    rows = {
        model_id: {
            layer: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            for layer in range(10)
        }
        for model_id in ("src/model", "tgt/model")
    }
    payload_calls: list[tuple[str, set[int] | None]] = []

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        requested = (
            None if requested_layers is None
            else {int(layer) for layer in requested_layers}  # type: ignore[union-attr]
        )
        payload_calls.append((model_id, requested))
        selected = set(rows[model_id]) if requested is None else requested
        return ({layer: rows[model_id][layer] for layer in selected}, sidecar(model_id))

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda model_id, **_kwargs: sidecar(model_id),
    )
    monkeypatch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    monkeypatch.setattr(
        fp_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    requested_map_layers: list[set[int]] = []

    def load_map(*_args: Any, **kwargs: Any) -> tuple[dict[int, Any], dict[str, Any]]:
        requested_map_layers.append(set(kwargs["requested_layers"]))
        return ({}, {"quality_per_layer": {"0": 0.5}})

    monkeypatch.setattr(alignment_mod, "load_alignment_map", load_map)
    fitted: list[tuple[set[int], dict[str, Any]]] = []

    def fit(src: dict[int, Any], _tgt: dict[int, Any], **kwargs: Any) -> dict[int, Any]:
        fitted.append((set(src), kwargs))
        return {5: torch.eye(2)}

    monkeypatch.setattr(alignment_mod, "fit_alignment", fit)
    monkeypatch.setattr(
        alignment_mod, "alignment_quality", lambda *_args, **_kwargs: {5: 0.9},
    )
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **kwargs: saved.append(kwargs) or tmp_path / "alignment",
    )
    monkeypatch.setattr(alignment_mod, "_loaded_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("force refit recaptured neutral activations")
        ))

    result = alignment_mod.load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=True,
        requested_layers=[5],
    )
    assert requested_map_layers == [set()]
    assert payload_calls == [("tgt/model", {5}), ("src/model", {5})]
    assert fitted[0][0] == {5}
    assert fitted[0][1]["requested_layers"] == {5}
    assert fitted[0][1]["available_shared_layers"] == set(range(10))
    assert saved[0]["extend"] is True
    assert set(result.alignment) == {5} and result.target_whitener.layers == {5}


def test_target_neutral_generation_race_replans_cached_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    target_generation = "old"

    def sidecar(model_id: str) -> dict[str, Any]:
        generation = target_generation if model_id == "tgt/model" else "stable"
        return {
            "method": "neutral_activations", "format_version": 4, "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}:{generation}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": generation,
            "tensor_sha256": {"0": "0" * 64},
            "tensor_files": {"0": "neutral.layer-0.safetensors"},
            "layers": [0],
            "tensor_schema": {"0": {"shape": [3, 2], "dtype": "torch.float32"}},
            "n_prompts": 3,
        }

    rows = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        nonlocal target_generation
        del requested_layers
        if model_id == "tgt/model" and target_generation == "old":
            target_generation = "new"
            raise ValueError("target cache changed during payload load")
        return rows, sidecar(model_id)

    monkeypatch.setattr(
        alignment_mod, "validate_neutral_cache_metadata",
        lambda model_id, **_kwargs: sidecar(model_id),
    )
    monkeypatch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    monkeypatch.setattr(
        fp_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    map_generations: list[str] = []

    def load_map(*_args: Any, **kwargs: Any):
        generation = kwargs["target_identity"]["capture_sha256"]
        map_generations.append(generation)
        factor = 1.0 if generation == "old" else 2.0
        return (
            {0: factor * torch.eye(2)},
            {"quality_per_layer": {"0": 0.8, "9": -5.0}},
        )

    monkeypatch.setattr(alignment_mod, "load_alignment_map", load_map)
    monkeypatch.setattr(alignment_mod, "_loaded_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity replan loaded a model")
        ))

    result = alignment_mod.load_or_fit_transfer_alignment(
        "src/model", "tgt/model", force=False,
        requested_layers=[0],
    )

    assert map_generations == ["old", "new"]
    assert torch.equal(result.alignment[0].to_dense(), 2.0 * torch.eye(2))
    assert result.quality_per_layer == {0: 0.8}
    assert result.target_identity["capture_sha256"] == "new"


def test_concurrent_distinct_alignments_single_flight_shared_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import torch

    from drowse.model import fingerprint as fp_mod
    import drowse.fit.transfer as alignment_mod

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    cached: dict[str, tuple[dict[int, torch.Tensor], dict[str, Any]]] = {}
    loads: dict[str, int] = {}

    def sidecar(model_id: str) -> dict[str, Any]:
        return {
            "method": "neutral_activations", "format_version": 4, "capture_version": 1,
            "model_fingerprint": f"fp:{model_id}",
            "model_source_fingerprint": f"source:{model_id}",
            "capture_sha256": model_id,
            "tensor_sha256": {str(layer): "0" * 64 for layer in range(10)},
            "tensor_files": {
                str(layer): f"neutral.layer-{layer}.safetensors"
                for layer in range(10)
            },
            "layers": list(range(10)), "tensor_schema": {}, "n_prompts": 3,
        }

    def metadata(model_id: str, **_kwargs: Any) -> dict[str, Any]:
        if model_id not in cached:
            raise FileNotFoundError(model_id)
        return cached[model_id][1]

    def load_rows(
        model_id: str, *, requested_layers: object = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        rows, sc = cached[model_id]
        if requested_layers is None:
            return rows, sc
        wanted = {int(layer) for layer in requested_layers}  # type: ignore[union-attr]
        return {layer: rows[layer] for layer in wanted if layer in rows}, sc

    class Session:
        def __init__(self, model_id: str) -> None:
            loads[model_id] = loads.get(model_id, 0) + 1
            self.model_id = model_id
            self.model = object()
            self.tokenizer = object()
            self.layers = [object()]

        def __enter__(self) -> tuple[object, object, object]:
            return self.model, self.tokenizer, self.layers

        def __exit__(self, *_args: object) -> None:
            return None

    def capture(
        _model: object, _tokenizer: object, _layers: object, *,
        model_id: str, force: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert force is False
        rows = {0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}
        cached[model_id] = (rows, sidecar(model_id))
        return cached[model_id]

    monkeypatch.setattr(alignment_mod, "validate_neutral_cache_metadata", metadata)
    monkeypatch.setattr(alignment_mod, "load_validated_neutral_cache", load_rows)
    monkeypatch.setattr(
        alignment_mod, "_load_or_compute_neutral_activations_with_metadata_locked",
        capture,
    )
    monkeypatch.setattr(
        fp_mod, "model_source_fingerprint",
        lambda model_id: f"source:{model_id}",
    )
    monkeypatch.setattr(alignment_mod, "_loaded_model", lambda model_id, **_kwargs: Session(model_id))
    monkeypatch.setattr(alignment_mod, "load_alignment_map", lambda *_a, **_k: None)
    monkeypatch.setattr(
        alignment_mod, "fit_alignment",
        lambda *_args, **_kwargs: {0: torch.eye(2)},
    )
    monkeypatch.setattr(
        alignment_mod, "alignment_quality", lambda *_args: {0: 1.0},
    )
    monkeypatch.setattr(
        alignment_mod, "save_alignment_map",
        lambda *_args, **_kwargs: tmp_path / "alignment",
    )
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run(target: str) -> None:
        try:
            barrier.wait()
            alignment_mod.load_or_fit_transfer_alignment(
                "shared/source", target, force=False,
                requested_layers=[0],
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(target,))
        for target in ("target/a", "target/b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert loads == {"shared/source": 1, "target/a": 1, "target/b": 1}


def test_cross_process_distinct_alignments_single_flight_shared_model_load(
    tmp_path: Path,
) -> None:
    """The filesystem neutral lock precedes model construction across commands."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    load_count = ctx.Value("i", 0)
    start = ctx.Barrier(3)
    errors = ctx.Queue()

    processes = [
        ctx.Process(
            target=_cross_process_alignment_worker,
            args=(str(tmp_path), target, load_count, start, errors),
        )
        for target in ("target/a", "target/b")
    ]
    timed_out: list[str] = []
    exitcodes: list[int | None] = []
    child_errors: list[str] = []
    try:
        for process in processes:
            process.start()
        start.wait(timeout=120)
        for process in processes:
            process.join(timeout=20)
        timed_out = [process.name for process in processes if process.is_alive()]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        exitcodes = [process.exitcode for process in processes]
        while True:
            try:
                child_errors.append(errors.get_nowait())
            except queue.Empty:
                break
        errors.close()
        errors.join_thread()

    assert timed_out == []
    assert exitcodes == [0, 0]
    assert child_errors == []
    assert load_count.value == 1
