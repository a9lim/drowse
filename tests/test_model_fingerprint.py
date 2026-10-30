"""Loaded-model and model-source fingerprints (the identities lens and
manifold artifacts bind to)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.model.fingerprint import loaded_model_fingerprint, model_source_fingerprint


def test_loaded_model_fingerprint_hashes_buffers_and_original_dtype_bits() -> None:
    class _Stateful(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([1.0], dtype=torch.float64),
            )
            self.register_buffer("scale", torch.tensor([2], dtype=torch.int64))
            self.config = SimpleNamespace(
                model_type="toy", _commit_hash="same-claimed-commit",
                _name_or_path="toy",
            )

    base = _Stateful()
    weight_changed = _Stateful()
    weight_changed.weight.data[0] = torch.nextafter(
        weight_changed.weight.data[0], torch.tensor(2.0, dtype=torch.float64),
    )
    buffer_changed = _Stateful()
    buffer_scale: Any = buffer_changed.scale
    buffer_scale[0] = 3
    fp = loaded_model_fingerprint(base, "toy")
    assert loaded_model_fingerprint(weight_changed, "toy") != fp
    assert loaded_model_fingerprint(buffer_changed, "toy") != fp


def test_loaded_model_fingerprint_ignores_registration_order() -> None:
    class _SameState(torch.nn.Module):
        def __init__(self, buffer_names: tuple[str, ...]) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            for name in buffer_names:
                self.register_buffer(name, torch.tensor([ord(name[0])]))
            self.config = SimpleNamespace(model_type="toy", _name_or_path="toy")

    forward = _SameState(("alpha", "zeta"))
    reversed_order = _SameState(("zeta", "alpha"))

    assert [name for name, _ in forward.named_buffers()] != [
        name for name, _ in reversed_order.named_buffers()
    ]
    assert loaded_model_fingerprint(forward, "toy") == loaded_model_fingerprint(
        reversed_order, "toy",
    )


def test_loaded_model_fingerprint_memoizes_until_sanctioned_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(8, 8, bias=False)
    real_to = torch.Tensor.to
    transfers = 0

    def counted_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal transfers
        if kwargs.get("device") == "cpu":
            transfers += 1
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counted_to)
    first = loaded_model_fingerprint(model, "toy")
    after_first = transfers
    assert loaded_model_fingerprint(model, "toy") == first
    assert transfers == after_first
    with torch.no_grad():
        model.weight.reshape(-1)[3].add_(0.5)
    assert loaded_model_fingerprint(model, "toy") != first
    assert transfers > after_first


def test_loaded_model_fingerprint_explicitly_invalidates_data_writes() -> None:
    from drowse.model.fingerprint import invalidate_loaded_model_fingerprint

    model = torch.nn.Linear(10, 10, bias=False)
    first = loaded_model_fingerprint(model, "toy")
    model.weight.data.reshape(-1)[7].add_(10)
    invalidate_loaded_model_fingerprint(model)
    assert loaded_model_fingerprint(model, "toy") != first


def test_local_source_fingerprint_includes_remote_code_and_tokenizer_files(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text("{}")
    code = model_dir / "modeling_custom.py"
    vocab = model_dir / "vocab.json"
    code.write_text("VALUE = 1\n")
    vocab.write_text('{"a": 0}')
    config = SimpleNamespace(
        model_type="custom", _commit_hash=None,
        _name_or_path=str(model_dir),
    )
    first = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert first is not None
    code.write_text("VALUE = 2\n")
    second = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert second != first
    vocab.write_text('{"b": 0}')
    third = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert third != second
    arbitrary_resource = model_dir / "bpe.codes"
    arbitrary_resource.write_text("old rules")
    fourth = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert fourth != third


def test_local_source_fingerprint_detects_same_size_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    import os

    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"weights-a")
    config = SimpleNamespace(
        model_type="custom", _commit_hash=None, _name_or_path=str(model_dir),
    )
    first = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    original = weights.stat()
    weights.write_bytes(b"weights-b")
    os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns))

    second = model_source_fingerprint(
        str(model_dir), config=config, device="cpu",
        parameter_dtype=torch.float32,
    )
    assert second != first


