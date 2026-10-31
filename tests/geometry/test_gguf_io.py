"""Roundtrip and edge-case tests for drowse.gguf_io."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

gguf = pytest.importorskip("gguf")

from drowse.geometry.gguf import read_gguf_profile, write_gguf_profile


def test_roundtrip_preserves_tensors(tmp_path: Path):
    profile = {
        0: torch.randn(16),
        3: torch.randn(16),
        14: torch.randn(16),
    }
    path = tmp_path / "happy.gguf"
    write_gguf_profile(profile, path, model_hint="llama")
    assert path.is_file()

    loaded, meta = read_gguf_profile(path)
    assert sorted(loaded.keys()) == [0, 3, 14]
    for k in profile:
        assert torch.allclose(profile[k].float(), loaded[k].float(), atol=1e-6)
    assert meta == {
        "method": "profile", "source_format": "gguf", "model_hint": "llama",
    }


def test_roundtrip_preserves_layer_indices_out_of_order(tmp_path: Path):
    profile = {14: torch.randn(8), 3: torch.randn(8), 0: torch.randn(8)}
    path = tmp_path / "x.gguf"
    write_gguf_profile(profile, path, model_hint="gemma3")
    loaded, _ = read_gguf_profile(path)
    assert set(loaded.keys()) == {0, 3, 14}


def test_write_casts_to_float32(tmp_path: Path):
    # fp16 input — must be readable back as fp32 without loss of indexing.
    profile = {0: torch.randn(8, dtype=torch.float16)}
    path = tmp_path / "fp16.gguf"
    write_gguf_profile(profile, path, model_hint="qwen2")
    loaded, _ = read_gguf_profile(path)
    assert loaded[0].dtype == torch.float32


def test_read_rejects_wrong_architecture(tmp_path: Path):
    # Write a GGUF with a non-controlvector architecture and verify read rejects it.
    path = tmp_path / "not_cv.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")  # wrong arch
    writer.add_string("controlvector.model_hint", "llama")
    writer.add_tensor("direction.0", torch.randn(4).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="architecture"):
        read_gguf_profile(path)


def test_read_requires_model_hint(tmp_path: Path):
    path = tmp_path / "no_hint.gguf"
    writer = gguf.GGUFWriter(str(path), "controlvector")
    writer.add_tensor("direction.0", torch.randn(4).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="model_hint"):
        read_gguf_profile(path)


def test_read_rejects_empty(tmp_path: Path):
    path = tmp_path / "empty.gguf"
    writer = gguf.GGUFWriter(str(path), "controlvector")
    writer.add_string("controlvector.model_hint", "llama")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="no direction"):
        read_gguf_profile(path)


def test_load_profile_dispatches_on_extension(tmp_path: Path):
    """drowse.geometry.profile.load_profile should route .gguf to the GGUF loader."""
    from drowse.geometry.profile import load_profile
    profile = {0: torch.randn(8), 5: torch.randn(8)}
    path = tmp_path / "dispatch.gguf"
    write_gguf_profile(profile, path, model_hint="llama")
    loaded, meta = load_profile(str(path))
    assert sorted(loaded.keys()) == [0, 5]
    assert meta["method"] == "profile"


def test_gguf_profile_saves_back_to_safetensors(tmp_path: Path):
    """Regression: ``Profile.load(x.gguf).save(y)`` raised on the GGUF-only
    ``gguf_import`` method; the load now round-trips through ``save``."""
    from drowse.geometry.profile import Profile

    original = {0: torch.randn(8), 5: torch.randn(8)}
    gguf_path = tmp_path / "v.gguf"
    write_gguf_profile(original, gguf_path, model_hint="gemma3")

    target = tmp_path / "v.safetensors"
    Profile.load(gguf_path).save(target)
    reloaded = Profile.load(target)

    assert sorted(reloaded.layers) == [0, 5]
    for layer, tensor in original.items():
        assert torch.allclose(reloaded[layer], tensor.float())
    assert reloaded.metadata["method"] == "profile"
    assert reloaded.metadata["source_format"] == "gguf"
    assert reloaded.metadata["model_hint"] == "gemma3"
