import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch):
    runtime = Path(__file__).parents[1] / "browser-runtime"
    monkeypatch.syspath_prepend(str(runtime))
    spec = importlib.util.spec_from_file_location("promote_mlc_precision", runtime / "promote_mlc_precision.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fixture(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    weights = np.array([0x76543210, 0xFEDCBA98], dtype="<u4")
    scales = np.array([0.125, 0.33325], dtype="<f2")
    (source / "params.bin").write_bytes(weights.tobytes() + scales.tobytes())
    files = {
        "tensor-cache.json": {"metadata": {"ParamSize": 2, "ParamBytes": 12, "BitsPerParam": 3}, "records": [{
            "dataPath": "params.bin", "nbytes": 12, "records": [
                {"name": "model.q_weight", "shape": [1, 2], "dtype": "uint32", "format": "raw", "byteOffset": 0, "nbytes": 8},
                {"name": "model.q_scale", "shape": [1, 2], "dtype": "float16", "format": "raw", "byteOffset": 8, "nbytes": 4},
            ],
        }]},
        "mlc-chat-config.json": {"quantization": "q4f16_1"},
        "CONVERSION-SOURCE.json": {"source": "fixture"},
    }
    for path, value in files.items():
        (source / path).write_text(json.dumps(value))
    build = {"quantization": "q4f16_1", "source": {"revision": "a" * 40}, "files": []}
    for path in [*files, "params.bin"]:
        data = (source / path).read_bytes()
        build["files"].append({"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (source / "drowse-build.json").write_text(json.dumps(build))
    return source


def test_promotion_preserves_packed_weights_scales_and_provenance(tmp_path: Path, module: ModuleType):
    source = source_fixture(tmp_path)
    target = tmp_path / "target"
    result = module.promote(source, target)
    assert result["allStoredValuesPreserved"] and result["tensorCount"] == 2
    old, new = module.MlcTensorCache(source), module.MlcTensorCache(target)
    for name in old.records:
        np.testing.assert_array_equal(old.tensor(name), new.tensor(name))
    assert new.tensor("model.q_scale").dtype == np.float32
    assert new.tensor("model.q_weight").dtype == np.uint32
    manifest = json.loads((target / "tensor-cache.json").read_text())
    assert manifest["metadata"]["ParamBytes"] == 16
    assert manifest["metadata"]["BitsPerParam"] == 4
    assert manifest["records"][0]["md5sum"] == hashlib.md5((target / "params.bin").read_bytes(), usedforsecurity=False).hexdigest()
    build = json.loads((target / "drowse-build.json").read_text())
    assert build["quantization"] == "q4f32_1"
    assert build["source"] == {"revision": "a" * 40}
    for file in build["files"]:
        data = (target / file["path"]).read_bytes()
        assert len(data) == file["bytes"] and hashlib.sha256(data).hexdigest() == file["sha256"]
    provenance = json.loads((target / "CONVERSION-SOURCE.json").read_text())
    assert provenance["source"] == "fixture"
    assert provenance["precisionPromotion"]["inputBuildSha256"] == hashlib.sha256((source / "drowse-build.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already exists"):
        module.promote(source, target)


@pytest.mark.parametrize("mutation", ["hash", "path", "quantization", "shard"])
def test_promotion_rejects_invalid_inputs_without_output(tmp_path: Path, module: ModuleType, mutation: str):
    source = source_fixture(tmp_path)
    build = json.loads((source / "drowse-build.json").read_text())
    if mutation == "hash":
        build["files"][0]["sha256"] = "0" * 64
    elif mutation == "path":
        build["files"][0]["path"] = "../tensor-cache.json"
    elif mutation == "quantization":
        build["quantization"] = "q4f32_1"
    else:
        build["files"] = [file for file in build["files"] if file["path"] != "params.bin"]
    (source / "drowse-build.json").write_text(json.dumps(build))
    with pytest.raises(ValueError):
        module.promote(source, tmp_path / "target")
    assert not (tmp_path / "target").exists()
