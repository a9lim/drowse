"""Promote a verified q4f16_1 conversion to q4f32_1 without requantizing weights."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from mlc_q4_torch import MlcTensorCache


def promote(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    if target.exists():
        raise ValueError("precision promotion output already exists")
    build_bytes = (source / "drowse-build.json").read_bytes()
    build = json.loads(build_bytes)
    config = json.loads((source / "mlc-chat-config.json").read_text())
    if build["quantization"] != "q4f16_1" or config["quantization"] != "q4f16_1":
        raise ValueError("precision promotion requires q4f16_1 inputs")
    paths = set()
    for file in build["files"]:
        path = file["path"]
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", path) or path in {".", "..", "drowse-build.json"} or path in paths:
            raise ValueError("invalid or repeated build path")
        paths.add(path)
        payload = (source / path).read_bytes()
        if len(payload) != file["bytes"] or digest(payload) != file["sha256"]:
            raise ValueError(f"build file integrity mismatch: {path}")
    if not {"mlc-chat-config.json", "tensor-cache.json", "CONVERSION-SOURCE.json"} <= paths:
        raise ValueError("precision promotion requires complete conversion provenance")
    manifest = json.loads((source / "tensor-cache.json").read_text())
    shard_paths = [shard["dataPath"] for shard in manifest["records"]]
    if len(set(shard_paths)) != len(shard_paths) or any(path not in paths for path in shard_paths):
        raise ValueError("tensor shards do not match the verified build closure")
    input_parameter_bytes = 0
    cache = MlcTensorCache(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".drowse-precision-", dir=target.parent))
    try:
        for shard in manifest["records"]:
            offset = 0
            with (stage / shard["dataPath"]).open("wb") as output:
                for record in shard["records"]:
                    array = np.asarray(cache.tensor(record["name"]))
                    input_parameter_bytes += array.nbytes
                    if array.dtype == np.float16:
                        array = array.astype("<f4")
                    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                        raise ValueError(f"non-finite tensor: {record['name']}")
                    payload = array.tobytes()
                    record.update(dtype=str(array.dtype), format="raw", byteOffset=offset, nbytes=len(payload))
                    output.write(payload)
                    offset += len(payload)
            shard["nbytes"] = offset
            shard["md5sum"] = hashlib.md5((stage / shard["dataPath"]).read_bytes(), usedforsecurity=False).hexdigest()
        parameter_bytes = sum(shard["nbytes"] for shard in manifest["records"])
        if "BitsPerParam" in manifest["metadata"]:
            manifest["metadata"]["BitsPerParam"] *= parameter_bytes / input_parameter_bytes
        manifest["metadata"]["ParamBytes"] = parameter_bytes
        write_json(stage / "tensor-cache.json", manifest)
        config["quantization"] = "q4f32_1"
        write_json(stage / "mlc-chat-config.json", config)
        for path in paths:
            if not (stage / path).exists():
                shutil.copy2(source / path, stage / path)
        provenance = json.loads((stage / "CONVERSION-SOURCE.json").read_text())
        provenance["precisionPromotion"] = {
            "inputBuildSha256": digest(build_bytes),
            "from": "q4f16_1",
            "to": "q4f32_1",
            "method": "lossless-fp16-to-fp32-storage-cast",
            "packedQ4WeightsUnchanged": True,
        }
        write_json(stage / "CONVERSION-SOURCE.json", provenance)
        build["quantization"] = "q4f32_1"
        for file in build["files"]:
            payload = (stage / file["path"]).read_bytes()
            file.update(bytes=len(payload), sha256=digest(payload))
        write_json(stage / "drowse-build.json", build)
        promoted = MlcTensorCache(stage)
        for name in cache.records:
            np.testing.assert_array_equal(promoted.tensor(name), cache.tensor(name))
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"target": str(target), "tensorCount": len(cache.records), "allStoredValuesPreserved": True}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(json.dumps(promote(args.source, args.target)))
