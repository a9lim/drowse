from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from transformers import AutoTokenizer

from mlc_q4_torch import build_transformers_model
from drowse.core.jlens import resolve_word_token
from drowse.io.integrity import hash_file


def import_jlens(args: argparse.Namespace) -> dict[str, object]:
    runtime_lock = json.loads(args.runtime_lock.read_text())
    model = next((entry for entry in runtime_lock["models"] if entry["id"] == args.model_id), None)
    if model is None:
        raise ValueError(f"unknown runtime-lock model {args.model_id}")
    if args.output.exists():
        raise ValueError(f"output directory already exists: {args.output}")
    if len(args.runtime_fingerprint) != 64:
        raise ValueError("runtime fingerprint must be a SHA-256 digest")

    config_raw = args.config.read_bytes()
    config = yaml.safe_load(config_raw)
    if not isinstance(config, dict) or config.get("hf_model_name") != model["sourceRepository"]:
        raise ValueError("provider config does not match the runtime model")
    fit = config.get("fit")
    results = config.get("results")
    dataset = config.get("dataset")
    if not isinstance(fit, dict) or not isinstance(results, dict) or not isinstance(dataset, dict):
        raise ValueError("provider config is incomplete")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"J", "n_prompts", "source_layers", "d_model"}:
        raise ValueError("provider checkpoint has an invalid schema")
    hidden_size = model["hiddenSize"]
    if checkpoint["d_model"] != hidden_size:
        raise ValueError("provider checkpoint hidden size does not match the runtime")
    provider_layers = checkpoint["source_layers"]
    jacobians = checkpoint["J"]
    if not isinstance(provider_layers, list) or not isinstance(jacobians, dict):
        raise ValueError("provider checkpoint layer map is invalid")
    selected_layers = parse_layers(args.source_layers)
    valid_layers = set(model["layerMap"][:-1])
    if any(layer not in valid_layers or layer not in provider_layers for layer in selected_layers):
        raise ValueError("selected J-lens layer is unavailable or outside the runtime")

    root = args.output / "packs" / "jlens"
    root.mkdir(parents=True)
    tensor_files: dict[str, str] = {}
    tensor_sha256: dict[str, str] = {}
    for layer in selected_layers:
        tensor = jacobians[layer] if layer in jacobians else jacobians[str(layer)]
        tensor = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if tuple(tensor.shape) != (hidden_size, hidden_size) or not torch.isfinite(tensor).all():
            raise ValueError(f"provider J-lens layer {layer} is invalid")
        name = f"jlens.layer-{layer}.safetensors"
        save_file({f"layer_{layer}": tensor}, root / name)
        tensor_files[str(layer)] = name
        tensor_sha256[str(layer)] = hash_file(root / name)

    if args.vocabulary_pack is not None:
        copy_vocabulary(args.vocabulary_pack, root, hidden_size)
    else:
        create_vocabulary(args.model_directory, args.words, args.device, root, hidden_size)

    checkpoint_sha256 = hash_file(args.checkpoint)
    prompts = int(checkpoint["n_prompts"])
    if prompts != int(results.get("prompts_fitted", -1)) or prompts < 1:
        raise ValueError("provider prompt count does not match its config")
    corpus = ":".join(str(dataset.get(key, "")) for key in ("name", "config", "split"))
    manifest = {
        "format_version": 6,
        "method": "provider_jacobian_lens",
        "n_prompts": prompts,
        "d_model": hidden_size,
        "source_layers": selected_layers,
        "dtype": "float32",
        "corpus_spec": f"provider:{args.provider_repository}:{args.provider_revision}:{corpus}",
        "corpus_sha256": checkpoint_sha256,
        "corpus_hash_kind": "provider_checkpoint_v1",
        "seq_len": int(fit["max_seq_len"]),
        "dim_batch": int(fit["dim_batch"]),
        "skip_first_positions": 16,
        "estimator_policy": {
            "method": "provider_jacobian_lens",
            "provider": args.provider_repository,
            "revision": args.provider_revision,
            "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        },
        "tensor_sha256": tensor_sha256,
        "tensor_files": tensor_files,
        "raw_corpus_sha256": checkpoint_sha256,
        "raw_prompt_count": prompts,
        "usable_prompt_count": prompts,
        "model_layer_count": len(model["layerMap"]),
        "model_fingerprint": args.runtime_fingerprint,
        "model_source_fingerprint": model["sourceRevision"],
        "checkpoint": False,
        "base_n_prompts": None,
        "partial_n_prompts": None,
        "consumed_prefix_sha256": None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "modelId": args.model_id,
        "prompts": prompts,
        "layers": selected_layers,
        "hiddenSize": hidden_size,
        "outputDirectory": str(args.output),
    }


def copy_vocabulary(source: Path, target: Path, hidden_size: int) -> None:
    manifest_path = source / "browser-vocabulary.json"
    manifest = json.loads(manifest_path.read_text())
    tensor_path = source / manifest.get("tensor_file", "")
    if (
        manifest.get("format") != "drowse-jlens-vocabulary-v2"
        or manifest.get("hidden_size") != hidden_size
        or hash_file(tensor_path) != manifest.get("tensor_sha256")
    ):
        raise ValueError("browser vocabulary pack does not match the runtime")
    shutil.copy2(manifest_path, target / manifest_path.name)
    shutil.copy2(tensor_path, target / tensor_path.name)


def create_vocabulary(
    model_directory: Path,
    words_path: Path,
    device: str,
    target: Path,
    hidden_size: int,
) -> None:
    words = json.loads(words_path.read_text())
    if not isinstance(words, list) or not words or any(not isinstance(word, str) for word in words):
        raise ValueError("J-lens words must be a non-empty string list")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_directory), local_files_only=True, trust_remote_code=False,
    )
    model = build_transformers_model(model_directory, device)
    try:
        weight = getattr(model.get_output_embeddings(), "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.shape[1] != hidden_size:
            raise ValueError("converted model does not expose the expected unembedding")
        token_ids = [resolve_word_token(tokenizer, word) for word in words]
        rows = weight[token_ids].detach().to(device="cpu", dtype=torch.float32).contiguous()
    finally:
        del model
    tensor_path = target / "browser-vocabulary.safetensors"
    save_file({"unembedding": rows}, tensor_path)
    manifest = {
        "format": "drowse-jlens-vocabulary-v2",
        "hidden_size": hidden_size,
        "tensor_file": tensor_path.name,
        "tensor_sha256": hash_file(tensor_path),
        "words": [
            {"word": word, "row": index, "token_id": token_id}
            for index, (word, token_id) in enumerate(zip(words, token_ids, strict=True))
        ],
    }
    (target / "browser-vocabulary.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_layers(value: str) -> list[int]:
    layers = [int(item) for item in value.split(",")]
    if not layers or layers != sorted(set(layers)) or any(layer < 0 for layer in layers):
        raise ValueError("source layers must be unique sorted nonnegative integers")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--provider-repository", required=True)
    parser.add_argument("--provider-revision", required=True)
    parser.add_argument("--source-layers", required=True)
    parser.add_argument("--vocabulary-pack", type=Path)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--words", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    has_pack = args.vocabulary_pack is not None
    has_model = args.model_directory is not None and args.words is not None
    if has_pack == has_model:
        parser.error("supply either --vocabulary-pack or both --model-directory and --words")
    return args


if __name__ == "__main__":
    print(json.dumps(import_jlens(parse_args()), indent=2))
