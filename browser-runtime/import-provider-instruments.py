#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


DEFAULT_WORDS = (
    "yes", "no", "true", "false", "good", "bad", "happy", "sad",
    "calm", "angry", "safe", "danger", "love", "hate", "help", "stop",
    "think", "know", "want", "need", "can", "cannot", "always", "never",
    "person", "people", "world", "time", "work", "home", "friend", "enemy",
    "honest", "fake", "confident", "uncertain", "formal", "casual", "kind", "rude",
    "science", "art", "code", "human", "machine", "answer", "question", "reason",
)
JLENS_TARGET_RESIDENT_BYTES = 448 * 1024 * 1024
JLENS_MAX_LAYERS = 8


def main() -> None:
    args = parse_args()
    runtime_lock = json.loads(args.runtime_lock.read_text(encoding="utf-8"))
    model = next(
        (entry for entry in runtime_lock["models"] if entry["id"] == args.model_id),
        None,
    )
    if model is None:
        raise SystemExit(f"unknown runtime-lock model {args.model_id}")
    if len(args.runtime_fingerprint) != 64:
        raise SystemExit("runtime fingerprint must be a SHA-256 digest")
    args.output.mkdir(parents=True, exist_ok=False)
    create_jlens_pack(args, model)
    create_sae_pack(args, model)
    print(json.dumps({
        "jlens": str(args.output / "jlens-pack"),
        "sae": str(args.output / "sae-pack"),
    }, sort_keys=True))


def create_jlens_pack(args, model: dict) -> None:
    checkpoint = torch.load(args.jlens_checkpoint, map_location="cpu", weights_only=False)
    raw_layers = checkpoint.get("J") if isinstance(checkpoint, dict) else None
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise SystemExit("provider J-lens checkpoint does not contain a J layer dictionary")
    layers: dict[str, torch.Tensor] = {}
    provider_layers = sorted(int(layer) for layer in raw_layers)
    layer_ids = select_jlens_layers(
        [layer for layer in model["layerMap"][:-1] if layer in provider_layers],
        model["hiddenSize"],
    )
    hidden_size = model["hiddenSize"]
    for layer in layer_ids:
        tensor = raw_layers[layer] if layer in raw_layers else raw_layers[str(layer)]
        tensor = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if tuple(tensor.shape) != (hidden_size, hidden_size) or not torch.isfinite(tensor).all():
            raise SystemExit(f"provider J-lens layer {layer} has an invalid matrix")
        layers[f"layer_{layer}"] = tensor

    pack = args.output / "jlens-pack" / "packs" / "jlens"
    pack.mkdir(parents=True)
    tensor_path = pack / "layers.safetensors"
    save_file(layers, tensor_path)
    tensor_digest = sha256(tensor_path)
    checkpoint_digest = sha256(args.jlens_checkpoint)
    prompt_count = int(checkpoint.get("n_prompts", checkpoint.get("prompts_fitted", 278)))
    manifest = {
        "format_version": 6,
        "method": "provider_jacobian_lens",
        "n_prompts": prompt_count,
        "checkpoint": False,
        "dtype": "float32",
        "d_model": hidden_size,
        "source_layers": layer_ids,
        "tensor_files": {str(layer): tensor_path.name for layer in layer_ids},
        "tensor_sha256": {str(layer): tensor_digest for layer in layer_ids},
        "corpus_spec": "Salesforce/wikitext:wikitext-103-raw-v1:train",
        "corpus_sha256": checkpoint_digest,
        "corpus_hash_kind": "provider_checkpoint_v1",
        "seq_len": 128,
        "dim_batch": 128,
        "skip_first_positions": 0,
        "estimator_policy": {"method": "provider_jacobian_lens"},
        "raw_corpus_sha256": None,
        "raw_prompt_count": None,
        "usable_prompt_count": prompt_count,
        "model_layer_count": len(model["layerMap"]),
        "model_fingerprint": args.runtime_fingerprint,
        "model_source_fingerprint": model["sourceRevision"],
        "base_n_prompts": None,
        "partial_n_prompts": None,
        "consumed_prefix_sha256": None,
    }
    write_json(pack / "manifest.json", manifest)
    create_browser_vocabulary(args, model, pack)


def select_jlens_layers(layer_map: list[int], hidden_size: int) -> list[int]:
    if not layer_map:
        raise SystemExit("provider J-lens has no compatible source layers")
    matrix_bytes = hidden_size * hidden_size * 4
    count = max(
        1,
        min(JLENS_MAX_LAYERS, len(layer_map), JLENS_TARGET_RESIDENT_BYTES // matrix_bytes),
    )
    if count == len(layer_map):
        return layer_map
    if count == 1:
        return [layer_map[-1]]
    return [
        layer_map[round(index * (len(layer_map) - 1) / (count - 1))]
        for index in range(count)
    ]


def create_browser_vocabulary(args, model: dict, pack: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_source,
        local_files_only=True,
        trust_remote_code=False,
    )
    embedding_path, embedding_key = embedding_tensor(args.model_source)
    with safe_open(embedding_path, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_tensor(embedding_key)
    if embedding.ndim != 2 or embedding.shape[1] != model["hiddenSize"]:
        raise SystemExit("source model embedding has the wrong shape")
    rows = []
    words = []
    seen_tokens = set()
    for word in DEFAULT_WORDS:
        token_id = single_token_id(tokenizer, word)
        if token_id is None or token_id in seen_tokens or token_id >= embedding.shape[0]:
            continue
        seen_tokens.add(token_id)
        rows.append(embedding[token_id].to(dtype=torch.float32, device="cpu"))
        words.append({"word": word, "row": len(words), "token_id": token_id})
    if not words:
        raise SystemExit("no default J-lens words are single tokens for this tokenizer")
    tensor_path = pack / "browser-vocabulary.safetensors"
    save_file({"unembedding": torch.stack(rows).contiguous()}, tensor_path)
    write_json(pack / "browser-vocabulary.json", {
        "format": "drowse-jlens-vocabulary-v2",
        "hidden_size": model["hiddenSize"],
        "tensor_file": tensor_path.name,
        "tensor_sha256": sha256(tensor_path),
        "words": words,
    })


def embedding_tensor(model_source: Path) -> tuple[Path, str]:
    keys = ("model.embed_tokens.weight", "language_model.model.embed_tokens.weight")
    single = model_source / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as tensors:
            matches = [key for key in keys if key in tensors.keys()]
        if len(matches) == 1:
            return single, matches[0]
    index_path = model_source / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        matches = [key for key in keys if key in weight_map]
        if len(matches) == 1:
            path = model_source / weight_map[matches[0]]
            if path.is_file():
                return path, matches[0]
    raise SystemExit("source model has no supported tied embedding tensor")


def create_sae_pack(args, model: dict) -> None:
    config = json.loads(args.sae_config.read_text(encoding="utf-8"))
    layer = hook_layer(config.get("hf_hook_point_out"))
    provider_repository = args.sae_provider_repository or gemma_scope_repository(config)
    with safe_open(args.sae_params, framework="pt", device="cpu") as source:
        tensors = {
            "W_enc": source.get_tensor("w_enc").to(dtype=torch.float32).contiguous(),
            "W_dec": source.get_tensor("w_dec").to(dtype=torch.float32).contiguous(),
            "b_enc": source.get_tensor("b_enc").to(dtype=torch.float32).contiguous(),
            "b_dec": source.get_tensor("b_dec").to(dtype=torch.float32).contiguous(),
            "threshold": source.get_tensor("threshold").to(dtype=torch.float32).contiguous(),
        }
    hidden_size = model["hiddenSize"]
    feature_count = int(config["width"])
    expected = {
        "W_enc": (hidden_size, feature_count),
        "W_dec": (feature_count, hidden_size),
        "b_enc": (feature_count,),
        "b_dec": (hidden_size,),
        "threshold": (feature_count,),
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name] or not torch.isfinite(tensor).all():
            raise SystemExit(f"provider SAE tensor {name} has an invalid shape or value")
    if torch.any(tensors["threshold"] < 0):
        raise SystemExit("provider JumpReLU SAE contains a negative threshold")

    pack = args.output / "sae-pack" / "packs" / "sae"
    pack.mkdir(parents=True)
    tensor_path = pack / f"layer-{layer}.safetensors"
    save_file(tensors, tensor_path)
    tensor_digest = sha256(tensor_path)
    width_label = f"{feature_count // 1024}k" if feature_count % 1024 == 0 else str(feature_count)
    name = f"gemma_scope_2_l{layer}_{width_label}"
    manifest = {
        "format_version": 2,
        "kind": "local",
        "name": name,
        "release": f"local:{name}",
        "model_id": model["sourceRepository"],
        "model_fingerprint": args.runtime_fingerprint,
        "model_source_fingerprint": model["sourceRevision"],
        "layer": layer,
        "d_model": hidden_size,
        "d_sae": feature_count,
        "activation": "jump_relu",
        "tensor_file": tensor_path.name,
        "tensor_sha256": tensor_digest,
        "corpus_spec": f"provider:{provider_repository}",
        "corpus_sha256": sha256(args.sae_params),
        "tokens_trained": 0,
        "seq_len": 0,
        "batch_size": 0,
        "learning_rate": 0.0,
        "l1_coefficient": 0.0,
        "dead_feature_threshold": 0,
    }
    write_json(pack / "manifest.json", manifest)


def gemma_scope_repository(config: dict) -> str:
    model_name = config.get("model_name")
    prefix = "google/gemma-3-"
    if not isinstance(model_name, str) or not model_name.startswith(prefix):
        raise SystemExit("--sae-provider-repository is required for this provider SAE")
    return f"google/gemma-scope-2-{model_name[len(prefix):]}"


def single_token_id(tokenizer, word: str) -> int | None:
    for candidate in (f" {word}", word):
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1 and tokenizer.decode(token_ids).strip() == word:
            return int(token_ids[0])
    return None


def hook_layer(value) -> int:
    prefix = "model.layers."
    suffix = ".output"
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith(suffix):
        raise SystemExit("provider SAE hook point is invalid")
    layer = int(value[len(prefix):-len(suffix)])
    if layer < 0:
        raise SystemExit("provider SAE layer is invalid")
    return layer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--jlens-checkpoint", required=True, type=Path)
    parser.add_argument("--sae-config", required=True, type=Path)
    parser.add_argument("--sae-params", required=True, type=Path)
    parser.add_argument("--sae-provider-repository")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
