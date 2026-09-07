"""Check pinned upstream instruments without fitting or publishing anything."""

import hashlib
import io
import json
import urllib.request

import torch
from safetensors.torch import load as load_safetensors


LENS_REVISION = "0731326edff4ae730ffc5356fe1a4728c748b3a6"
LENSES = [
    ("google/gemma-3-1b-pt", "gemma-3-1b", "gemma-3-1b-pt", 1152,
     66363652, "39f61074bfdb7c896b5027612a21e9648ca348d6516f0fbdbca6e5690c8b1ab1"),
    ("openai-community/gpt2", "gpt2-small", "gpt2", 768,
     12980477, "d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762"),
    ("EleutherAI/pythia-70m-deduped", "pythia-70m-deduped", "pythia-70m-deduped", 512,
     2624492, "96385b7a44000fe4682bf0957b2ce9f1639333833d71620f27ba9bb29bbaba3d"),
]


def download(url, expected_bytes, expected_sha256):
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read(expected_bytes + 1)
    if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Artifact size or SHA-256 mismatch: {url}")
    return payload


def main():
    torch.set_num_threads(2)
    for model, folder, stem, hidden, size, digest in LENSES:
        prefix = f"https://huggingface.co/neuronpedia/jacobian-lens/resolve/{LENS_REVISION}/{folder}/jlens/Salesforce-wikitext"
        with urllib.request.urlopen(f"{prefix}/config.yaml", timeout=30) as response:
            config = response.read(32768).decode()
        if f'hf_model_name: "{model}"' not in config:
            raise ValueError(f"J-lens configuration does not name {model}")
        payload = download(f"{prefix}/{stem}_jacobian_lens.pt", size, digest)
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        matrices = checkpoint["J"]
        if not isinstance(matrices, dict) or not matrices:
            raise ValueError("Missing J-lens matrices")
        for layer, matrix in matrices.items():
            if matrix.shape != (hidden, hidden) or not torch.isfinite(matrix).all():
                raise ValueError(f"Invalid J-lens matrix: {model}, layer {layer}")
        print(json.dumps({
            "model": model, "check": "pinned_jlens_tensor_integrity",
            "layers": sorted(map(int, matrices)), "hidden_size": hidden,
            "download_bytes": size,
            "fp32_matrix_bytes": sum(matrix.numel() * 4 for matrix in matrices.values()),
            "generation_tested": False,
        }), flush=True)
        del checkpoint, matrices, payload

    prefix = "https://huggingface.co/google/gemma-scope-2-1b-pt/resolve/b738dc06961818c011fb2e44a316352ca0f4e873/resid_post/layer_13_width_16k_l0_medium"
    with urllib.request.urlopen(f"{prefix}/config.json", timeout=30) as response:
        config = json.load(response)
    expected_config = {
        "model_name": "google/gemma-3-1b-pt", "architecture": "jump_relu",
        "hf_hook_point_in": "model.layers.13.output",
        "hf_hook_point_out": "model.layers.13.output",
        "width": 16384, "affine_connection": False, "type": "sae",
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("SAE model, hook, or activation does not match")
    payload = download(f"{prefix}/params.safetensors", 151131000,
                       "2b57a8bfb01d73b5cb7d26b48d6d8f1ec5c2b1d8554488e5a81796a4b710829d")
    tensors = load_safetensors(payload)
    shapes = {"w_enc": (1152, 16384), "w_dec": (16384, 1152),
              "b_enc": (16384,), "b_dec": (1152,), "threshold": (16384,)}
    if set(tensors) != set(shapes):
        raise ValueError("Unexpected SAE tensors")
    for key, shape in shapes.items():
        tensor = tensors[key]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid SAE tensor: {key}")
    if (tensors["threshold"] < 0).any():
        raise ValueError("Negative JumpReLU thresholds")
    print(json.dumps({
        "model": "google/gemma-3-1b-pt", "check": "pinned_sae_tensor_integrity",
        "layer": 13, "activation": "jump_relu", "features": 16384,
        "download_bytes": len(payload), "generation_tested": False,
    }), flush=True)


if __name__ == "__main__":
    main()
