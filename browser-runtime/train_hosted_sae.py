from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def train(
    activation_path: Path,
    metadata_path: Path,
    output_directory: Path,
    *,
    name: str,
    features: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l1_coefficient: float,
    dead_feature_threshold: float,
    seed: int,
    device_name: str,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text())
    required = {
        "schema_version", "model_id", "model_source_fingerprint",
        "runtime_identity_sha256", "layer", "hidden_size", "rows", "seq_len",
        "corpus_spec", "corpus_sha256", "capture_plan_sha256",
        "activation_sha256",
    }
    if set(metadata) != required or metadata["schema_version"] != 1:
        raise ValueError("browser SAE capture metadata has an invalid schema")
    rows = int(metadata["rows"])
    hidden_size = int(metadata["hidden_size"])
    if rows <= 0 or hidden_size <= 0 or features <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("browser SAE dimensions and training settings must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise ValueError("SAE name contains unsupported characters")
    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    expected_bytes = rows * hidden_size * 4
    if activation_path.stat().st_size != expected_bytes:
        raise ValueError("browser SAE activation file has the wrong size")
    if hash_file(activation_path) != metadata["activation_sha256"]:
        raise ValueError("browser SAE activation digest does not match its metadata")
    if learning_rate <= 0 or l1_coefficient < 0 or dead_feature_threshold < 0:
        raise ValueError("browser SAE optimizer settings are invalid")

    device = choose_device(device_name)
    values = np.memmap(activation_path, dtype="<f4", mode="r", shape=(rows, hidden_size))
    if not np.isfinite(values).all():
        raise ValueError("browser SAE activations contain non-finite values")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    decoder_init = torch.randn(features, hidden_size, generator=generator)
    decoder_init /= decoder_init.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w_dec = torch.nn.Parameter(decoder_init.to(device))
    w_enc = torch.nn.Parameter(decoder_init.T.contiguous().to(device))
    b_enc = torch.nn.Parameter(torch.zeros(features, device=device))
    b_dec = torch.nn.Parameter(torch.from_numpy(np.asarray(values.mean(axis=0))).to(device))
    optimizer = torch.optim.Adam((w_enc, w_dec, b_enc, b_dec), lr=learning_rate)
    feature_fires = torch.zeros(features, dtype=torch.int64)
    losses: list[float] = []
    mse_values: list[float] = []
    activation_values: list[float] = []
    for _epoch in range(epochs):
        order = torch.randperm(rows, generator=generator).numpy()
        for start in range(0, rows, batch_size):
            indices = order[start : start + batch_size]
            batch = torch.from_numpy(np.array(values[indices], copy=True)).to(device)
            encoded = torch.relu((batch - b_dec) @ w_enc + b_enc)
            reconstruction = encoded @ w_dec + b_dec
            mse = (reconstruction - batch).square().mean()
            sparsity = encoded.abs().mean()
            loss = mse + l1_coefficient * sparsity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                w_dec.div_(w_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
                feature_fires.add_(
                    (encoded.detach().cpu() > dead_feature_threshold).sum(dim=0)
                )
            losses.append(float(loss.detach().cpu()))
            mse_values.append(float(mse.detach().cpu()))
            activation_values.append(float(sparsity.detach().cpu()))

    tensors = {
        "W_enc": w_enc.detach().cpu().to(torch.float32).contiguous(),
        "W_dec": w_dec.detach().cpu().to(torch.float32).contiguous(),
        "b_enc": b_enc.detach().cpu().to(torch.float32).contiguous(),
        "b_dec": b_dec.detach().cpu().to(torch.float32).contiguous(),
    }
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("browser SAE training produced non-finite tensors")
    root = output_directory / "packs" / "sae"
    root.mkdir(parents=True)
    tensor_name = f"layer-{int(metadata['layer'])}.safetensors"
    tensor_path = root / tensor_name
    save_file(tensors, str(tensor_path))
    tensor_sha256 = hash_file(tensor_path)
    manifest = {
        "format_version": 1,
        "kind": "local",
        "name": name,
        "release": f"local:{name}",
        "model_id": metadata["model_id"],
        "model_fingerprint": metadata["runtime_identity_sha256"],
        "model_source_fingerprint": metadata["model_source_fingerprint"],
        "layer": int(metadata["layer"]),
        "d_model": hidden_size,
        "d_sae": features,
        "activation": "relu",
        "tensor_file": tensor_name,
        "tensor_sha256": tensor_sha256,
        "corpus_spec": metadata["corpus_spec"],
        "corpus_sha256": metadata["corpus_sha256"],
        "tokens_trained": rows * epochs,
        "seq_len": int(metadata["seq_len"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "l1_coefficient": l1_coefficient,
        "dead_feature_threshold": dead_feature_threshold,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "output": str(output_directory),
        "device": str(device),
        "tokens_trained": rows * epochs,
        "features": features,
        "mean_loss": math.fsum(losses) / len(losses),
        "mean_mse": math.fsum(mse_values) / len(mse_values),
        "mean_feature_activation": math.fsum(activation_values) / len(activation_values),
        "dead_features": int((feature_fires == 0).sum()),
        "tensor_sha256": tensor_sha256,
    }


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("activation_path", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--name", default="browser-default")
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    parser.add_argument("--dead-feature-threshold", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    result = train(
        arguments.activation_path,
        arguments.metadata_path,
        arguments.output_directory,
        name=arguments.name,
        features=arguments.features,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        l1_coefficient=arguments.l1_coefficient,
        dead_feature_threshold=arguments.dead_feature_threshold,
        seed=arguments.seed,
        device_name=arguments.device,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
