from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from mlc_q4_torch import BrowserResidualCorrector, build_transformers_model

from drowse.core.jlens import SKIP_FIRST_POSITIONS, resolve_word_token
from drowse.core.jlens_fit import fit_jacobian_lens
from drowse.io.integrity import hash_file
from drowse.io.lens import _save_lens_at


def fit_pack(
    model_directory: Path,
    capture_directory: Path,
    words_path: Path,
    output_directory: Path,
    *,
    device: str,
    dim_batch: int,
    source_layers: list[int] | None,
) -> dict[str, object]:
    if output_directory.exists():
        raise ValueError(f"output directory already exists: {output_directory}")
    if dim_batch <= 0:
        raise ValueError("dim_batch must be positive")
    metadata_path = capture_directory / "capture.json"
    residual_path = capture_directory / "residuals.f32"
    metadata = json.loads(metadata_path.read_text())
    required = {
        "schema_version", "model_id", "source_model_id",
        "model_source_fingerprint", "runtime_identity_sha256", "layer_map",
        "hidden_size", "seq_len", "skip_first_positions", "corpus_spec",
        "raw_corpus_sha256", "capture_plan_sha256", "residual_sha256", "rows",
    }
    if set(metadata) != required or metadata["schema_version"] != 1:
        raise ValueError("browser J-lens capture metadata has an invalid schema")
    layers = _integer_list(metadata["layer_map"], "layer_map")
    hidden_size = _positive_integer(metadata["hidden_size"], "hidden_size")
    seq_len = _positive_integer(metadata["seq_len"], "seq_len")
    skip_first = _nonnegative_integer(
        metadata["skip_first_positions"], "skip_first_positions",
    )
    if skip_first != SKIP_FIRST_POSITIONS:
        raise ValueError(
            f"browser J-lens capture uses skip_first_positions={skip_first}; "
            f"the artifact schema requires {SKIP_FIRST_POSITIONS}"
        )
    for field in (
        "model_id", "source_model_id", "model_source_fingerprint",
        "runtime_identity_sha256", "corpus_spec", "raw_corpus_sha256",
        "capture_plan_sha256", "residual_sha256",
    ):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"browser J-lens capture has an invalid {field}")
    for field in (
        "runtime_identity_sha256", "raw_corpus_sha256",
        "capture_plan_sha256", "residual_sha256",
    ):
        if not _sha256(metadata[field]):
            raise ValueError(f"browser J-lens capture has an invalid {field}")
    if not _hex_digest(metadata["model_source_fingerprint"], 40):
        raise ValueError("browser J-lens capture has an invalid model_source_fingerprint")
    if hash_file(residual_path) != metadata["residual_sha256"]:
        raise ValueError("browser J-lens residual digest does not match")
    rows = _capture_rows(
        metadata["rows"],
        residual_path.stat().st_size,
        layer_count=len(layers),
        hidden_size=hidden_size,
        sequence_limit=seq_len,
        skip_first=skip_first,
    )
    plan = hashlib.sha256()
    for row in rows:
        plan.update(_canonical_json({"inputIds": row["input_ids"]}))
    if plan.hexdigest() != metadata["capture_plan_sha256"]:
        raise ValueError("browser J-lens capture plan digest does not match")

    model = build_transformers_model(model_directory, device)
    model_layers = list(model.model.layers)
    if layers != list(range(len(model_layers))) or hidden_size != int(model.config.hidden_size):
        raise ValueError("browser J-lens capture does not match the dequantized model shape")
    requested_layers = (
        list(range(len(model_layers) - 1))
        if source_layers is None
        else sorted(set(source_layers))
    )
    if not requested_layers or any(
        layer < 0 or layer >= len(model_layers) - 1 for layer in requested_layers
    ):
        raise ValueError(
            f"source layers must lie in [0, {len(model_layers) - 1})"
        )

    residuals = np.memmap(residual_path, dtype="<f4", mode="r")
    corrector = BrowserResidualCorrector(model_layers)
    lens = None
    try:
        for index, row in enumerate(rows):
            count = len(row["input_ids"])
            start = row["byte_offset"] // 4
            stop = start + len(layers) * count * hidden_size
            reference = np.asarray(residuals[start:stop]).reshape(
                len(layers), count, hidden_size,
            )
            if not np.isfinite(reference).all():
                raise ValueError(f"browser J-lens residual row {index} is non-finite")
            corrector.set(torch.from_numpy(np.array(reference, copy=True)))
            lens = fit_jacobian_lens(
                model,
                _PreparedTokenizer(),
                [f"browser-capture-row-{index}"],
                model_layers,
                source_layers=requested_layers,
                dim_batch=dim_batch,
                prompt_batch=1,
                max_seq_len=seq_len,
                skip_first=skip_first,
                checkpoint_every=1,
                input_id_rows=[row["input_ids"]],
                vjp_mode="batched",
                suppress_terminal_checkpoint=True,
                initial_lens=lens,
                on_progress=lambda message: print(message, flush=True),
                progress_base=index,
            )
    finally:
        corrector.close()
    if lens is None:
        raise ValueError("browser J-lens capture contains no usable rows")

    words = _load_words(words_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_directory), local_files_only=True, trust_remote_code=False,
    )
    vocabulary = []
    token_ids = []
    for row, word in enumerate(words):
        token_id = resolve_word_token(tokenizer, word)
        vocabulary.append({"word": word, "row": row, "token_id": token_id})
        token_ids.append(token_id)
    output_directory.mkdir(parents=True)
    try:
        root = output_directory / "packs" / "jlens"
        root.mkdir(parents=True)
        _save_lens_at(
            lens,
            root / "jlens.safetensors",
            root / "manifest.json",
            corpus_spec=metadata["corpus_spec"],
            corpus_sha256=_token_corpus_sha256(rows),
            seq_len=seq_len,
            dim_batch=dim_batch,
            skip_first=skip_first,
            corpus_hash_kind="token_ids_v1",
            durable=True,
            raw_corpus_sha256=metadata["raw_corpus_sha256"],
            raw_prompt_count=len(rows),
            usable_prompt_count=len(rows),
            model_layer_count=len(model_layers),
            model_fingerprint=metadata["runtime_identity_sha256"],
            model_source_fingerprint=metadata["model_source_fingerprint"],
        )
        shutil.rmtree(root / ".locks", ignore_errors=True)
        _stabilize_tensor_names(root)
        output_embeddings = model.get_output_embeddings()
        weight = getattr(output_embeddings, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError("dequantized model does not expose its language-head weight")
        selected = weight[token_ids].detach().to(device="cpu", dtype=torch.float32).contiguous()
        vocabulary_tensor = root / "browser-vocabulary.safetensors"
        save_file({"unembedding": selected}, vocabulary_tensor)
        vocabulary_sha256 = hash_file(vocabulary_tensor)
        (root / "browser-vocabulary.json").write_text(
            json.dumps(
                {
                    "format": "drowse-jlens-vocabulary-v2",
                    "hidden_size": hidden_size,
                    "tensor_file": vocabulary_tensor.name,
                    "tensor_sha256": vocabulary_sha256,
                    "words": vocabulary,
                },
                indent=2,
            ) + "\n"
        )
    except BaseException:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise
    return {
        "modelId": metadata["model_id"],
        "runtimeIdentitySha256": metadata["runtime_identity_sha256"],
        "prompts": lens.n_prompts,
        "layers": lens.source_layers,
        "hiddenSize": lens.d_model,
        "words": len(words),
        "outputDirectory": str(output_directory),
    }


class _PreparedTokenizer:
    pad_token_id = 0
    eos_token_id = 0


def _capture_rows(
    value: object,
    file_bytes: int,
    *,
    layer_count: int,
    hidden_size: int,
    sequence_limit: int,
    skip_first: int,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("browser J-lens capture rows must be a non-empty list")
    rows = []
    expected_offset = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "input_ids", "byte_offset", "byte_length",
        }:
            raise ValueError(f"browser J-lens capture row {index} has an invalid schema")
        input_ids = item["input_ids"]
        if (
            not isinstance(input_ids, list)
            or len(input_ids) <= skip_first + 1
            or len(input_ids) > sequence_limit
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in input_ids)
        ):
            raise ValueError(f"browser J-lens capture row {index} has invalid input IDs")
        offset = _nonnegative_integer(item["byte_offset"], f"rows[{index}].byte_offset")
        length = _positive_integer(item["byte_length"], f"rows[{index}].byte_length")
        expected_length = layer_count * len(input_ids) * hidden_size * 4
        if offset != expected_offset or length != expected_length:
            raise ValueError(f"browser J-lens capture row {index} has an invalid extent")
        rows.append({"input_ids": input_ids, "byte_offset": offset, "byte_length": length})
        expected_offset += length
    if expected_offset != file_bytes:
        raise ValueError("browser J-lens capture rows do not close the residual file")
    return rows


def _load_words(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(word, str) or not word or word.strip() != word for word in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("J-lens words must be a non-empty JSON array of unique trimmed strings")
    return value


def _stabilize_tensor_names(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tensor_files = manifest.get("tensor_files")
    source_layers = manifest.get("source_layers")
    if not isinstance(tensor_files, dict) or not isinstance(source_layers, list):
        raise ValueError("J-lens writer produced invalid tensor pointers")
    stable = {}
    for layer in source_layers:
        source_name = tensor_files.get(str(layer))
        if not isinstance(source_name, str):
            raise ValueError(f"J-lens writer omitted tensor layer {layer}")
        target_name = f"jlens.layer-{layer}.safetensors"
        os.replace(root / source_name, root / target_name)
        stable[str(layer)] = target_name
    manifest["tensor_files"] = stable
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _token_corpus_sha256(rows: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        _canonical_json([row["input_ids"] for row in rows])
    ).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _integer_list(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value)
        or any(value[index - 1] >= value[index] for index in range(1, len(value)))
    ):
        raise ValueError(f"browser J-lens capture has an invalid {label}")
    return value


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"browser J-lens capture has an invalid {label}")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"browser J-lens capture has an invalid {label}")
    return value


def _sha256(value: object) -> bool:
    return _hex_digest(value, 64)


def _hex_digest(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("words", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dim-batch", type=int, default=64)
    parser.add_argument("--source-layers", type=str)
    arguments = parser.parse_args()
    source_layers = (
        None
        if arguments.source_layers is None
        else [int(value) for value in arguments.source_layers.split(",") if value]
    )
    result = fit_pack(
        arguments.model_directory,
        arguments.capture_directory,
        arguments.words,
        arguments.output_directory,
        device=arguments.device,
        dim_batch=arguments.dim_batch,
        source_layers=source_layers,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
