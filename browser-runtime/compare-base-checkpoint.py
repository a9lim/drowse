#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_source_architecture_reference(source, state, quantization):
    from transformers import initialization
    from mlc_q4_torch import replace_linears, replace_rms_norms

    config = AutoConfig.from_pretrained(source, local_files_only=True)
    if config.model_type == "qwen3_5":
        config = config.text_config
    if config.model_type not in {"gemma3_text", "qwen3_5_text"} or quantization not in {"q4f16_1", "q4f32_1"}:
        raise ValueError("The converted-weight source reference supports only Gemma 3 q4 or Qwen3.5 q4")
    dtype = torch.float16 if quantization == "q4f16_1" else torch.float32
    with initialization.no_init_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="eager")
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    # The converter has already folded the ordinary RMSNorm weight +1 into these tensors.
    replace_rms_norms(model)
    replace_linears(model)
    return model.eval().requires_grad_(False)


def verify_files(directory, entries):
    for entry in entries:
        path = directory / entry["path"]
        with path.open("rb") as source_file:
            digest = hashlib.file_digest(source_file, "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError(f"Artifact differs from browser build: {entry['path']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--browser-report", type=Path, required=True)
    parser.add_argument("--converted", type=Path,
                        help="Compare exact dequantized browser weights with the source architecture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    browser = json.loads(args.browser_report.read_text())
    prompt = browser["prompt"]
    if not prompt.startswith("I love marmots because"):
        raise SystemExit("Generation tests must use the marmot prefix")
    verify_files(args.source, browser["artifact"]["source"]["files"])
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    prefix_ids = browser.get("promptPrefixTokenIds", [])
    tokens = tokenizer.encode(prompt, add_special_tokens=bool(prefix_ids))
    if tokens != prefix_ids + tokenizer.encode(prompt, add_special_tokens=False):
        raise SystemExit("Browser completion prefix differs from the source tokenizer")
    if tokens != browser["inputIds"]:
        raise SystemExit("Browser tokenization differs from reference")
    if args.converted:
        from mlc_q4_torch import transformers_state_dict

        manifest_path = next((args.converted / f"{slug}-build.json"
                              for slug in ("drowse", "polythetic", "saklas")
                              if (args.converted / f"{slug}-build.json").is_file()), None)
        if manifest_path is None:
            raise SystemExit("Converted-weight build manifest is missing")
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != browser["artifact"]["modelBuildSha256"]:
            raise SystemExit("Converted-weight manifest differs from the browser build")
        converted = json.loads(manifest_bytes)
        verify_files(args.converted, converted["files"])
        state, config = transformers_state_dict(args.converted)
        model = load_source_architecture_reference(args.source, state, config["quantization"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.source, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
        ).eval()
    if model.config.model_type == "gpt2":
        layers = model.transformer.h
    elif model.config.model_type == "gpt_neox":
        layers = model.gpt_neox.layers
    elif model.config.model_type in {"gemma3_text", "qwen3_5_text", "qwen3_5"}:
        layers = model.model.layers
    else:
        raise SystemExit(f"Unsupported reference architecture: {model.config.model_type}")
    captured = []

    def capture(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value[0, browser["capture"]["positions"]].float().cpu().numpy())

    handles = [layer.register_forward_hook(capture) for layer in layers]
    input_ids = torch.tensor([tokens])
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)
    for handle in handles:
        handle.remove()
    reference_logits = output.logits[0, -1].float().numpy()
    gpu_logits = np.asarray(browser["firstLogits"])
    reference_capture = np.stack(captured)
    gpu_capture = np.asarray(browser["capture"]["values"]).reshape(reference_capture.shape)
    if (not all(np.isfinite(value.astype("float64")).all()
                for value in [reference_logits, gpu_logits, reference_capture, gpu_capture])):
        raise SystemExit("Non-finite model outputs cannot produce an accuracy result")
    count = min(len(reference_logits), len(gpu_logits), len(tokenizer))
    logit_metrics = metrics(reference_logits[:count], gpu_logits[:count])
    reference_top = np.argsort(reference_logits[:count])[-8:][::-1]
    gpu_top = np.argsort(gpu_logits[:count])[-8:][::-1]
    with torch.inference_mode():
        generated = model.generate(input_ids, max_new_tokens=24, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    result = {
        "schemaVersion": 1, "prompt": prompt, "inputIds": tokens,
        "referenceAddSpecialTokens": bool(prefix_ids),
        "torchVersion": torch.__version__, "referencePrecision": str(next(model.parameters()).dtype),
        "referenceWeights": "exact-dequantized-browser" if args.converted else "original-checkpoint",
        "referenceArchitecture": "source-config",
        "referenceAttention": "eager", "referenceDevice": "cpu",
        "browserArtifact": browser["artifact"],
        "logits": logit_metrics,
        "centeredLogits": metrics(reference_logits[:count] - reference_logits[:count].mean(),
                                  gpu_logits[:count] - gpu_logits[:count].mean()),
        "referenceTop": [{"id": int(token), "text": tokenizer.decode([token]), "logit": float(reference_logits[token])} for token in reference_top],
        "browserTop": [{"id": int(token), "text": tokenizer.decode([token]), "logit": float(gpu_logits[token])} for token in gpu_top],
        "captureByLayer": [metrics(ref, gpu) for ref, gpu in zip(reference_capture, gpu_capture)],
        "captureByPosition": [[metrics(ref, gpu) for ref, gpu in zip(ref_layer, gpu_layer)]
                              for ref_layer, gpu_layer in zip(reference_capture, gpu_capture)],
        "referenceResidualStats": [[{"mean": float(row.mean()), "std": float(row.std()), "maxAbs": float(np.max(np.abs(row)))}
                                    for row in layer] for layer in reference_capture],
        "referenceCompletion": tokenizer.decode(generated[0, len(tokens):]),
        "referenceGeneratedIds": generated[0, len(tokens):].tolist(),
        "browserGeneratedIds": browser["generatedIds"],
        "greedyTokenIdsMatch": generated[0, len(tokens):].tolist() == browser["generatedIds"],
        "browserCompletion": browser["completion"],
        "validation": "diagnostic-metrics-only",
    }
    with args.output.open("x") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "browserArtifact"}, indent=2))


def metrics(reference, actual):
    reference, actual = reference.astype("float64").reshape(-1), actual.astype("float64").reshape(-1)
    difference = actual - reference
    return {"cosine": float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual))),
            "relativeL2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "maxAbs": float(np.max(np.abs(difference)))}


if __name__ == "__main__":
    main()
