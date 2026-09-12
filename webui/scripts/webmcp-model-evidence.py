"""Run the small-model WebMCP control experiment in new temporary caches."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


MODELS = (
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
    ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON destination; defaults to the temporary workspace")
    parser.add_argument("--device", choices=("cpu", "mps"), help="Default: MPS if available, otherwise CPU")
    parser.add_argument("--download-cache", type=Path, help="Optional existing HF Hub cache; defaults to a new temporary cache")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="drowse-webmcp-model-evidence-"))
    os.environ["DROWSE_HOME"] = str(root / "drowse")
    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(args.download_cache or root / "huggingface/hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    import transformers
    import drowse
    from drowse import DrowseSession, SamplingConfig
    from drowse.io.manifold_authoring import create_discover_manifold_folder
    from drowse.server.session_models import default_role_labels, role_support
    from huggingface_hub import snapshot_download

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[2]
    corpus_path = repo / "drowse/data/manifolds/personas/nodes/50_pirate.json"
    output = args.output or root / "evidence.json"
    evidence = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(), "device": device,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "drowse": drowse.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "dtype": "torch.float32", "max_tokens": 40,
            "seed": 1729, "temperature": 0, "prompt_count": 1,
            "scope": "isolated Python Drowse sessions; not hosted browser inference",
            "workspace": str(root), "artifacts": [],
            "corpus_sha256": {
                str(corpus_path.relative_to(repo)): hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            },
            "limitations": [
                "One prompt on a 135M model is not a general style benchmark.",
                "The 40-token cap truncates most outputs.",
                "Steering uses a new one-node pirate ray, not the full personas manifold.",
            ],
        },
        "runs": [],
    }

    def progress(message):
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    def record(row):
        evidence["runs"].append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n")
        progress(json.dumps({key: value for key, value in row.items() if key not in ("tokens", "model_info")}))

    progress("Isolated workspace: " + str(root))
    for model_id, revision in MODELS:
        started = time.monotonic()
        progress("Downloading pinned weights: " + model_id + "@" + revision)
        snapshot = Path(snapshot_download(model_id, revision=revision,
            cache_dir=os.environ["HF_HUB_CACHE"], allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        files = {}
        for path in sorted(snapshot.iterdir()):
            if path.is_file():
                with path.open("rb") as source:
                    files[path.name] = {"sha256": hashlib.file_digest(source, "sha256").hexdigest(), "bytes": path.stat().st_size}
        evidence["metadata"]["artifacts"].append({"model_id": model_id, "revision": revision, "files": files})
        model_path = root / ("instruct" if model_id.endswith("Instruct") else "base")
        model_path.symlink_to(snapshot, target_is_directory=True)
        session = DrowseSession.from_pretrained(str(model_path), device=device, dtype=torch.float32,
            probes=[], max_tokens=40, compile=False, trust_remote_code=False, on_progress=progress)
        try:
            assistant_role_ok, user_role_ok = role_support(session)
            default_assistant, default_user = default_role_labels(session)
            capabilities = {"role_substitution_supported": assistant_role_ok,
                "user_role_supported": user_role_ok, "scene_mode": session.scene_grammar is not None,
                "role_headers_apply_to_raw": False, "default_assistant_role": default_assistant,
                "default_user_role": default_user}
            record({"model": model_id, "phase": "load", "seconds": time.monotonic() - started,
                "is_base_model": session.is_base_model, "model_info": session.model_info,
                "template_capabilities": capabilities})
            prompt = "Tell me how to make a cup of tea in two sentences."
            if session.is_base_model:
                prompt = "Question: " + prompt + "\nAnswer:"
            for phase, system, role in (
                ("baseline", None, None),
                ("pirate_system", "Speak like a friendly pirate. Keep the answer useful and clear. Use a little nautical vocabulary.", None),
                ("pirate_role_header", None, "pirate"),
            ):
                session.config = replace(session.config, system_prompt=system)
                started = time.monotonic()
                try:
                    ids = session._prepare_input(prompt, raw=session.is_base_model, thinking=False,
                        stateless=True, assistant_role=role, to_device=False)
                    rendered = session.tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
                    effective_role = None if session.is_base_model else "pirate" if rendered.endswith("<|im_start|>pirate\n") else "assistant" if rendered.endswith("<|im_start|>assistant\n") else "unidentified"
                    run = session.generate(prompt, stateless=True, raw=session.is_base_model,
                        sampling=SamplingConfig(temperature=0, max_tokens=40, seed=1729, assistant_role=role),
                        steering=None, thinking=False).first
                    record({"model": model_id, "phase": phase, "prompt": prompt, "system_prompt": system,
                        "assistant_role": role, "raw": session.is_base_model, "text": run.text,
                        "requested_assistant_role": role, "effective_assistant_role": effective_role,
                        "rendered_prompt_tail": rendered[-240:],
                        "effective_role_source": "actual tokenized generation prompt inspected with the same Drowse input preparation function",
                        "tokens": run.tokens, "finish_reason": run.finish_reason, "seconds": time.monotonic() - started})
                except Exception as error:
                    record({"model": model_id, "phase": phase, "error": type(error).__name__ + ": " + str(error)})
            session.config = replace(session.config, system_prompt=None)
            if session.is_base_model:
                scaffold = ("A friendly pirate gives clear, useful advice.\n"
                    "Question: Tell me how to make a cup of tea in two sentences.\nPirate: Arrr, matey,")
                run = session.generate(scaffold, raw=True, stateless=True, thinking=False, steering=None,
                    sampling=SamplingConfig(temperature=0, max_tokens=40, seed=1729)).first
                record({"model": model_id, "phase": "raw_pirate_scaffold", "prompt": scaffold, "text": run.text})
            else:
                started = time.monotonic()
                try:
                    corpus = json.loads(corpus_path.read_text())
                    folder = create_discover_manifold_folder("webmcp", "pirate",
                        "Isolated evidence fit from bundled pirate corpus", fit_mode="pca", node_corpora={"pirate": corpus})
                    fitted = session.fit(folder, layers="workspace", on_progress=progress)
                    record({"model": model_id, "phase": "fit_pirate", "corpus_count": len(corpus),
                        "layer_count": len(fitted.layers), "seconds": time.monotonic() - started})
                    for coefficient in (0.01, 0.05, 0.25, 0.5):
                        expression = str(coefficient) + " webmcp/pirate%pirate"
                        run = session.generate(prompt, stateless=True, thinking=False, steering=expression,
                            sampling=SamplingConfig(temperature=0, max_tokens=40, seed=1729)).first
                        record({"model": model_id, "phase": "pirate_steering", "steering": expression,
                            "text": run.text, "tokens": run.tokens})
                except Exception as error:
                    record({"model": model_id, "phase": "fit_or_steer_pirate", "error": type(error).__name__ + ": " + str(error)})
        finally:
            session.close()
            del session
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    base = [row for row in evidence["runs"] if row["model"].endswith("135M") and row["phase"] in
        ("baseline", "pirate_system", "pirate_role_header")]
    evidence["metadata"]["base_raw_token_invariance"] = len(base) == 3 and all("tokens" in row for row in base) and base[0]["tokens"] == base[1]["tokens"] == base[2]["tokens"]
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    progress("Evidence JSON: " + str(output.resolve()))


if __name__ == "__main__":
    main()
