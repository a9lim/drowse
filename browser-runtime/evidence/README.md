# Engineering evidence records

These optional records support regression tracking and support-policy decisions.
Their status does not gate downloads, generation, fitting, or release builds;
see [the runtime policy](../README.md).

The recorder executes registered observers and labels its output
`candidate-unattested`. The current CI workflow does not include an evidence
attestation job or evidence-target dispatch input.

Producer arguments live in a checked-in JSON config so the tested command is
part of the clean source revision. `arguments` is one string array for a
single observer run, or an array of string arrays when a gate combines multiple
physical reports:

```json
{
  "$schema": "../evidence-config.schema.json",
  "schemaVersion": 1,
  "evidenceType": "physicalBrowserBenchmark",
  "observer": "webui/scripts/browser-full-runtime.mjs",
  "arguments": ["--model-id", "MODEL", "--model-directory", "MODEL_DIR"]
}
```

After committing the executable source and config, record a new immutable
evidence file from the repository root:

```bash
node webui/scripts/record-hosted-evidence.mjs \
  benchmark \
  --evidence-type physicalBrowserBenchmark \
  --producer-config browser-runtime/evidence-configs/benchmark.json \
  --output browser-runtime/evidence/benchmark-macos-chrome.json
```

Use `gate` for runtime and authoring gates. The config must be a tracked regular
file, the worktree must otherwise be clean, and the output must be a new JSON
file directly under `browser-runtime/evidence/`. The recorder executes the
registered observer itself; it does not accept a prewritten measurement file.

Evidence records use `../release-evidence-record.schema.json` schema version 4.
Each record also binds the registered producer ID, the checked-in config path,
the config SHA-256, and a tool version that hashes both the producer closure and
the exact config bytes. Benchmark rows carry the same four fields. Renaming or
editing an observer config therefore invalidates old evidence instead of
silently changing what its command means.
Every gate result names a repository-relative `fixturePath` and the SHA-256 of
that exact file. The path must resolve to a regular, checked-in file inside the
Drowse repository. Symlinks, missing or untracked fixtures, path traversal, and
digest drift fail the gate. Runtime gates retain their numerical tolerances.
Gemma 3 270M is the long-prefill regression model and its record must identify
stock WebLLM `0.2.84` exactly. Qwen3-1.7B has independent tiny-fp32 and
production-q4 parity records; a Gemma long-prefill result cannot satisfy them.

Authoring evidence does not accept an open-ended assertions object. Each gate
has exact result fields in `release-evidence-record.schema.json`: activation
capture covers the post-block boundary and requested positions; fitting covers
neutral centering, whitening, PCA/spectral projection, RBF, and geometry error
bounds; topology covers flat, curved, periodic, and automatic selection;
serialization covers manifold v10, safetensors, `.drowse`, and runtime
fingerprint rejection; instruments cover core geometry, J-lens v6, SAE v1, and
binding/tensor validation; shared goldens cover every shared contract family;
physical fitting covers desktop and Android Chrome, OPFS spooling, a confirmed
non-fallback adapter, responsiveness, and a 50 ms maximum main-thread task.

Deterministic local observer configs are documented in
[the config README](../evidence-configs/README.md).
