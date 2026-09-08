# Release evidence producer configs

These files pin the exact observer and arguments used by the release evidence
recorder. Their repository path and SHA-256 are stored in each evidence record,
and their bytes are included in the producer tool version. Editing or renaming a
config invalidates evidence made with the old config.

Four authoring gates have direct, deterministic local observers:

- `fitting-kernel-parity.json`
- `topology-orchestration.json`
- `manifold-serialization.json`
- `shared-golden-results.json`

Run the observer without recording evidence by invoking the registered producer:

```bash
node webui/scripts/hosted-release-evidence-producer.mjs \
  --evidence-type fittingKernelParity \
  --config browser-runtime/evidence-configs/fitting-kernel-parity.json \
  --nonce 0000000000000000000000000000000000000000000000000000000000000000
```

Creating a candidate evidence record requires a committed clean worktree and
an immutable tested Drowse revision. Physical observer configs also require
the matching compiled model artifacts and hardware. These are optional
engineering records; the current CI workflow has no evidence-target dispatch
input or evidence attestation job.

The shared-golden observer runs both the Python and browser consumers for the
structured-program, measurement, loom-transcript, topology/RBF, sampling, and
reciprocal artifact fixtures. Use the recorder from a clean, committed revision to capture its results.
