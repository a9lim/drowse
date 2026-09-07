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

Creating a candidate evidence record still requires a committed clean worktree
and an immutable tested Polythetic revision. Acceptance additionally requires a
GitHub OIDC/Sigstore attestation from
`OWNER/REPOSITORY/.github/workflows/ci.yml`; the CI trust job rejects otherwise
valid but unattested JSON. The four configs above can be run through that
workflow's `release_evidence_target` dispatch input. The remaining configs
cannot be pinned truthfully until their compiled model artifacts or physical
environments exist:
tiny and production model parity, Gemma long-prefill, lifecycle device loss,
production activation capture, production instrument rejection checks, physical
desktop/Android fitting, and the 48-cell physical benchmark matrix.

Physical configs and their records must be executed and attested on controlled
self-hosted runners. Adding an attestation after copying a local result does not
establish the observation: the registered producer and attestation action must
share the source-bound workflow run. The resulting bundle may be retained for
offline audit, while normal CI verification uses GitHub's attestation service.

The shared-golden observer runs both the Python and browser consumers for the
structured-program, measurement, loom-transcript, topology/RBF, sampling, and
reciprocal artifact fixtures. The gate remains null until that observer is run
from a clean, committed revision through the recorder.
