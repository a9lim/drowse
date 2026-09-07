# Release evidence records

This directory is intentionally empty while the browser runtime is in the
`feasibility-required` state. Once the authoring implementation is complete,
`BROWSER_AUTHORING_IMPLEMENTATION_INTEGRATED = true` and
`BROWSER_AUTHORING_RUNTIME_INTEGRATED = true` record that the production path
is connected. Preview builds may exercise that path, while release builds keep
it unavailable unless `BROWSER_AUTHORING_RELEASE_VERIFIED` is backed by a
verified authoring evidence set. An evidence set with partial records may use
the parallel `integrated-unverified` status. Release tooling accepts
evidence only through a
content-addressed reference in `runtime-feasibility-evidence.json` or
`authoring-evidence.json`.

Every release gate and physical benchmark has a registered executable
producer. The recorder is deliberately fail-closed about its inputs:
hand-authored measurements, tool-version strings, commands, device labels, and
throughput claims cannot create a candidate record through that process. Its
in-process receipt marker is only an API-misuse guard, not a durable trust
boundary. A candidate becomes acceptable release evidence only after GitHub
Actions signs its exact bytes with the repository CI workflow's OIDC identity
and the `release-evidence-trust` job verifies that attestation against both the
signer workflow and the tested source commit.

`workflow_dispatch` on `.github/workflows/ci.yml` can execute and attest the
four deterministic local observers. It emits the exact record plus its
Sigstore bundle as a workflow artifact; the attestation is also retained by
GitHub's attestations service. Download the record without changing its bytes,
add its `{path, sha256}` reference to the appropriate evidence set, and let CI
verify it before merge. A locally generated record without that attestation is
intentionally rejected by the trust job even if its JSON is otherwise valid.
Repository rules must require the `release-evidence-trust` check for changes to
reach a release branch; configuring and preserving that branch protection is an
external repository-administration step.

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

Every referenced JSON file must contain the exact runtime and hook ABIs, the
canonical SHA-256 of `runtime-lock.json` with only its mutable `status` omitted,
the tested Polythetic source revision, production time, tool version, a passing
result, and gate-specific results. The lock checker reads the referenced file,
verifies its SHA-256, validates numerical tolerances and lifecycle assertions,
and rejects records older than 180 days. Opaque digests or status-marker changes
cannot satisfy a release gate, and toolchain or model identity drift changes the
canonical runtime digest.

Evidence records use `../release-evidence-record.schema.json` schema version 4.
Each record also binds the registered producer ID, the checked-in config path,
the config SHA-256, and a tool version that hashes both the producer closure and
the exact config bytes. Benchmark rows carry the same four fields. Renaming or
editing an observer config therefore invalidates old evidence instead of
silently changing what its command means.
Every record, evidence-set lock, and physical benchmark run carries the same
lowercase 40-character `polytheticRevision`: the commit whose executable source was
actually tested. The release revision supplied with `--polythetic-revision`,
`POLYTHETIC_RELEASE_REVISION`, or `CF_PAGES_COMMIT_SHA` names the release HEAD and
must equal `git rev-parse HEAD`; conflicting CLI and environment values are
rejected.

The tested revision must be an ancestor of the release HEAD. Between those
commits, only the evidence JSON records and locks, runtime/distribution
promotion locks, and hosted header promotion may change. Runtime, frontend,
package, and WASM source drift is rejected. The checker also reads
`runtime-lock.json` at the tested commit and requires its canonical identity to
match the release lock, so the permitted runtime-lock path can change only in
non-identity fields such as `status`. This ancestor-plus-allowlist model permits
evidence to be committed after testing without asking a Git commit to contain
its own hash.

The attestation verifier checks referenced gate files directly. Benchmark rows
are canonicalized with the same recursive key ordering and trailing newline as
the recorder, then verified by digest against the attestation for the original
standalone run record. This keeps the inline benchmark schema while preventing
a copied or edited row from inheriting another run's attestation.

Physical evidence still has to be produced on controlled hardware. The same
record-and-attest sequence must run in this repository's CI identity on a
trusted physical runner; an artifact attestation proves which workflow and
source revision produced the bytes, not that an untrusted self-hosted runner's
hardware labels were honest. Runner custody, environment protection/approval,
and actually executing the workflow therefore remain release operations rather
than facts local tests can synthesize.

Every gate result names a repository-relative `fixturePath` and the SHA-256 of
that exact file. The path must resolve to a regular, checked-in file inside the
Polythetic repository. Symlinks, missing or untracked fixtures, path traversal, and
digest drift fail the gate. Runtime gates retain their numerical tolerances.
Gemma 3 270M is the long-prefill regression model and its record must identify
stock WebLLM `0.2.84` exactly. Qwen3-1.7B has independent tiny-fp32 and
production-q4 parity records; a Gemma long-prefill result cannot satisfy them.

Authoring evidence does not accept an open-ended assertions object. Each gate
has exact result fields in `release-evidence-record.schema.json`: activation
capture covers the post-block boundary and requested positions; fitting covers
neutral centering, whitening, PCA/spectral projection, RBF, and geometry error
bounds; topology covers flat, curved, periodic, and automatic selection;
serialization covers manifold v10, safetensors, `.polythetic`, and runtime
fingerprint rejection; instruments cover core geometry, J-lens v6, SAE v1, and
binding/tensor validation; shared goldens cover every shared contract family;
physical fitting covers desktop and Android Chrome, OPFS spooling, a confirmed
non-fallback adapter, responsiveness, and a 50 ms maximum main-thread task.

Authoring release promotion remains evidence-gated. The runtime wiring stays
enabled for preview testing, while the release build requires every authoring
record to be verified and source-bound. `capabilities.ts` checks that evidence,
the runtime evidence, and the benchmark evidence before advertising fitting in
a release build, and the release checker rejects the build before bundling if
any gate is unresolved.

`benchmark-evidence.json` declares the release matrix. Verified evidence needs
at least one successful, current, confirmed-hardware run for every launch model
at 2K and 4K on every declared platform/browser combination. The development
files intentionally retain `polytheticRevision: null`, empty runs, and null evidence
references; never replace those nulls with synthetic results.

Deterministic local configs and their remaining prerequisites are documented in
`../evidence-configs/README.md`. The current immutable-pin audit is recorded in
`PROVENANCE_AUDIT.md`.
