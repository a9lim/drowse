# Hosted fitting runtime contract

Browser manifold fitting is integrated with the hosted model backend. It is
available when the browser passes the normal runtime capability check and the
loaded model exposes compatible activation capture. Production currently
enables linear PCA fits and explicitly authored domains only.

Automatic topology discovery remains disabled by `BrowserModelBackend` through
`allowAutomaticTopologyDiscovery: false`. The acceptance criterion for geometric
detection is now synthetic manifolds in R^n, including held-out sampling and
embedding variants; real-model experiments are not a prerequisite for that
scope. Do not remove the gate based only on native Python tests: the hosted
detector still proposes periodic charts, and the compiled browser runtime does
not implement Klein-bottle or projective-plane quotient domains. The new native
surface evidence API does not recover coordinate charts automatically.
See [the original validation report](../TOPOLOGY_VALIDATION_2026-09-05.md) and
[the quotient-domain update](../SYNTHETIC_SURFACE_VALIDATION_2026-09-05.md).

The numerical and orchestration implementation includes:

- post-block WebGPU capture with exact prompt rendering, requested-position
  pooling, runtime/context fingerprints, and bounded OPFS row spooling;
- crash recovery, checkpoints, rollback, writer exclusion, cancellation, and
  stateless fitting-worker reads of committed fp32 rows;
- neutral centering, reduced-space whitening, Fisher/PCA and spectral topology
  selection, periodic-coordinate handling, automatic topology selection, RBF
  smoothing, sigma fields, and Python-matching DLS selection;
- authored, discover, template, extraction, and template-scoring orchestration;
- flat and curved manifold-v10/safetensors serialization into validated
  `.drowse` closures;
- validation-before-install, transactional installation, compiler refresh,
  probe reattachment, and fail-closed model invalidation when a committed
  artifact cannot be refreshed;
- exact core-pack, SAE, and J-lens binding checks at the runtime boundary.

Regression coverage must continue to compare browser and Python results for
capture, fitting geometry, topology selection, serialization, instruments, and
shared golden fixtures. Production artifacts must still match the selected
runtime ABI, hook ABI, model fingerprint, tokenizer/chat template, quantization,
layer map, and context profile. Compatibility checks do not supersede the
automatic-topology release gate or validate semantic topology claims.
