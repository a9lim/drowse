# Post-block residual hook ABI v4

The browser runtime hook is attached after each zero-based transformer block and
before its output becomes the next block's input. Qwen3 and Llama/Smol adapters
must expose the same logical boundary.

## Residual contract

- Input and output shape: `[batch, sequence, hidden]`.
- If an architecture returns a tuple, element zero is the residual and all
  other elements pass through unchanged.
- Decode capture uses `[0, sequence - 1, :]`.
- Prompt capture may select explicit non-padding sequence positions.
- Hook-disabled execution must not allocate steering/probe buffers or change
  model logits.

## Program contract

One immutable program is uploaded before a generation. Each layer entry names
up to four affine groups, four curved terms, and eight probes. GPU buffers
remain valid for the generation and are released on stop, unload, worker
termination, or device loss.

The rank-one feasibility program uses layer-major structure-of-arrays buffers:
`uint32 enabled[layer]`, fp32 `basis[layer, hidden]`, `neutral[layer, hidden]`,
`target[layer]`, `along[layer]`, `collapse[layer]`, `probe_basis[layer, hidden]`,
and `probe_neutral[layer, hidden]`. The application validates dimensions,
finiteness, and unit-length bases before GPU upload. The shared
`fixtures/post-block-rank-one-v2.json` file is the Python/browser golden for
this subset; passing it is necessary but not sufficient to mark the production
hook runtime verified.

Affine rank-one injection uses fp32 intermediates:

```text
centered = h - neutral
coordinate = dot(centered, basis)
delta = along * (target - collapse * coordinate)
h' = h + delta * basis
```

Affine injection is deliberately uncapped, matching Python's current
`subspace_inject` semantics. ABI v2 replaced v1, whose feasibility fixture
incorrectly applied the curved-manifold safety cap to affine injection. The
result is converted once to the residual runtime dtype before the next block.
A `standard-v1` library accepts `drowse-structured-v2`. `standard-v2` and
`standard-v3` libraries accept `drowse-structured-v3`, which adds fixed geometry
inputs and outputs without changing the rank-one compatibility contract.
`standard-v3` also attests `exact-readout-v1`, top-k width eight, and the bounded
SAE feature-chunk capacity. These readout fields do not alter the structured
program buffers, but they require the matching compiled VM entry points. The compiled library
descriptor attests the accepted format and all fixed capacities before upload;
a v2 library never receives v3 buffers.

A structured-v3 program also carries packed multi-curve state and exact J-lens
probability probes without changing the rank-one compatibility contract.
A probe reads the post-injection residual at the same boundary. Dynamic terms
receive phase, decode step, thinking state, and prior-step gate scalars; they
never read a partially computed current-step measurement.

The scalar structured-probe output remains `[layer, 8]` in both structured
formats. V3 adds a separate geometry output `[layer, 8, 41]`; each record is:

```text
[valid, fraction, residual, membership,
 coords[4], node_distance[33]]
```

Only the first `intrinsic_dim` coordinates and `candidate_count` distances are
meaningful. Distances are raw in the probe's whitened reduced metric. The host
share-weights them across layers, divides nearest-label reports by the fixed
typical-label-spacing scale, and computes the assignment posterior with the
fixed per-candidate `tau` and `-rank * log(tau)` terms. A geometry gate reads
the complete prior-token host score map; it does not consume an approximate
scalar slot from the current forward.

When `persist_subspace_coords` is explicitly enabled, the host reconstructs the
exact per-layer whitened reduced coordinate from these candidate distances and
the immutable `geometry_node_white` anchors. Each active slot must provide at
least `rank + 1` finite anchors whose affine span has the slot's actual rank;
the reconstructed point must reproduce every reported candidate distance
within the fp32 readout tolerance. The persisted map uses the model-layer
numbers from the program's layer map and each vector uses that layer slot's
actual rank. Invalid anchors, distances, or residuals fail the opted-in
generation with a typed error. With the flag disabled the field is omitted and
the legacy geometry aggregation remains unchanged. Reconstruction reuses the
existing geometry read and device synchronization; it adds no GPU output,
readback, or synchronization.

The rank-one compatibility output has shape `[layer]`; selective capture output
has shape `[layer, position, hidden]`. Implementations write each layer into one
shared output allocation. They must not lower a final all-layer concatenation
into a single shader, because its storage-binding count grows with layer count
and is not valid on baseline WebGPU adapters.

Probe kind 1 is a linear scalar `dot(h, direction) + bias`. Probe kind 2 applies
ReLU to that scalar and is the exact SAE encoder-channel primitive. Probe kind
3 selects a probability from the model's full-vocabulary softmax after an fp32
J-lens transport and the model's native final norm and language head. J-lens
steering uses an fp32 `W_U[v] @ J_l` direction prepared before generation.

Both operations accept a dynamic fitted-layer dimension bounded at compile time
by the model's layer count. WebLLM partitions the fp32 matrices into
`[chunk, hidden, hidden]` inputs no larger than the smaller of
`maxBufferSize` and `maxStorageBufferBindingSize`; a pack is rejected before
generation if even one matrix cannot fit. WebLLM uses one matrix per chunk to
avoid runtime-dependent failures in larger dynamic J-lens batches. Partial v6
lenses preserve fitted source-layer order and do not allocate zero-padded
all-layer Jacobian buffers.

The probability entry point also receives the complete
`[model_layer, hidden]` capture and the chunk's layer indices, so residuals and
validated Jacobian chunks remain GPU-resident. WebLLM uploads those chunks once
when installing the structured program, invokes the exact full-vocabulary
softmax once per chunk on each read, performs one combined device
synchronization, and scatters the `[chunk, 8]` rows into model-layer order.
Program replacement, clearing, unload, device loss, and worker termination
dispose the persistent inputs. Direction preparation uses the same bounded
chunk plan, with temporary inputs disposed after each chunk.

This batched transport remains part of the unreleased implementation. It keeps
the residual hook ABI and structured-program format unchanged, but it is bound
by the `standard-v3` capacity profile and `exact-readout-v1` descriptor because
it adds compiled MLC entry points. The MLC model library and WebLLM bridge must
be rebuilt and digest-bound together; an older experimental build cannot pass
attestation.

Curved terms retain Python's safety cap, carry the final prompt foot into the
first decode step, and then carry the previous decode foot forward. Each curve
uses a fixed padded program: rank eight, thirty-two RBF nodes, four intrinsic
dimensions, and eight embedded dimensions. Up to four curves compose in slot
order. Domain tags are `0` for an unused slot, `1` for box/custom domains, and
`2` for a sphere. Periodic dimensions use cosine/sine embedding and shortest
wrapped target displacement. Sphere `S^n` uses `n + 1` contiguous embedded
components, hyperspherical embed/Jacobian, polar clamp plus azimuth wrap, and
the Python curvature-correct translation: log at the neutral origin, parallel
transport to the current foot, scaled exponential map, then unembed. The exact
surface math remains the `subspace_inject` contract; an implementation that
substitutes Euclidean projection is incompatible with this ABI.

The browser packs every curve slot into one 671-float parameter record. Curve
basis, neutral, packed parameters, and feet are the only persistent curve
inputs. Affine application and curved reconstruction use fixed-layout
scheduled TIR kernels. This prevents Relax memory-planner aliasing and keeps
every generated WebGPU stage within WebGPU's guaranteed eight-storage-buffer
limit.

## Structured-v3 geometry program

The geometry kernel exposes seven public buffers. Its immutable geometry is
packed into one payload; ten payload sections use direct flat indexing and the
curve-parameter section uses one alias view. The generated WGSL therefore has
exactly eight storage bindings. Release compilation checks the emitted shader,
not only the source-level TIR signature.

The `standard-v2` capacity profile fixes eight geometry probes, whitener rank
96, thirty-three candidates, rank eight, four intrinsic dimensions, and the
41-float output stride above. It supplies these layer-major arrays:

```text
whitener_rank[layer]
whitener_ridge[layer]
whitener_basis[layer, 96, hidden]
whitener_correction[layer, 96]

geometry_active/kind/rank/intrinsic_dim/candidate_count/curve_node_count[layer, 8]
geometry_mean/inverse_mean[layer, 8, hidden]
geometry_basis[layer, 8, 8, hidden]
geometry_gram_inverse/cholesky[layer, 8, 8, 8]
geometry_node_white[layer, 8, 33, 8]
geometry_coord_map[layer, 8, 4, 8]
geometry_coord_bias[layer, 8, 4]
geometry_curve_parameters[layer, 8, 671]
geometry_curve_node_coords[layer, 8, 32, 4]
geometry_curve_node_values[layer, 8, 32, 8]
geometry_feet[layer, 8, 4]
geometry_domain_kind[layer, 8]
```

The low-rank inverse is `Sigma^-1 h = h / ridge + U diag(correction) U^T h`.
Each probe subtracts the precomputed `Sigma^-1 mean`, projects through its basis
and reduced Gram inverse, and whitens with its Cholesky factor. Flat probes map
that reduced coordinate through `coord_map + coord_bias`. Curved probes solve
the nearest foot with the same packed 671-float RBF program as steering: cold
prefill uses three restarts and twelve iterations; warm decode uses the carried
foot plus nearest-node seeds, two restarts, and four iterations. The output
fraction is reduced Mahalanobis norm divided by total centered Mahalanobis norm,
residual is reduced off-surface norm divided by reduced query norm, and
membership is `exp(-raw_residual^2 / (2 * sigma(foot)^2))` when a sigma field
exists, otherwise one.

These layouts and formulas describe the implemented TypeScript compiler,
simulator, WebLLM bridge, and compiled hook entry points. They are ABI evidence,
not a claim that production q4 models or the physical browser matrix are
release-verified.

## Event contract

The worker persists each authoritative tree mutation before emitting
`tree_mutated`. `started` must open a generation stream before any mutation or
token, and `done` closes it. A stateful generation emits at least one
authoritative mutation after `started`, including its final mutation before
`done`; a stateless generation emits none. Both runtimes use:

```text
started -> (tree_mutated | token)* -> done
```

Every stream event contains a monotonic sequence number and generation ID. A
sequence gap requires a full tree snapshot before further deltas are applied.

## Runtime identity

An artifact binding includes the source-model commit, converted weight manifest
digest, quantization, tokenizer and chat-template digests, compiled library
digest, MLC/TVM runtime ABI, this hook ABI, layer map, hidden size, and context
profile. A friendly model ID is never sufficient for activation compatibility.
