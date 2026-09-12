import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const tests = [];
const test = (name, run) => tests.push({ name, run });
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const loweringGolden = JSON.parse(readFileSync(
  new URL("../../browser-runtime/fixtures/affine-rank-one-lowering-v1.json", import.meta.url),
  "utf8",
));
const composedGolden = JSON.parse(readFileSync(
  new URL("../../browser-runtime/fixtures/affine-composed-lowering-v1.json", import.meta.url),
  "utf8",
));
const ablationGolden = JSON.parse(readFileSync(
  new URL("../../browser-runtime/fixtures/affine-ablation-lowering-v1.json", import.meta.url),
  "utf8",
));

try {
  const { BrowserFeasibilityCorePackCompiler } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserCorePack.ts",
  );
  const { runRankOneHookLayer } = await server.ssrLoadModule(
    "/src/hosted/runtime/rankOneHookProgram.ts",
  );
  const {
    initialStructuredHookState,
    runStructuredHookLayer,
    structuredHookControlsFor,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/structuredHookProgram.ts",
  );
  const { BrowserInstrumentRuntime } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserInstrumentRuntime.ts",
  );
  const { measurementEnvelope } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmGeneration.ts",
  );
  const {
    buildDrowseArchive,
    encodeFp32Safetensors,
    jsonBytes,
  } = await server.ssrLoadModule("/src/hosted/artifacts/index.ts");

  async function fixture(options = {}) {
    const fingerprint = options.fingerprint ?? "a".repeat(64);
    const contextBindingSha256 = options.contextBindingSha256 ?? "b".repeat(64);
    const fittedContextBindingSha256 = Object.hasOwn(options, "fittedContextBindingSha256")
      ? options.fittedContextBindingSha256 : contextBindingSha256;
    const secondContextBindingSha256 = "c".repeat(64);
    const sphere = options.sphere === true;
    const curved = options.curved === true || sphere;
    const fitMode = options.auto ? "auto" : curved ? "spectral" : "pca";
    const curveOrigin = options.curveOrigin ?? 0;
    const affineRank2 = options.affineRank2 === true || options.nonPoised === true;
    const saeFit = options.saeFit === true;
    const nodeRole = options.role ?? null;
    const labels = affineRank2 ? ["center", "east", "north"] : ["calm", "alert"];
    const nodeRoles = options.nodeRoles ?? labels.map(() => nodeRole);
    const tensorName = saeFit
      ? "_zdGVzdC9tb2RlbA_sae-_zojswyzlbonss2yi.safetensors"
      : "_zdGVzdC9tb2RlbA.safetensors";
    const sidecarName = tensorName.replace(".safetensors", ".json");
    const nodeFiles = Object.fromEntries(labels.map((label, index) => [
      `manifolds/local/demo/nodes/${String(index).padStart(2, "0")}_${label}.json`,
      jsonBytes([label]),
    ]));
    const tensors = encodeFp32Safetensors(curved ? {
      node_coords: sphere
        ? { shape: [2, 2], data: new Float32Array([Math.PI / 2, 0, Math.PI / 2, Math.PI / 2]) }
        : { shape: [2, 1], data: new Float32Array([0, 1]) },
      "layer_0.mean": { shape: [2], data: new Float32Array([0.1, 0.2]) },
      "layer_0.basis": { shape: [2, 2], data: new Float32Array([1, 0, 0, 1]) },
      "layer_0.node_params": sphere
        ? { shape: [2, 3], data: new Float32Array([0, 1, 0, 0, 0, 1]) }
        : options.periodic === true
        ? { shape: [2, 2], data: new Float32Array([0, 1, 0, -1]) }
        : { shape: [2, 1], data: new Float32Array([0, 1]) },
      "layer_0.rbf_weights": { shape: [2, 2], data: new Float32Array(4) },
      "layer_0.poly_coeffs": sphere
        ? { shape: [4, 2], data: new Float32Array([0, 0, 0, 0, 1, 0, 0, 1]) }
        : options.periodic === true
        ? { shape: [3, 2], data: new Float32Array([0, 0, 1, 0, 0, 1]) }
        : { shape: [2, 2], data: new Float32Array([0, 0, 1, 0]) },
      "layer_0.coord_offset": sphere
        ? { shape: [3], data: new Float32Array(3) }
        : options.periodic === true
        ? { shape: [2], data: new Float32Array([0, 0]) }
        : { shape: [1], data: new Float32Array([0]) },
      "layer_0.coord_scale": sphere
        ? { shape: [3], data: new Float32Array([1, 1, 1]) }
        : options.periodic === true
        ? { shape: [2], data: new Float32Array([1, 1]) }
        : { shape: [1], data: new Float32Array([1]) },
      "layer_0.sigma_rbf_weights": { shape: [2, 1], data: new Float32Array(2) },
      "layer_0.sigma_poly_coeffs": sphere
        ? { shape: [4, 1], data: new Float32Array([-20, 0, 0, 0]) }
        : options.periodic === true
        ? { shape: [3, 1], data: new Float32Array([-20, 0, 0]) }
        : { shape: [2, 1], data: new Float32Array([-20, 0]) },
      "layer_1.mean": { shape: [2], data: new Float32Array([0.3, 0.4]) },
      "layer_1.basis": options.mixedRank
        ? { shape: [1, 2], data: new Float32Array([1, 0]) }
        : { shape: [2, 2], data: new Float32Array([1, 0, 0, 1]) },
      "layer_1.node_params": sphere
        ? { shape: [2, 3], data: new Float32Array([0, 1, 0, 0, 0, 1]) }
        : options.periodic === true
        ? { shape: [2, 2], data: new Float32Array([0, 1, 0, -1]) }
        : { shape: [2, 1], data: new Float32Array([0, 1]) },
      "layer_1.rbf_weights": options.mixedRank
        ? { shape: [2, 1], data: new Float32Array(2) }
        : { shape: [2, 2], data: new Float32Array(4) },
      "layer_1.poly_coeffs": options.mixedRank
        ? sphere
          ? { shape: [4, 1], data: new Float32Array([0, 0, 1, 0]) }
          : options.periodic === true
          ? { shape: [3, 1], data: new Float32Array([0, 1, 0]) }
          : { shape: [2, 1], data: new Float32Array([0, 1]) }
        : sphere
        ? { shape: [4, 2], data: new Float32Array([0, 0, 0, 0, 1, 0, 0, 1]) }
        : options.periodic === true
        ? { shape: [3, 2], data: new Float32Array([0, 0, 1, 0, 0, 1]) }
        : { shape: [2, 2], data: new Float32Array([0, 0, 1, 0]) },
      "layer_1.coord_offset": sphere
        ? { shape: [3], data: new Float32Array(3) }
        : options.periodic === true
        ? { shape: [2], data: new Float32Array([0, 0]) }
        : { shape: [1], data: new Float32Array([0]) },
      "layer_1.coord_scale": sphere
        ? { shape: [3], data: new Float32Array([1, 1, 1]) }
        : options.periodic === true
        ? { shape: [2], data: new Float32Array([1, 1]) }
        : { shape: [1], data: new Float32Array([1]) },
      "layer_1.sigma_rbf_weights": { shape: [2, 1], data: new Float32Array(2) },
      "layer_1.sigma_poly_coeffs": sphere
        ? { shape: [4, 1], data: new Float32Array([-20, 0, 0, 0]) }
        : options.periodic === true
        ? { shape: [3, 1], data: new Float32Array([-20, 0, 0]) }
        : { shape: [2, 1], data: new Float32Array([-20, 0]) },
    } : affineRank2 ? {
      node_coords: {
        shape: [3, 2],
        data: options.nonPoised
          ? new Float32Array([0, 0, 1, 1, 2, 2])
          : new Float32Array([0, 0, 1, 0, 0, 1]),
      },
      "layer_0.mean": { shape: [2], data: new Float32Array([0.1, 0.2]) },
      "layer_0.basis": { shape: [2, 2], data: new Float32Array([1, 0, 0, 1]) },
      "layer_0.node_coords": {
        shape: [3, 2],
        data: new Float32Array([0, 0, 2, 0, 0, 4]),
      },
      "layer_1.mean": { shape: [2], data: new Float32Array([0.3, 0.4]) },
      "layer_1.basis": { shape: [2, 2], data: new Float32Array([1, 0, 0, 1]) },
      "layer_1.node_coords": {
        shape: [3, 2],
        data: new Float32Array([0, 0, 0, 3, 5, 0]),
      },
    } : {
      node_coords: { shape: [2, 1], data: new Float32Array([-1, 1]) },
      "layer_0.mean": { shape: [2], data: new Float32Array([0.1, 0.2]) },
      "layer_0.basis": { shape: [1, 2], data: new Float32Array([1, 0]) },
      "layer_0.node_coords": { shape: [2, 1], data: new Float32Array([-1, 1]) },
      "layer_1.mean": { shape: [2], data: new Float32Array([0.3, 0.4]) },
      "layer_1.basis": { shape: [1, 2], data: new Float32Array([0, 1]) },
      "layer_1.node_coords": { shape: [2, 1], data: new Float32Array([-2, 2]) },
    });
    const sidecar = jsonBytes({
      format_version: 10,
      name: "demo",
      method: saeFit
        ? "manifold_discover_sae"
        : options.auto ? "manifold_discover_auto"
        : curved ? "manifold_discover_spectral" : "manifold_discover_pca",
      drowse_version: "test",
      fit_mode: fitMode,
      hyperparams: {},
      diagnostics: options.diagnostics ?? {},
      node_count: labels.length,
      node_labels: labels,
      node_roles: nodeRoles,
      node_kinds: labels.map(() => "abstract"),
      domain: sphere
        ? { type: "sphere", dim: 2 }
        : curved
        ? {
            type: "box",
            axes: [{
              name: "u",
              periodic: options.periodic === true,
              period: options.periodic === true ? 2 : 1,
              lo: 0,
              hi: options.periodic === true ? 2 : 1,
            }],
          }
        : { type: "custom", embed_dim: affineRank2 ? 2 : 1, bounds: null },
      node_spread_per_layer: {},
      fitted_layers: [0, 1],
      mahalanobis_share_per_layer: Object.fromEntries(
        loweringGolden.layers.map((layer) => [String(layer.layer), layer.mahalanobisShare]),
      ),
      origin_per_layer: sphere
        ? { "0": [Math.PI / 2, 0], "1": [Math.PI / 2, 0] }
        : curved ? { "0": [curveOrigin], "1": [curveOrigin] } : {},
      feature_space: saeFit ? "sae-release-a" : "raw",
      nodes_sha256: discoverNodesHash(
        labels,
        Object.values(nodeFiles),
        fitMode,
        nodeRoles,
      ),
      sae_release: saeFit ? "release-a" : null,
      sae_revision: null,
      sae_fingerprint: null,
      sae_ids_by_layer: saeFit ? { "0": "feature-0", "1": "feature-1" } : {},
      sae_full_coverage: false,
      model_fingerprint: fingerprint,
      context_binding_sha256: fittedContextBindingSha256,
      model_source_fingerprint: null,
      capture_sha256: null,
      capture_version: null,
      capture_render_sha256: null,
      baseline_prompts_sha256: null,
      fit_policy_version: 1,
      share_metric: "mahalanobis",
      subspace_metric: "mahalanobis",
      rbf_smoothing_per_layer: curved ? {
        "0": { lambda: 0, edf: 2, gcv: 0 },
        "1": { lambda: 0, edf: 2, gcv: 0 },
      } : {},
      sigma_field_per_layer: curved ? {
        "0": { sigma_mean: 0, sigma_min: 0, sigma_max: 0, lambda: 0 },
        "1": { sigma_mean: 0, sigma_min: 0, sigma_max: 0, lambda: 0 },
      } : {},
      resolved_fit_mode: options.auto ? curved ? "spectral" : "pca" : null,
      topology_winner: options.auto ? "fixture" : null,
      topology_candidates: options.auto ? [{
        name: "fixture", fit_mode: curved ? "spectral" : "pca", intrinsic_dim: 1,
        score: 0.5, viable: true, reason: null,
      }] : [],
      components: null,
      bake_policy: null,
      source_model_id: null,
      source_model_fingerprint: null,
      transfer_quality_estimate: null,
    });
    const manifold = jsonBytes({
      format_version: 10,
      name: "demo",
      description: "core-pack fixture",
      fit_mode: fitMode,
      hyperparams: {},
      nodes: labels.map((label, index) => ({
        label,
        role: nodeRoles[index],
        kind: "abstract",
      })),
      files: {
        [tensorName]: sha256(tensors),
        [sidecarName]: sha256(sidecar),
      },
      source: "local",
      tags: [],
      template_ref: null,
    });
    const archive = await buildDrowseArchive({
      primary: "manifolds/local/demo",
      template: null,
      producerVersion: "browser-core-test",
      source: { uri: "local", repository: null, revision: null },
      files: {
        "manifolds/local/demo/manifold.json": manifold,
        ...nodeFiles,
        [`manifolds/local/demo/${tensorName}`]: tensors,
        [`manifolds/local/demo/${sidecarName}`]: sidecar,
      },
    });
    const bytes = new Uint8Array(await archive.arrayBuffer());
    const archiveName = options.archiveName ?? "demo.drowse";
    const manifest = {
      path: `core/${archiveName}`,
      role: "core_pack",
      url: `https://example.test/core/${archiveName}`,
      revision: "1".repeat(40),
      bytes: bytes.length,
      sha256: sha256(bytes),
    };
    const runtimeIdentity = {
      sourceModel: "test/model",
      sourceRevision: "2".repeat(40),
      convertedManifestSha256: "3".repeat(64),
      quantization: "q4f32_1",
      tokenizerSha256: "4".repeat(64),
      chatTemplateSha256: "5".repeat(64),
      modelLibrarySha256: "6".repeat(64),
      runtimeAbi: options.runtimeAbi ?? "drowse-browser-v1",
      hookAbi: "post-block-residual-v4",
      hiddenSize: 2,
      layerMap: [0, 1],
    };
    const manifests = [manifest];
    const artifacts = [{ manifest, file: new File([bytes], manifest.path) }];
    if (options.withSecondManifold) {
      const labels = ["soft", "sharp"];
      const secondRole = options.secondRole ?? null;
      const focusTensorName = "_zdGVzdC9tb2RlbA.safetensors";
      const focusSidecarName = focusTensorName.replace(".safetensors", ".json");
      const focusNodeFiles = {
        "manifolds/local/focus/nodes/00_soft.json": jsonBytes(["soft"]),
        "manifolds/local/focus/nodes/01_sharp.json": jsonBytes(["sharp"]),
      };
      const focusTensors = encodeFp32Safetensors({
        node_coords: { shape: [2, 1], data: new Float32Array([-1, 1]) },
        "layer_0.mean": { shape: [2], data: new Float32Array([0.2, 0.1]) },
        "layer_0.basis": { shape: [1, 2], data: new Float32Array([0.6, 0.8]) },
        "layer_0.node_coords": { shape: [2, 1], data: new Float32Array([-1.5, 1.5]) },
        "layer_1.mean": { shape: [2], data: new Float32Array([0.4, 0.3]) },
        "layer_1.basis": { shape: [1, 2], data: new Float32Array([0.8, 0.6]) },
        "layer_1.node_coords": { shape: [2, 1], data: new Float32Array([-0.5, 0.5]) },
      });
      const focusSidecar = jsonBytes({
        format_version: 10,
        name: "focus",
        method: "manifold_discover_pca",
        drowse_version: "test",
        fit_mode: "pca",
        hyperparams: {},
        diagnostics: {},
        node_count: 2,
        node_labels: labels,
        node_roles: [secondRole, secondRole],
        node_kinds: ["abstract", "abstract"],
        domain: { type: "custom", embed_dim: 1, bounds: null },
        node_spread_per_layer: {},
        fitted_layers: [0, 1],
        mahalanobis_share_per_layer: {
          "0": 3.4985713086386614,
          "1": 0.8315218588478542,
        },
        origin_per_layer: {},
        feature_space: "raw",
        nodes_sha256: discoverNodesHash(
          labels,
          Object.values(focusNodeFiles),
          "pca",
          [secondRole, secondRole],
        ),
        sae_release: null,
        sae_revision: null,
        sae_fingerprint: null,
        sae_ids_by_layer: {},
        sae_full_coverage: false,
        model_fingerprint: fingerprint,
        context_binding_sha256: fittedContextBindingSha256,
        model_source_fingerprint: "2".repeat(40),
        capture_sha256: null,
        capture_version: null,
        capture_render_sha256: null,
        baseline_prompts_sha256: null,
        fit_policy_version: 1,
        share_metric: "mahalanobis",
        subspace_metric: "mahalanobis",
        rbf_smoothing_per_layer: {},
        sigma_field_per_layer: {},
        resolved_fit_mode: null,
        topology_winner: null,
        topology_candidates: [],
        components: null,
        bake_policy: null,
        source_model_id: null,
        source_model_fingerprint: null,
        transfer_quality_estimate: null,
      });
      const focusManifold = jsonBytes({
        format_version: 10,
        name: "focus",
        description: "second core-pack fixture",
        fit_mode: "pca",
        hyperparams: {},
        nodes: labels.map((label) => ({ label, role: secondRole, kind: "abstract" })),
        files: {
          [focusTensorName]: sha256(focusTensors),
          [focusSidecarName]: sha256(focusSidecar),
        },
        source: "local",
        tags: [],
        template_ref: null,
      });
      const focusArchive = await buildDrowseArchive({
        primary: "manifolds/local/focus",
        template: null,
        producerVersion: "browser-core-test",
        source: { uri: "local", repository: null, revision: null },
        files: {
          "manifolds/local/focus/manifold.json": focusManifold,
          ...focusNodeFiles,
          [`manifolds/local/focus/${focusTensorName}`]: focusTensors,
          [`manifolds/local/focus/${focusSidecarName}`]: focusSidecar,
        },
      });
      const focusBytes = new Uint8Array(await focusArchive.arrayBuffer());
      const focusManifest = {
        path: "core/focus.drowse",
        role: "core_pack",
        url: "https://example.test/core/focus.polythetic",
        revision: "1".repeat(40),
        bytes: focusBytes.length,
        sha256: sha256(focusBytes),
      };
      manifests.push(focusManifest);
      artifacts.push({
        manifest: focusManifest,
        file: new File([focusBytes], focusManifest.path),
      });
    }
    if (options.withWhitener) {
      const whitenerTensors = encodeFp32Safetensors({
        "layer_0.mean": { shape: [2], data: new Float32Array([0.5, -0.5]) },
        "layer_0.basis": { shape: [1, 2], data: new Float32Array([1, 0]) },
        "layer_0.eigenvalues": { shape: [1], data: new Float32Array([2]) },
        "layer_0.inverse_scales": { shape: [1], data: new Float32Array([0.5]) },
        "layer_1.mean": { shape: [2], data: new Float32Array([1.5, -1.5]) },
        "layer_1.basis": { shape: [1, 2], data: new Float32Array([0, 1]) },
        "layer_1.eigenvalues": { shape: [1], data: new Float32Array([3]) },
        "layer_1.inverse_scales": { shape: [1], data: new Float32Array([0.25]) },
      });
      const tensorManifest = {
        path: "core/neutral-whitener.safetensors",
        role: "core_pack",
        url: "https://example.test/core/neutral-whitener.safetensors",
        revision: "1".repeat(40),
        bytes: whitenerTensors.length,
        sha256: sha256(whitenerTensors),
      };
      const whitenerSidecar = jsonBytes({
        format_version: 1,
        runtime_identity_sha256: fingerprint,
        context_binding_sha256: contextBindingSha256,
        hidden_size: 2,
        layer_map: [0, 1],
        tensors_sha256: tensorManifest.sha256,
        ridge_per_layer: { "0": 0.25, "1": 0.5 },
      });
      const sidecarManifest = {
        path: "core/neutral-whitener.json",
        role: "core_pack",
        url: "https://example.test/core/neutral-whitener.json",
        revision: "1".repeat(40),
        bytes: whitenerSidecar.length,
        sha256: sha256(whitenerSidecar),
      };
      manifests.push(tensorManifest, sidecarManifest);
      artifacts.push(
        { manifest: tensorManifest, file: new File([whitenerTensors], tensorManifest.path) },
        { manifest: sidecarManifest, file: new File([whitenerSidecar], sidecarManifest.path) },
      );
      if (options.withSecondWhitener) {
        const secondTensorManifest = {
          ...tensorManifest,
          path: "core/4096/neutral-whitener.safetensors",
          url: "https://example.test/core/4096/neutral-whitener.safetensors",
        };
        const secondWhitenerSidecar = jsonBytes({
          format_version: 1,
          runtime_identity_sha256: fingerprint,
          context_binding_sha256: secondContextBindingSha256,
          hidden_size: 2,
          layer_map: [0, 1],
          tensors_sha256: secondTensorManifest.sha256,
          ridge_per_layer: { "0": 0.25, "1": 0.5 },
        });
        const secondSidecarManifest = {
          path: "core/4096/neutral-whitener.json",
          role: "core_pack",
          url: "https://example.test/core/4096/neutral-whitener.json",
          revision: "1".repeat(40),
          bytes: secondWhitenerSidecar.length,
          sha256: sha256(secondWhitenerSidecar),
        };
        manifests.push(secondTensorManifest, secondSidecarManifest);
        artifacts.push(
          {
            manifest: secondTensorManifest,
            file: new File([whitenerTensors], secondTensorManifest.path),
          },
          {
            manifest: secondSidecarManifest,
            file: new File([secondWhitenerSidecar], secondSidecarManifest.path),
          },
        );
      }
    }
    const optionalPacks = [];
    if (options.withSae) {
      const saeTensors = encodeFp32Safetensors({
        W_enc: { shape: [2, 2], data: new Float32Array([1, 0, 0, 2]) },
        W_dec: { shape: [2, 2], data: new Float32Array([1, 0, 0, 1]) },
        b_enc: { shape: [2], data: new Float32Array([0.5, -1]) },
        b_dec: { shape: [2], data: new Float32Array([0.25, 0.5]) },
        ...(options.saeFormatVersion === 2 ? {
          threshold: { shape: [2], data: new Float32Array([1.25, 2.5]) },
        } : {}),
      });
      const tensorFile = instrumentFile("packs/sae/layer-1.safetensors", saeTensors);
      const saeManifest = jsonBytes({
        format_version: options.saeFormatVersion ?? 1,
        kind: "local",
        name: "fixture",
        release: "local:fixture",
        model_id: "test/model",
        model_fingerprint: options.instrumentModelFingerprint ?? fingerprint,
        model_source_fingerprint: "2".repeat(40),
        activation: options.saeActivation ?? "relu",
        layer: 1,
        d_model: 2,
        d_sae: 2,
        tensor_file: "layer-1.safetensors",
        tensor_sha256: tensorFile.sha256,
        corpus_spec: "fixture",
        corpus_sha256: "8".repeat(64),
        tokens_trained: 2,
        seq_len: 2,
        batch_size: 1,
        learning_rate: 0.001,
        l1_coefficient: 0.01,
        dead_feature_threshold: 0,
        ...options.saeManifestOverrides,
      });
      const manifestFile = instrumentFile("packs/sae/manifest.json", saeManifest);
      const saeFiles = [manifestFile, tensorFile];
      if (options.withSaeMetadata) {
        const metadata = jsonBytes({
          format_version: 1,
          model_id: "test/model",
          release: "local:fixture",
          features: options.invalidSaeMetadata
            ? { "2": { label: "outside width", max_act: 2.5 } }
            : {
                "0": { label: "opening delimiters", max_act: 2.5 },
                "1": { label: null, max_act: null },
              },
        });
        saeFiles.push(instrumentFile("packs/sae/features.json", metadata));
      }
      optionalPacks.push(instrumentPack("fixture-sae", "sae", saeFiles, fingerprint));
    }
    if (options.withJlens) {
      const jlensLayers = options.partialJlens ? [1] : [0, 1];
      const lensTensors = encodeFp32Safetensors(Object.fromEntries(
        jlensLayers.map((layer) => [
          `layer_${layer}`,
          {
            shape: [2, 2],
            data: layer === 0
              ? new Float32Array([1, 0, 0, 1])
              : new Float32Array([1, 2, 0, 1]),
          },
        ]),
      ));
      const lensFile = instrumentFile("packs/jlens/layers.safetensors", lensTensors);
      const lensManifestBody = {
        format_version: 6,
        method: "jlens_cotangent_sum",
        n_prompts: 1,
        checkpoint: false,
        dtype: "float32",
        d_model: 2,
        source_layers: jlensLayers,
        tensor_files: Object.fromEntries(
          jlensLayers.map((layer) => [String(layer), "layers.safetensors"]),
        ),
        tensor_sha256: Object.fromEntries(
          jlensLayers.map((layer) => [String(layer), lensFile.sha256]),
        ),
        corpus_spec: "fixture",
        corpus_sha256: "9".repeat(64),
        corpus_hash_kind: "text_v1",
        seq_len: 2,
        dim_batch: 1,
        skip_first_positions: 0,
        estimator_policy: { method: "jlens_cotangent_sum" },
        raw_corpus_sha256: null,
        raw_prompt_count: null,
        usable_prompt_count: null,
        model_layer_count: 2,
        model_fingerprint: options.instrumentModelFingerprint ?? fingerprint,
        model_source_fingerprint: "2".repeat(40),
        base_n_prompts: null,
        partial_n_prompts: null,
        consumed_prefix_sha256: null,
      };
      const lensManifest = jsonBytes(lensManifestBody);
      const manifestFile = instrumentFile("packs/jlens/manifest.json", lensManifest);
      const vocabularyTensors = encodeFp32Safetensors({
        unembedding: { shape: [2, 2], data: new Float32Array([2, 3, -1, 4]) },
      });
      const vocabularyTensorFile = instrumentFile(
        "packs/jlens/browser-vocabulary.safetensors",
        vocabularyTensors,
      );
      const vocabularyManifest = jsonBytes({
        format: options.jlensVocabularyFormat ?? "drowse-jlens-vocabulary-v2",
        hidden_size: 2,
        tensor_file: "browser-vocabulary.safetensors",
        tensor_sha256: vocabularyTensorFile.sha256,
        words: [
          { word: "hello", row: 0, token_id: 0 },
          { word: "world", row: 1, token_id: 1 },
        ],
      });
      const vocabularyFile = instrumentFile(
        "packs/jlens/browser-vocabulary.json",
        vocabularyManifest,
      );
      optionalPacks.push(instrumentPack(
        "fixture-jlens",
        "jlens",
        [manifestFile, lensFile, vocabularyFile, vocabularyTensorFile],
        fingerprint,
      ));
      if (options.withSecondJlens) {
        const rLensFile = instrumentFile("packs/rlens/layers.safetensors", lensTensors);
        const rLensManifest = jsonBytes({
          ...lensManifestBody,
          method: "relp_cotangent_sum",
          estimator_policy: { method: "relp_cotangent_sum" },
          tensor_files: Object.fromEntries(
            jlensLayers.map((layer) => [String(layer), "layers.safetensors"]),
          ),
          tensor_sha256: Object.fromEntries(
            jlensLayers.map((layer) => [String(layer), rLensFile.sha256]),
          ),
        });
        const rLensManifestFile = instrumentFile("packs/rlens/manifest.json", rLensManifest);
        const rVocabularyTensorFile = instrumentFile(
          "packs/rlens/browser-vocabulary.safetensors",
          vocabularyTensors,
        );
        const rVocabularyManifest = jsonBytes({
          format: "drowse-jlens-vocabulary-v2",
          hidden_size: 2,
          tensor_file: "browser-vocabulary.safetensors",
          tensor_sha256: rVocabularyTensorFile.sha256,
          words: [
            { word: "hello", row: 0, token_id: 0 },
            { word: "world", row: 1, token_id: 1 },
          ],
        });
        const rVocabularyFile = instrumentFile(
          "packs/rlens/browser-vocabulary.json",
          rVocabularyManifest,
        );
        optionalPacks.push(instrumentPack(
          "fixture-rlens",
          "jlens",
          [rLensManifestFile, rLensFile, rVocabularyFile, rVocabularyTensorFile],
          fingerprint,
        ));
      }
    }
    const request = {
      model: { id: "fixture-model" },
      variant: {
        structuredHookProfile: options.structuredHookProfile ?? "standard-v1",
        runtimeIdentity,
        runtimeIdentitySha256: fingerprint,
        contextProfiles: [
          { contextTokens: 2048, bindingSha256: contextBindingSha256 },
          ...(options.withSecondWhitener
            ? [{ contextTokens: 4096, bindingSha256: secondContextBindingSha256 }]
            : []),
        ],
      },
      requiredCorePack: { files: manifests },
      artifacts,
      optionalPacks,
      contextTokens: 2048,
      signal: new AbortController().signal,
    };
    return { request, manifest, artifacts };
  }

  function instrumentFile(path, bytes) {
    const manifest = {
      path,
      role: "instrument",
      url: `https://example.test/${path}`,
      revision: "7".repeat(40),
      bytes: bytes.length,
      sha256: sha256(bytes),
    };
    return { ...manifest, manifest, file: new File([bytes], path) };
  }

  function instrumentPack(id, kind, files, fingerprint) {
    const manifests = files.map(({ manifest }) => manifest);
    return {
      pack: {
        id,
        kind,
        displayName: id,
        runtimeIdentitySha256: fingerprint,
        files: manifests,
      },
      artifacts: files.map(({ manifest, file }) => ({ manifest, file })),
    };
  }

  test("compiles an exact-runtime fitted node into per-layer hook buffers", async () => {
    const { request } = await fixture();
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile("0.5 alert");
    assert.equal(program.hiddenSize, 2);
    assert.equal(program.layerCount, 2);
    assert.deepEqual([...program.enabled], [1, 1]);
    assert.deepEqual([...program.basis], [1, 0, 0, 1]);
    assertFloatArray(
      program.target,
      loweringGolden.layers.map((layer) => layer.expectedTarget),
    );
    assertFloatArray(
      program.along,
      loweringGolden.layers.map((layer) => layer.expectedAlong),
    );
    assert.deepEqual([...program.collapse], [0, 0]);
  });

  test("caches immutable generation payloads and isolates public mutable programs", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const borrowed = compiler.compileForGeneration("0.5 alert");
    assert.equal(compiler.compileForGeneration("0.5 alert"), borrowed);
    const publicProgram = compiler.compile("0.5 alert");
    publicProgram.basis.fill(0);
    assert.deepEqual([...borrowed.basis], [1, 0, 0, 1]);
    assert.notEqual(compiler.compileForGeneration("0.3 alert"), borrowed);
    const plain = compiler.compileForGeneration("");
    compiler.attachProbe({ selector: "local/demo" });
    assert.notEqual(compiler.compileForGeneration(""), plain);
  });

  test("routes auto fits by their fitted geometry and preserves inspector diagnostics", async () => {
    for (const curved of [false, true]) {
      const diagnostics = curved
        ? { eigenvalues: [0, 0.5], picked_k: 1, gap_index: 1, gap_magnitude: 0.5, bandwidth: 1, k_nn: 1, component_count: 1 }
        : { per_component_variance: [1], cumulative_variance: [1], picked_k: 1, threshold: 0.9 };
      const { request } = await fixture({ auto: true, curved, diagnostics, withWhitener: true, structuredHookProfile: "standard-v3" });
      const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      const [summary] = compiler.listManifolds();
      assert.equal(summary.fit_mode, "auto");
      assert.equal(summary.resolved_fit_mode, curved ? "spectral" : "pca");
      assert.equal(summary.fitted_for_session, true);
      assert.equal(compiler.attachProbe({ selector: "local/demo" }).is_affine, !curved);
      const detail = compiler.getManifold("local", "demo");
      assert.deepEqual(detail.fitted[0].diagnostics, diagnostics);
      assert.equal(detail.fitted[0].fit_mode, "auto");
      detail.fitted[0].diagnostics.picked_k = 99;
      assert.equal(compiler.getManifold("local", "demo").fitted[0].diagnostics.picked_k, 1);
      assert.ok(compiler.compile("0.1 local/demo%alert").layerCount > 0);
    }
  });

  test("loads a verified legacy-named core archive", async () => {
    for (const extension of ["polythetic", "saklas", "saklaspack"]) {
      const { request } = await fixture({ archiveName: `demo.${extension}` });
      const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      assert.equal(compiler.compile("0.5 alert").layerCount, 2);
    }
  });

  test("rejects explicitly context-bound core fits from a different context", async () => {
    const { request } = await fixture({ fittedContextBindingSha256: "e".repeat(64) });
    await assert.rejects(() => BrowserFeasibilityCorePackCompiler.load(request),
      /has no fit for this exact browser runtime/);
    const unbound = await fixture({ fittedContextBindingSha256: null });
    assert.equal((await BrowserFeasibilityCorePackCompiler.load(unbound.request))
      .compile("0.5 alert").layerCount, 2);
  });

  test("selects the exact-context fit from a bundled archive collection", async () => {
    const first = await fixture({ archiveName: "demo-2048.drowse" });
    const second = await fixture({ archiveName: "demo-4096.drowse",
      fittedContextBindingSha256: "e".repeat(64) });
    const request = first.request;
    request.variant.contextProfiles.push({ contextTokens: 4096, bindingSha256: "e".repeat(64) });
    request.artifacts.push(...second.request.artifacts);
    request.requiredCorePack.files.push(...second.request.requiredCorePack.files);
    for (const contextTokens of [2048, 4096]) {
      request.contextTokens = contextTokens;
      const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      assert.equal(compiler.listManifolds().length, 1);
      assert.equal(compiler.compile("0.5 alert").layerCount, 2);
    }
    request.variant.contextProfiles.push({ contextTokens: 8192, bindingSha256: "f".repeat(64) });
    request.contextTokens = 8192;
    await assert.rejects(() => BrowserFeasibilityCorePackCompiler.load(request),
      /has no fit for this exact browser runtime/);
  });

  test("accepts only the deterministic legacy runtime identity alias", async () => {
    const baseline = await fixture();
    const currentIdentity = {
      ...baseline.request.variant.runtimeIdentity,
      runtimeAbi: "drowse-web-runtime-v1",
    };
    const currentFingerprint = sha256(canonicalJson(currentIdentity));
    for (const slug of ["polythetic", "saklas"]) {
      const legacyFingerprint = sha256(canonicalJson({
        ...currentIdentity,
        runtimeAbi: `${slug}-web-runtime-v1`,
      }));
      const { contextBindingSha256 } = await server.ssrLoadModule("/src/lib/runtime/catalog.ts");
      const legacy = await fixture({ fingerprint: legacyFingerprint,
        fittedContextBindingSha256: contextBindingSha256(legacyFingerprint, 2048) });
      legacy.request.variant.runtimeIdentity = currentIdentity;
      legacy.request.variant.runtimeIdentitySha256 = currentFingerprint;

      const compiler = await BrowserFeasibilityCorePackCompiler.load(legacy.request);
      assert.equal(compiler.compile("0.5 alert").layerCount, 2);
    }
  });

  test("requires explicit role selectors and stamps the active generation role", async () => {
    const { request } = await fixture({ role: "scholar" });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const [profile] = compiler.listProfiles().profiles;
    assert.equal(profile.name, "local/demo:role-scholar");
    assert.equal(profile.metadata.selector_key, "local/demo:role-scholar");
    assert.equal(profile.metadata.feature_space, "raw");
    assert.equal(profile.metadata.variant, "raw");
    assert.equal(profile.metadata.variant_identity, null);
    assert.throws(
      () => compiler.compile("0.5 local/demo"),
      (error) => error.code === "CORE_PACK_ROLE_VARIANT_REQUIRED",
    );
    const program = compiler.compile("0.5 local/demo:role-scholar");
    assert.equal(program.activeRole, "scholar");
  });

  test("infers mixed per-node roles from labels and authoring coordinates", async () => {
    const { request } = await fixture({
      nodeRoles: ["pirate", "scholar"],
      withWhitener: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.throws(
      () => compiler.compile("0.5 local/demo"),
      (error) => error.code === "CORE_PACK_ROLE_VARIANT_REQUIRED",
    );
    assert.equal(compiler.compile("0.5 local/demo%calm").activeRole, "pirate");
    assert.equal(compiler.compile("0.5 local/demo%alert").activeRole, "scholar");
    assert.equal(compiler.compile("0.5 local/demo%0.75").activeRole, "scholar");
    assert.equal(
      compiler.compile("0.2 local/demo%calm + 0.8 local/demo%alert").activeRole,
      "scholar",
    );
    assert.throws(
      () => compiler.compile("0.5 local/demo:role-pirate"),
      (error) => error.code === "CORE_PACK_VARIANT_UNAVAILABLE",
    );
  });

  test("uses chordal domain geometry for per-node role selection", async () => {
    const periodic = await fixture({
      curved: true,
      periodic: true,
      nodeRoles: ["seam", "opposite"],
      withWhitener: true,
    });
    const periodicCompiler = await BrowserFeasibilityCorePackCompiler.load(periodic.request);
    assert.equal(periodicCompiler.compile("0.5 local/demo%1.8").activeRole, "seam");
    assert.equal(periodicCompiler.compile("0.5 local/demo%1.2").activeRole, "opposite");

    const sphere = await fixture({
      sphere: true,
      nodeRoles: ["prime", "quarter"],
      withWhitener: true,
      structuredHookProfile: "standard-v3",
    });
    const sphereCompiler = await BrowserFeasibilityCorePackCompiler.load(sphere.request);
    assert.equal(
      sphereCompiler.compile(`0.5 local/demo%${Math.PI / 2},${Math.PI * 0.45}`).activeRole,
      "quarter",
    );
  });

  test("lets an explicit role override a manifold-implied role", async () => {
    const { request } = await fixture({
      nodeRoles: ["pirate", "scholar"],
      withSecondManifold: true,
      secondRole: "narrator",
      withWhitener: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile(
      "0.8 local/demo%alert + 0.2 local/focus:role-narrator",
    );
    assert.equal(program.activeRole, "narrator");
  });

  test("infers a role from a bare multi-axis node label", async () => {
    const { request } = await fixture({
      affineRank2: true,
      nodeRoles: ["centered", "eastern", "northern"],
      withWhitener: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.equal(compiler.compile("0.5 east").activeRole, "eastern");
  });

  test("keys SAE analytics by full selector and preserves fitted metadata", async () => {
    const { request } = await fixture({ saeFit: true, withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const [profile] = compiler.listProfiles().profiles;
    assert.equal(profile.name, "local/demo:sae-_zojswyzlbonss2yi");
    assert.equal(profile.metadata.manifold, "local/demo");
    assert.equal(profile.metadata.selector_key, profile.name);
    assert.equal(profile.metadata.feature_space, "sae-release-a");
    assert.equal(profile.metadata.variant, "sae");
    assert.equal(profile.metadata.variant_identity, "release-a");
    const probe = compiler.attachProbe({ selector: profile.name, name: "SAE geometry" });
    assert.equal(probe.manifold, profile.name);
  });

  test("keeps co-installed analytics variants distinct", async () => {
    const raw = await fixture();
    const sae = await fixture({ saeFit: true });
    raw.request.requiredCorePack.files.push(sae.manifest);
    raw.request.artifacts.push(...sae.artifacts);
    const compiler = await BrowserFeasibilityCorePackCompiler.load(raw.request);
    assert.deepEqual(
      compiler.listProfiles().profiles.map((profile) => profile.name),
      ["local/demo", "local/demo:sae-_zojswyzlbonss2yi"],
    );
    assert.equal(compiler.getProfile("local/demo").metadata.variant, "raw");
    assert.equal(
      compiler.getProfile("local/demo:sae-_zojswyzlbonss2yi").metadata.variant,
      "sae",
    );
  });

  test("rejects conflicting explicit role baselines before generation", async () => {
    const { request } = await fixture({
      role: "scholar",
      withSecondManifold: true,
      secondRole: "critic",
      withWhitener: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.throws(
      () => compiler.compile(
        "0.5 local/demo:role-scholar + 0.5 local/focus:role-critic",
      ),
      (error) => error.code === "CORE_PACK_ROLE_CONFLICT",
    );
  });

  test("matches Python rank-one mean-replacement ablation", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile(ablationGolden.expression);
    assert.deepEqual([...program.enabled], [1, 1]);
    assert.deepEqual([...program.target], [0, 0]);
    assert.deepEqual([...program.along], [1, 1]);
    assert.deepEqual([...program.collapse], [
      ablationGolden.coefficient,
      ablationGolden.coefficient,
    ]);
    for (const layer of ablationGolden.layers) {
      assert.deepEqual(
        [...runRankOneHookLayer(program, layer.layer, new Float32Array(layer.input)).residual],
        layer.expected,
      );
    }
    const mixed = compiler.compile("0.5 !alert + 0.25 calm");
    assert.equal(mixed.format, "drowse-structured-v2");
    assert.deepEqual([...mixed.affineActive.slice(0, 2)], [1, 0]);
  });

  test("preserves coefficients and Python affine composition in structured mode", async () => {
    const { request } = await fixture({ withWhitener: true, withSecondManifold: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const input = new Float32Array([0.4, -0.2]);
    const execute = (expression) => {
      const program = compiler.compile(expression);
      assert.equal(program.format, "drowse-structured-v2");
      return runStructuredHookLayer(
        program,
        initialStructuredHookState(program),
        0,
        input,
      ).residual;
    };
    const deltaNorm = (residual) => Math.hypot(
      residual[0] - input[0],
      residual[1] - input[1],
    );

    const small = execute("0.1 alert@response");
    const large = execute("0.9 alert@response");
    assert.ok(Math.abs(deltaNorm(large) / deltaNorm(small) - 9) < 1e-5);

    const explicitStructuredCompiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const explicitStructured = explicitStructuredCompiler.compile("0.1 alert@both");
    compiler.attachProbe({ selector: "demo", name: "demo" });
    const probeForcedStructured = compiler.compile("0.1 alert");
    assertFloatArray(
      runStructuredHookLayer(
        probeForcedStructured,
        initialStructuredHookState(probeForcedStructured),
        0,
        input,
      ).residual,
      runStructuredHookLayer(
        explicitStructured,
        initialStructuredHookState(explicitStructured),
        0,
        input,
      ).residual,
      1e-5,
    );

    const pushOnly = execute("0.5 alert@response");
    const sharedPushAndAblation = execute(
      "0.5 alert@response + 0.5 !alert@response",
    );
    assertFloatArray(sharedPushAndAblation, pushOnly, 1e-5);

    const nonorthogonal = compiler.compile(
      "0.25 !alert@response + 0.75 !sharp@response",
    );
    assert.equal(nonorthogonal.affineActive[0], 1);
    assert.equal(nonorthogonal.affineActive[1], 0);
    assert.equal(nonorthogonal.affineKappa.filter((value) => value !== 0).length, 4);
    assert.doesNotThrow(() => compiler.compile(
      "0.1 alert@response + 0.2 calm@response + 0.3 alert@response + " +
      "0.4 calm@response + 0.5 alert@response",
    ));
  });

  test("folds duplicate resolved terms and rejects conflicting triggers", async () => {
    const { request } = await fixture({
      curved: true,
      withWhitener: true,
      structuredHookProfile: "standard-v2",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const combined = compiler.compile("0.2,0.1 demo%alert + 0.3,0.2 demo%alert");
    const expected = compiler.compile("0.5,0.3 demo%alert");
    assertFloatArray(combined.curveAlong, expected.curveAlong);
    assertFloatArray(combined.curveOnto, expected.curveOnto);
    assertFloatArray(combined.curveTarget, expected.curveTarget);
    assert.equal(combined.curveActive.filter(Boolean).length, 2);
    assert.throws(
      () => compiler.compile("0.2 demo%alert@response + 0.3 demo%alert@prompt"),
      (error) => error.code === "CORE_PACK_TRIGGER_CONFLICT",
    );

    const flatFixture = await fixture({ withWhitener: true, withSecondManifold: true });
    const flat = await BrowserFeasibilityCorePackCompiler.load(flatFixture.request);
    const duplicateProjection = flat.compile(
      "0.2 alert~sharp@response + 0.3 alert~sharp@response",
    );
    const singleProjection = flat.compile("0.5 alert~sharp@response");
    assertFloatArray(duplicateProjection.affineTarget, singleProjection.affineTarget);
    assertFloatArray(duplicateProjection.affineAlong, singleProjection.affineAlong);
    const duplicateAblation = flat.compile(
      "0.2 !alert@response + 0.3 !alert@response",
    );
    const singleAblation = flat.compile("0.5 !alert@response");
    assertFloatArray(duplicateAblation.affineKappa, singleAblation.affineKappa);
    assert.throws(
      () => flat.compile("0.2 !alert@response + 0.3 !alert@prompt"),
      (error) => error.code === "CORE_PACK_TRIGGER_CONFLICT",
    );
  });

  test("matches Python bare-selector tier gating and canonicalization", async () => {
    const rawFixture = await fixture({ withWhitener: true });
    const raw = await BrowserFeasibilityCorePackCompiler.load(rawFixture.request);
    const canonical = raw.compile("0.5 demo");
    const mixedCase = raw.compile("0.5 Demo");
    assertFloatArray(mixedCase.basis, canonical.basis);
    assertFloatArray(mixedCase.target, canonical.target);
    assert.throws(
      () => raw.compile("0.5 local/alert"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND",
    );

    const saeFixture = await fixture({ saeFit: true, withWhitener: true });
    const sae = await BrowserFeasibilityCorePackCompiler.load(saeFixture.request);
    assert.throws(
      () => sae.compile("0.5 alert:sae"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND",
    );
    assert.doesNotThrow(() => sae.compile("0.5 Demo:sae"));
  });

  test("lowers phase and prior-step probe gates into structured controls", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const phased = compiler.compile("0.5 alert@response");
    assert.equal(phased.format, "drowse-structured-v2");
    assert.equal(structuredHookControlsFor(phased, {
      prefill: true,
      thinking: false,
      generatedTokens: 0,
    }).affineActive[0], 0);
    assert.equal(structuredHookControlsFor(phased, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
    }).affineActive[0], 1);

    const gated = compiler.compile("0.5 alert@when:demo>0");
    assert.equal(gated.probeKind[0], 1);
    const highProbe = new Float32Array(16);
    highProbe[0] = 0.25;
    highProbe[8] = 0.25;
    assert.equal(structuredHookControlsFor(gated, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorMeasurements: highProbe,
    }).affineActive[0], 1);
    assert.equal(structuredHookControlsFor(gated, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorMeasurements: new Float32Array(16),
    }).affineActive[0], 0);
  });

  test("lowers flat manifold positions through affine structured groups", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const direct = compiler.compile("0.5 alert@response&when:demo>0");
    const byLabel = compiler.compile("0.5,1 demo%alert@response&when:demo>0");
    const byCoordinates = compiler.compile("0.5 demo%1@response&when:demo>0");

    for (const program of [byLabel, byCoordinates]) {
      assert.equal(program.format, "drowse-structured-v2");
      assert.deepEqual([...program.curveActive], new Array(program.curveActive.length).fill(0));
      assert.deepEqual([...program.affineActive], [...direct.affineActive]);
      assertFloatArray(program.affineBasis, direct.affineBasis);
      assertFloatArray(program.affineTarget, direct.affineTarget);
      assertFloatArray(program.affineAlong, direct.affineAlong);
      assertFloatArray(program.affineKappa, direct.affineKappa);
    }

    assert.throws(
      () => compiler.compile("0.5 demo%missing@response"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND",
    );
    assert.throws(
      () => compiler.compile("0.5 demo%0,1@response"),
      (error) => error.code === "CORE_PACK_AFFINE_POSITION_INVALID",
    );
  });

  test("lowers bare labels and off-node coordinates on rank-R flat manifolds", async () => {
    const { request } = await fixture({ affineRank2: true, withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const bare = compiler.compile("0.5 east");
    const explicit = compiler.compile("0.5 demo%east");

    assert.equal(bare.format, "drowse-structured-v2");
    assert.deepEqual([...bare.curveActive], new Array(bare.curveActive.length).fill(0));
    assertFloatArray(bare.affineBasis, explicit.affineBasis);
    assertFloatArray(bare.affineTarget, explicit.affineTarget);
    assertFloatArray(bare.affineAlong, explicit.affineAlong);

    const offNode = compiler.compile("0.5 demo%0.25,0.25@response");
    const layerZeroResidual = runStructuredHookLayer(
      offNode,
      initialStructuredHookState(offNode),
      0,
      [0, 0],
    ).residual;
    const layerZeroNorm = Math.hypot(...layerZeroResidual);
    assertFloatArray(
      Float32Array.from(layerZeroResidual, (value) => value / layerZeroNorm),
      [1 / Math.sqrt(5), 2 / Math.sqrt(5)],
    );
    const layerOneResidual = runStructuredHookLayer(
      offNode,
      initialStructuredHookState(offNode),
      1,
      [0, 0],
    ).residual;
    const layerOneNorm = Math.hypot(...layerOneResidual);
    assertFloatArray(
      Float32Array.from(layerOneResidual, (value) => value / layerOneNorm),
      [5 / Math.sqrt(34), 3 / Math.sqrt(34)],
    );
    assert.deepEqual([...offNode.curveActive], new Array(offNode.curveActive.length).fill(0));
  });

  test("rejects free coordinates on a non-poised flat layout while labels remain valid", async () => {
    const { request } = await fixture({ nonPoised: true, withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);

    assert.doesNotThrow(() => compiler.compile("0.5 demo%east@response"));
    assert.throws(
      () => compiler.compile("0.5 demo%1,1@response"),
      (error) => error.code === "CORE_PACK_AFFINE_POSITION_INVALID",
    );
  });

  test("lowers an SAE-gated open curved manifold with sigma, onto, and carried feet", async () => {
    const { request } = await fixture({ curved: true, withWhitener: true, withSae: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const bareLabel = compiler.compile("calm");
    assert.equal(bareLabel.format, "drowse-structured-v2");
    assert.equal(bareLabel.curveActive.some(Boolean), true);
    const program = compiler.compile("0.5,1 demo%alert@response&when:sae/0>0.2");
    assert.equal(program.format, "drowse-structured-v2");
    assert.deepEqual([...program.curveRank.filter(Boolean)], [2, 2]);
    assert.deepEqual([...program.curveNodeCount.filter(Boolean)], [2, 2]);
    assert.deepEqual([...program.curveSigmaPresent.filter(Boolean)], [1, 1]);
    assert.deepEqual([...program.curveTarget.filter(Boolean)], [1, 1]);
    assert.equal(program.probeKind[8], 2);
    assert.equal(structuredHookControlsFor(program, {
      prefill: true,
      thinking: false,
      generatedTokens: 0,
    }).curveActive[0], 0);
    const priorMeasurements = new Float32Array(16);
    assert.equal(structuredHookControlsFor(program, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorMeasurements,
    }).curveActive[0], 0);
    priorMeasurements[8] = 0.25;
    const enabled = structuredHookControlsFor(program, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorMeasurements,
    });
    program.curveActive.set(enabled.curveActive);
    const state = initialStructuredHookState(program);
    const first = runStructuredHookLayer(program, state, 0, [0.2, 1], { decode: false });
    const second = runStructuredHookLayer(program, state, 0, [0.3, 1]);
    assert.ok(Number.isFinite(first.curveFeet[0]));
    assert.ok(Number.isFinite(second.curveFeet[0]));
    assert.notEqual(first.residual[0], Math.fround(0.2));
  });

  test("rejects overlapping curved terms before generation", async () => {
    const { request } = await fixture({ curved: true, withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.throws(
      () => compiler.compile("0.5,1 demo%alert + 0.5,1 demo%calm"),
      (error) => error.code === "CORE_PACK_CURVED_OVERLAP",
    );
  });

  test("projects affine steering out of curved spans and clamps periodic gains", async () => {
    const { request } = await fixture({
      curved: true,
      periodic: true,
      withWhitener: true,
      withSae: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile("0.5 sae/0 + 0.5,1 demo%alert");
    assert.deepEqual([...program.affineActive], new Array(program.affineActive.length).fill(0));
    assert.deepEqual(
      [...program.curveAlong].filter((value) => value !== 0),
      [1, 1],
    );
  });

  test("loads a local v1 SAE and lowers decoder steering plus an exact ReLU gate", async () => {
    const { request } = await fixture({ withWhitener: true, withSae: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile("0.5 sae/0@when:sae/0>0.2");
    assert.equal(program.format, "drowse-structured-v2");
    assert.equal(structuredHookControlsFor(program, {
      prefill: true,
      thinking: false,
      generatedTokens: 0,
    }).affineActive[4], 0);
    assert.equal(program.probeKind[8], 2);
    program.affineActive.fill(0);
    const result = runStructuredHookLayer(program, initialStructuredHookState(program), 1, [1, 2]);
    assert.ok(Math.abs(result.probes[0] - 1.25) <= 1e-6);
    const prior = new Float32Array(16);
    prior[8] = result.probes[0];
    assert.equal(structuredHookControlsFor(program, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorMeasurements: prior,
    }).affineActive[4], 1);
  });

  test("uses uncentered Gemma Scope 2 encoding for discovery, probes, gates, and fitting", async () => {
    for (const size of ["270m", "1b", "4b"]) {
      for (const explicit of [false, true]) {
        const { request } = await fixture({
          withWhitener: true,
          withSae: true,
          saeFormatVersion: 2,
          saeActivation: "jump_relu",
          saeManifestOverrides: {
            corpus_spec: `provider:google/gemma-scope-2-${size}-it`,
            ...(explicit ? { apply_b_dec_to_input: false } : {}),
          },
        });
        const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
        const dictionary = compiler.saeGpuDictionary();
        assertFloatArray(dictionary.decoderBias, [0, 0]);
        assertFloatArray(dictionary.encoderBias, [0.5, -1]);
        compiler.attachProbe({ selector: "sae/0", name: "feature strength" });
        const program = compiler.compile("");
        const result = runStructuredHookLayer(program, initialStructuredHookState(program), 1, [1, 2]);
        assert.equal(result.probes[0], 1.5);
        const atThreshold = runStructuredHookLayer(program, initialStructuredHookState(program), 1, [0.75, 1.75]);
        assert.equal(atThreshold.probes[0], 0, "JumpReLU is inactive at its threshold");
        const gated = compiler.compile("0.5 alert@when:sae/0>1.4");
        const prior = new Float32Array(gated.layerCount * gated.profile.maxProbes);
        prior[gated.profile.maxProbes] = result.probes[0];
        assert.equal(structuredHookControlsFor(gated, {
          prefill: false, thinking: false, generatedTokens: 1, priorMeasurements: prior,
        }).affineActive.some(Boolean), true);
        const fitting = compiler.exactSaeFitting("fixture-sae");
        const transformed = fitting.transformCentroids(1, {
          rows: 2, columns: 2, values: new Float64Array([1, 2, 0.75, 1.75]),
        });
        assertFloatArray(transformed.values, [1.75, 3.5, 0.25, 0.5], 1e-6);
        assert.notEqual(fitting.provenance.fingerprint, fitting.provenance.revision,
          "uncentered reads and fitted artifacts must not reuse the old centered identity");
        assert.equal(dictionary.bindingId, fitting.provenance.fingerprint);
      }
    }
  });

  test("preserves centered local SAEs and honors explicit encoder conventions", async () => {
    const compilers = [];
    for (const apply of [undefined, true, false]) {
      const { request } = await fixture({
        withWhitener: true, withSae: true, withSaeMetadata: true,
        saeManifestOverrides: apply === undefined ? {} : { apply_b_dec_to_input: apply },
      });
      const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      compiler.attachProbe({ selector: "sae/0", name: "feature strength" });
      const program = compiler.compile("");
      const result = runStructuredHookLayer(program, initialStructuredHookState(program), 1, [1, 2]);
      assert.ok(Math.abs(result.probes[0] - (apply === false ? 0.6 : 0.5)) < 1e-6);
      compilers.push(compiler);
    }
    assert.deepEqual(compilers[0].probeHashes(), compilers[1].probeHashes());
    assert.notDeepEqual(compilers[0].probeHashes(), compilers[2].probeHashes());
  });

  test("rejects invalid or unknown provider SAE input conventions", async () => {
    for (const overrides of [
      { apply_b_dec_to_input: null },
      { apply_b_dec_to_input: "false" },
      { apply_b_dec_to_input: 0 },
      { corpus_spec: "provider:unknown/sae" },
    ]) {
      const { request } = await fixture({ withSae: true, saeManifestOverrides: overrides });
      await assert.rejects(BrowserFeasibilityCorePackCompiler.load(request),
        error => error.code === "SAE_PACK_INVALID");
    }
  });

  test("backfills published descriptions without changing activation calibration", async () => {
    const binding = { model: "gemma-3-270m-it", source: "12-gemmascope-2-res-16k",
      repository: "google/gemma-scope-2-270m-it", folder: "resid_post/layer_12_width_16k_l0_medium" };
    const compiler = {
      instrumentDescriptor: () => ({ sae: { descriptionSource: binding }, jlens: null }),
      validateSaeFeature: id => ({ id, label: id === 0 ? "bundled label" : null, max_act: id === 1 ? 42 : null }),
    };
    const runtime = new BrowserInstrumentRuntime(compiler, null, null, false);
    const originalFetch = globalThis.fetch;
    const requested = [];
    try {
      globalThis.fetch = async url => {
        if (url.includes("/sae-descriptions/")) return new Response("unavailable", { status: 503 });
        const id = url.split("/").at(-1);
        requested.push(id);
        return Response.json({ modelId: binding.model, layer: binding.source, index: id,
          source: { hfRepoId: binding.repository, hfFolderId: binding.folder },
          maxActApprox: 999, explanations: id === "1" ? [{ description: "published label" }] : [] });
      };
      const request = { service: "instruments", method: "saeFeaturesMetadata", args: [[0, 1, 2]] };
      const expected = { features: {
        "0": { label: "bundled label", max_act: null },
        "1": { label: "published label", max_act: 42 },
        "2": { label: null, max_act: null },
      } };
      assert.deepEqual(await runtime.request(request, () => null), expected);
      assert.deepEqual(await runtime.request(request, () => null), expected);
      assert.deepEqual(requested, ["1", "2"], "bundled labels and cached misses never trigger duplicate requests");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("keeps successful SAE descriptions when another lookup fails and retries the missing feature", async () => {
    const binding = { model: "partial-model", source: "partial-source",
      repository: "test/partial", folder: "layer_1" };
    const compiler = {
      instrumentDescriptor: () => ({ sae: { descriptionSource: binding }, jlens: null }),
      validateSaeFeature: id => ({ id, label: null, max_act: 42 }),
    };
    const runtime = new BrowserInstrumentRuntime(compiler, null, null, false);
    const originalFetch = globalThis.fetch;
    const requested = [];
    try {
      globalThis.fetch = async url => {
        const id = url.split("/").at(-1);
        requested.push(id);
        if (id === "2" && requested.filter(value => value === "2").length === 1) {
          return new Response("unavailable", { status: 503 });
        }
        return Response.json({ modelId: binding.model, layer: binding.source, index: id,
          source: { hfRepoId: binding.repository, hfFolderId: binding.folder },
          explanations: [{ description: `published ${id}` }] });
      };
      const request = { service: "instruments", method: "saeFeaturesMetadata", args: [[1, 2]] };
      assert.deepEqual(await runtime.request(request, () => null), {
        features: { "1": { label: "published 1", max_act: 42 } },
      });
      assert.deepEqual(await runtime.request(request, () => null), {
        features: { "1": { label: "published 1", max_act: 42 }, "2": { label: "published 2", max_act: 42 } },
      });
      assert.deepEqual(requested, ["1", "2", "2"]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("loads validated SAE feature metadata and normalizes the GPU readout channel", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withSae: true,
      withSaeMetadata: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.deepEqual(compiler.validateSaeFeature(0), {
      id: 0,
      label: "opening delimiters",
      layer: 1,
      max_act: 2.5,
    });
    assert.deepEqual(compiler.validateSaeFeature(1), {
      id: 1,
      label: null,
      layer: 1,
      max_act: null,
    });
    const instruments = new BrowserInstrumentRuntime(compiler);
    const sources = instruments.request({ service: "instruments", method: "sources", args: ["sae"] }, () => null);
    assert.deepEqual(sources.sources[0].model_layers, request.variant.runtimeIdentity.layerMap);
    assert.equal(sources.sources[0].description_source, null, "Unverified dictionaries must not borrow published descriptions");
    assert.deepEqual(await instruments.request({
      service: "instruments",
      method: "saeFeaturesMetadata",
      args: [[0, 1]],
    }, () => null), {
      features: {
        "0": { label: "opening delimiters", max_act: 2.5 },
        "1": { label: null, max_act: null },
      },
    });
    const attached = compiler.attachProbe({ selector: "sae/0", name: "delimiter strength" });
    assert.equal(attached.label, "opening delimiters");
    assert.equal(attached.max_act, 2.5);
    const program = compiler.compile("");
    assert.equal(program.measurementSchema.probes[0].label, "opening delimiters");
    assert.equal(program.measurementSchema.probes[0].maxAct, 2.5);
    const result = runStructuredHookLayer(
      program,
      initialStructuredHookState(program),
      1,
      [1, 2],
    );
    assert.ok(Math.abs(result.probes[0] - 0.5) <= 1e-6);
    const values = new Float32Array(program.layerCount * program.measurementSchema.probes.length);
    values.set(result.probes, program.measurementSchema.probes.length);
    const envelope = measurementEnvelope(program, values, null);
    assert.equal(envelope.instruments.sae.readings["delimiter strength"].value, 0.5);
    assert.equal(
      envelope.instruments.sae.readings["delimiter strength"].unit,
      "activation_over_max",
    );
    assert.equal(envelope.instruments.sae.readout, undefined);
    const gated = compiler.compile("0.5 alert@when:sae/0>0.4");
    const prior = new Float32Array(gated.layerCount * gated.profile.maxProbes);
    prior[gated.profile.maxProbes] = 0.5;
    assert.equal(structuredHookControlsFor(gated, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorMeasurements: prior,
    }).affineActive.some(Boolean), true);
    prior[gated.profile.maxProbes] = 0.3;
    assert.equal(structuredHookControlsFor(gated, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorMeasurements: prior,
    }).affineActive.some(Boolean), false);
  });

  test("rejects Top-K SAE packs instead of silently encoding them as ReLU", async () => {
    for (const saeActivation of ["topk", "top_k", "batch_topk"]) {
      const { request } = await fixture({ withSae: true, saeActivation });
      await assert.rejects(
        BrowserFeasibilityCorePackCompiler.load(request),
        (error) => error.code === "SAE_PACK_INVALID",
      );
    }
  });

  test("activates only an exact instrument pack already loaded for this context", async () => {
    const binding = "b".repeat(64);
    const { request } = await fixture({
      withWhitener: true,
      withSae: true,
      withJlens: true,
      contextBindingSha256: binding,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const instruments = new BrowserInstrumentRuntime(compiler, null, binding);
    const tokenLookup = () => null;

    assert.deepEqual(
      instruments.blocks().filter((block) => block.family !== "geometry").map((block) => ({
        family: block.family,
        preparations: block.capabilities.preparations,
        sourceSwitch: block.capabilities.source_switch,
      })),
      [
        { family: "lens", preparations: [], sourceSwitch: false },
        { family: "sae", preparations: [], sourceSwitch: false },
      ],
    );
    assert.deepEqual(instruments.request({
      service: "instruments",
      method: "activateInstalledPack",
      args: ["lens", { source: "fixture-jlens", contextBindingSha256: binding }],
    }, tokenLookup), {
      state: "active",
      family: "lens",
      source: "fixture-jlens",
      live: { enabled: true, layers: [0, 1] },
      contextBindingSha256: binding,
      reloadRequired: false,
    });
    assert.deepEqual(instruments.request({
      service: "instruments",
      method: "activateInstalledPack",
      args: ["sae", { source: "fixture-sae", layer: 1 }],
    }, tokenLookup), {
      state: "active",
      family: "sae",
      source: "fixture-sae",
      live: { enabled: true, layer: 1, source: "fixture-sae" },
      contextBindingSha256: binding,
      reloadRequired: false,
    });

    for (const [args, code] of [
      [["lens", { source: "other-jlens" }], "INSTRUMENT_PACK_SOURCE_MISMATCH"],
      [["lens", { source: "fixture-jlens", contextBindingSha256: "c".repeat(64) }], "INSTRUMENT_PACK_CONTEXT_MISMATCH"],
      [["sae", { source: "fixture-sae", layer: 0 }], "INSTRUMENT_PACK_LAYER_MISMATCH"],
    ]) {
      assert.throws(
        () => instruments.request({
          service: "instruments",
          method: "activateInstalledPack",
          args,
        }, tokenLookup),
        (error) => error.code === code,
      );
    }
    assert.throws(
      () => instruments.request({
        service: "instruments",
        method: "startPreparation",
        args: ["lens", { operation: "fetch", source: "fixture-jlens" }],
      }, tokenLookup),
      (error) => error.code === "HOSTED_PACK_DOWNLOAD_MANAGED",
    );
    assert.match(
      instruments.request({
        service: "instruments",
        method: "cancelPreparation",
        args: ["lens"],
      }, tokenLookup).message,
      /Model settings/u,
    );

    const withoutOptional = await fixture({ withWhitener: true });
    const emptyCompiler = await BrowserFeasibilityCorePackCompiler.load(withoutOptional.request);
    const emptyInstruments = new BrowserInstrumentRuntime(emptyCompiler, null, binding);
    assert.throws(
      () => emptyInstruments.request({
        service: "instruments",
        method: "activateInstalledPack",
        args: ["sae", { source: "fixture-sae" }],
      }, tokenLookup),
      (error) => error.code === "INSTRUMENT_PACK_NOT_LOADED" &&
        /close and reopen/u.test(error.message),
    );
  });

  test("rejects an invalid SAE feature metadata sidecar as one failed pack", async () => {
    const { request } = await fixture({
      withSae: true,
      withSaeMetadata: true,
      invalidSaeMetadata: true,
    });
    await assert.rejects(
      BrowserFeasibilityCorePackCompiler.load(request),
      (error) => error.code === "SAE_FEATURE_METADATA_INVALID",
    );
  });

  test("reconstructs manifold centroids through the exact installed SAE", async () => {
    const { request } = await fixture({ withWhitener: true, withSae: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const fitting = compiler.exactSaeFitting("fixture-sae");
    assert.ok(fitting);
    assert.deepEqual(fitting.layers, [1]);
    assert.equal(fitting.provenance.release, "local:fixture");
    assert.equal(fitting.provenance.fullCoverage, false);
    assert.equal(fitting.provenance.idsByLayer.get(1), "local:fixture:layer-1");
    const transformed = fitting.transformCentroids(1, {
      rows: 2,
      columns: 2,
      values: new Float64Array([1, 2, 0, 0]),
    });
    assertFloatArray(transformed.values, [1.5, 2.5, 0.5, 0.5]);
    assert.ok(compiler.exactSaeFitting("fixture-sae"));
    assert.equal(compiler.exactSaeFitting("another-sae"), null);
  });

  test("loads local v6 J-lens matrices and lowers curated token directions", async () => {
    const { request } = await fixture({ withWhitener: true, withJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile("0.5 jlens/hello");
    assert.equal(program.format, "drowse-structured-v2");
    assert.deepEqual([...program.affineActive.filter(Boolean)], [1, 1]);
    const layer0 = program.affineBasis.slice(0, 2);
    const layer1 = program.affineBasis.slice(4 * 8 * 2, 4 * 8 * 2 + 2);
    assertFloatArray(layer0, [2 / Math.sqrt(13), 3 / Math.sqrt(13)], 2e-6);
    assertFloatArray(layer1, [2 / Math.sqrt(53), 7 / Math.sqrt(53)], 2e-6);
    assert.throws(
      () => compiler.compile("0.5 jlens/missing"),
      (error) => error.code === "JLENS_TOKEN_UNAVAILABLE",
    );
  });

  test("loads the deterministic legacy J-lens identity and vocabulary alias", async () => {
    const baseline = await fixture();
    const currentIdentity = {
      ...baseline.request.variant.runtimeIdentity,
      runtimeAbi: "drowse-web-runtime-v1",
    };
    const currentFingerprint = sha256(canonicalJson(currentIdentity));
    for (const slug of ["polythetic", "saklas"]) {
      const legacyFingerprint = sha256(canonicalJson({
        ...currentIdentity,
        runtimeAbi: `${slug}-web-runtime-v1`,
      }));
      const { request } = await fixture({
        withWhitener: true,
        withJlens: true,
        fingerprint: currentFingerprint,
        runtimeAbi: currentIdentity.runtimeAbi,
        instrumentModelFingerprint: legacyFingerprint,
        jlensVocabularyFormat: `${slug}-jlens-vocabulary-v2`,
      });
      const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      assert.equal(compiler.compile("0.5 jlens/hello").layerCount, 2);
    }
  });

  test("resolves arbitrary single-token J-lens selectors through the loaded model", async () => {
    const { request } = await fixture({ withWhitener: true, withJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.equal(
      compiler.steeringDelta("0.5 jlens/uncached", "0.25 jlens/uncached"),
      "-0.25 jlens/uncached",
    );
    const tokenByWord = new Map([
      ["novel", 7],
      ...Array.from({ length: 10 }, (_, index) => [`word${index}`, 20 + index]),
    ]);
    const wordByToken = new Map([...tokenByWord].map(([word, token]) => [token, word]));
    const batches = [];
    const dictionary = compiler.jlensGpuDictionary();
    assert.ok(dictionary);
    const resolver = {
      async tokenizeText(text) {
        const word = text.trim();
        if (word === "split") return [100, 101];
        const token = tokenByWord.get(word);
        return token === undefined ? [100, 101] : [token];
      },
      async decodeTokens(ids) {
        return ids.length === 1 && wordByToken.has(ids[0])
          ? ` ${wordByToken.get(ids[0])}`
          : ids.map((id) => `<${id}>`).join("");
      },
      async resolveJlensTokenDirections(bindingId, layerIndices, tokenIds) {
        assert.equal(bindingId, dictionary.bindingId);
        assert.deepEqual(layerIndices, [0, 1]);
        assert.ok(tokenIds.length > 0 && tokenIds.length <= 8);
        batches.push([...tokenIds]);
        const output = new Float32Array(2 * tokenIds.length * 2);
        for (let layer = 0; layer < 2; layer += 1) {
          const matrix = dictionary.matrices[layer];
          tokenIds.forEach((token, tokenIndex) => {
            const row = [token, token + 1];
            const offset = (layer * tokenIds.length + tokenIndex) * 2;
            output[offset] = row[0] * matrix[0] + row[1] * matrix[2];
            output[offset + 1] = row[0] * matrix[1] + row[1] * matrix[3];
          });
        }
        return output;
      },
    };
    await compiler.prepareJlensSelectors(
      "0.5 jlens/novel@when:jlens/novel>0.01",
      resolver,
    );
    const program = compiler.compile("0.5 jlens/novel@when:jlens/novel>0.01");
    assertFloatArray(program.affineBasis.slice(0, 2), [7 / Math.sqrt(113), 8 / Math.sqrt(113)]);
    assertFloatArray(
      program.affineBasis.slice(4 * 8 * 2, 4 * 8 * 2 + 2),
      [7 / Math.sqrt(533), 22 / Math.sqrt(533)],
    );
    assert.equal(program.jLensTokenIds[0], 7);
    assert.deepEqual(batches, [[7]]);

    await compiler.prepareJlensWords(
      Array.from({ length: 10 }, (_, index) => `word${index}`),
      resolver,
    );
    assert.deepEqual(batches.slice(1).map((batch) => batch.length), [8, 2]);
    assert.equal(compiler.validateLensToken("word9").token_id, 29);
    await assert.rejects(
      compiler.prepareJlensWords(["split"], resolver),
      (error) => error.code === "MULTI_TOKEN_WORD",
    );
  });

  test("bounds unpinned J-lens directions while attached probes stay resident", async () => {
    const { request } = await fixture({ withWhitener: true, withJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const first = Array.from({ length: 64 }, (_, index) => `cached${index}`);
    const later = Array.from({ length: 10 }, (_, index) => `later${index}`);
    const tokenByWord = new Map(
      [...first, ...later].map((word, index) => [word, 100 + index]),
    );
    const wordByToken = new Map([...tokenByWord].map(([word, token]) => [token, word]));
    const dictionary = compiler.jlensGpuDictionary();
    assert.ok(dictionary);
    const resolver = {
      async tokenizeText(text) {
        return [tokenByWord.get(text.trim())];
      },
      async decodeTokens(ids) {
        return ` ${wordByToken.get(ids[0])}`;
      },
      async resolveJlensTokenDirections(bindingId, layerIndices, tokenIds) {
        assert.equal(bindingId, dictionary.bindingId);
        assert.deepEqual(layerIndices, [0, 1]);
        return Float32Array.from(
          { length: layerIndices.length * tokenIds.length * 2 },
          (_value, index) => index + 1,
        );
      },
    };

    await compiler.prepareJlensWords(first, resolver);
    compiler.attachProbe({ selector: "jlens/cached0", name: "pinned token" });
    await compiler.prepareJlensWords(later, resolver);

    assert.equal(compiler.validateLensToken("cached0").token_id, 100);
    assert.throws(
      () => compiler.validateLensToken("cached1"),
      (error) => error.code === "JLENS_TOKEN_UNAVAILABLE",
    );
    const jlens = compiler.instruments.jlens;
    assert.equal(jlens.resolvedByTokenId.size, 65);
    assert.equal(jlens.resolvedBytes, jlens.resolvedByTokenId.size * 16);

    compiler.detachProbe("pinned token");
    assert.equal(jlens.resolvedByTokenId.size, 64);
    assert.equal(jlens.resolvedBytes, jlens.resolvedByTokenId.size * 16);
  });

  test("passes partial J-lens Jacobians in fitted-layer order without zero padding", async () => {
    const { request } = await fixture({ withWhitener: true, withJlens: true, partialJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const dictionary = compiler.jlensGpuDictionary();
    assert.ok(dictionary);
    const resolver = {
      async tokenizeText(text) {
        return text.trim() === "novel" ? [7] : [100, 101];
      },
      async decodeTokens(ids) {
        return ids.length === 1 && ids[0] === 7 ? " novel" : "not-novel";
      },
      async resolveJlensTokenDirections(bindingId, layerIndices, tokenIds) {
        assert.equal(bindingId, dictionary.bindingId);
        assert.deepEqual(layerIndices, [1]);
        assert.deepEqual(tokenIds, [7]);
        assert.equal(dictionary.matrices.length, 1);
        assertFloatArray(dictionary.matrices[0], [1, 2, 0, 1]);
        return new Float32Array([7, 22]);
      },
    };

    await compiler.prepareJlensSelectors("0.5 jlens/novel", resolver);
    const program = compiler.compile("0.5 jlens/novel@when:jlens/novel>0.01");
    assert.equal(program.affineActive.filter(Boolean).length, 1);
    assert.equal(program.affineActive[0], 0);
    assert.equal(program.affineActive[4], 1);
    assert.equal(program.jLensBindingId, dictionary.bindingId);
    assert.deepEqual([...program.jLensLayerIndices], [1]);
    assert.equal(program.probeKind[0], 0);
    assert.equal(program.probeKind[8], 3);
    assertFloatArray(
      program.affineBasis.slice(4 * 8 * 2, 4 * 8 * 2 + 2),
      [7 / Math.sqrt(533), 22 / Math.sqrt(533)],
    );
  });

  test("composes manifold, SAE, J-lens, and ablation into one group per trigger", async () => {
    const { request } = await fixture({ withWhitener: true, withSae: true, withJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile("0.25 alert + 0.1 sae/1 + 0.2 jlens/world + 0.5 !calm");
    assert.equal(program.format, "drowse-structured-v2");
    assert.equal(program.affineActive.filter(Boolean).length, 2);
  });

  test("omits v3 geometry requests when only lens and SAE measurements are active", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withSae: true,
      withJlens: true,
      structuredHookProfile: "standard-v3",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    compiler.setInstrumentLive("lens", true);
    compiler.setInstrumentLive("sae", true);
    const program = compiler.compile("");

    assert.equal(program.format, "drowse-structured-v3");
    assert.equal(program.measurementSchema.probes.some((probe) => probe?.family === "lens"), false);
    assert.equal(program.measurementSchema.probes.some((probe) => probe?.family === "sae"), false);
    assert.equal(program.measurementSchema.lensReadout, true);
    assert.equal(program.measurementSchema.saeReadout, true);
    assert.equal(program.saeBindingId, compiler.saeGpuDictionary().bindingId);
    assert.equal(program.probeKind.some(Boolean), false);
    assert.deepEqual([...program.jLensLayerIndices], [0, 1]);
    assert.equal(program.geometryActive.some(Boolean), false);
    assert.equal(Object.hasOwn(program.measurementSchema, "geometryProbes"), false);
  });

  test("keeps lens gate and pinned means independent of live display layers", async () => {
    const { request } = await fixture({ withWhitener: true, withJlens: true, structuredHookProfile: "standard-v3" });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    for (const pinned of [false, true]) {
      if (pinned) compiler.attachProbe({ selector: "jlens/hello", name: "jlens/hello" });
      for (const [enabled, layers] of [[false, undefined], [true, undefined], [true, [1]], [true, [0]]]) {
        compiler.setInstrumentLive("lens", enabled, layers);
        const program = compiler.compile("0.5 alert@when:jlens/hello>0.6");
        assert.deepEqual([...program.jLensLayerIndices], [0, 1]);
        assert.deepEqual([...program.jLensReadoutLayerIndices], enabled ? layers ?? [0, 1] : []);
        const values = new Float32Array(program.layerCount * program.profile.maxProbes);
        values[0] = 0.1;
        values[program.profile.maxProbes] = 0.9;
        const controls = structuredHookControlsFor(program, {
          prefill: false, thinking: false, generatedTokens: 1, priorMeasurements: values,
        });
        assert.equal(controls.affineActive.some(Boolean), false);
        values[0] = 0.7;
        assert.equal(structuredHookControlsFor(program, {
          prefill: false, thinking: false, generatedTokens: 1, priorMeasurements: values,
        }).affineActive.some(Boolean), true);
      }
    }
  });

  test("compiles exact selected J-lens live layers", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withJlens: true,
      structuredHookProfile: "standard-v3",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    compiler.setInstrumentLive("lens", true, [1]);
    const program = compiler.compile("");

    assert.deepEqual(compiler.instrumentLiveState(), {
      geometry: true,
      lens: true,
      lensLayers: [1],
      sae: false,
    });
    assert.deepEqual([...program.jLensLayerIndices], [1]);
    assert.equal(program.measurementSchema.lensReadout, true);
    assert.throws(
      () => compiler.setInstrumentLive("lens", true, [2]),
      (error) => error.code === "JLENS_PACK_LAYER_MISMATCH",
    );
    assert.deepEqual(compiler.instrumentLiveState().lensLayers, [1]);
  });

  test("swaps installed standard and R-lens packs without fitting or losing probes", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withJlens: true,
      withSecondJlens: true,
      structuredHookProfile: "standard-v3",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const installedDictionaries = [];
    let activeBinding = compiler.jlensGpuDictionary().bindingId;
    const resolver = {
      async tokenizeText() { return [0]; },
      async decodeTokens() { return "hello"; },
      async resolveJlensTokenDirections(bindingId) {
        assert.equal(bindingId, activeBinding);
        return new Float32Array(4);
      },
      async setJlensDictionary(dictionary) {
        activeBinding = dictionary.bindingId;
        installedDictionaries.push(dictionary);
      },
    };
    const runtime = new BrowserInstrumentRuntime(compiler, resolver);
    compiler.attachProbe({ selector: "jlens/hello", name: "watched hello" });
    compiler.setInstrumentLive("lens", true, [1]);

    assert.equal(runtime.blocks()[1].capabilities.source_switch, true);
    assert.deepEqual(runtime.request({
      service: "instruments",
      method: "sources",
      args: ["lens"],
    }, () => null), {
      sources: [
        { source: "fixture-jlens", name: "fixture-jlens", kind: "catalog", provider: "catalog", active: true },
        { source: "fixture-rlens", name: "fixture-rlens", kind: "catalog", provider: "catalog", active: false },
      ],
    });

    const switched = await runtime.request({
      service: "instruments",
      method: "setLensSource",
      args: ["fixture-rlens"],
    }, () => null);
    assert.deepEqual(switched, { source: "fixture-rlens", live_layers: [0, 1] });
    assert.equal(installedDictionaries.length, 1);
    assert.equal(installedDictionaries[0].bindingId, compiler.jlensGpuDictionary().bindingId);
    assert.equal(compiler.listProbes()[0].token_id, 0);
    assert.deepEqual(compiler.instrumentLiveState().lensLayers, [0, 1]);
    assert.doesNotThrow(() => compiler.compile("0.25 !jlens/hello"));
    const sources = runtime.request({
      service: "instruments",
      method: "sources",
      args: ["lens"],
    }, () => null).sources;
    assert.equal(sources.find((source) => source.source === "fixture-rlens").active, true);
    const refreshed = await compiler.withInstalledManifolds([], request.signal);
    await runtime.replaceCompiler(refreshed);
    assert.equal(installedDictionaries.length, 1);
    assert.equal(refreshed.jlensGpuDictionary().bindingId, activeBinding);
    assert.deepEqual(refreshed.listProbes(), compiler.listProbes());
    assert.deepEqual(refreshed.instrumentLiveState(), compiler.instrumentLiveState());
    assert.doesNotThrow(() => refreshed.compile("0.25 !jlens/hello"));
  });

  test("captures installed J-lens readouts during browser generation by default", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withJlens: true,
      structuredHookProfile: "standard-v3",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const resolver = {
      async tokenizeText() { return [0]; },
      async decodeTokens() { return "hello"; },
      async resolveJlensTokenDirections() { return new Float32Array(4); },
    };
    const runtime = new BrowserInstrumentRuntime(compiler, resolver);

    assert.deepEqual(runtime.blocks()[1].live, {
      enabled: true,
      layers: [0, 1],
    });
    assert.equal(compiler.compile("").measurementSchema.lensReadout, true);

    assert.deepEqual(runtime.request({
      service: "instruments",
      method: "setLive",
      args: ["lens", { enabled: false }],
    }, () => null), {
      enabled: false,
      layers: null,
    });
    assert.equal(compiler.compile("").measurementSchema.lensReadout, false);
  });

  test("optional live readouts can stay off without disabling their tools", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withJlens: true,
      withSae: true,
      structuredHookProfile: "standard-v3",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const resolver = {
      async tokenizeText() { return [0]; },
      async decodeTokens() { return "hello"; },
      async resolveJlensTokenDirections() { return new Float32Array(4); },
    };
    const runtime = new BrowserInstrumentRuntime(compiler, resolver, null, false);

    assert.deepEqual(runtime.blocks()[1].live, {
      enabled: false,
      layers: null,
    });
    assert.equal(compiler.compile("").measurementSchema.lensReadout, false);
    assert.equal(compiler.compile("").measurementSchema.saeReadout, false);
    assert.equal(runtime.blocks()[2].live.enabled, false);
    assert.deepEqual(runtime.request({
      service: "instruments",
      method: "setLive",
      args: ["lens", { enabled: true }],
    }, () => null), {
      enabled: true,
      layers: [0, 1],
    });
  });

  test("keeps all eight scalar slots for explicit probes while exact live readouts stay separate", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withSae: true,
      withJlens: true,
      structuredHookProfile: "standard-v3",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    for (let index = 0; index < 8; index += 1) {
      compiler.attachProbe({ selector: "sae/0", name: `watched-${index}` });
    }
    compiler.setInstrumentLive("lens", true);
    compiler.setInstrumentLive("sae", true);
    const program = compiler.compile("");

    assert.equal(program.measurementSchema.probes.filter(Boolean).length, 8);
    assert.equal(program.probeKind.filter(Boolean).length, 8);
    assert.equal(program.measurementSchema.lensReadout, true);
    assert.equal(program.measurementSchema.saeReadout, true);
    assert.deepEqual([...program.jLensLayerIndices], [0, 1]);
  });

  test("lowers thinking-only controls and exact J-lens probability gates", async () => {
    const { request } = await fixture({ withWhitener: true, withJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const thinking = compiler.compile("0.5 alert@thinking");
    assert.equal(thinking.controls.affine[0].phase.kind, "thinking_only");
    const jlensGate = compiler.compile("0.5 alert@when:jlens/hello>0.1");
    assert.equal(jlensGate.probeKind[0], 3);
    assert.equal(jlensGate.probeKind[8], 3);
    assert.equal(jlensGate.jLensBindingId, compiler.jlensGpuDictionary().bindingId);
    assert.deepEqual([...jlensGate.jLensLayerIndices], [0, 1]);
    assert.deepEqual([...jlensGate.jLensTokenIds], [0, 0, 0, 0, 0, 0, 0, 0]);
  });

  test("resolves a two-node PCA manifold name to its first pole", async () => {
    const { request } = await fixture();
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const program = compiler.compile("0.25 local/demo");
    assertFloatArray(program.target, [-0.25, -0.25]);
    assertFloatArray(program.along, [32 / 3, 64 / 3]);
  });

  test("canonicalizes steering deltas through installed manifold identities", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.equal(
      compiler.steeringDelta("0.5 calm", "0.25 alert"),
      "-0.5 local/demo%calm +0.25 local/demo%alert",
    );
    assert.equal(compiler.steeringDelta(null, "0.5 demo%alert"), "0.5 local/demo%alert");
    assert.equal(compiler.steeringDelta(null, "0.25 demo%0.25"), "0.25 local/demo%0.25");
    assert.doesNotThrow(() =>
      compiler.compile(compiler.steeringDelta(null, "0.25 demo%0.25"))
    );
  });

  test("keeps projection steering deltas parseable", async () => {
    const { request } = await fixture({ withWhitener: true, withSecondManifold: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.equal(compiler.steeringDelta(null, "0.5 calm~sharp"), "0.5 calm~sharp");
    assert.doesNotThrow(() => compiler.compile(compiler.steeringDelta(null, "0.5 calm~sharp")));
  });

  test("loads an exact-runtime neutral whitener for every fitting layer", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const whiteners = compiler.fittingWhiteners();
    assert.deepEqual([...whiteners.keys()], [0, 1]);
    assert.deepEqual([...whiteners.get(0).mean], [0.5, -0.5]);
    assert.deepEqual([...whiteners.get(1).basis], [0, 1]);
    assert.equal(whiteners.get(0).ridge, 0.25);
    assert.equal(whiteners.get(1).rank, 1);
    whiteners.get(0).mean[0] = 99;
    assert.equal(compiler.fittingWhiteners().get(0).mean[0], 0.5);
  });

  test("attaches a real affine geometry probe and lowers its Mahalanobis readout", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const attached = compiler.attachProbe({ selector: "local/demo", name: "calm scale" });
    assert.equal(attached.family, "geometry");
    assert.equal(attached.name, "calm scale");
    assert.deepEqual(attached.layers, [0, 1]);
    assert.deepEqual(compiler.listProbes().map((probe) => probe.name), ["calm scale"]);
    assert.equal(compiler.hasAttachedProbes(), true);

    const program = compiler.compile("");
    assert.equal(program.format, "drowse-structured-v2");
    assert.equal(program.measurementSchema.probes[0].name, "calm scale");
    assert.equal(program.measurementSchema.probes[0].family, "geometry");
    const layer0 = runStructuredHookLayer(
      program,
      initialStructuredHookState(program),
      0,
      [0.85, 0.2],
    );
    assert.ok(Math.abs(layer0.probes[0] - 0.75) <= 1e-6);

    const geometry = compiler.probeGeometry("calm scale");
    assert.equal(geometry.manifold, "local/demo");
    assert.deepEqual(geometry.node_labels, ["calm", "alert"]);
    assert.equal(geometry.layers["0"].node_white.length, 2);
    assert.equal(geometry.layers["0"].rank, 1);

    compiler.detachProbe("calm scale");
    assert.deepEqual(compiler.listProbes(), []);
    assert.equal(compiler.hasAttachedProbes(), false);

    const aliased = compiler.attachProbe({ selector: "demo", name: "friendly alias" });
    assert.equal(aliased.name, "friendly alias");
    assert.equal(aliased.manifold, "local/demo");
  });

  test("serializes exact multidimensional geometry and simulates its v3 readout", async () => {
    const { request } = await fixture({
      curved: true,
      mixedRank: true,
      withWhitener: true,
      structuredHookProfile: "standard-v2",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const attached = compiler.attachProbe({ selector: "local/demo", top_n: 3 });
    assert.equal(attached.intrinsic_dim, 1);
    const program = compiler.compile("");
    assert.equal(program.format, "drowse-structured-v3");
    assert.equal(program.profile.geometryOutputStride, 41);
    assert.deepEqual([...program.whitenerRank], [1, 1]);
    assert.deepEqual([...program.geometryKind.filter(Boolean)], [2, 2]);
    assert.deepEqual([...program.geometryRank.filter(Boolean)], [2, 1]);
    assert.deepEqual([...program.geometryCandidateCount.filter(Boolean)], [3, 3]);
    assert.equal(program.measurementSchema.probes[0], null);
    assert.deepEqual(program.measurementSchema.geometryProbes[0].labels, [
      "calm",
      "alert",
      "neutral",
    ]);
    const state = initialStructuredHookState(program);
    const layer = runStructuredHookLayer(program, state, 0, [0.6, 0.4], {
      decode: false,
    });
    assert.equal(layer.geometry[0], 1);
    assert.ok(layer.geometry[1] > 0 && layer.geometry[1] <= 1);
    assert.ok(Math.abs(layer.geometry[4] - 0.5) <= 1e-3);
    assert.ok(layer.geometry[2] > 0);
    assert.ok(layer.geometry[3] >= 0 && layer.geometry[3] <= 1);
    assert.ok([...layer.geometry].every(Number.isFinite));

    const gated = compiler.compile(
      "0.5 local/demo%calm@when:local/demo:membership>0.6",
    );
    const gate = gated.controls.curve.find((control) => control !== null)?.gate;
    assert.equal(gate.scoreKey, "local/demo:membership");
    assert.deepEqual(gate.slots, []);
    const inactive = structuredHookControlsFor(gated, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorScores: { "local/demo:membership": 0.5 },
    });
    const active = structuredHookControlsFor(gated, {
      prefill: false,
      thinking: false,
      generatedTokens: 1,
      priorScores: { "local/demo:membership": 0.7 },
    });
    assert.equal(inactive.curveActive.some(Boolean), false);
    assert.equal(active.curveActive.some(Boolean), true);
  });

  test("matches Python curved neutral whitening at a nonzero manifold origin", async () => {
    const curveOrigin = 0.25;
    const { request } = await fixture({
      curved: true,
      mixedRank: true,
      withWhitener: true,
      structuredHookProfile: "standard-v2",
      curveOrigin,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    compiler.attachProbe({ selector: "local/demo", top_n: 3 });
    const program = compiler.compile("");
    const geometry = compiler.probeGeometry("local/demo");
    let sawNonzeroNeutral = false;
    for (let layerIndex = 0; layerIndex < program.layerCount; layerIndex += 1) {
      const slot = layerIndex * program.profile.maxGeometryProbes;
      const rank = program.geometryRank[slot];
      const neutralCandidate = program.geometryCandidateCount[slot] - 1;
      const packedNeutral = Array.from({ length: rank }, (_, axis) =>
        program.geometryNodeWhite[
          (slot * program.profile.maxGeometryCandidates + neutralCandidate) *
            program.profile.maxRank + axis
        ]
      );
      const expected = Array.from({ length: rank }, (_, column) =>
        curveOrigin * program.geometryCholesky[
          (slot * program.profile.maxRank) * program.profile.maxRank + column
        ]
      );
      const responseNeutral = geometry.layers[String(layerIndex)].neutral_white;
      for (let axis = 0; axis < rank; axis += 1) {
        assert.ok(Math.abs(responseNeutral[axis] - expected[axis]) < 1e-6);
        assert.ok(Math.abs(packedNeutral[axis] - expected[axis]) < 1e-6);
        sawNonzeroNeutral ||= Math.abs(expected[axis]) > 1e-6;
      }
    }
    assert.equal(sawNonzeroNeutral, true);
  });

  test("loads sphere artifacts into exact v3 steering and geometry programs", async () => {
    const { request } = await fixture({
      sphere: true,
      withWhitener: true,
      structuredHookProfile: "standard-v2",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    compiler.attachProbe({ selector: "local/demo" });
    const program = compiler.compile("0.5 local/demo%alert");
    assert.equal(program.format, "drowse-structured-v3");
    assert.deepEqual([...program.curveDomainKind.filter(Boolean)], [2, 2]);
    assert.deepEqual([...program.geometryDomainKind.filter(Boolean)], [2, 2]);
    const result = runStructuredHookLayer(
      program,
      initialStructuredHookState(program),
      0,
      [1.1, 0.2],
      { decode: false },
    );
    assert.ok(Math.abs(result.residual[0] - 0.1) < 2e-3);
    assert.ok(Math.abs(result.residual[1] - 1.2) < 2e-3);
    assert.ok(Math.abs(result.geometry[4] - Math.PI / 2) < 3e-3);
    assert.ok(Math.abs(result.geometry[5] - Math.PI / 2) < 3e-3);
    assert.ok(Number.isFinite(result.geometry[3]));
    assert.ok(Math.abs(result.geometry[3] - 1) < 2e-3);
  });

  test("computes real Mahalanobis profile correlation and cross-layer comparison", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withSecondManifold: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.deepEqual(
      compiler.listProfiles().profiles.map((profile) => profile.name),
      ["local/demo", "local/focus"],
    );
    assert.deepEqual(compiler.getProfile("local/demo").layers, [0, 1]);
    assert.equal(
      compiler.getProfile("local/demo").metadata.share_metric,
      "mahalanobis",
    );

    const correlation = compiler.profileCorrelation();
    assert.deepEqual(correlation.names, ["local/demo", "local/focus"]);
    assert.equal(correlation.matrix["local/demo"]["local/demo"], 1);
    assert.equal(correlation.matrix["local/focus"]["local/focus"], 1);
    assert.equal(correlation.layers_shared["local/demo__local/focus"], 2);
    assert.equal(
      correlation.matrix["local/demo"]["local/focus"],
      correlation.matrix["local/focus"]["local/demo"],
    );
    assert.ok(Number.isFinite(correlation.matrix["local/demo"]["local/focus"]));

    const pairwise = compiler.profilePairwise("local/demo", "local/focus");
    assert.equal(pairwise.metric, "mahalanobis");
    assert.equal(pairwise.model, "fixture-model");
    assert.deepEqual(pairwise.layers_a, [0, 1]);
    assert.deepEqual(pairwise.layers_b, [0, 1]);
    assert.deepEqual(pairwise.matrix.map((row) => row.length), [2, 2]);
    assert.ok(pairwise.matrix.flat().every((value) => value === null || Number.isFinite(value)));

    compiler.attachProbe({ selector: "local/demo", name: "calm scale" });
    assert.ok(compiler.listProfiles().profiles.some((profile) => profile.name === "calm scale"));
    assert.equal(compiler.getProfile("calm scale").metadata.manifold, "local/demo");
    assert.throws(
      () => compiler.profileCorrelation(["missing"]),
      (error) => error.code === "PROFILE_NOT_FOUND",
    );
  });

  test("attaches aliased J-lens and SAE probes to the real structured program", async () => {
    const { request } = await fixture({ withWhitener: true, withSae: true, withJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const lens = compiler.attachProbe({ selector: "jlens/hello", name: "next hello" });
    const sae = compiler.attachProbe({ selector: "sae/1", name: "feature one" });
    assert.equal(lens.family, "lens");
    assert.equal(lens.token_id, 0);
    assert.equal(sae.family, "sae");
    assert.equal(sae.feature_id, 1);
    const program = compiler.compile("");
    assert.deepEqual(
      program.measurementSchema.probes.slice(0, 2).map((probe) => ({
        name: probe.name,
        family: probe.family,
      })),
      [
        { name: "next hello", family: "lens" },
        { name: "feature one", family: "sae" },
      ],
    );
    assert.equal(program.jLensTokenIds[0], 0);
    assert.equal(program.probeKind[0], 3);
    assert.equal(program.probeKind[9], 2);
    assert.equal(program.measurementSchema.lensReadout, false);
    assert.equal(program.measurementSchema.saeReadout, false);
  });

  test("preflights a prospective probe roster without mutating the attached roster", async () => {
    const { request } = await fixture({ withWhitener: true, withSae: true, withJlens: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    compiler.attachProbe({ selector: "local/demo", name: "existing geometry" });
    const before = compiler.listProbes();
    const program = compiler.compile("", [
      { selector: "jlens/hello", name: "next token" },
      { selector: "sae/1", name: "latent alias" },
    ]);
    assert.deepEqual(
      program.measurementSchema.probes.slice(0, 2).map((probe) => probe.name),
      ["next token", "latent alias"],
    );
    assert.deepEqual(compiler.listProbes(), before);

    assert.throws(
      () => compiler.compile("", [{ selector: "local/demo:sae-missing", name: "wrong variant" }]),
      (error) => error.code === "CORE_PACK_VARIANT_UNAVAILABLE",
    );
    assert.deepEqual(compiler.listProbes(), before);
    assert.throws(
      () => compiler.compile("", Array.from({ length: 9 }, (_, index) => ({
        selector: "sae/0",
        name: `capacity ${index}`,
      }))),
      (error) => error.code === "CORE_PACK_PROBE_LIMIT",
    );
    assert.deepEqual(compiler.listProbes(), before);
  });

  test("prospective probe preflight rejects an unavailable instrument", async () => {
    const { request } = await fixture({ withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.throws(
      () => compiler.compile("", [{ selector: "sae/0", name: "missing SAE" }]),
      (error) => error.code === "SAE_PACK_UNAVAILABLE",
    );
    assert.deepEqual(compiler.listProbes(), []);
  });

  test("matches Python geometry, J-lens, and SAE probe identity digests", async () => {
    const { request } = await fixture({
      withWhitener: true,
      withSae: true,
      withSaeMetadata: true,
      withJlens: true,
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    compiler.attachProbe({ selector: "local/demo", name: "geometry identity" });
    compiler.attachProbe({ selector: "jlens/hello", name: "lens identity" });
    compiler.attachProbe({ selector: "sae/0", name: "sae identity" });
    const saeTensorSha256 = request.optionalPacks
      .find(({ pack }) => pack.kind === "sae")
      .artifacts.find(({ manifest }) => manifest.path.endsWith(".safetensors"))
      .manifest.sha256;
    assert.deepEqual(compiler.probeHashes(), {
      "geometry identity": "c10a9ef9572329ac589e570f44593ead935178f314ff850078f1770937e5c061",
      "lens identity": "a308592c267e293b1286d1160261cc54ed4454445d48f6d00b499f55a468fbf5",
      "sae identity": sha256(Buffer.from(
        `('sae-readout-v2', 'fixture-model', '${saeTensorSha256}', 'local:fixture', 1, 0, 2.5)`,
      )),
    });
    const instruments = new BrowserInstrumentRuntime(compiler);
    assert.deepEqual(instruments.probeHashes(), compiler.probeHashes());
    compiler.detachProbe("lens identity");
    assert.equal(instruments.probeHashes()["lens identity"], undefined);
  });

  test("matches Python curved geometry tensor-walk identity", async () => {
    const { request } = await fixture({
      curved: true,
      withWhitener: true,
      structuredHookProfile: "standard-v2",
    });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    compiler.attachProbe({ selector: "local/demo", name: "curved identity" });
    assert.deepEqual(compiler.probeHashes(), {
      "curved identity": "cd50e11ceaaef63c1603be13a7a31c99f3ed3c50cc06bcd1a45ed0905e3c91ca",
    });
  });

  test("rejects curved geometry attachment before generation", async () => {
    const { request } = await fixture({ curved: true, withWhitener: true });
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.throws(
      () => compiler.attachProbe({ selector: "local/demo" }),
      (error) => error.code === "CORE_PACK_CURVED_GATE_UNAVAILABLE",
    );
    assert.deepEqual(compiler.listProbes(), []);
  });

  test("selects the neutral whitener bound to the active context", async () => {
    const { request } = await fixture({ withWhitener: true, withSecondWhitener: true,
      fittedContextBindingSha256: null });
    request.contextTokens = 4096;
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.deepEqual([...compiler.fittingWhiteners().keys()], [0, 1]);
  });

  test("activates exact-runtime fitted manifolds from the local artifact repository", async () => {
    const fixtureWithLocal = await fixture({ withSecondManifold: true });
    const focus = fixtureWithLocal.artifacts.find((artifact) =>
      artifact.manifest.path.endsWith("/focus.drowse")
    );
    assert.ok(focus);
    fixtureWithLocal.request.artifacts = fixtureWithLocal.request.artifacts.filter(
      (artifact) => artifact !== focus,
    );
    fixtureWithLocal.request.requiredCorePack.files =
      fixtureWithLocal.request.requiredCorePack.files.filter(
        (file) => file.sha256 !== focus.manifest.sha256,
      );
    const compiler = await BrowserFeasibilityCorePackCompiler.load(
      fixtureWithLocal.request,
      { installedManifoldArchives: [focus.file] },
    );
    assert.doesNotThrow(() => compiler.compile("0.5 sharp"));
  });

  test("refreshes only installed manifolds while reusing verified core and instrument tensors", async () => {
    const { request, artifacts } = await fixture({
      withWhitener: true,
      withSecondManifold: true,
      withSae: true,
      withJlens: true,
      structuredHookProfile: "standard-v3",
    });
    const focus = artifacts.find((artifact) => artifact.manifest.path.endsWith("/focus.drowse"));
    request.artifacts = request.artifacts.filter((artifact) => artifact !== focus);
    request.requiredCorePack.files = request.requiredCorePack.files.filter(
      (file) => file.sha256 !== focus.manifest.sha256,
    );
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const baseline = compiler.compile("0.5 alert");
    const sae = compiler.saeGpuDictionary();
    const lens = compiler.jlensGpuDictionary();
    for (const { file } of [
      ...request.artifacts,
      ...request.optionalPacks.flatMap((pack) => pack.artifacts),
    ]) {
      for (const method of ["arrayBuffer", "stream", "slice"]) {
        file[method] = () => { throw new Error("unchanged artifact was reread"); };
      }
    }
    const next = await compiler.withInstalledManifolds([focus.file], request.signal);
    assert.doesNotThrow(() => next.compile("0.5 sharp"));
    assert.deepEqual(next.compile("0.5 alert"), baseline);
    assert.equal(next.saeGpuDictionary().encoder, sae.encoder);
    assert.equal(next.jlensGpuDictionary().matrices[0], lens.matrices[0]);
    assert.deepEqual(next.fittingWhiteners(), compiler.fittingWhiteners());
    assert.throws(() => compiler.compile("0.5 sharp"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND");
    const removed = await next.withInstalledManifolds([], request.signal);
    assert.deepEqual(removed.compile("0.5 alert"), baseline);
    assert.throws(() => removed.compile("0.5 sharp"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND");
    await assert.rejects(compiler.withInstalledManifolds([focus.file, focus.file], request.signal),
      (error) => error.code === "CORE_PACK_IDENTITY_COLLISION");
    await assert.rejects(compiler.withInstalledManifolds([new Blob(["corrupt"])], request.signal));
    const aborted = new AbortController();
    aborted.abort();
    await assert.rejects(compiler.withInstalledManifolds([], aborted.signal),
      (error) => error.name === "AbortError");
  });

  test("keeps incompatible local fitted manifolds installed but inactive", async () => {
    const compatible = await fixture();
    const incompatible = await fixture({
      fingerprint: "b".repeat(64),
      withSecondManifold: true,
    });
    const focus = incompatible.artifacts.find((artifact) =>
      artifact.manifest.path.endsWith("/focus.drowse")
    );
    assert.ok(focus);
    const compiler = await BrowserFeasibilityCorePackCompiler.load(
      compatible.request,
      { installedManifoldArchives: [focus.file] },
    );
    assert.doesNotThrow(() => compiler.compile("0.5 alert"));
    assert.throws(
      () => compiler.compile("0.5 sharp"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND",
    );
    const refreshed = await compiler.withInstalledManifolds([focus.file], compatible.request.signal);
    assert.throws(() => refreshed.compile("0.5 sharp"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND");
  });

  test("keeps a local fit for another context installed but inactive", async () => {
    const compatible = await fixture();
    const otherContext = await fixture({
      withSecondManifold: true,
      fittedContextBindingSha256: "e".repeat(64),
    });
    const focus = otherContext.artifacts.find((artifact) =>
      artifact.manifest.path.endsWith("/focus.drowse")
    );
    assert.ok(focus);
    const compiler = await BrowserFeasibilityCorePackCompiler.load(
      compatible.request,
      { installedManifoldArchives: [focus.file] },
    );
    assert.doesNotThrow(() => compiler.compile("0.5 alert"));
    assert.throws(
      () => compiler.compile("0.5 sharp"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND",
    );
    const refreshed = await compiler.withInstalledManifolds([focus.file], compatible.request.signal);
    assert.throws(() => refreshed.compile("0.5 sharp"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND");
  });

  test("rejects local manifold identities that collide with the core pack", async () => {
    const compatible = await fixture();
    await assert.rejects(
      BrowserFeasibilityCorePackCompiler.load(compatible.request, {
        installedManifoldArchives: [compatible.artifacts[0].file],
      }),
      (error) => error.code === "CORE_PACK_IDENTITY_COLLISION",
    );
  });

  for (const golden of composedGolden.cases) {
    test(`matches Python composed lowering: ${golden.name}`, async () => {
      const { request } = await fixture({ withWhitener: true, withSecondManifold: true });
      const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      const program = compiler.compile(golden.expression);
      assert.deepEqual([...program.enabled], [1, 1]);
      golden.expectedLayers.forEach((expected, programIndex) => {
        const offset = programIndex * program.hiddenSize;
        assertFloatArray(
          program.basis.slice(offset, offset + program.hiddenSize),
          expected.basis,
          2e-6,
        );
        assertFloatArray(
          program.neutral.slice(offset, offset + program.hiddenSize),
          expected.neutral,
        );
        assert.ok(Math.abs(program.target[programIndex] - expected.target) <= 2e-6);
        assert.ok(Math.abs(program.along[programIndex] - expected.along) <= 2e-5);
        const delta = program.basis.slice(offset, offset + program.hiddenSize)
          .map((value) => value * program.target[programIndex] * program.along[programIndex]);
        assertFloatArray(delta, expected.delta, 3e-5);
      });

      compiler.attachProbe({ selector: "demo", name: "structured parity" });
      const structured = compiler.compile(golden.expression);
      assert.equal(structured.format, "drowse-structured-v2");
      golden.expectedLayers.forEach((expected, programIndex) => {
        const residual = runStructuredHookLayer(
          structured,
          initialStructuredHookState(structured),
          programIndex,
          [0, 0],
        ).residual;
        assertFloatArray(residual, expected.delta, 4e-5);
      });
    });
  }

  test("fails closed when browser fitting has no neutral whitener", async () => {
    const { request } = await fixture();
    const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    assert.throws(
      () => compiler.fittingWhiteners(),
      (error) => error.code === "CORE_PACK_WHITENER_UNAVAILABLE",
    );
  });

  test("rejects an optional instrument pack from another runtime", async () => {
    const { request } = await fixture({ withSae: true });
    request.optionalPacks[0].pack.runtimeIdentitySha256 = "f".repeat(64);
    await assert.rejects(
      BrowserFeasibilityCorePackCompiler.load(request),
      (error) => error.code === "INSTRUMENT_PACK_RUNTIME_MISMATCH",
    );
  });

  test("rejects runtime, catalog, and unsupported expression mismatches", async () => {
    const mismatched = await fixture({ fingerprint: "b".repeat(64) });
    mismatched.request.variant.runtimeIdentitySha256 = "c".repeat(64);
    await assert.rejects(
      BrowserFeasibilityCorePackCompiler.load(mismatched.request),
      (error) => error.code === "CORE_PACK_RUNTIME_MISMATCH",
    );

    const missing = await fixture();
    missing.request.requiredCorePack.files = [{
      ...missing.manifest,
      sha256: "d".repeat(64),
    }];
    await assert.rejects(
      BrowserFeasibilityCorePackCompiler.load(missing.request),
      (error) => error.code === "CORE_PACK_ARTIFACT_MISMATCH",
    );

    const valid = await fixture();
    const compiler = await BrowserFeasibilityCorePackCompiler.load(valid.request);
    for (const expression of ["!calm", "demo%alert", "calm@response"]) {
      assert.throws(
        () => compiler.compile(expression),
        (error) => error.code?.startsWith("CORE_PACK_"),
        expression,
      );
    }
  });

  let passed = 0;
  for (const { name, run } of tests) {
    await run();
    passed += 1;
    process.stdout.write(`ok ${passed} - ${name}\n`);
  }
  process.stdout.write(`1..${passed}\n`);
} finally {
  await server.close();
}

function discoverNodesHash(
  labels,
  nodePayloads,
  fitMode = "pca",
  roles = [null, null],
) {
  const hash = createHash("sha256");
  hash.update(canonicalJson(labels));
  for (const payload of nodePayloads) hash.update(payload);
  hash.update(canonicalJson({ fit_mode: fitMode, hyperparams: {} }));
  hash.update(canonicalJson(roles));
  hash.update(canonicalJson(roles.map(() => "abstract")));
  return hash.digest("hex");
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => (
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    )).join(",")}}`;
  }
  return JSON.stringify(value);
}

function assertFloatArray(actual, expected, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) <= tolerance,
      `${value} != ${expected[index]} at ${index}`,
    );
  });
}
