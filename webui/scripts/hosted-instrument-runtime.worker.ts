import {
  contextBindingSha256,
  runtimeIdentitySha256,
} from "../src/lib/runtime/catalog";
import type {
  ModelVariant,
  RuntimeIdentity,
} from "../src/lib/runtime/contracts";
import { BrowserFeasibilityCorePackCompiler } from "../src/hosted/runtime/browserCorePack";
import { BrowserInstrumentRuntime } from "../src/hosted/runtime/browserInstrumentRuntime";
import {
  DrowseWebLlmRuntime,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";
import type { BrowserJlensGpuDictionary } from "../src/hosted/runtime/browserInstrumentPacks";
import type { StructuredHookProgramBuffers } from "../src/hosted/runtime/structuredHookProgram";

interface FileEntry {
  path: string;
  role: string;
  bytes: number;
  sha256: string;
}

interface Request {
  type: "run";
  modelId: string;
  runtimeIdentity: RuntimeIdentity;
  structuredHookProfile: ModelVariant["structuredHookProfile"];
  thinkingProfile: ModelVariant["thinkingProfile"];
  quantization: "q4f16_1" | "q4f32_1" | "q0f32";
  requiredFeatures: GPUFeatureName[];
  contextTokens: number;
  contextProfiles: number[];
  modelFiles: FileEntry[];
  coreFiles: FileEntry[];
  jlensFiles: FileEntry[];
  saeFiles: FileEntry[];
  jlensLayerLimit: number | null;
  diagnosticFamily: "combined" | "jlens" | "sae";
  jlensWord: string;
  saeFeature: number;
  stepTimeoutMs: number;
}

const REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X = 256;
const REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP = 256;

self.onmessage = async (event: MessageEvent<Request>) => {
  if (event.data?.type !== "run") return;
  let runtime: DrowseWebLlmRuntime | null = null;
  try {
    if (event.data.structuredHookProfile !== "standard-v3") {
      throw new Error(
        "physical precomputed instruments require standard-v3 exact-readout-v1",
      );
    }
    if (
      event.data.diagnosticFamily !== "sae" &&
      event.data.jlensFiles.length === 0
    ) {
      throw new Error(
        "the focused physical proof requires a precomputed J-lens pack",
      );
    }
    if (
      event.data.diagnosticFamily !== "jlens" &&
      event.data.saeFiles.length === 0
    ) {
      throw new Error(
        "the focused physical proof requires a precomputed SAE pack",
      );
    }
    const webLlmUrl = "/webllm.js";
    const webllm = (await import(
      /* @vite-ignore */ webLlmUrl
    )) as unknown as DrowseWebLlmModule;
    const fingerprint = runtimeIdentitySha256(event.data.runtimeIdentity);
    const contextBinding = contextBindingSha256(
      fingerprint,
      event.data.contextTokens,
    );
    const timings: Record<string, number> = {};
    const staged = await timedStep(
      "artifact_staging",
      event.data.stepTimeoutMs,
      () =>
        Promise.all([
          stage("model", event.data.modelFiles),
          stage("core", event.data.coreFiles),
          stage("jlens", event.data.jlensFiles),
          stage("sae", event.data.saeFiles),
        ]),
    );
    timings.artifactStagingMs = staged.elapsedMs;
    const [modelArtifacts, coreArtifacts, jlensArtifacts, saeArtifacts] =
      staged.value;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter was returned");
    if (
      adapter.limits.maxComputeWorkgroupSizeX <
        REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X ||
      adapter.limits.maxComputeInvocationsPerWorkgroup <
        REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP
    ) {
      throw new Error(
        "the WebGPU adapter cannot provide the exact readout workgroup limits",
      );
    }
    const variant = {
      id: `${event.data.modelId}-${event.data.quantization}`,
      structuredHookProfile: event.data.structuredHookProfile,
      thinkingProfile: event.data.thinkingProfile,
      runtimeIdentity: event.data.runtimeIdentity,
      runtimeIdentitySha256: fingerprint,
      contextProfiles: event.data.contextProfiles.map((contextTokens) => ({
        contextTokens,
        bindingSha256: contextBindingSha256(fingerprint, contextTokens),
      })),
      requirements: {
        features: event.data.requiredFeatures,
        limits: {
          maxStorageBufferBindingSize: 134217728,
          maxComputeWorkgroupSizeX: REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X,
          maxComputeInvocationsPerWorkgroup:
            REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
        },
      },
    };
    const optionalPacks = [];
    if (jlensArtifacts.length > 0) {
      optionalPacks.push({
        pack: {
          id: `${event.data.modelId}-jlens`,
          kind: "jlens",
          displayName: `${event.data.modelId} J-lens`,
          runtimeIdentitySha256: fingerprint,
          files: jlensArtifacts.map(({ manifest }) => manifest),
        },
        artifacts: jlensArtifacts,
      });
    }
    if (saeArtifacts.length > 0) {
      optionalPacks.push({
        pack: {
          id: `${event.data.modelId}-sae`,
          kind: "sae",
          displayName: `${event.data.modelId} SAE`,
          runtimeIdentitySha256: fingerprint,
          files: saeArtifacts.map(({ manifest }) => manifest),
        },
        artifacts: saeArtifacts,
      });
    }
    const request = {
      model: { id: event.data.modelId },
      variant,
      requiredCorePack: {
        id: `${event.data.modelId}-core`,
        files: coreArtifacts.map(({ manifest }) => manifest),
      },
      contextTokens: event.data.contextTokens,
      adapter,
      artifacts: [...modelArtifacts, ...coreArtifacts],
      optionalPacks,
      activationSpool: {},
      signal: new AbortController().signal,
      onDeviceLost() {},
    };
    runtime = new DrowseWebLlmRuntime(webllm);
    const loaded = await timedStep(
      "runtime_load",
      event.data.stepTimeoutMs,
      async () => {
        await runtime!.load({ ...request, artifacts: modelArtifacts } as never);
        const compiler = await BrowserFeasibilityCorePackCompiler.load(
          request as never,
        );
        const saeDictionary = compiler.saeGpuDictionary();
        const jlensDictionary = limitJlensDictionary(
          compiler.jlensGpuDictionary(),
          event.data.jlensLayerLimit,
        );
        if (
          event.data.diagnosticFamily !== "jlens" &&
          saeDictionary === null
        ) {
          throw new Error(
            "The verified precomputed SAE pack exposed no GPU dictionary",
          );
        }
        if (saeDictionary !== null) {
          await runtime!.setSaeDictionary(saeDictionary);
        }
        if (
          event.data.diagnosticFamily !== "sae" &&
          jlensDictionary === null
        ) {
          throw new Error(
            "The verified precomputed J-lens pack exposed no GPU dictionary",
          );
        }
        if (jlensDictionary !== null) {
          await runtime!.setJlensDictionary(jlensDictionary);
        }
        return { compiler, saeDictionary, jlensDictionary };
      },
      () => runtime!.stop(),
    );
    timings.runtimeLoadMs = loaded.elapsedMs;
    const { compiler, saeDictionary, jlensDictionary } = loaded.value;
    const instruments = new BrowserInstrumentRuntime(
      compiler,
      runtime,
      contextBinding,
    );
    const tokenLookup = () => null;
    const lensProbeName = "physical_jlens";
    const saeProbeName = "physical_sae";
    const configured = await timedStep(
      "instrument_setup",
      event.data.stepTimeoutMs,
      async () => {
        const descriptor = compiler.instrumentDescriptor();
        if (
          (event.data.diagnosticFamily !== "sae" &&
            descriptor.jlens === null) ||
          (event.data.diagnosticFamily !== "jlens" && descriptor.sae === null)
        ) {
          throw new Error(
            "a required verified instrument descriptor was not loaded",
          );
        }
        const manifolds = instruments.request(
          { service: "manifolds", method: "list", args: [] },
          tokenLookup,
        ) as { manifolds: unknown[] };
        if (
          !Array.isArray(manifolds.manifolds) ||
          manifolds.manifolds.length === 0
        ) {
          throw new Error(
            "The required core pack exposed no browser manifolds",
          );
        }
        if (event.data.diagnosticFamily !== "sae") {
          await instruments.request(
            {
              service: "probes",
              method: "attach",
              args: [
                {
                  selector: `jlens/${event.data.jlensWord}`,
                  name: lensProbeName,
                },
              ],
            },
            tokenLookup,
          );
        }
        if (event.data.diagnosticFamily !== "jlens") {
          await instruments.request(
            {
              service: "probes",
              method: "attach",
              args: [
                {
                  selector: `sae/${event.data.saeFeature}`,
                  name: saeProbeName,
                },
              ],
            },
            tokenLookup,
          );
        }
        const liveFamilies: Array<"lens" | "sae"> =
          event.data.diagnosticFamily === "combined"
            ? ["lens", "sae"]
            : event.data.diagnosticFamily === "jlens"
              ? ["lens"]
              : ["sae"];
        for (const family of liveFamilies) {
          await instruments.request(
            {
              service: "instruments",
              method: "setLive",
              args: [family, { enabled: true }],
            },
            tokenLookup,
          );
        }
        return descriptor;
      },
    );
    timings.instrumentSetupMs = configured.elapsedMs;
    const descriptor = configured.value;
    const measuredDescriptor = limitDescriptorLayers(
      descriptor,
      event.data.jlensLayerLimit,
    );

    const exact = await timedStep(
      "exact_readout",
      event.data.stepTimeoutMs,
      async () => {
        const program = limitJlensLayers(
          requireStructured(compiler.compile("")),
          event.data.jlensLayerLimit,
          event.data.runtimeIdentity.layerMap.length,
        );
        let measurements: Record<string, any> | null = null;
        const started = performance.now();
        const generation = await runtime!.streamGeneration(
          {
            input: { kind: "raw", prompt: "hello" },
            sampling: { max_tokens: 1, seed: 1, temperature: 0 },
            thinking: false,
            steeringExpression: null,
            hookProgram: program,
            onRawToken(token) {
              measurements =
                (token.measurements as Record<string, any> | undefined) ??
                measurements;
              tokenMilestone(
                "exact_readout",
                token.rawIndex,
                performance.now() - started,
              );
            },
          },
          () => undefined,
        );
        if (generation.tokens !== 1 || measurements === null) {
          throw new Error(
            "one-token exact precomputed readout did not complete",
          );
        }
        const lensReading =
          measurements.instruments?.lens?.readings?.[lensProbeName]?.value;
        const saeReading =
          measurements.instruments?.sae?.readings?.[saeProbeName]?.value;
        if (
          (event.data.diagnosticFamily !== "sae" &&
            (typeof lensReading !== "number" || !Number.isFinite(lensReading))) ||
          (event.data.diagnosticFamily !== "jlens" &&
            (typeof saeReading !== "number" || !Number.isFinite(saeReading)))
        ) {
          throw new Error(
            "one-token exact readout lost its pinned J-lens or SAE reading",
          );
        }
        const chatArtifact = modelArtifacts.find((entry) => entry.manifest.role === "chat_template")!;
        const chatConfig = JSON.parse(await chatArtifact.file.text());
        const completionPrefix = chatConfig.drowse_completion_prefix_token_ids
          ?? chatConfig.polythetic_completion_prefix_token_ids
          ?? chatConfig.saklas_completion_prefix_token_ids
          ?? [];
        const inputIds = [...completionPrefix, ...await runtime!.tokenizeText("hello")];
        if (inputIds.length !== generation.usage.promptTokens) {
          throw new Error("instrument reference prompt does not match the measured generation");
        }
        const position = inputIds.length - 1;
        const capture = await runtime!.capturePreparedRow({ inputIds, position });
        return {
          generation,
          lensReading,
          saeReading,
          referenceCapture: {
            inputIds,
            position,
            hiddenSize: capture.hiddenSize,
            layerCount: capture.layerCount,
            residuals: Array.from(capture.values),
            measurements,
          },
          readouts: requireExactReadouts(
            measurements,
            measuredDescriptor,
            event.data.diagnosticFamily !== "sae",
            event.data.diagnosticFamily !== "jlens",
          ),
        };
      },
      () => runtime!.stop(),
    );
    timings.exactReadoutMs = exact.elapsedMs;

    const gated = event.data.diagnosticFamily === "combined" ? await timedStep(
      "prior_step_gated_generation",
      event.data.stepTimeoutMs,
      async () => {
        const expression =
          `0.01 jlens/${event.data.jlensWord}@generated&when:${lensProbeName}>=0` +
          ` + 0.01 sae/${event.data.saeFeature}@generated&when:${saeProbeName}>-1000000`;
        await compiler.prepareJlensSelectors(expression, runtime!);
        const program = limitJlensLayers(
          requireStructured(compiler.compile(expression)),
          event.data.jlensLayerLimit,
          event.data.runtimeIdentity.layerMap.length,
        );
        if (
          !program.controls ||
          program.jLensBindingId !== jlensDictionary?.bindingId ||
          !program.jLensLayerIndices?.length
        ) {
          throw new Error(
            "precomputed J-lens and SAE steering did not lower into GPU controls",
          );
        }
        const gatedSlots = program.controls.affine.flatMap((control, index) =>
          control?.gate ? [index] : [],
        );
        const gatedFamilies = new Set(
          program.controls.affine.flatMap(
            (control) =>
              control?.gate?.slots.flatMap((slot) => {
                const probe = program.measurementSchema?.probes[slot.probe];
                return probe ? [probe.family] : [];
              }) ?? [],
          ),
        );
        for (const family of ["lens", "sae"] as const) {
          if (!gatedFamilies.has(family)) {
            throw new Error(
              `${family} did not lower into a prior-step GPU gate`,
            );
          }
        }
        const engine = runtime!.requireEngine();
        const originalInstall = engine.setDrowseStructuredProgram.bind(engine);
        const originalUpdate =
          engine.updateDrowseStructuredControls.bind(engine);
        const transitions: number[][] = [];
        engine.setDrowseStructuredProgram = async (installed, modelId) => {
          transitions.push(
            gatedSlots.map((slot) => installed.affineActive[slot]),
          );
          return originalInstall(installed, modelId);
        };
        engine.updateDrowseStructuredControls = async (
          affine,
          curve,
          modelId,
        ) => {
          transitions.push(gatedSlots.map((slot) => affine[slot]));
          return originalUpdate(affine, curve, modelId);
        };
        let measurements: Record<string, any> | null = null;
        const started = performance.now();
        let generation;
        try {
          generation = await runtime!.streamGeneration(
            {
              input: { kind: "raw", prompt: "A calm ocean feels" },
              sampling: { max_tokens: 2, seed: 1, temperature: 0 },
              thinking: false,
              steeringExpression: expression,
              hookProgram: program,
              onRawToken(token) {
                measurements =
                  (token.measurements as Record<string, any> | undefined) ??
                  measurements;
                tokenMilestone(
                  "prior_step_gated_generation",
                  token.rawIndex,
                  performance.now() - started,
                );
              },
            },
            () => undefined,
          );
        } finally {
          engine.setDrowseStructuredProgram = originalInstall;
          engine.updateDrowseStructuredControls = originalUpdate;
        }
        const activationTransitionIndices = gatedSlots.map((_, slotIndex) =>
          transitions.findIndex((values) => values[slotIndex] === 1),
        );
        if (
          generation.tokens !== 2 ||
          gatedSlots.length < 2 ||
          activationTransitionIndices.some((index) => index < 1) ||
          transitions[0]?.some((value) => value !== 0) ||
          measurements === null
        ) {
          throw new Error(
            "two-token generation did not activate every J-lens/SAE GPU gate from prior-step readings",
          );
        }
        const lensReading =
          measurements.instruments?.lens?.readings?.[lensProbeName]?.value;
        const saeReading =
          measurements.instruments?.sae?.readings?.[saeProbeName]?.value;
        if (
          typeof lensReading !== "number" ||
          !Number.isFinite(lensReading) ||
          typeof saeReading !== "number" ||
          !Number.isFinite(saeReading)
        ) {
          throw new Error(
            "gated generation lost its physical J-lens or SAE reading",
          );
        }
        return {
          generation,
          expression,
          lensReading,
          saeReading,
          jLensBytes: jlensDictionary.matrices.reduce(
            (bytes, matrix) => bytes + matrix.byteLength,
            0,
          ),
          gatedTerms: gatedSlots.length,
          activatedGatedTerms: activationTransitionIndices.length,
          activationTransitionIndices,
          gateUpdates: transitions.length,
          gatedFamilies: [...gatedFamilies].sort(),
          readouts: requireExactReadouts(
            measurements,
            measuredDescriptor,
            true,
            true,
          ),
        };
      },
      () => runtime!.stop(),
    ) : null;
    if (gated !== null) {
      timings.priorStepGatedGenerationMs = gated.elapsedMs;
    }

    const unloaded = await timedStep("unload", event.data.stepTimeoutMs, () =>
      runtime!.unload(),
    );
    timings.unloadMs = unloaded.elapsedMs;
    runtime = null;
    self.postMessage({
      type: "done",
      result: {
        releaseEvidence: false,
        runtimeIdentitySha256: fingerprint,
        structuredHookProfile: event.data.structuredHookProfile,
        exactReadoutAbi: "exact-readout-v1",
        layers: event.data.runtimeIdentity.layerMap.length,
        diagnosticJlensLayerLimit: event.data.jlensLayerLimit,
        diagnosticFamily: event.data.diagnosticFamily,
        saeDictionaryFeatureCount: saeDictionary?.featureCount ?? null,
        saeDictionaryRuntimeLayerIndex:
          saeDictionary?.runtimeLayerIndex ?? null,
        jlensWord: event.data.jlensWord,
        saeFeature: event.data.saeFeature,
        timings,
        exactReadout: {
          tokens: exact.value.generation.tokens,
          finishReason: exact.value.generation.finishReason,
          lensReading: exact.value.lensReading,
          saeReading: exact.value.saeReading,
          referenceCapture: exact.value.referenceCapture,
          ...exact.value.readouts,
        },
        priorStepGatedGeneration: gated === null ? null : {
          tokens: gated.value.generation.tokens,
          finishReason: gated.value.generation.finishReason,
          expression: gated.value.expression,
          lensReading: gated.value.lensReading,
          saeReading: gated.value.saeReading,
          jLensBytes: gated.value.jLensBytes,
          gatedTerms: gated.value.gatedTerms,
          activatedGatedTerms: gated.value.activatedGatedTerms,
          activationTransitionIndices: gated.value.activationTransitionIndices,
          gateUpdates: gated.value.gateUpdates,
          gatedFamilies: gated.value.gatedFamilies,
          ...gated.value.readouts,
        },
      },
    });
  } catch (error) {
    try {
      await runtime?.unload();
    } catch {}
    self.postMessage({
      type: "error",
      message:
        error instanceof Error ? (error.stack ?? error.message) : String(error),
    });
  }
};

async function timedStep<T>(
  stageName: string,
  timeoutMs: number,
  run: () => Promise<T> | T,
  onTimeout?: () => Promise<void> | void,
): Promise<{ value: T; elapsedMs: number }> {
  const started = performance.now();
  self.postMessage({ type: "milestone", stage: stageName, status: "started" });
  let timeout: number | undefined;
  try {
    const value = await Promise.race([
      Promise.resolve().then(run),
      new Promise<never>((_, reject) => {
        timeout = self.setTimeout(() => {
          void Promise.resolve(onTimeout?.()).catch(() => undefined);
          reject(new Error(`${stageName} timed out after ${timeoutMs} ms`));
        }, timeoutMs);
      }),
    ]);
    const elapsedMs = performance.now() - started;
    self.postMessage({
      type: "milestone",
      stage: stageName,
      status: "passed",
      elapsedMs,
    });
    return { value, elapsedMs };
  } catch (error) {
    self.postMessage({
      type: "milestone",
      stage: stageName,
      status: "failed",
      elapsedMs: performance.now() - started,
      message:
        error instanceof Error ? (error.stack ?? error.message) : String(error),
    });
    throw error;
  } finally {
    if (timeout !== undefined) self.clearTimeout(timeout);
  }
}

function tokenMilestone(
  stageName: string,
  rawIndex: number,
  elapsedMs: number,
): void {
  self.postMessage({
    type: "token_milestone",
    stage: stageName,
    rawIndex,
    elapsedMs,
  });
}

function requireStructured(program: unknown): StructuredHookProgramBuffers {
  if (!program || typeof program !== "object" || !("affineActive" in program)) {
    throw new Error(
      "browser expression did not lower into the structured GPU program",
    );
  }
  return program as StructuredHookProgramBuffers;
}

function limitJlensLayers(
  program: StructuredHookProgramBuffers,
  limit: number | null,
  modelLayerCount: number,
): StructuredHookProgramBuffers {
  if (limit === null) return program;
  const layerIndices = program.jLensLayerIndices;
  if (layerIndices === undefined || program.jLensBindingId === undefined) {
    throw new Error("The diagnostic J-lens layer limit requires a J-lens program");
  }
  if (limit > layerIndices.length) {
    throw new Error(
      `The diagnostic J-lens layer limit ${limit} exceeds the ${layerIndices.length} fitted layers`,
    );
  }
  if (program.probeKind.length % modelLayerCount !== 0) {
    throw new Error("The diagnostic J-lens program has an invalid probe layout");
  }
  const probesPerLayer = program.probeKind.length / modelLayerCount;
  const retainedLayers = new Set(layerIndices.slice(0, limit));
  const probeKind = new Uint32Array(program.probeKind);
  for (let layer = 0; layer < modelLayerCount; layer += 1) {
    if (retainedLayers.has(layer)) continue;
    const offset = layer * probesPerLayer;
    for (let probe = 0; probe < probesPerLayer; probe += 1) {
      if (probeKind[offset + probe] === 3) probeKind[offset + probe] = 0;
    }
  }
  return {
    ...program,
    probeKind,
    jLensLayerIndices: layerIndices.slice(0, limit),
  };
}

function limitJlensDictionary(
  dictionary: BrowserJlensGpuDictionary | null,
  limit: number | null,
): BrowserJlensGpuDictionary | null {
  if (dictionary === null || limit === null) return dictionary;
  if (limit > dictionary.layerIndices.length) {
    throw new Error(
      `The diagnostic J-lens layer limit ${limit} exceeds the ${dictionary.layerIndices.length} fitted layers`,
    );
  }
  return {
    ...dictionary,
    layerIndices: dictionary.layerIndices.slice(0, limit),
    matrices: dictionary.matrices.slice(0, limit),
  };
}

function limitDescriptorLayers(
  descriptor: ReturnType<
    BrowserFeasibilityCorePackCompiler["instrumentDescriptor"]
  >,
  limit: number | null,
): ReturnType<BrowserFeasibilityCorePackCompiler["instrumentDescriptor"]> {
  if (limit === null) return descriptor;
  if (descriptor.jlens === null || limit > descriptor.jlens.layers.length) {
    throw new Error("The diagnostic J-lens layer limit exceeds the fitted descriptor");
  }
  return {
    ...descriptor,
    jlens: {
      ...descriptor.jlens,
      layers: descriptor.jlens.layers.slice(0, limit),
    },
  };
}

function requireExactReadouts(
  measurements: Record<string, any>,
  descriptor: ReturnType<
    BrowserFeasibilityCorePackCompiler["instrumentDescriptor"]
  >,
  requireJlens: boolean,
  requireSae: boolean,
): Record<string, number> {
  const aggregate = measurements.instruments?.lens?.readout?.aggregate;
  const layers = measurements.instruments?.lens?.readout?.layers;
  const result: Record<string, number> = {};
  if (requireJlens) {
    if (
      descriptor.jlens === null ||
      !Array.isArray(aggregate) ||
      aggregate.length !== 8 ||
      !Array.isArray(layers) ||
      layers.length !== descriptor.jlens.layers.length
    ) {
      throw new Error("The GPU returned no exact full-vocabulary J-lens readout");
    }
    result.exactJlensAggregateTokens = aggregate.length;
    result.exactJlensLayerRows = layers.length;
  }
  if (!requireSae) return result;
  const features = measurements.instruments?.sae?.readout?.features;
  if (
    descriptor.sae === null ||
    !Array.isArray(features) ||
    features.length < 1 ||
    features.length > 8 ||
    features.some(
      (feature) =>
        !Number.isSafeInteger(feature?.id) ||
        feature.id < 0 ||
        feature.id >= descriptor.sae!.features ||
        typeof feature?.activation !== "number" ||
        !Number.isFinite(feature.activation) ||
        feature.activation < 0,
    )
  ) {
    throw new Error("The GPU returned no exact full-dictionary SAE readout");
  }
  return {
    ...result,
    exactSaeDiscoveredFeatures: features.length,
    exactSaeDictionaryFeatureCount: descriptor.sae.features,
  };
}

async function stage(set: string, files: FileEntry[]) {
  const root = await navigator.storage.getDirectory();
  const directory = await root.getDirectoryHandle(`drowse-instrument-${set}`, {
    create: true,
  });
  const revision = "a".repeat(40);
  const artifacts = [];
  for (const entry of files) {
    const response = await fetch(
      `/artifact?set=${set}&path=${encodeURIComponent(entry.path)}`,
      { cache: "no-store" },
    );
    if (!response.ok)
      throw new Error(`artifact fetch failed for ${set}/${entry.path}`);
    const handle = await fileHandle(directory, entry.path);
    const writable = await handle.createWritable();
    await writable.write(await response.blob());
    await writable.close();
    artifacts.push({
      manifest: {
        ...entry,
        revision,
        url: `https://huggingface.co/polythetic/local/resolve/${revision}/${entry.path}`,
      },
      file: await handle.getFile(),
    });
  }
  return artifacts;
}

async function fileHandle(
  root: FileSystemDirectoryHandle,
  path: string,
): Promise<FileSystemFileHandle> {
  const parts = path.split("/");
  if (
    parts.length === 0 ||
    parts.some(
      (part) => !part || part === "." || part === ".." || part.includes("\\"),
    )
  )
    throw new Error(`artifact has an unsafe path: ${path}`);
  let directory = root;
  for (const part of parts.slice(0, -1)) {
    directory = await directory.getDirectoryHandle(part, { create: true });
  }
  return directory.getFileHandle(parts.at(-1)!, { create: true });
}
