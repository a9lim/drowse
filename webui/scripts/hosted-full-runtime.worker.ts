import {
  contextBindingSha256,
  runtimeIdentitySha256,
} from "../src/lib/runtime/catalog";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import type {
  CatalogFile,
  CatalogInstrumentPack,
  ModelVariant,
  RuntimeIdentity,
  RuntimeProgressEvent,
} from "../src/lib/runtime/contracts";
import { BrowserDrowseArchiveService } from "../src/hosted/artifacts/service";
import { BrowserActivationSpool } from "../src/hosted/fitting/activationSpool";
import { BrowserCapabilityChecker } from "../src/hosted/runtime/capabilities";
import { BrowserFeasibilityCorePackCompiler } from "../src/hosted/runtime/browserCorePack";
import { BrowserInstrumentRuntime } from "../src/hosted/runtime/browserInstrumentRuntime";
import { BrowserManifoldFitting } from "../src/hosted/runtime/browserManifoldFitting";
import type {
  BrowserInstrumentPackArtifacts,
  BrowserModelArtifact,
  BrowserModelLoadRequest,
} from "../src/hosted/runtime/modelBackend";
import {
  DrowseWebLlmRuntime,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";
import type { StructuredHookProgramBuffers } from "../src/hosted/runtime/structuredHookProgram";
import { validatePairedSteeringControls } from "./browser-full-runtime-contract.mjs";

type FileEntry = CatalogFile;

interface RunRequest {
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
  maxTokens: number;
  stopMaxTokens: number;
  fitLayer: number | null;
  jlensWord: string;
  saeFeature: number;
}

type CheckName =
  | "webgpu_adapter"
  | "load"
  | "ordinary_generation"
  | "forced_replay"
  | "explicit_capture"
  | "flat_gated_instruments"
  | "curved_fitting"
  | "curved_generation"
  | "stop"
  | "unload_reload"
  | "offline_reuse";

const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 10;
const REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X = 256;
const REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP = 256;
const REQUIRED_STRUCTURED_HOOK_PROFILE = "standard-v3";
const nativeFetch = globalThis.fetch.bind(globalThis);
let offline = false;
const offlineNetworkAttempts: string[] = [];
const externalNetworkAttempts: string[] = [];
globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;
  const resolved = new URL(url, globalThis.location.href);
  if (resolved.protocol === "http:" || resolved.protocol === "https:") {
    if (offline) {
      offlineNetworkAttempts.push(resolved.href);
      return Promise.reject(
        new TypeError(`offline E2E blocked network request ${resolved.href}`),
      );
    }
    if (resolved.origin !== globalThis.location.origin) {
      externalNetworkAttempts.push(resolved.href);
      return Promise.reject(
        new TypeError(
          `real-browser E2E blocked external request ${resolved.href}`,
        ),
      );
    }
  }
  return nativeFetch(input, init);
}) as typeof globalThis.fetch;

self.onmessage = async (event: MessageEvent<RunRequest>) => {
  if (event.data?.type !== "run") return;
  let currentCheck: CheckName | "start" = "start";
  let runtime: DrowseWebLlmRuntime | null = null;
  let artifacts: BrowserDrowseArchiveService | null = null;
  const harnessManifold = {
    namespace: "local",
    name: "browser_e2e_curve",
  } as const;
  try {
    if (event.data.structuredHookProfile !== REQUIRED_STRUCTURED_HOOK_PROFILE) {
      throw new Error(
        `real-browser instruments require ${REQUIRED_STRUCTURED_HOOK_PROFILE} exact-readout-v1`,
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
    const checker = new BrowserCapabilityChecker();
    let capabilities: Awaited<ReturnType<BrowserCapabilityChecker["check"]>>;
    await check("webgpu_adapter", async () => {
      currentCheck = "webgpu_adapter";
      capabilities = await checker.check();
      if (!capabilities.supported) {
        const failures = capabilities.issues
          .filter((issue) => issue.severity === "hard")
          .map((issue) => `${issue.code}: ${issue.message}`);
        throw new Error(`browser compatibility failed: ${failures.join("; ")}`);
      }
      if (capabilities.webGpu.fallback !== "hardware") {
        throw new Error("browser did not confirm a hardware WebGPU adapter");
      }
      const storageBindings =
        capabilities.webGpu.limits.maxStorageBuffersPerShaderStage;
      if (
        typeof storageBindings !== "number" ||
        storageBindings < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE
      ) {
        throw new Error(
          `browser exposes ${storageBindings ?? "unknown"} storage buffers per shader stage; ` +
            `${REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE} are required`,
        );
      }
      const workgroupSizeX =
        capabilities.webGpu.limits.maxComputeWorkgroupSizeX;
      const workgroupInvocations =
        capabilities.webGpu.limits.maxComputeInvocationsPerWorkgroup;
      if (
        typeof workgroupSizeX !== "number" ||
        workgroupSizeX < REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X ||
        typeof workgroupInvocations !== "number" ||
        workgroupInvocations < REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP
      ) {
        throw new Error(
          "browser cannot provide the exact readout workgroup limits " +
            `(sizeX=${workgroupSizeX ?? "unknown"}, ` +
            `invocations=${workgroupInvocations ?? "unknown"})`,
        );
      }
      return {
        fallback: capabilities.webGpu.fallback,
        featureCount: capabilities.webGpu.features.length,
        adapterFeatures: [...capabilities.webGpu.features].sort(),
        adapterLimits: Object.fromEntries(
          Object.entries(capabilities.webGpu.limits)
            .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
            .sort(([left], [right]) => left.localeCompare(right)),
        ),
        maxStorageBufferBindingSize:
          capabilities.webGpu.limits.maxStorageBufferBindingSize ?? null,
        maxStorageBuffersPerShaderStage:
          capabilities.webGpu.limits.maxStorageBuffersPerShaderStage ?? null,
        requiredMaxStorageBuffersPerShaderStage:
          REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
        maxComputeWorkgroupSizeX: workgroupSizeX,
        maxComputeInvocationsPerWorkgroup: workgroupInvocations,
        requiredMaxComputeWorkgroupSizeX: REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X,
        requiredMaxComputeInvocationsPerWorkgroup:
          REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
        calibrationScore: capabilities.signals.calibrationScore,
        crossOriginIsolated: capabilities.crossOriginIsolated,
      };
    });
    currentCheck = "load";
    const modelArtifacts = await stage("model", event.data.modelFiles);
    const coreArtifacts = await stage("core", event.data.coreFiles);
    const jlensArtifacts = await stage("jlens", event.data.jlensFiles);
    const saeArtifacts = await stage("sae", event.data.saeFiles);
    const artifactVerification = {
      schemaVersion: 1,
      sets: {
        model: artifactSetReceipt("model", event.data.modelFiles),
        core: artifactSetReceipt("core", event.data.coreFiles),
        jlens: artifactSetReceipt("jlens", event.data.jlensFiles),
        sae: artifactSetReceipt("sae", event.data.saeFiles),
      },
    };
    const variant: ModelVariant = {
      id: `${event.data.modelId}-${event.data.quantization}-e2e`,
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
          maxStorageBufferBindingSize: 134_217_728,
          maxComputeWorkgroupStorageSize: 32_768,
          maxComputeWorkgroupSizeX: REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X,
          maxComputeInvocationsPerWorkgroup:
            REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
          maxStorageBuffersPerShaderStage:
            REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
        },
      },
    } as unknown as ModelVariant;
    const requiredCorePack: CatalogInstrumentPack = {
      id: `${event.data.modelId}-core-e2e`,
      kind: "core",
      displayName: "Local E2E core pack",
      runtimeIdentitySha256: fingerprint,
      files: coreArtifacts.map(({ manifest }) => manifest),
    } as CatalogInstrumentPack;
    const optionalPacks: BrowserInstrumentPackArtifacts[] = [
      optionalPack("jlens", event.data.modelId, fingerprint, jlensArtifacts),
      optionalPack("sae", event.data.modelId, fingerprint, saeArtifacts),
    ];
    const loadRequest = async (): Promise<BrowserModelLoadRequest> => ({
      model: {
        id: event.data.modelId,
      } as unknown as BrowserModelLoadRequest["model"],
      variant,
      requiredCorePack,
      contextTokens: event.data.contextTokens,
      adapter: await checker.adapterForLoad(),
      artifacts: [...modelArtifacts, ...coreArtifacts],
      optionalPacks,
      activationSpool: new BrowserActivationSpool(),
      signal: new AbortController().signal,
      onDeviceLost(failure) {
        deviceLosses.push(failure);
      },
    });
    const deviceLosses: Array<{
      code: string;
      message: string;
      confirmedOom: boolean;
    }> = [];
    let request = await loadRequest();
    const instrumentReleaseAssertions = await verifyInstrumentRejections(request);
    runtime = new DrowseWebLlmRuntime(webllm);
    let compiler: BrowserFeasibilityCorePackCompiler;
    let instruments: BrowserInstrumentRuntime;
    let saeDictionary: ReturnType<
      BrowserFeasibilityCorePackCompiler["saeGpuDictionary"]
    >;
    let jlensDictionary: ReturnType<
      BrowserFeasibilityCorePackCompiler["jlensGpuDictionary"]
    >;
    await check("load", async () => {
      currentCheck = "load";
      await runtime!.load(request);
      compiler = await BrowserFeasibilityCorePackCompiler.load(request);
      saeDictionary = compiler.saeGpuDictionary();
      jlensDictionary = compiler.jlensGpuDictionary();
      if (saeDictionary === null) {
        throw new Error(
          "the verified precomputed SAE pack exposed no GPU dictionary",
        );
      }
      if (jlensDictionary === null) {
        throw new Error(
          "the verified precomputed J-lens pack exposed no GPU dictionary",
        );
      }
      await runtime!.setSaeDictionary(saeDictionary);
      await runtime!.setJlensDictionary(jlensDictionary);
      instruments = new BrowserInstrumentRuntime(
        compiler,
        runtime!,
        contextBinding,
      );
      const hooks = runtime!.hookCapabilities();
      for (const capability of [
        "curved_manifold",
        "phase_trigger",
        "probe_gate",
        "sae",
        "jlens",
      ] as const) {
        if (!hooks[capability])
          throw new Error(`loaded runtime lacks ${capability}`);
      }
      const runtimeCapabilities = runtime!.runtimeCapabilities();
      if (!runtimeCapabilities.tokenizer)
        throw new Error("loaded runtime lacks its exact tokenizer port");
      assertNoDeviceLoss(deviceLosses);
      return {
        runtimeIdentitySha256: fingerprint,
        layers: event.data.runtimeIdentity.layerMap.length,
        hiddenSize: event.data.runtimeIdentity.hiddenSize,
        rankOne: hooks.rank_one,
        multiTerm: hooks.multi_term,
        curved: hooks.curved_manifold,
        sae: hooks.sae,
        jlens: hooks.jlens,
        tokenizer: runtimeCapabilities.tokenizer,
        exactReadoutProfile: event.data.structuredHookProfile,
        saeDictionaryFeatureCount: saeDictionary.featureCount,
        saeDictionaryRuntimeLayerIndex: saeDictionary.runtimeLayerIndex,
        stagedArtifactFiles:
          modelArtifacts.length +
          coreArtifacts.length +
          jlensArtifacts.length +
          saeArtifacts.length,
        stagedArtifactBytes: [
          ...event.data.modelFiles,
          ...event.data.coreFiles,
          ...event.data.jlensFiles,
          ...event.data.saeFiles,
        ].reduce((sum, file) => sum + file.bytes, 0),
        allOpfsHashesVerified: true,
      };
    });

    let ordinaryRawTokens: number[] = [];
    await check("ordinary_generation", async () => {
      currentCheck = "ordinary_generation";
      const rawTokens: number[] = [];
      const result = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: { ...deterministicSampling(event.data.maxTokens), return_top_k: 8 },
          thinking: false,
          steeringExpression: null,
          hookProgram: null,
          onRawToken(token) {
            if (token.tokenId !== null) rawTokens.push(token.tokenId);
          },
        },
        () => undefined,
      );
      if (result.tokens < 1 || rawTokens.length !== result.tokens) {
        throw new Error(
          "ordinary generation returned no exact raw-token stream",
        );
      }
      ordinaryRawTokens = [...rawTokens];
      if (result.measurements !== undefined) {
        throw new Error(
          "ordinary generation unexpectedly returned Drowse measurements",
        );
      }
      if (
        result.prefillTokensPerSecond === null ||
        result.prefillTokensPerSecond <= 0 ||
        result.decodeTokensPerSecond === null ||
        result.decodeTokensPerSecond <= 0
      ) {
        throw new Error(
          "ordinary generation returned no measured prefill/decode speed",
        );
      }
      const chatChecks: Array<{ prompt: string; response: string; passed: boolean; promptTokens: number; tokenIds: number[]; finishReason: string }> = [];
      if (runtime!.runtimeCapabilities().baseModel !== true) {
        for (const [prompt, expected] of [
          ["What is 2 + 2? Answer with only the number.", /^\s*4[.!]?\s*$/u],
          ["What is the capital of France? Answer with only the city.", /^\s*Paris[.!]?\s*$/iu],
        ] as const) {
          const tokenIds: number[] = [];
          const answer = await runtime!.streamGeneration({
            input: { kind: "chat", messages: [{ role: "user", content: prompt }] },
            sampling: { ...deterministicSampling(24), return_top_k: 8 },
            thinking: false,
            steeringExpression: null,
            hookProgram: null,
            onRawToken(token) { if (token.tokenId !== null) tokenIds.push(token.tokenId); },
          }, () => undefined);
          chatChecks.push({ prompt, response: answer.text, passed: expected.test(answer.text), promptTokens: answer.usage.promptTokens, tokenIds, finishReason: answer.finishReason });
        }
        if (chatChecks.some(({ passed }) => !passed)) {
          throw new Error(`unsteered chat sanity checks failed: ${JSON.stringify(chatChecks)}`);
        }
      }
      return {
        tokens: result.tokens,
        text: result.text,
        chatChecks,
        finishReason: result.finishReason,
        prefillTokensPerSecond: result.prefillTokensPerSecond,
        decodeTokensPerSecond: result.decodeTokensPerSecond,
      };
    });

    await check("forced_replay", async () => {
      currentCheck = "forced_replay";
      const runtimeCapabilities = runtime!.runtimeCapabilities();
      if (!runtimeCapabilities.forcedReplay || !runtimeCapabilities.replayScoring) {
        throw new Error(
          "loaded runtime lacks exact forced-token replay or compact replay scoring",
        );
      }
      const forcedPrefix = ordinaryRawTokens.slice(
        0,
        Math.min(2, ordinaryRawTokens.length),
      );
      if (forcedPrefix.length === 0) {
        throw new Error("ordinary generation left no token prefix to replay");
      }
      const requestedTokenIds = [...new Set(ordinaryRawTokens.slice(0, 4))];
      const replayed: Array<{
        tokenId: number | null;
        replayScore?: {
          emittedTokenId: number;
          sampledTokenId: number;
          forcedTokenId: number | null;
          requestedLogprobs: Array<{ tokenId: number; logprob: number }>;
          argmax: { tokenId: number; logprob: number };
          topLogprobs: Array<{ tokenId: number; logprob: number }>;
        };
      }> = [];
      const result = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: {
            ...deterministicSampling(forcedPrefix.length),
            temperature: 1,
            return_top_k: 5,
          },
          thinking: false,
          replay: {
            forcedPrefixTokenIds: forcedPrefix,
            scoreTokenIds: requestedTokenIds,
          },
          steeringExpression: null,
          hookProgram: null,
          onRawToken(token) {
            replayed.push({
              tokenId: token.tokenId,
              ...(token.replayScore === undefined
                ? {}
                : { replayScore: token.replayScore }),
            });
          },
        },
        () => undefined,
      );
      if (result.tokens !== forcedPrefix.length || replayed.length !== forcedPrefix.length) {
        throw new Error("forced replay returned an unexpected token count");
      }
      for (let index = 0; index < replayed.length; index += 1) {
        const row = replayed[index];
        const score = row.replayScore;
        const missingRequested = score === undefined
          ? requestedTokenIds
          : requestedTokenIds.filter(
              (tokenId) =>
                !score.requestedLogprobs.some(
                  (candidate) =>
                    candidate.tokenId === tokenId &&
                    (Number.isFinite(candidate.logprob) ||
                      candidate.logprob === Number.NEGATIVE_INFINITY),
                ),
            );
        if (
          row.tokenId !== forcedPrefix[index] ||
          score === undefined ||
          score.emittedTokenId !== forcedPrefix[index] ||
          score.forcedTokenId !== forcedPrefix[index] ||
          score.topLogprobs.length < 2 ||
          missingRequested.length > 0
        ) {
          throw new Error(
            `forced replay row ${index} did not preserve its token and exact score envelope: ` +
              JSON.stringify({
                expectedTokenId: forcedPrefix[index],
                actualTokenId: row.tokenId,
                score,
                missingRequested,
              }),
          );
        }
      }
      const firstScore = replayed[0].replayScore!;
      const alternative = firstScore.topLogprobs.find(
        (candidate) => candidate.tokenId !== forcedPrefix[0],
      );
      if (alternative === undefined) {
        throw new Error("forced replay exposed no alternate token for a physical fork");
      }
      let forkedToken: (typeof replayed)[number] | null = null;
      const forked = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: {
            ...deterministicSampling(1),
            temperature: 1,
            return_top_k: 5,
          },
          thinking: false,
          replay: {
            forcedPrefixTokenIds: [alternative.tokenId],
            scoreTokenIds: [forcedPrefix[0], alternative.tokenId],
          },
          steeringExpression: null,
          hookProgram: null,
          onRawToken(token) {
            forkedToken = {
              tokenId: token.tokenId,
              ...(token.replayScore === undefined
                ? {}
                : { replayScore: token.replayScore }),
            };
          },
        },
        () => undefined,
      );
      if (
        forked.tokens !== 1 ||
        forkedToken?.tokenId !== alternative.tokenId ||
        forkedToken.replayScore?.forcedTokenId !== alternative.tokenId
      ) {
        throw new Error("alternate-token fork did not follow the exact forced token");
      }
      return {
        replayedTokens: replayed.length,
        requestedScoresPerToken: requestedTokenIds.length,
        topScoresPerToken: firstScore.topLogprobs.length,
        originalFirstTokenId: forcedPrefix[0],
        alternateFirstTokenId: alternative.tokenId,
        alternateTokenLogprob: alternative.logprob,
        forkedTokens: forked.tokens,
      };
    });

    await check("explicit_capture", async () => {
      currentCheck = "explicit_capture";
      const prepared = await runtime!.prepareCaptureRows([
        {
          system: "",
          messages: [
            { role: "user", content: "Describe the weather." },
            { role: "assistant", content: "The day is calm and clear." },
          ],
        },
      ]);
      if (prepared.length !== 1)
        throw new Error("explicit capture row preparation failed");
      const capture = await runtime!.capturePreparedRow(prepared[0]);
      const expectedValues =
        event.data.runtimeIdentity.layerMap.length *
        event.data.runtimeIdentity.hiddenSize;
      if (
        capture.layerCount !== event.data.runtimeIdentity.layerMap.length ||
        capture.hiddenSize !== event.data.runtimeIdentity.hiddenSize ||
        capture.positionCount !== 1 ||
        capture.values.length !== expectedValues ||
        capture.values.some((value) => !Number.isFinite(value))
      ) {
        throw new Error(
          "explicit post-block capture returned an invalid tensor envelope",
        );
      }
      let absoluteMax = 0;
      for (const value of capture.values)
        absoluteMax = Math.max(absoluteMax, Math.abs(value));
      if (!(absoluteMax > 0))
        throw new Error("explicit post-block capture was identically zero");
      return {
        layers: capture.layerCount,
        positions: capture.positionCount,
        hiddenSize: capture.hiddenSize,
        values: capture.values.length,
        absoluteMax,
      };
    });

    let flatSelector = "";
    let flatLabel = "";
    await check("flat_gated_instruments", async () => {
      currentCheck = "flat_gated_instruments";
      const flat = await firstGeometry(instruments!, true, "browser_e2e_flat");
      flatSelector = flat.selector;
      flatLabel = flat.label;
      const controlLogprobs: Record<string, number> = {};
      const controlOutcomes: Array<{ name: string; tokens: number; finishReason: string; logprob: string }> = [];
      for (const [name, coefficient] of [["baseline", null], ["zero", 0], ["positive", 0.5], ["negative", -0.5]] as const) {
        const controlExpression = coefficient === null
          ? null : `${coefficient} ${flat.selector}%${flat.label}@both`;
        const control = await runtime!.streamGeneration({
          input: { kind: "raw", prompt: "The sky looks blue because" },
          sampling: { ...deterministicSampling(1), temperature: 1 },
          thinking: false,
          replay: { forcedPrefixTokenIds: ordinaryRawTokens.slice(0, 1), scoreTokenIds: ordinaryRawTokens.slice(0, 1) },
          steeringExpression: controlExpression,
          hookProgram: controlExpression === null ? null : compiler!.compile(controlExpression),
          onRawToken(token) {
            const score = token.replayScore?.requestedLogprobs.find(row => row.tokenId === ordinaryRawTokens[0]);
            if (score) controlLogprobs[name] = score.logprob;
          },
        }, () => undefined);
        controlOutcomes.push({ name, tokens: control.tokens, finishReason: control.finishReason, logprob: String(controlLogprobs[name]) });
      }
      const controlMeasurements = validatePairedSteeringControls(controlLogprobs);
      await instruments!.request(
        {
          service: "probes",
          method: "attach",
          args: [
            {
              selector: `jlens/${event.data.jlensWord}`,
              name: "browser_e2e_jlens",
            },
          ],
        },
        () => null,
      );
      await instruments!.request(
        {
          service: "probes",
          method: "attach",
          args: [
            {
              selector: `sae/${event.data.saeFeature}`,
              name: "browser_e2e_sae",
            },
          ],
        },
        () => null,
      );
      for (const family of ["geometry", "lens", "sae"] as const) {
        await instruments!.request(
          {
            service: "instruments",
            method: "setLive",
            args: [family, { enabled: true }],
          },
          () => null,
        );
      }
      const expression =
        `0.05 ${flat.selector}%${flat.label}@generated&when:browser_e2e_flat>-1000000` +
        ` + 0.01 jlens/${event.data.jlensWord}@generated&when:browser_e2e_jlens>=0` +
        ` + 0.01 sae/${event.data.saeFeature}@generated&when:browser_e2e_sae>-1000000`;
      await compiler!.prepareJlensSelectors(expression, runtime!);
      const program = requireStructured(compiler!.compile(expression));
      if (!program.controls)
        throw new Error("flat steering gate exposed no runtime controls");
      const gatedSlots = program.controls.affine.flatMap((control, index) =>
        control?.gate ? [index] : [],
      );
      if (gatedSlots.length === 0)
        throw new Error("flat steering gate did not lower into GPU controls");
      const geometryScoreKeys = new Set(
        program.measurementSchema?.geometryProbes?.flatMap(
          (probe) => probe?.scoreKeys ?? [],
        ) ?? [],
      );
      const gatedFamilies = new Set(
        program.controls.affine.flatMap((control) => {
          const gate = control?.gate;
          if (!gate) return [];
          const families = gate.slots.flatMap((slot) => {
            const probe = program.measurementSchema?.probes[slot.probe];
            return probe ? [probe.family] : [];
          });
          if (gate.scoreKey && geometryScoreKeys.has(gate.scoreKey)) {
            families.push("geometry");
          }
          return families;
        }),
      );
      for (const family of ["geometry", "lens", "sae"] as const) {
        if (!gatedFamilies.has(family)) {
          throw new Error(
            `${family} measurements did not lower into a physical GPU gate`,
          );
        }
      }
      if (
        program.jLensBindingId !== jlensDictionary.bindingId ||
        !program.jLensLayerIndices?.length
      ) {
        throw new Error("J-lens steering did not lower into the GPU program");
      }
      const engine = runtime!.requireEngine();
      const originalUpdate = engine.updateDrowseStructuredControls.bind(engine);
      const gateTransitions: number[][] = [];
      engine.updateDrowseStructuredControls = async (
        affineActive,
        curveActive,
        modelId,
      ) => {
        gateTransitions.push(gatedSlots.map((slot) => affineActive[slot]));
        return originalUpdate(affineActive, curveActive, modelId);
      };
      let measurements: any = null;
      try {
        const result = await runtime!.streamGeneration(
          {
            input: { kind: "raw", prompt: "A welcoming person says" },
            sampling: deterministicSampling(event.data.maxTokens),
            thinking: false,
            steeringExpression: expression,
            hookProgram: program,
            onRawToken(token) {
              measurements = token.measurements ?? measurements;
            },
          },
          () => undefined,
        );
        if (result.tokens < 2 || result.measurements?.scope !== "aggregate") {
          throw new Error(
            "flat gated generation did not return aggregate measurements",
          );
        }
      } finally {
        engine.updateDrowseStructuredControls = originalUpdate;
      }
      const flatReading = measurements?.scores?.browser_e2e_flat;
      const lensReading =
        measurements?.instruments?.lens?.readings?.browser_e2e_jlens?.value;
      const saeReading =
        measurements?.instruments?.sae?.readings?.browser_e2e_sae?.value;
      for (const [name, value] of Object.entries({
        flatReading,
        lensReading,
        saeReading,
      })) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new Error(`${name} was not a finite physical GPU measurement`);
        }
      }
      const exactReadouts = requireExactPrecomputedReadouts(
        measurements,
        compiler!.instrumentDescriptor(),
        "initial generation",
      );
      const activatedGateSlots = gatedSlots.filter((_, slotIndex) =>
        gateTransitions.some((values) => values[slotIndex] === 1),
      );
      if (activatedGateSlots.length !== gatedSlots.length) {
        throw new Error(
          "not every physical geometry, J-lens, and SAE gate activated after a measured token",
        );
      }
      return {
        flatSelector,
        controlTokenId: ordinaryRawTokens[0],
        ...controlMeasurements,
        controlOutcomes,
        gateUpdates: gateTransitions.length,
        gatedTerms: gatedSlots.length,
        activatedGatedTerms: activatedGateSlots.length,
        gatedFamilies: [...gatedFamilies].sort(),
        flatReading,
        lensReading,
        saeReading,
        ...instrumentReleaseAssertions,
        ...exactReadouts,
        jLensBytes: jlensDictionary.matrices.reduce(
          (bytes, matrix) => bytes + matrix.byteLength,
          0,
        ),
      };
    });

    let curveSelector = "";
    let curveLabel = "";
    await check("curved_fitting", async () => {
      currentCheck = "curved_fitting";
      for (const name of [
        "browser_e2e_flat",
        "browser_e2e_jlens",
        "browser_e2e_sae",
      ]) {
        await instruments!.request(
          { service: "probes", method: "detach", args: [name] },
          () => null,
        );
      }
      for (const family of ["lens", "sae"] as const) {
        await instruments!.request(
          {
            service: "instruments",
            method: "setLive",
            args: [family, { enabled: false }],
          },
          () => null,
        );
      }
      artifacts = new BrowserDrowseArchiveService(undefined, undefined, "0.0.0");
      await artifacts.request(
        {
          service: "manifolds",
          method: "delete",
          args: [harnessManifold.namespace, harnessManifold.name],
        },
        () => undefined,
      );
      await artifacts.request(
        {
          service: "manifolds",
          method: "createDiscover",
          args: [
            {
              namespace: harnessManifold.namespace,
              name: harnessManifold.name,
              description: "Real-browser E2E curved fitting fixture",
              fit_mode: "spectral",
              hyperparams: {
                max_dim: 1,
                min_dim: 1,
                k_nn: 2,
                max_subspace_dim: 2,
                smoothing: "auto",
              },
              nodes: [
                {
                  label: "distant",
                  role: "distant_voice",
                  statements: [
                    "I keep my distance from everyone.",
                    "I avoid inviting anyone closer.",
                  ],
                },
                {
                  label: "reserved",
                  role: "reserved_voice",
                  statements: [
                    "I remain quiet around new people.",
                    "I respond politely but keep to myself.",
                  ],
                },
                {
                  label: "open",
                  role: "open_voice",
                  statements: [
                    "I greet people and listen with interest.",
                    "I am comfortable sharing this moment.",
                  ],
                },
                {
                  label: "warm",
                  role: "warm_voice",
                  statements: [
                    "I make everyone feel welcome and at home.",
                    "I speak with generous warmth and care.",
                  ],
                },
                {
                  label: "exuberant",
                  role: "exuberant_voice",
                  statements: [
                    "I celebrate meeting everyone with delight.",
                    "I am overflowing with joyful enthusiasm.",
                  ],
                },
              ],
            },
          ],
        },
        () => undefined,
      );
      const fitLayer =
        event.data.fitLayer ??
        event.data.runtimeIdentity.layerMap[
          Math.floor(event.data.runtimeIdentity.layerMap.length / 2)
        ];
      if (!event.data.runtimeIdentity.layerMap.includes(fitLayer)) {
        throw new Error(
          `requested fit layer ${fitLayer} is outside the compiled layer map`,
        );
      }
      const progress: RuntimeProgressEvent[] = [];
      const fitting = new BrowserManifoldFitting(artifacts, "0.0.0");
      const fitted = (await fitting.request({
        request: {
          service: "manifolds",
          method: "fit",
          args: [
            harnessManifold.namespace,
            harnessManifold.name,
            {
              layers: [fitLayer],
              fit_mode: "spectral",
              hyperparams: {
                max_dim: 1,
                min_dim: 1,
                k_nn: 2,
                max_subspace_dim: 2,
                smoothing: "auto",
              },
              force: true,
            },
          ],
        },
        loadRequest: request,
        runtime: runtime!,
        onProgress(event) {
          progress.push(event);
        },
        signal: new AbortController().signal,
      })) as { fitted_models?: unknown };
      if (
        !Array.isArray(fitted.fitted_models) ||
        fitted.fitted_models.length === 0
      ) {
        throw new Error("browser curved fit did not persist a fitted artifact");
      }
      const stages = [
        ...new Set(
          progress.flatMap((event) => {
            const data = event.data as { stage?: unknown };
            return typeof data?.stage === "string" ? [data.stage] : [];
          }),
        ),
      ];
      for (const stage of [
        "capturing",
        "committing",
        "pooling",
        "fitting",
        "topology",
        "surface",
      ]) {
        if (!stages.includes(stage))
          throw new Error(`browser curved fit omitted ${stage} progress`);
      }
      const archives = await installedArchives(artifacts);
      compiler = await BrowserFeasibilityCorePackCompiler.load(request, {
        installedManifoldArchives: archives,
      });
      instruments = new BrowserInstrumentRuntime(
        compiler,
        runtime!,
        contextBinding,
      );
      const curve = await exactGeometry(
        instruments,
        `${harnessManifold.namespace}/${harnessManifold.name}`,
        false,
        "browser_e2e_curve",
      );
      curveSelector = curve.selector;
      curveLabel = curve.label;
      if (!Object.keys(curve.geometry.layers).includes(String(fitLayer))) {
        throw new Error(
          "installed curved manifold did not retain the selected physical fit layer",
        );
      }
      return {
        selector: curveSelector,
        fitLayer,
        captureRows: 10,
        progressStages: stages,
        fittedLayers: Object.keys(curve.geometry.layers).length,
        installedArchives: archives.length,
        opfsSpoolingObserved:
          stages.includes("committing") && stages.includes("pooling"),
      };
    });

    await check("curved_generation", async () => {
      currentCheck = "curved_generation";
      await instruments!.request(
        {
          service: "instruments",
          method: "setLive",
          args: ["geometry", { enabled: true }],
        },
        () => null,
      );
      const expression = `0.05 ${curveSelector}%${curveLabel}@generated`;
      const program = requireStructured(compiler!.compile(expression));
      if (program.activeRole !== "distant_voice") {
        throw new Error(
          `curved manifold selected the wrong nearest-node role ${String(program.activeRole)}`,
        );
      }
      if (!runtime!.runtimeCapabilities().namedRoles) {
        throw new Error("the physical model chat template does not support named roles");
      }
      if (!program.curveRank.some((rank) => rank > 0)) {
        throw new Error(
          "curved manifold did not lower into a physical GPU curve program",
        );
      }
      let measurements: any = null;
      const engine = runtime!.requireEngine();
      const originalReadGeometry =
        engine.readDrowseGeometryMeasurements.bind(engine);
      const geometryStride = program.profile.geometryOutputStride;
      const activeGeometrySlots = Array.from(
        program.geometryActive ?? [],
      ).flatMap((active, slot) => (active === 1 ? [slot] : []));
      const geometryReads: number[] = [];
      const activeGeometryRows: number[][][] = [];
      engine.readDrowseGeometryMeasurements = async (...args) => {
        const values = await originalReadGeometry(...args);
        geometryReads.push(values?.length ?? 0);
        activeGeometryRows.push(
          activeGeometrySlots.map((slot) =>
            values === undefined
              ? []
              : Array.from(
                  values.subarray(
                    slot * geometryStride,
                    (slot + 1) * geometryStride,
                  ),
                ),
          ),
        );
        return values;
      };
      let result;
      try {
        result = await runtime!.streamGeneration(
          {
            input: {
              kind: "chat",
              messages: [{
                role: "user",
                content: "The feeling slowly changes as",
              }],
            },
            sampling: deterministicSampling(event.data.maxTokens),
            thinking: false,
            generationSeat: "assistant",
            generationRoleName: program.activeRole,
            replay: { forcedPrefixTokenIds: ordinaryRawTokens.slice(0, 2) },
            steeringExpression: expression,
            hookProgram: program,
            onRawToken(token) {
              measurements = token.measurements ?? measurements;
            },
          },
          () => undefined,
        );
      } finally {
        engine.readDrowseGeometryMeasurements = originalReadGeometry;
      }
      const finiteGeometryRows = measurements?.instruments?.geometry?.readings
        ?.browser_e2e_curve
        ? 1
        : 0;
      const reading = measurements?.scores?.browser_e2e_curve;
      if (
        result.tokens < 1 ||
        typeof reading !== "number" ||
        !Number.isFinite(reading)
      ) {
        throw new Error(
          "curved physical GPU generation returned no finite geometry reading " +
            `(tokens=${result.tokens}, reads=${geometryReads.join(",") || "none"}, ` +
            `active=${activeGeometrySlots.length}, finite=${finiteGeometryRows}, ` +
            `row=${JSON.stringify(activeGeometryRows.at(-1)?.[0] ?? [])})`,
        );
      }
      return {
        tokens: result.tokens,
        finishReason: result.finishReason,
        reading,
        forcedPrefixTokenIds: ordinaryRawTokens.slice(0, 2),
        activeCurveSlots: program.curveRank.filter((rank) => rank > 0).length,
        activeGeometrySlots: activeGeometrySlots.length,
        finiteGeometryRows,
        geometryReadElements: geometryReads,
        activeRole: program.activeRole,
        namedRoleSupported: true,
      };
    });

    await check("stop", async () => {
      currentCheck = "stop";
      const engine = runtime!.requireEngine();
      const originalInterrupt = engine.interruptGenerate.bind(engine);
      let interruptCalls = 0;
      engine.interruptGenerate = async () => {
        interruptCalls += 1;
        return originalInterrupt();
      };
      let rawTokens = 0;
      let result;
      try {
        result = await runtime!.streamGeneration(
          {
            input: {
              kind: "raw",
              prompt: "Count upward without stopping: 1, 2, 3,",
            },
            sampling: deterministicSampling(event.data.stopMaxTokens),
            thinking: false,
            steeringExpression: null,
            hookProgram: null,
            async onRawToken() {
              rawTokens += 1;
              if (rawTokens === 1) await runtime!.stop();
            },
          },
          () => undefined,
        );
      } finally {
        engine.interruptGenerate = originalInterrupt;
      }
      if (interruptCalls !== 1 || rawTokens !== 1 || result.tokens !== 1) {
        throw new Error(
          "generation stop did not interrupt an in-flight physical decode",
        );
      }
      return {
        interruptCalls,
        tokensBeforeSettled: result.tokens,
        requestedMaxTokens: event.data.stopMaxTokens,
        finishReason: result.finishReason,
      };
    });

    await check("unload_reload", async () => {
      currentCheck = "unload_reload";
      await runtime!.unload();
      let unloadedRejected = false;
      try {
        runtime!.requireEngine();
      } catch {
        unloadedRejected = true;
      }
      if (!unloadedRejected)
        throw new Error("runtime remained usable after unload");
      request = await loadRequest();
      await runtime!.load(request);
      const restoredDictionary = compiler!.saeGpuDictionary();
      const restoredJlensDictionary = compiler!.jlensGpuDictionary();
      if (restoredDictionary === null) {
        throw new Error(
          "the precomputed SAE dictionary disappeared before runtime reload",
        );
      }
      if (restoredJlensDictionary === null) {
        throw new Error(
          "the precomputed J-lens dictionary disappeared before runtime reload",
        );
      }
      await runtime!.setSaeDictionary(restoredDictionary);
      await runtime!.setJlensDictionary(restoredJlensDictionary);
      await instruments!.request(
        {
          service: "probes",
          method: "attach",
          args: [
            {
              selector: `jlens/${event.data.jlensWord}`,
              name: "browser_e2e_jlens_reload",
            },
          ],
        },
        () => null,
      );
      await instruments!.request(
        {
          service: "probes",
          method: "attach",
          args: [
            {
              selector: `sae/${event.data.saeFeature}`,
              name: "browser_e2e_sae_reload",
            },
          ],
        },
        () => null,
      );
      for (const family of ["lens", "sae"] as const) {
        await instruments!.request(
          {
            service: "instruments",
            method: "setLive",
            args: [family, { enabled: true }],
          },
          () => null,
        );
      }
      const reloadExpression =
        `0.01 jlens/${event.data.jlensWord}@generated` +
        ` + 0.01 sae/${event.data.saeFeature}@generated`;
      await compiler!.prepareJlensSelectors(reloadExpression, runtime!);
      const reloadProgram = requireStructured(
        compiler!.compile(reloadExpression),
      );
      let reloadMeasurements: any = null;
      const result = await runtime!.streamGeneration(
        {
          input: { kind: "raw", prompt: "Reloaded model:" },
          sampling: deterministicSampling(1),
          thinking: false,
          steeringExpression: reloadExpression,
          hookProgram: reloadProgram,
          onRawToken(token) {
            reloadMeasurements = token.measurements ?? reloadMeasurements;
          },
        },
        () => undefined,
      );
      if (result.tokens !== 1)
        throw new Error("reloaded model did not complete a one-token decode");
      const reloadLensReading =
        reloadMeasurements?.instruments?.lens?.readings
          ?.browser_e2e_jlens_reload?.value;
      const reloadSaeReading =
        reloadMeasurements?.instruments?.sae?.readings?.browser_e2e_sae_reload
          ?.value;
      for (const [name, value] of Object.entries({
        reloadLensReading,
        reloadSaeReading,
      })) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new Error(`${name} was not restored after runtime reload`);
        }
      }
      const exactReadouts = requireExactPrecomputedReadouts(
        reloadMeasurements,
        compiler!.instrumentDescriptor(),
        "runtime reload",
      );
      assertNoDeviceLoss(deviceLosses);
      return {
        unloadInvalidatedEngine: unloadedRejected,
        reloadTokens: result.tokens,
        restoredSaeDictionaryFeatureCount: restoredDictionary.featureCount,
        reloadLensReading,
        reloadSaeReading,
        ...exactReadouts,
        deviceLosses: deviceLosses.length,
      };
    });

    await check("offline_reuse", async () => {
      currentCheck = "offline_reuse";
      await runtime!.unload();
      runtime = null;
      await artifacts!.close();
      artifacts = null;
      const cutoff = await fetch("/offline-arm", { method: "POST" });
      if (!cutoff.ok)
        throw new Error("failed to arm offline E2E network cutoff");
      await cutoff.arrayBuffer();
      offline = true;
      const [offlineModel, offlineCore, offlineJlens, offlineSae] =
        await Promise.all([
          reopen("model", event.data.modelFiles),
          reopen("core", event.data.coreFiles),
          reopen("jlens", event.data.jlensFiles),
          reopen("sae", event.data.saeFiles),
        ]);
      artifacts = new BrowserDrowseArchiveService(undefined, undefined, "0.0.0");
      const archives = await installedArchives(artifacts);
      const offlineOptional = [
        optionalPack("jlens", event.data.modelId, fingerprint, offlineJlens),
        optionalPack("sae", event.data.modelId, fingerprint, offlineSae),
      ];
      const offlineRequest: BrowserModelLoadRequest = {
        ...(await loadRequest()),
        artifacts: [...offlineModel, ...offlineCore],
        requiredCorePack: {
          ...requiredCorePack,
          files: offlineCore.map(({ manifest }) => manifest),
        },
        optionalPacks: offlineOptional,
      };
      runtime = new DrowseWebLlmRuntime(webllm);
      await runtime.load(offlineRequest);
      const offlineCompiler = await BrowserFeasibilityCorePackCompiler.load(
        offlineRequest,
        {
          installedManifoldArchives: archives,
        },
      );
      const offlineSaeDictionary = offlineCompiler.saeGpuDictionary();
      const offlineJlensDictionary = offlineCompiler.jlensGpuDictionary();
      if (offlineSaeDictionary === null) {
        throw new Error(
          "offline OPFS reuse lost the precomputed SAE dictionary",
        );
      }
      if (offlineJlensDictionary === null) {
        throw new Error(
          "offline OPFS reuse lost the precomputed J-lens dictionary",
        );
      }
      await runtime.setSaeDictionary(offlineSaeDictionary);
      await runtime.setJlensDictionary(offlineJlensDictionary);
      const offlineInstruments = new BrowserInstrumentRuntime(
        offlineCompiler,
        runtime,
        contextBinding,
      );
      const curve = await exactGeometry(
        offlineInstruments,
        curveSelector,
        false,
        "browser_e2e_curve_offline",
      );
      await offlineInstruments.request(
        {
          service: "probes",
          method: "attach",
          args: [
            {
              selector: `jlens/${event.data.jlensWord}`,
              name: "browser_e2e_jlens_offline",
            },
          ],
        },
        () => null,
      );
      await offlineInstruments.request(
        {
          service: "probes",
          method: "attach",
          args: [
            {
              selector: `sae/${event.data.saeFeature}`,
              name: "browser_e2e_sae_offline",
            },
          ],
        },
        () => null,
      );
      for (const family of ["geometry", "lens", "sae"] as const) {
        await offlineInstruments.request(
          {
            service: "instruments",
            method: "setLive",
            args: [family, { enabled: true }],
          },
          () => null,
        );
      }
      const expression =
        `0.03 ${curve.selector}%${curve.label}@generated` +
        ` + 0.01 jlens/${event.data.jlensWord}@generated` +
        ` + 0.01 sae/${event.data.saeFeature}@generated`;
      await offlineCompiler.prepareJlensSelectors(expression, runtime);
      const program = requireStructured(offlineCompiler.compile(expression));
      if (
        !program.curveRank.some((rank) => rank > 0) ||
        !program.geometryActive?.some((active) => active === 1)
      ) {
        throw new Error(
          "offline curved manifold did not lower into physical GPU programs",
        );
      }
      let offlineMeasurements: any = null;
      const result = await runtime.streamGeneration(
        {
          input: { kind: "raw", prompt: "Offline model:" },
          sampling: deterministicSampling(1),
          thinking: false,
          steeringExpression: expression,
          hookProgram: program,
          onRawToken(token) {
            offlineMeasurements = token.measurements ?? offlineMeasurements;
          },
        },
        () => undefined,
      );
      const offlineReading =
        offlineMeasurements?.scores?.browser_e2e_curve_offline;
      const offlineLensReading =
        offlineMeasurements?.instruments?.lens?.readings
          ?.browser_e2e_jlens_offline?.value;
      const offlineSaeReading =
        offlineMeasurements?.instruments?.sae?.readings?.browser_e2e_sae_offline
          ?.value;
      if (
        result.tokens !== 1 ||
        typeof offlineReading !== "number" ||
        !Number.isFinite(offlineReading) ||
        typeof offlineLensReading !== "number" ||
        !Number.isFinite(offlineLensReading) ||
        typeof offlineSaeReading !== "number" ||
        !Number.isFinite(offlineSaeReading) ||
        offlineNetworkAttempts.length !== 0
      ) {
        throw new Error(
          "offline OPFS reuse lost a physical geometry/J-lens/SAE readout or attempted network access: " +
            JSON.stringify(offlineNetworkAttempts),
        );
      }
      const exactReadouts = requireExactPrecomputedReadouts(
        offlineMeasurements,
        offlineCompiler.instrumentDescriptor(),
        "offline reuse",
      );
      await runtime.unload();
      runtime = null;
      await artifacts.request(
        {
          service: "manifolds",
          method: "delete",
          args: [harnessManifold.namespace, harnessManifold.name],
        },
        () => undefined,
      );
      await artifacts.close();
      artifacts = null;
      return {
        reopenedModelFiles: offlineModel.length,
        reopenedCoreFiles: offlineCore.length,
        reopenedJlensFiles: offlineJlens.length,
        reopenedSaeFiles: offlineSae.length,
        persistedManifoldArchives: archives.length,
        offlineTokens: result.tokens,
        offlineGeometryReading: offlineReading,
        offlineLensReading,
        offlineSaeReading,
        restoredSaeDictionaryFeatureCount: offlineSaeDictionary.featureCount,
        ...exactReadouts,
        networkRequests: offlineNetworkAttempts.length,
      };
    });
    assertNoDeviceLoss(deviceLosses);
    if (externalNetworkAttempts.length !== 0) {
      throw new Error(
        `verified local runtime attempted external fetches: ${JSON.stringify(externalNetworkAttempts)}`,
      );
    }
    self.postMessage({
      type: "done",
      artifactVerification,
      environment: {
        modelId: event.data.modelId,
        contextTokens: event.data.contextTokens,
        userAgent: navigator.userAgent,
        language: navigator.language,
        runtimeIdentitySha256: fingerprint,
        adapterFallback: capabilities!.webGpu.fallback,
        adapterInfo: capabilities!.webGpu.adapterInfo,
        deviceMemoryGiB: capabilities!.signals.deviceMemoryGiB,
        logicalCpuCount: capabilities!.signals.logicalCpuCount,
        deviceLosses: deviceLosses.map((failure) => ({ ...failure })),
      },
    });
    setTimeout(() => self.close(), 0);
  } catch (error) {
    try {
      await runtime?.unload();
    } catch {}
    try {
      await cleanupArtifacts(artifacts, harnessManifold);
    } catch {}
    self.postMessage({
      type: "error",
      check: currentCheck,
      message:
        error instanceof Error ? (error.stack ?? error.message) : String(error),
    });
  }
};

async function check(
  name: CheckName,
  run: () => Promise<Record<string, unknown>>,
): Promise<void> {
  self.postMessage({ type: "check_started", check: name });
  const measurements = await run();
  if (Object.keys(measurements).length === 0)
    throw new Error(`check ${name} produced no measurements`);
  self.postMessage({ type: "check_passed", check: name, measurements });
}

async function stage(
  set: string,
  files: FileEntry[],
): Promise<BrowserModelArtifact[]> {
  const root = await navigator.storage.getDirectory();
  const harness = await root.getDirectoryHandle("drowse-real-browser-e2e-v1", {
    create: true,
  });
  const directory = await harness.getDirectoryHandle(set, { create: true });
  const revision = "a".repeat(40);
  const artifacts: BrowserModelArtifact[] = [];
  for (const entry of files) {
    const response = await fetch(localArtifactUrl(set, entry.path), {
      cache: "no-store",
    });
    if (!response.ok)
      throw new Error(`artifact fetch failed for ${set}/${entry.path}`);
    if (response.body === null)
      throw new Error(
        `artifact fetch returned no body for ${set}/${entry.path}`,
      );
    const handle = await fileHandle(directory, entry.path);
    const writable = await handle.createWritable();
    await response.body.pipeTo(writable);
    const file = await handle.getFile();
    await verifyPersistedFile(file, entry, set);
    artifacts.push({
      manifest: localManifest(entry, revision, set),
      file,
    });
  }
  return artifacts;
}

function localArtifactUrl(set: string, path: string): string {
  const url = new URL("/artifact", globalThis.location.origin);
  url.searchParams.set("set", set);
  url.searchParams.set("path", path);
  return url.href;
}

function artifactSetReceipt(set: string, files: FileEntry[]) {
  const tuples = files.map((file) => [set, file.path, file.bytes, file.sha256]);
  return {
    fileCount: files.length,
    totalBytes: files.reduce((sum, file) => sum + file.bytes, 0),
    entriesSha256: bytesToHex(
      sha256(new TextEncoder().encode(JSON.stringify(tuples))),
    ),
  };
}

async function reopen(
  set: string,
  files: FileEntry[],
): Promise<BrowserModelArtifact[]> {
  const root = await navigator.storage.getDirectory();
  const harness = await root.getDirectoryHandle("drowse-real-browser-e2e-v1");
  const directory = await harness.getDirectoryHandle(set);
  const revision = "a".repeat(40);
  const artifacts: BrowserModelArtifact[] = [];
  for (const entry of files) {
    const handle = await fileHandle(directory, entry.path, false);
    const file = await handle.getFile();
    await verifyPersistedFile(file, entry, set);
    artifacts.push({ manifest: localManifest(entry, revision, set), file });
  }
  return artifacts;
}

async function verifyPersistedFile(
  file: File,
  entry: FileEntry,
  set: string,
): Promise<void> {
  if (file.size !== entry.bytes) {
    throw new Error(`persisted artifact ${set}/${entry.path} changed size`);
  }
  const hasher = sha256.create();
  const reader = file.stream().getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    hasher.update(value);
  }
  const actual = bytesToHex(hasher.digest());
  if (actual !== entry.sha256) {
    throw new Error(`persisted artifact ${set}/${entry.path} changed SHA-256`);
  }
}

function localManifest(
  entry: FileEntry,
  revision: string,
  set: string,
): CatalogFile {
  return {
    ...entry,
    revision,
    url: `https://huggingface.co/polythetic/local-${set}/resolve/${revision}/${entry.path}`,
  };
}

async function fileHandle(
  root: FileSystemDirectoryHandle,
  path: string,
  create = true,
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
    directory = await directory.getDirectoryHandle(part, { create });
  }
  return directory.getFileHandle(parts.at(-1)!, { create });
}

function optionalPack(
  kind: "jlens" | "sae",
  modelId: string,
  fingerprint: string,
  artifacts: BrowserModelArtifact[],
): BrowserInstrumentPackArtifacts {
  return {
    pack: {
      id: `${modelId}-${kind}-e2e`,
      kind,
      displayName: `Local E2E ${kind} pack`,
      runtimeIdentitySha256: fingerprint,
      files: artifacts.map(({ manifest }) => manifest),
    },
    artifacts,
  } as BrowserInstrumentPackArtifacts;
}

async function verifyInstrumentRejections(
  request: BrowserModelLoadRequest,
): Promise<{
  incompatibleBindingRejected: true;
  fp32TensorValidationPassed: true;
}> {
  const mismatchedPacks = request.optionalPacks.map((entry, index) =>
    index === 0
      ? {
          ...entry,
          pack: { ...entry.pack, runtimeIdentitySha256: "f".repeat(64) },
        }
      : entry,
  );
  await requireCorePackRejection(
    { ...request, optionalPacks: mismatchedPacks },
    "INSTRUMENT_PACK_RUNTIME_MISMATCH",
  );

  const saeIndex = request.optionalPacks.findIndex((entry) => entry.pack.kind === "sae");
  if (saeIndex < 0) throw new Error("physical instrument check has no SAE pack");
  const original = request.optionalPacks[saeIndex];
  const manifestIndex = original.artifacts.findIndex((artifact) =>
    artifact.manifest.path.endsWith("/manifest.json"),
  );
  const tensorIndex = original.artifacts.findIndex((artifact) =>
    artifact.manifest.path.endsWith(".safetensors"),
  );
  if (manifestIndex < 0 || tensorIndex < 0)
    throw new Error("physical SAE pack has no manifest or tensor payload");
  const manifestArtifact = original.artifacts[manifestIndex];
  const tensorArtifact = original.artifacts[tensorIndex];
  const invalidTensorBytes = new Uint8Array(await tensorArtifact.file.arrayBuffer());
  const headerLength = Number(
    new DataView(
      invalidTensorBytes.buffer,
      invalidTensorBytes.byteOffset,
      8,
    ).getBigUint64(0, true),
  );
  const headerStart = 8;
  const headerEnd = headerStart + headerLength;
  const header = new TextDecoder().decode(
    invalidTensorBytes.subarray(headerStart, headerEnd),
  );
  const invalidHeader = header.replaceAll('"F32"', '"F16"');
  if (invalidHeader === header)
    throw new Error("physical SAE tensor has no fp32 header to reject");
  invalidTensorBytes.set(new TextEncoder().encode(invalidHeader), headerStart);
  const invalidTensorSha256 = bytesToHex(sha256(invalidTensorBytes));
  const invalidTensorCatalogFile = {
    ...tensorArtifact.manifest,
    bytes: invalidTensorBytes.byteLength,
    sha256: invalidTensorSha256,
  };
  const invalidManifest = JSON.parse(await manifestArtifact.file.text());
  invalidManifest.tensor_sha256 = invalidTensorSha256;
  const invalidBytes = new TextEncoder().encode(JSON.stringify(invalidManifest));
  const invalidSha256 = bytesToHex(sha256(invalidBytes));
  const invalidCatalogFile = {
    ...manifestArtifact.manifest,
    bytes: invalidBytes.byteLength,
    sha256: invalidSha256,
  };
  const invalidArtifacts = original.artifacts.map((artifact, index) =>
    index === manifestIndex
      ? {
          manifest: invalidCatalogFile,
          file: new File([invalidBytes], "manifest.json", {
            type: "application/json",
          }),
        }
      : index === tensorIndex
        ? {
            manifest: invalidTensorCatalogFile,
            file: new File([invalidTensorBytes], "invalid-fp16.safetensors", {
              type: "application/octet-stream",
            }),
          }
        : artifact,
  );
  const invalidPacks = request.optionalPacks.map((entry, index) =>
    index === saeIndex
      ? {
          ...entry,
          pack: {
            ...entry.pack,
            files: entry.pack.files.map((file) =>
              file.path === invalidCatalogFile.path
                ? invalidCatalogFile
                : file.path === invalidTensorCatalogFile.path
                  ? invalidTensorCatalogFile
                  : file,
            ),
          },
          artifacts: invalidArtifacts,
        }
      : entry,
  );
  await requireCorePackRejection(
    { ...request, optionalPacks: invalidPacks },
    "SAFETENSORS_INVALID",
  );
  return {
    incompatibleBindingRejected: true,
    fp32TensorValidationPassed: true,
  };
}

async function requireCorePackRejection(
  request: BrowserModelLoadRequest,
  expectedCode: string,
): Promise<void> {
  try {
    await BrowserFeasibilityCorePackCompiler.load(request);
  } catch (error) {
    if ((error as { code?: unknown })?.code === expectedCode) return;
    throw error;
  }
  throw new Error(`physical instrument check did not reject ${expectedCode}`);
}

function requireExactPrecomputedReadouts(
  measurements: any,
  descriptor: ReturnType<
    BrowserFeasibilityCorePackCompiler["instrumentDescriptor"]
  >,
  lifecycle: string,
): Record<string, unknown> {
  if (descriptor.jlens === null || descriptor.sae === null) {
    throw new Error(
      `${lifecycle} lost its verified precomputed instrument descriptors`,
    );
  }
  const lensReadout = measurements?.instruments?.lens?.readout;
  const aggregate = lensReadout?.aggregate;
  const layers = lensReadout?.layers;
  if (
    !Array.isArray(aggregate) ||
    aggregate.length !== 8 ||
    aggregate.some(
      (row) =>
        typeof row?.token !== "string" ||
        typeof row?.strength !== "number" ||
        !Number.isFinite(row.strength) ||
        row.strength < 0 ||
        row.strength > 1 ||
        typeof row?.com !== "number" ||
        !Number.isFinite(row.com) ||
        typeof row?.spread !== "number" ||
        !Number.isFinite(row.spread) ||
        row.spread < 0,
    ) ||
    !Array.isArray(layers) ||
    layers.length !== descriptor.jlens.layers.length ||
    layers.some(
      (row) =>
        !descriptor.jlens!.layers.includes(row?.layer) ||
        !Array.isArray(row?.tokens) ||
        row.tokens.length !== 8 ||
        row.tokens.some(
          (token: any) =>
            !Number.isSafeInteger(token?.id) ||
            typeof token?.token !== "string" ||
            typeof token?.logprob !== "number" ||
            !Number.isFinite(token.logprob),
        ),
    )
  ) {
    throw new Error(
      `${lifecycle} returned no valid exact full-vocabulary J-lens readout`,
    );
  }
  const features = measurements?.instruments?.sae?.readout?.features;
  if (
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
    throw new Error(
      `${lifecycle} returned no valid exact full-dictionary SAE readout`,
    );
  }
  return {
    exactJlensAggregateTokens: aggregate.length,
    exactJlensLayerRows: layers.length,
    exactJlensPerLayerTokens: layers.reduce(
      (sum: number, row: { tokens: unknown[] }) => sum + row.tokens.length,
      0,
    ),
    exactSaeDiscoveredFeatures: features.length,
    exactSaeDictionaryFeatureCount: descriptor.sae.features,
    exactSaeLayer: descriptor.sae.layer,
  };
}

async function firstGeometry(
  instruments: BrowserInstrumentRuntime,
  affine: boolean,
  probeName: string,
): Promise<{
  selector: string;
  label: string;
  geometry: { layers: Record<string, unknown> };
}> {
  const listed = (await instruments.request(
    { service: "manifolds", method: "list", args: [] },
    () => null,
  )) as {
    manifolds: Array<{
      namespace: string;
      name: string;
      node_labels: string[];
    }>;
  };
  const failures: string[] = [];
  for (const manifold of listed.manifolds) {
    const selector = `${manifold.namespace}/${manifold.name}`;
    try {
      return await exactGeometry(instruments, selector, affine, probeName);
    } catch (error) {
      failures.push(
        `${selector}: ${error instanceof Error ? error.message : String(error)}`,
      );
      await instruments.request(
        { service: "probes", method: "detach", args: [probeName] },
        () => null,
      );
    }
  }
  throw new Error(
    `no ${affine ? "affine" : "curved"} physical geometry was available: ${failures.join("; ")}`,
  );
}

async function exactGeometry(
  instruments: BrowserInstrumentRuntime,
  selector: string,
  affine: boolean,
  probeName: string,
): Promise<{
  selector: string;
  label: string;
  geometry: { layers: Record<string, unknown> };
}> {
  const parts = selector.split("/");
  if (parts.length !== 2)
    throw new Error(`geometry selector must be canonical: ${selector}`);
  const detail = (await instruments.request(
    { service: "manifolds", method: "get", args: parts },
    () => null,
  )) as { node_labels: string[] };
  if (!Array.isArray(detail.node_labels) || detail.node_labels.length === 0) {
    throw new Error(`${selector} has no addressable node label`);
  }
  await instruments.request(
    {
      service: "probes",
      method: "attach",
      args: [{ selector, name: probeName }],
    },
    () => null,
  );
  const geometry = (await instruments.request(
    { service: "probes", method: "geometry", args: [probeName] },
    () => null,
  )) as { is_affine?: boolean; layers?: Record<string, unknown> };
  if (
    geometry.is_affine !== affine ||
    !geometry.layers ||
    Object.keys(geometry.layers).length === 0
  ) {
    throw new Error(
      `${selector} is not a fitted ${affine ? "affine" : "curved"} geometry`,
    );
  }
  return {
    selector,
    label: detail.node_labels[0],
    geometry: { layers: geometry.layers },
  };
}

async function installedArchives(
  service: BrowserDrowseArchiveService,
): Promise<Blob[]> {
  const listed = (await service.request(
    { service: "manifolds", method: "drowseArchiveList", args: [] },
    () => undefined,
  )) as { packs?: Array<{ id?: unknown }> };
  if (!Array.isArray(listed.packs))
    throw new Error("local manifold repository returned no pack index");
  const archives: Blob[] = [];
  for (const pack of listed.packs) {
    if (typeof pack.id !== "string")
      throw new Error("local manifold repository returned an invalid pack ID");
    const archive = await service.request(
      { service: "manifolds", method: "drowseArchiveExport", args: [pack.id] },
      () => undefined,
    );
    if (!(archive instanceof Blob))
      throw new Error(`local manifold archive ${pack.id} is unreadable`);
    archives.push(archive);
  }
  return archives;
}

async function cleanupArtifacts(
  service: BrowserDrowseArchiveService | null,
  manifold: { namespace: string; name: string },
): Promise<void> {
  if (service === null) return;
  await service.request(
    {
      service: "manifolds",
      method: "delete",
      args: [manifold.namespace, manifold.name],
    },
    () => undefined,
  );
  await service.close();
}

function requireStructured(program: unknown): StructuredHookProgramBuffers {
  if (!program || typeof program !== "object" || !("affineActive" in program)) {
    throw new Error(
      "browser expression did not lower into the structured GPU program",
    );
  }
  return program as StructuredHookProgramBuffers;
}

function deterministicSampling(maxTokens: number) {
  return {
    max_tokens: maxTokens,
    seed: 1,
    temperature: 0,
    top_p: 1,
    top_k: 0,
    presence_penalty: 0,
    frequency_penalty: 0,
  };
}

function assertNoDeviceLoss(
  losses: Array<{ code: string; message: string; confirmedOom: boolean }>,
): void {
  if (losses.length > 0)
    throw new Error(
      `physical WebGPU device was lost: ${JSON.stringify(losses)}`,
    );
}
