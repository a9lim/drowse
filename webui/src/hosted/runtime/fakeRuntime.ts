import type {
  CatalogInstrumentPack,
  InstalledInstrumentPackActivationRequest,
  InstalledInstrumentPackActivationResult,
  RuntimeClient,
  RuntimeEventChannel,
  RuntimeEventChannelState,
  RuntimeInstrumentsService,
  RuntimeManifoldsService,
  RuntimeProfilesService,
  RuntimeProbesService,
  RuntimeSessionsService,
  RuntimeTemplatesService,
  RuntimeTreeService,
} from "../../lib/runtime/contracts";
import type {
  InstrumentFamily,
  InstrumentFamilyBlock,
  InstrumentSourceJSON,
  LoomNodeJSON,
  LoomTreeJSON,
  MeasurementsEnvelopeJSON,
  PreparationStatusJSON,
  ProbeInfo,
  ProbeReadingJSON,
  ProbeRequest,
  ScalarReadingJSON,
  SessionInfo,
  WSClientMessage,
  WSGenerateRequest,
  WSServerMessage,
  WSSubmitRequest,
} from "../../lib/types";
import { DROWSE_HOOK_ABI, type RankOneHookProgramBuffers } from "./rankOneHookProgram";
import { BrowserLoomRuntime, type BrowserGenerationPort } from "./browserLoom";
import type {
  WebLlmGeneratedToken,
  WebLlmGenerationPlan,
  WebLlmGenerationResult,
} from "./webLlmGeneration";

type FakeInstrumentPack = Pick<CatalogInstrumentPack, "id" | "kind" | "displayName">;

export interface FakeRuntimeOptions {
  modelId?: string;
  response?: string;
  session?: SessionInfo;
  sessionOverrides?: Partial<SessionInfo>;
  tree?: LoomTreeJSON;
  rootId?: string;
  tokenDelayMs?: number;
  instrumentPacks?: readonly FakeInstrumentPack[];
  contextBindingSha256?: string;
}

export class DeterministicFakeRuntime implements RuntimeClient {
  readonly mode = "fake" as const;
  readonly sessions: RuntimeSessionsService;
  readonly profiles: RuntimeProfilesService;
  readonly probes: RuntimeProbesService;
  readonly manifolds: RuntimeManifoldsService;
  readonly templates: RuntimeTemplatesService;
  readonly tree: RuntimeTreeService;
  readonly instruments: RuntimeInstrumentsService;
  readonly events: RuntimeEventChannel;
  private session: SessionInfo;
  private loom: LoomTreeJSON;
  private readonly generation: DeterministicGenerationPort;
  private readonly sourcesByFamily: Record<"lens" | "sae", InstrumentSourceJSON[]> = {
    lens: [],
    sae: [],
  };
  private readonly preparations: Record<"lens" | "sae", PreparationStatusJSON> = {
    lens: idlePreparation(),
    sae: idlePreparation(),
  };
  private probeRows: ProbeInfo[] = [];
  private idCounter = 0;
  private readonly contextBindingSha256: string;

  constructor(options: FakeRuntimeOptions = {}) {
    this.session = clone(options.session ?? {
      ...fakeSession(options.modelId ?? "fixture/drowse-tiny"),
      ...options.sessionOverrides,
    });
    this.loom = clone(options.tree ?? fakeTree(this.session.model_id, options.rootId));
    this.idCounter = nextFixtureIdCounter(this.loom);
    this.generation = new DeterministicGenerationPort(
      options.response ?? "This is a deterministic local Drowse runtime fixture.",
      options.tokenDelayMs ?? 0,
      (text, index, scope) => this.fixtureMeasurements(text, index, scope),
    );
    this.contextBindingSha256 = options.contextBindingSha256 ?? "fixture-context-binding";
    this.configureInstrumentPacks(options.instrumentPacks ?? []);

    this.sessions = {
      list: async () => ({ sessions: [clone(this.session)] }),
      get: async () => clone(this.session),
      patch: async (body: Partial<SessionInfo["config"]>) => {
        return this.loomRequest("sessions", "patch", [body]) as Promise<SessionInfo>;
      },
      validateSteering: (
        expression: string,
        _id?: string,
        options?: import("../../lib/runtime/contracts").SteeringValidationOptions,
      ) => this.loomRequest(
          "sessions",
          "validateSteering",
          options === undefined ? [expression] : [expression, undefined, options],
        ) as ReturnType<RuntimeSessionsService["validateSteering"]>,
    } satisfies RuntimeSessionsService;
    this.profiles = {
      list: async () => ({ profiles: [] }),
      get: async (name: string) => {
        throw fakeError(
          "PROFILE_NOT_FOUND",
          `Profile ${name} is not available in the deterministic fixture core pack`,
        );
      },
      correlation: async () => ({ names: [], matrix: {}, layers_shared: {} }),
      pairwise: async (a: string, b: string) => ({
        a,
        b,
        metric: "whitened cosine",
        layers_a: [],
        layers_b: [],
        matrix: [],
        model: this.session.model_id,
      }),
      extract: async () => {
        throw fakeFittingUnavailable();
      },
    } satisfies RuntimeProfilesService;
    this.probes = {
      list: async () => ({ probes: clone(this.probeRows) }),
      attach: async (request: ProbeRequest) => this.attachProbe(request),
      detach: async (name: string) => {
        this.probeRows = this.probeRows.filter((probe) => probe.name !== name);
        this.syncProbeState();
      },
      geometry: async (name: string) => this.probeGeometry(name),
    } satisfies RuntimeProbesService;
    this.manifolds = {
      list: async () => ({ manifolds: [] }),
      inspectSurface: async () => {
        throw fakeArtifactRoutingRequired("manifolds.inspectSurface");
      },
      get: async (namespace: string, name: string) => {
        throw fakeError(
          "CORE_PACK_SELECTOR_NOT_FOUND",
          `Core-pack manifold ${namespace}/${name} is not available in the fixture model runtime`,
        );
      },
      create: async () => {
        throw fakeArtifactRoutingRequired("manifolds.create");
      },
      createDiscover: async () => {
        throw fakeArtifactRoutingRequired("manifolds.createDiscover");
      },
      createFromTemplate: async () => {
        throw fakeArtifactRoutingRequired("manifolds.createFromTemplate");
      },
      delete: async () => {
        throw fakeArtifactRoutingRequired("manifolds.delete");
      },
      search: async () => ({ results: [] }),
      install: async () => {
        throw fakeArtifactRoutingRequired("manifolds.install");
      },
      merge: async () => {
        throw fakeArtifactRoutingRequired("manifolds.merge");
      },
      fit: async () => {
        throw fakeFittingUnavailable();
      },
      generate: async () => {
        throw fakeFittingUnavailable();
      },
      drowseArchiveList: async () => ({ packs: [] }),
      drowseArchiveInstall: async () => {
        throw fakeArtifactRoutingRequired("manifolds.drowseArchiveInstall");
      },
      drowseArchiveExport: async () => {
        throw fakeArtifactRoutingRequired("manifolds.drowseExport");
      },
      drowseArchiveDelete: async () => {
        throw fakeArtifactRoutingRequired("manifolds.drowseDelete");
      },
    } satisfies RuntimeManifoldsService;
    this.templates = {
      list: async () => ({ templates: [] }),
      get: async (namespace: string, name: string) => {
        throw fakeError(
          "TEMPLATE_NOT_FOUND",
          `Template ${namespace}/${name} is not installed in the deterministic fixture`,
        );
      },
      create: async () => {
        throw fakeArtifactRoutingRequired("templates.create");
      },
      delete: async () => {
        throw fakeArtifactRoutingRequired("templates.delete");
      },
      score: async () => {
        throw fakeFittingUnavailable();
      },
    } satisfies RuntimeTemplatesService;
    this.tree = {
      replayCapabilities: async () => ({
        jointLogprobs: { available: true, reason: null },
      }),
      get: async () => clone(this.loom),
      reset: async (...args: unknown[]) => this.loomRequest("tree", "reset", args),
      restore: async (...args: unknown[]) => this.loomRequest("tree", "restore", args),
      active: async (...args: unknown[]) => this.loomRequest("tree", "active", args),
      navigate: async (...args: unknown[]) => this.loomRequest("tree", "navigate", args),
      edit: async (...args: unknown[]) => this.loomRequest("tree", "edit", args),
      branch: async (...args: unknown[]) => this.loomRequest("tree", "branch", args),
      delete: async (...args: unknown[]) => this.loomRequest("tree", "delete", args),
      star: async (...args: unknown[]) => this.loomRequest("tree", "star", args),
      note: async (...args: unknown[]) => this.loomRequest("tree", "note", args),
      edgeLabel: async (...args: unknown[]) => this.loomRequest("tree", "edgeLabel", args),
      filter: async (...args: unknown[]) => this.loomRequest("tree", "filter", args),
      diff: async (...args: unknown[]) => this.loomRequest("tree", "diff", args),
      transcriptExport: async (...args: unknown[]) =>
        this.loomRequest("tree", "transcriptExport", args),
      transcriptLoad: async (...args: unknown[]) =>
        this.loomRequest("tree", "transcriptLoad", args),
      jointLogprobs: async (aId: string, bId: string) => this.jointLogprobs(aId, bId),
      cast: async (...args: unknown[]) => this.loomRequest("tree", "cast", args),
      castPut: async (...args: unknown[]) => this.loomRequest("tree", "castPut", args),
      castDelete: async (...args: unknown[]) => this.loomRequest("tree", "castDelete", args),
    } as RuntimeTreeService;
    this.instruments = {
      sources: async (family: InstrumentFamily) => this.instrumentSources(family),
      setLive: async (
        family: InstrumentFamily,
        body: { enabled: boolean; layers?: number[] | null },
      ) => this.setInstrumentLive(family, body),
      setLensSource: async (source: string) => this.setLensSource(source),
      activateInstalledPack: async (
        family: "lens" | "sae",
        body: InstalledInstrumentPackActivationRequest,
      ) => this.activateInstalledPack(family, body),
      startPreparation: async (
        family: InstrumentFamily,
        body: { operation: "fetch" | "fit" | "train" } & Record<string, unknown>,
      ) => this.startPreparation(family, body),
      preparationStatus: async (family: InstrumentFamily) =>
        clone(this.preparations[this.preparedFamily(family)]),
      cancelPreparation: async (family: InstrumentFamily) =>
        this.cancelPreparation(family),
      tokenReadout: async (family: InstrumentFamily) => ({
        measurements: this.tokenReadout(family),
      }),
      validateLensToken: async (word: string) => this.validateLensToken(word),
      validateSaeFeature: async (featureId: number) => this.validateSaeFeature(featureId),
      saeFeaturesMetadata: async (ids: number[]) => ({
        features: Object.fromEntries(ids.map((id) => [String(id), featureMetadata(id)])),
      }),
    } satisfies RuntimeInstrumentsService;
    this.events = new FakeEventChannel(
      (request, emit) => this.generate(request, emit),
      () => this.stopGeneration(),
    );
  }

  async dispose(): Promise<void> {
    this.events.close();
  }

  async generate(
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
  ): Promise<void> {
    const runtime = this.createLoomRuntime();
    await runtime.generate(request, async (message) => {
      this.applySnapshot(runtime.snapshot());
      await emit(message);
    });
    this.applySnapshot(runtime.snapshot());
  }

  stopGeneration(): Promise<void> {
    return this.generation.stop();
  }

  private createLoomRuntime(): BrowserLoomRuntime {
    return new BrowserLoomRuntime({
      session: this.session,
      initialTree: this.loom,
      generation: this.generation,
      createId: (kind) => `fake-${kind}-${++this.idCounter}`,
      compileSteering: () => fixtureHookProgram(),
      steeringDelta: (parent, child) => parent === child
        ? "unchanged"
        : `${parent ?? "none"} → ${child ?? "none"}`,
    });
  }

  private async loomRequest(
    service: "sessions" | "tree",
    method: string,
    args: unknown[],
  ): Promise<unknown> {
    const runtime = this.createLoomRuntime();
    const result = await runtime.request({ service, method, args });
    this.applySnapshot(runtime.snapshot());
    return result;
  }

  private applySnapshot(snapshot: { session: SessionInfo; tree: LoomTreeJSON }): void {
    this.session = snapshot.session;
    this.loom = snapshot.tree;
    this.idCounter = Math.max(this.idCounter, nextFixtureIdCounter(this.loom));
  }

  private configureInstrumentPacks(packs: readonly FakeInstrumentPack[]): void {
    const lensPacks = packs.filter((pack) => pack.kind === "jlens");
    const lensPack = lensPacks[0];
    const saePack = packs.find((pack) => pack.kind === "sae");
    if (lensPack) {
      this.sourcesByFamily.lens = lensPacks.map((pack, index) => ({
        source: pack.id,
        name: pack.displayName,
        kind: "catalog",
        provider: "catalog",
        active: index === 0,
      }));
      const block = this.instrumentBlock("lens");
      block.source = lensPack.id;
      block.live = { enabled: true, layers: [0, 1] };
      block.capabilities = {
        sources: true,
        preparations: [],
        token_readout: true,
        source_switch: lensPacks.length > 1,
      };
      this.session.jlens_fitted = true;
    }
    if (saePack) {
      this.sourcesByFamily.sae = [{
        source: saePack.id,
        name: saePack.displayName,
        kind: "catalog",
        provider: "catalog",
        active: true,
        layer: 1,
        features: 64,
      }];
      const block = this.instrumentBlock("sae");
      block.source = saePack.id;
      block.live = { enabled: true, layer: 1, source: saePack.id };
      block.capabilities = {
        sources: true,
        preparations: [],
        token_readout: true,
        source_switch: false,
      };
    }
  }

  private instrumentSources(
    family: InstrumentFamily,
  ): { sources: InstrumentSourceJSON[]; releases?: never[] } {
    if (family === "geometry") return { sources: [] };
    return { sources: clone(this.sourcesByFamily[family]), releases: [] };
  }

  private setInstrumentLive(
    family: InstrumentFamily,
    body: { enabled: boolean; layers?: number[] | null },
  ): InstrumentFamilyBlock["live"] {
    if (typeof body.enabled !== "boolean") {
      throw fakeError("INVALID_INSTRUMENT_STATE", "enabled must be boolean");
    }
    const block = this.instrumentBlock(family);
    if (family === "geometry") {
      block.live = { enabled: body.enabled };
    } else if (family === "lens") {
      if (body.enabled && !block.source) {
        throw fakeError(
          "INSTRUMENT_SOURCE_REQUIRED",
          "Install a J-lens pack before enabling live readout",
        );
      }
      block.live = {
        enabled: body.enabled,
        layers: body.enabled ? body.layers ?? [0, 1] : null,
      };
    } else {
      if (body.enabled && !block.source) {
        throw fakeError(
          "INSTRUMENT_SOURCE_REQUIRED",
          "Install an SAE pack before enabling live readout",
        );
      }
      const prior = "layer" in block.live ? block.live.layer : null;
      block.live = {
        enabled: body.enabled,
        layer: prior ?? 1,
        source: block.source,
      };
    }
    return clone(block.live);
  }

  private setLensSource(source: string): { source: string; live_layers: number[] } {
    const selected = this.sourcesByFamily.lens.find((candidate) => candidate.source === source);
    if (!selected) {
      throw fakeError(
        "INSTRUMENT_SOURCE_NOT_FOUND",
        `J-lens source ${source} is not installed`,
      );
    }
    for (const candidate of this.sourcesByFamily.lens) candidate.active = candidate === selected;
    const block = this.instrumentBlock("lens");
    block.source = source;
    block.live = { enabled: true, layers: [0, 1] };
    this.session.jlens_fitted = true;
    return { source, live_layers: [0, 1] };
  }

  private activateInstalledPack(
    family: "lens" | "sae",
    body: InstalledInstrumentPackActivationRequest,
  ): InstalledInstrumentPackActivationResult {
    if (typeof body.source !== "string" || body.source.trim() === "") {
      throw fakeError(
        "INVALID_INSTRUMENT_REQUEST",
        "installed instrument source must be a non-empty string",
      );
    }
    if (
      body.contextBindingSha256 != null &&
      body.contextBindingSha256 !== this.contextBindingSha256
    ) {
      throw fakeError(
        "INSTRUMENT_PACK_CONTEXT_MISMATCH",
        "This model tool was made for a different conversation length; choose a compatible tool in Model settings",
      );
    }
    const selected = this.sourcesByFamily[family].find((candidate) => candidate.active) ?? null;
    if (selected === null) {
      throw fakeError(
        "INSTRUMENT_PACK_NOT_LOADED",
        `No compatible ${family === "lens" ? "J-lens" : "SAE"} pack is loaded. Download it in Model settings, then close and reopen the model`,
      );
    }
    if (body.source !== selected.source) {
      throw fakeError(
        "INSTRUMENT_PACK_SOURCE_MISMATCH",
        `The loaded ${family === "lens" ? "J-lens" : "SAE"} pack is ${selected.source}; close the model before changing model tools`,
      );
    }
    if (
      family === "sae" && body.layer != null &&
      (!Number.isSafeInteger(body.layer) || body.layer !== selected.layer)
    ) {
      throw fakeError(
        "INSTRUMENT_PACK_LAYER_MISMATCH",
        `The loaded SAE pack is for layer ${selected.layer}, not layer ${String(body.layer)}`,
      );
    }
    const live = this.setInstrumentLive(family, { enabled: true });
    return {
      state: "active",
      family,
      source: selected.source,
      live,
      contextBindingSha256: this.contextBindingSha256,
      reloadRequired: false,
    };
  }

  private startPreparation(
    family: InstrumentFamily,
    body: { operation: "fetch" | "fit" | "train" } & Record<string, unknown>,
  ): PreparationStatusJSON {
    this.preparedFamily(family);
    if (family === "lens" && body.operation === "fit") {
      throw fakeError(
        "HOSTED_JLENS_FITTING_EXCLUDED",
        "J-lens fitting is available in the Python runtime; install a compatible browser pack instead",
      );
    }
    if (family === "sae" && body.operation === "train") {
      throw fakeError(
        "HOSTED_SAE_TRAINING_EXCLUDED",
        "SAE training is available in the Python runtime; install a compatible browser pack instead",
      );
    }
    if (body.operation === "fetch") {
      throw fakeError(
        "HOSTED_PACK_DOWNLOAD_MANAGED",
        "Browser tool downloads are managed in Model settings. Activate an already loaded compatible pack instead",
      );
    }
    throw fakeError(
      "INVALID_INSTRUMENT_REQUEST",
      `Operation ${JSON.stringify(body.operation)} is not valid for the browser ${family} runtime`,
    );
  }

  private cancelPreparation(family: InstrumentFamily): PreparationStatusJSON {
    const key = this.preparedFamily(family);
    if (this.preparations[key].state === "running") {
      this.preparations[key] = {
        ...this.preparations[key],
        state: "cancelled",
        message: "cancelled",
        finished_at: Date.now() / 1_000,
        cancellable: false,
      };
    }
    const status = clone(this.preparations[key]);
    if (status.state === "idle") {
      status.message = "No preparation is running. Pause an active download from Model settings";
    }
    return status;
  }

  private preparedFamily(family: InstrumentFamily): "lens" | "sae" {
    if (family === "geometry") {
      throw fakeError(
        "INSTRUMENT_PREPARATION_UNAVAILABLE",
        "Geometry has no downloadable preparation job",
      );
    }
    return family;
  }

  private tokenReadout(family: InstrumentFamily): MeasurementsEnvelopeJSON {
    const envelope: MeasurementsEnvelopeJSON = {
      version: 1,
      scope: "replay",
      provenance: "replayed",
      instruments: {},
      scores: {},
      per_layer_scores: {},
    };
    if (family === "lens" && this.instrumentBlock("lens").source) {
      envelope.instruments.lens = {
        binding: { source: this.instrumentBlock("lens").source, steering: null },
        readout: {
          layers: [0, 1].map((layer) => ({
            layer,
            tokens: [{ token: " fixture", id: 7, logprob: -0.1 - layer * 0.05 }],
          })),
          aggregate: [{ token: " fixture", strength: 0.88, com: 0.5, spread: 0.25 }],
        },
      };
    } else if (family === "sae" && this.instrumentBlock("sae").source) {
      const block = this.instrumentBlock("sae");
      envelope.instruments.sae = {
        binding: {
          source: block.source,
          steering: null,
          layer: "layer" in block.live ? block.live.layer : 1,
        },
        readout: {
          features: [
            { id: 7, activation: 0.7, label: "fixture feature", max_act: 1 },
            { id: 12, activation: 0.35, label: "local behavior", max_act: 1 },
          ],
        },
      };
    } else if (family === "geometry") {
      envelope.instruments.geometry = {
        binding: { source: "fixture-core", steering: null },
        readings: {},
      };
    }
    return envelope;
  }

  private fixtureMeasurements(
    text: string,
    index: number,
    scope: "token" | "aggregate",
  ): MeasurementsEnvelopeJSON | undefined {
    const geometryProbes = this.probeRows.filter((probe) => probe.family === "geometry");
    const lensProbes = this.probeRows.filter((probe) => probe.family === "lens");
    const saeProbes = this.probeRows.filter((probe) => probe.family === "sae");
    const lensBlock = this.instrumentBlock("lens");
    const saeBlock = this.instrumentBlock("sae");
    const lensLive = lensBlock.live.enabled;
    const saeLive = saeBlock.live.enabled;
    if (
      geometryProbes.length === 0 && lensProbes.length === 0 && saeProbes.length === 0 &&
      !lensLive && !saeLive
    ) return undefined;

    const envelope: MeasurementsEnvelopeJSON = {
      version: 1,
      scope,
      provenance: "captured",
      instruments: {},
      scores: {},
      per_layer_scores: { "0": {}, "1": {} },
    };
    const position = Math.max(-0.9, Math.min(0.9, -0.55 + index * 0.18));

    if (geometryProbes.length > 0) {
      const readings: Record<string, ProbeReadingJSON> = {};
      for (const probe of geometryProbes) {
        const nearestLabel = position < 0 ? "negative" : "positive";
        const reading: ProbeReadingJSON = {
          fraction: (position + 1) / 2,
          nearest: [[nearestLabel, Math.abs(Math.sign(position || 1) - position)]],
          coords: [position],
          residual: 0.03,
          fraction_per_layer: { "0": (position + 0.95) / 2, "1": (position + 1) / 2 },
          coords_per_layer: { "0": [position - 0.05], "1": [position] },
          residual_per_layer: { "0": 0.04, "1": 0.03 },
          subspace_coords_per_layer: { "0": [position - 0.05], "1": [position] },
        };
        readings[probe.name] = reading;
        envelope.scores![probe.name] = position;
        envelope.per_layer_scores!["0"][probe.name] = position - 0.05;
        envelope.per_layer_scores!["1"][probe.name] = position;
      }
      envelope.instruments.geometry = {
        binding: { source: "fixture-core", steering: null },
        readings,
      };
    }

    if (lensBlock.source && (lensLive || lensProbes.length > 0)) {
      const value = Math.min(0.96, 0.58 + index * 0.025);
      const readings: Record<string, ScalarReadingJSON> = {};
      for (const probe of lensProbes) {
        readings[probe.name] = fixtureScalarReading(value, "mean_token_probability");
        envelope.scores![probe.name] = value;
        envelope.per_layer_scores!["0"][probe.name] = value - 0.08;
        envelope.per_layer_scores!["1"][probe.name] = value;
      }
      envelope.instruments.lens = {
        binding: { source: lensBlock.source, steering: null },
        ...(lensProbes.length > 0 ? { readings } : {}),
        ...(lensLive ? {
          readout: {
            layers: [0, 1].map((layer) => ({
              layer,
              tokens: [
                { token: text, id: stableId(text), logprob: -0.08 - layer * 0.03 },
                { token: " honest", id: stableId("honest"), logprob: -0.35 - layer * 0.04 },
              ],
            })),
            aggregate: [
              { token: text, strength: value, com: 0.55, spread: 0.2 },
              { token: " honest", strength: value - 0.12, com: 0.62, spread: 0.18 },
            ],
          },
        } : {}),
      };
    }

    if (saeBlock.source && (saeLive || saeProbes.length > 0)) {
      const value = Math.min(0.95, 0.42 + index * 0.035);
      const readings: Record<string, ScalarReadingJSON> = {};
      for (const probe of saeProbes) {
        readings[probe.name] = fixtureScalarReading(value, "activation_over_max");
        envelope.scores![probe.name] = value;
        envelope.per_layer_scores!["1"][probe.name] = value;
      }
      const pinnedFeatureIds = saeProbes.map((probe) => probe.feature_id);
      const featureIds = [...new Set([...pinnedFeatureIds, 7, 12])];
      envelope.instruments.sae = {
        binding: { source: saeBlock.source, steering: null, layer: 1 },
        ...(saeProbes.length > 0 ? { readings } : {}),
        ...(saeLive ? {
          readout: {
            features: featureIds.map((id, featureIndex) => ({
              id,
              activation: Math.max(0.05, value - featureIndex * 0.11),
              label: featureMetadata(id).label,
              max_act: 1,
            })),
          },
        } : {}),
      };
    }

    if (Object.keys(envelope.scores!).length === 0) delete envelope.scores;
    if (
      Object.values(envelope.per_layer_scores!).every(
        (layer) => Object.keys(layer).length === 0,
      )
    ) delete envelope.per_layer_scores;
    return envelope;
  }

  private validateLensToken(word: string): { word: string; token_id: number } {
    if (!this.instrumentBlock("lens").source) {
      throw fakeError(
        "INSTRUMENT_SOURCE_REQUIRED",
        "Install a J-lens pack before selecting a token",
      );
    }
    const trimmed = word.trim().replace(/^jlens\//, "");
    if (!trimmed || /\s/.test(trimmed)) {
      throw fakeError(
        "INVALID_LENS_TOKEN",
        "J-lens tokens must be one non-empty tokenizer token",
      );
    }
    return { word: trimmed, token_id: stableId(trimmed) };
  }

  private validateSaeFeature(
    featureId: number,
  ): { id: number; label: string; layer: number; max_act: number } {
    if (!this.instrumentBlock("sae").source) {
      throw fakeError(
        "INSTRUMENT_SOURCE_REQUIRED",
        "Install an SAE pack before selecting a feature",
      );
    }
    if (!Number.isSafeInteger(featureId) || featureId < 0 || featureId >= 64) {
      throw fakeError(
        "INVALID_SAE_FEATURE",
        "Fixture SAE feature id must be between 0 and 63",
      );
    }
    return {
      id: featureId,
      label: featureMetadata(featureId).label,
      layer: 1,
      max_act: 1,
    };
  }

  private attachProbe(request: ProbeRequest): ProbeInfo {
    const selector = request.selector?.trim();
    if (!selector) throw fakeError("INVALID_PROBE", "Probe selector is required");
    const name = request.name?.trim() || selector;
    const existing = this.probeRows.find((probe) => probe.name === name);
    if (existing) return clone(existing);
    let probe: ProbeInfo;
    if (selector.startsWith("jlens/")) {
      const validated = this.validateLensToken(selector.slice("jlens/".length));
      probe = {
        family: "lens",
        name,
        layers: [0, 1],
        intrinsic_dim: 1,
        feature_space: "residual",
        word: validated.word,
        token_id: validated.token_id,
      };
    } else if (selector.startsWith("sae/")) {
      const validated = this.validateSaeFeature(Number(selector.slice("sae/".length)));
      probe = {
        family: "sae",
        name,
        layers: [validated.layer],
        intrinsic_dim: 1,
        feature_space: "sae",
        feature_id: validated.id,
        label: validated.label,
        max_act: validated.max_act,
      };
    } else {
      probe = {
        family: "geometry",
        name,
        manifold: selector,
        top_n: request.top_n ?? 3,
        layers: [0, 1],
        node_labels: ["negative", "positive"],
        node_count: 2,
        domain: {
          type: "box",
          axes: [{ name: "position", periodic: false, period: 2, lo: -1, hi: 1 }],
        },
        intrinsic_dim: 1,
        feature_space: "residual",
        is_affine: true,
        node_coords: [[-1], [1]],
      };
    }
    this.probeRows.push(probe);
    this.syncProbeState();
    return clone(probe);
  }

  private probeGeometry(name: string) {
    const probe = this.probeRows.find((candidate) => candidate.name === name);
    if (!probe) throw fakeError("PROBE_NOT_FOUND", `Probe ${name} is not attached`);
    if (probe.family !== "geometry") {
      throw fakeError(
        "PROBE_GEOMETRY_UNAVAILABLE",
        "Only geometry probes have a manifold plot",
      );
    }
    const fixtureRank = name === "fixture/mobile-3d" ? 3 : 1;
    const nodeWhite = fixtureRank === 3
      ? [[-1, -0.4, 0.2], [1, 0.5, -0.2]]
      : [[-1], [1]];
    const neutralWhite = fixtureRank === 3 ? [0, 0, 0] : [0];
    const pcaRotation = fixtureRank === 3
      ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      : [[1]];
    const explainedVariancePcs = fixtureRank === 3 ? [0.55, 0.3, 0.15] : [1];
    return {
      name: probe.name,
      manifold: probe.manifold,
      intrinsic_dim: fixtureRank,
      is_affine: probe.is_affine,
      node_labels: [...probe.node_labels],
      rank_uniform: true,
      layers: Object.fromEntries(probe.layers.map((layer) => [String(layer), {
        layer,
        rank: fixtureRank,
        intrinsic_dim: fixtureRank,
        is_affine: true,
        node_white: nodeWhite,
        neutral_white: neutralWhite,
        pca_rotation: pcaRotation,
        explained_variance_pcs: explainedVariancePcs,
        mahalanobis_share: 1,
        overlay: null,
      }])),
    };
  }

  private syncProbeState(): void {
    this.session.probes = this.probeRows.map((probe) => probe.name);
    for (const block of this.session.instruments) {
      block.probes = this.probeRows
        .filter((probe) => probe.family === block.family)
        .map((probe) => probe.name);
    }
  }

  private jointLogprobs(aId: string, bId: string) {
    const a = this.requireNode(aId);
    const b = this.requireNode(bId);
    if (a.parent_id !== b.parent_id) {
      throw fakeError(
        "JOINT_LOGPROB_PARENT_MISMATCH",
        "Compared nodes must be siblings",
      );
    }
    const aTokens = a.tokens ?? [];
    const bTokens = b.tokens ?? [];
    const rows = Array.from(
      { length: Math.max(aTokens.length, bTokens.length) },
      (_, index) => {
        const aToken = aTokens[index];
        const bToken = bTokens[index];
        const aligned = aToken !== undefined && bToken !== undefined;
        const lpA = aToken?.logprob ?? null;
        const lpB = bToken?.logprob ?? null;
        return {
          a_index: index,
          b_index: index,
          a_text: aToken?.text ?? "",
          b_text: bToken?.text ?? "",
          aligned,
          lp_a_in_a: lpA,
          lp_b_in_b: lpB,
          lp_a_in_b: aligned && lpA !== null ? lpA - 0.02 : null,
          lp_b_in_a: aligned && lpB !== null ? lpB - 0.02 : null,
          rank_changed: aToken?.text !== bToken?.text,
          approx_kl: aligned ? 0.02 : null,
        };
      },
    );
    return {
      a_id: aId,
      b_id: bId,
      parent_id: a.parent_id,
      rows,
      n_rank1_changed: rows.filter((row) => row.rank_changed).length,
    };
  }

  private requireNode(id: string): LoomNodeJSON {
    const node = this.loom.nodes.find((candidate) => candidate.id === id);
    if (!node) {
      throw fakeError("LOOM_NODE_NOT_FOUND", `Conversation node ${id} does not exist`);
    }
    return node;
  }

  private instrumentBlock(family: InstrumentFamily): InstrumentFamilyBlock {
    const block = this.session.instruments.find((candidate) => candidate.family === family);
    if (!block) {
      throw fakeError(
        "INSTRUMENT_FAMILY_MISSING",
        `Instrument family ${family} is unavailable`,
      );
    }
    return block;
  }
}

class DeterministicGenerationPort implements BrowserGenerationPort {
  private readonly response: string;
  private readonly tokenDelayMs: number;
  private readonly measurements: (
    text: string,
    index: number,
    scope: "token" | "aggregate",
  ) => MeasurementsEnvelopeJSON | undefined;
  private stopped = false;
  private readonly tokenText = new Map<number, string>();

  constructor(
    response: string,
    tokenDelayMs: number,
    measurements: DeterministicGenerationPort["measurements"],
  ) {
    this.response = response;
    this.tokenDelayMs = Math.max(0, tokenDelayMs);
    this.measurements = measurements;
  }

  async tokenizeText(text: string): Promise<number[]> {
    const pieces = text.match(/\S+\s*/g) ?? (text ? [text] : []);
    return pieces.map((piece) => {
      const tokenId = stableId(piece);
      this.tokenText.set(tokenId, piece);
      return tokenId;
    });
  }

  async streamGeneration(
    plan: WebLlmGenerationPlan,
    onToken: (token: WebLlmGeneratedToken) => void | Promise<void>,
  ): Promise<WebLlmGenerationResult> {
    this.stopped = false;
    const stopStrings = (plan.sampling?.stop ?? []).filter((value) => value.length > 0);
    const stopAt = stopStrings.reduce((earliest, value) => {
      const index = this.response.indexOf(value);
      return index < 0 ? earliest : Math.min(earliest, index);
    }, this.response.length);
    const stoppedBySequence = stopAt < this.response.length;
    const response = this.response.slice(0, stopAt);
    const pieces = response.match(/\S+\s*/g) ?? (response ? [response] : []);
    const maxTokens = plan.sampling?.max_tokens ?? pieces.length;
    const limit = Math.min(pieces.length, maxTokens);
    const forcedPrefix = plan.replay?.forcedPrefixTokenIds ?? [];
    const alternativeCount = Math.max(
      0,
      Math.min(5, Math.floor(plan.sampling?.return_top_k ?? 0)),
    );
    const emitted: string[] = [];
    for (let index = 0; index < limit; index += 1) {
      if (this.stopped) break;
      const originalText = pieces[index];
      const originalId = stableId(originalText);
      this.tokenText.set(originalId, originalText);
      const topAlts = alternativeCount > 0
        ? fixtureTokenAlternatives(originalText, index, alternativeCount)
        : null;
      for (const alternative of topAlts ?? []) {
        this.tokenText.set(alternative.id, alternative.text);
      }
      const forcedId = forcedPrefix[index];
      const tokenId = forcedId ?? originalId;
      const text = forcedId === undefined
        ? originalText
        : this.tokenText.get(forcedId) ?? originalText;
      const logprob = topAlts?.find((row) => row.id === tokenId)?.logprob ?? -0.1;
      emitted.push(text);
      const measurements = this.measurements(text, index, "token");
      await plan.onRawToken?.({
        text,
        tokenId,
        logprob,
        samplerEntropy: 0.1,
        perplexity: Math.exp(0.1),
        rawIndex: index,
        topAlts,
        measurements,
      });
      await onToken({
        text,
        thinking: false,
        tokenId,
        logprob,
        samplerEntropy: 0.1,
        perplexity: Math.exp(0.1),
        rawIndex: index,
        topAlts,
        measurements,
      });
      if (this.tokenDelayMs > 0) {
        await new Promise<void>((resolve) => setTimeout(resolve, this.tokenDelayMs));
      }
    }
    const tokens = emitted.length;
    const text = emitted.join("");
    const capped = limit < pieces.length;
    const finishReason = this.stopped || stoppedBySequence
      ? "stop"
      : capped
        ? "length"
        : "stop";
    const promptTokens = generationPromptTokens(plan);
    return {
      text,
      thinkingText: null,
      tokens,
      finishReason,
      terminalReason: this.stopped
        ? "external_stop"
        : stoppedBySequence
          ? "stop_sequence"
          : capped
            ? "length"
            : "eos",
      usage: {
        promptTokens,
        completionTokens: tokens,
        totalTokens: promptTokens + tokens,
      },
      meanLogprob: tokens > 0 ? -0.1 : null,
      meanSurprise: tokens > 0 ? 0.1 : null,
      prefillTokensPerSecond: 120,
      decodeTokensPerSecond: 48,
      ...(tokens > 0
        ? { measurements: this.measurements(emitted[tokens - 1], tokens - 1, "aggregate") }
        : {}),
    };
  }

  async stop(): Promise<void> {
    this.stopped = true;
  }

  runtimeCapabilities(): import("./webLlmGeneration").WebLlmRuntimeCapabilities {
    return {
      topK: true,
      forcedReplay: true,
      replayScoring: false,
      tokenizer: true,
      namedRoles: false,
      userSeatGeneration: false,
      sceneStitching: false,
    };
  }
}

class FakeEventChannel implements RuntimeEventChannel {
  private readonly runGeneration: (
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
  ) => Promise<void>;
  private readonly stopGeneration: () => Promise<void>;
  private readonly listeners = new Set<(message: WSServerMessage) => void>();
  private readonly stateListeners = new Set<(state: RuntimeEventChannelState) => void>();
  private openState = false;
  private running = false;

  constructor(
    runGeneration: FakeEventChannel["runGeneration"],
    stopGeneration: FakeEventChannel["stopGeneration"],
  ) {
    this.runGeneration = runGeneration;
    this.stopGeneration = stopGeneration;
  }

  get isOpen(): boolean {
    return this.openState;
  }

  async open(): Promise<void> {
    if (this.openState) return;
    this.openState = true;
    for (const listener of this.stateListeners) listener({ state: "open" });
  }

  subscribe(listener: (message: WSServerMessage) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  subscribeState(listener: (state: RuntimeEventChannelState) => void): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  send(message: WSClientMessage): void {
    if (!this.openState) throw new Error("Fake runtime event channel is not open");
    if (message.type === "stop") {
      void this.stop();
      return;
    }
    if (this.running) {
      this.emit({
        type: "error",
        code: "GENERATION_BUSY",
        message: "The fixture is already generating",
      });
      return;
    }
    this.running = true;
    queueMicrotask(() => {
      void this.runGeneration(message, (event) => this.emit(event))
        .catch((error) => this.emit({
          type: "error",
          code: readErrorCode(error),
          message: error instanceof Error ? error.message : String(error),
        }))
        .finally(() => {
          this.running = false;
        });
    });
  }

  async stop(): Promise<void> {
    await this.stopGeneration();
  }

  acknowledgeSnapshot(): void {}

  close(): void {
    const wasOpen = this.openState;
    this.openState = false;
    void this.stopGeneration();
    this.listeners.clear();
    if (wasOpen) {
      for (const listener of this.stateListeners) {
        listener({ state: "closed", expected: true, reason: null });
      }
    }
    this.stateListeners.clear();
  }

  private emit(message: WSServerMessage): void {
    for (const listener of this.listeners) listener(message);
  }
}

function fakeSession(modelId: string): SessionInfo {
  return {
    id: "default",
    model_id: modelId,
    device: "webgpu fixture",
    dtype: "float32",
    created: 0,
    config: {
      temperature: 0,
      top_p: 1,
      top_k: null,
      max_tokens: 64,
      system_prompt: null,
      thinking: false,
    },
    profiles: [],
    probes: [],
    history_length: 0,
    supports_thinking: false,
    thinking_is_optional: false,
    is_base_model: false,
    jlens_fitted: false,
    instruments: [instrument("geometry"), instrument("lens"), instrument("sae")],
    default_steering: null,
    role_substitution_supported: false,
    user_role_supported: false,
    default_assistant_role: "assistant",
    default_user_role: "user",
    scene_mode: false,
    thinking_input_supported: false,
    strips_history_thinking: false,
  };
}

function instrument(family: "geometry" | "lens" | "sae"): InstrumentFamilyBlock {
  const live = family === "geometry"
    ? { enabled: false }
    : family === "lens"
      ? { enabled: false, layers: null }
      : { enabled: false, layer: null, source: null };
  return {
    family,
    live,
    source: null,
    probes: [],
    capabilities: {
      sources: false,
      preparations: [],
      token_readout: family === "geometry",
      source_switch: false,
    },
  } as InstrumentFamilyBlock;
}

function fakeTree(modelId: string, rootId = "root"): LoomTreeJSON {
  const root: LoomNodeJSON = {
    id: rootId,
    parent_id: null,
    role: "system",
    text: "",
    role_label: null,
    thinking_text: null,
    aggregate_readings: {},
    applied_steering: null,
    finish_reason: null,
    starred: false,
    notes: "",
    created_at: 0,
    edited_at: null,
    edit_count: 0,
    mean_logprob: null,
    mean_surprise: null,
    recipe: null,
    tokens: null,
    thinking_tokens: null,
    raw_token_ids: null,
  };
  return {
    tree_format: 2,
    drowse_version: "fixture",
    model_id: modelId,
    session_id: "default",
    name: "fixture",
    rev: 0,
    root_id: root.id,
    active_node_id: root.id,
    nodes: [root],
    children_of: { [root.id]: [] },
    cast: {},
  };
}

function idlePreparation(): PreparationStatusJSON {
  return {
    state: "idle",
    operation: null,
    progress: null,
    message: null,
    error: null,
    started_at: null,
    finished_at: null,
    cancellable: false,
  };
}

function fixtureHookProgram(): RankOneHookProgramBuffers {
  return {
    hookAbi: DROWSE_HOOK_ABI,
    hiddenSize: 1,
    layerCount: 1,
    enabled: new Uint32Array([1]),
    basis: new Float32Array([1]),
    neutral: new Float32Array([0]),
    target: new Float32Array([0]),
    along: new Float32Array([0]),
    collapse: new Float32Array([0]),
    probeBasis: new Float32Array([1]),
    probeNeutral: new Float32Array([0]),
  };
}

function generationPromptTokens(plan: WebLlmGenerationPlan): number {
  const text = plan.input.kind === "raw"
    ? plan.input.prompt
    : plan.input.messages.map((message) => message.content).join(" ");
  return text.match(/\S+/g)?.length ?? 0;
}

function featureMetadata(id: number): { label: string; max_act: number } {
  return { label: `fixture feature ${id}`, max_act: 1 };
}

function fixtureScalarReading(value: number, unit: string): ScalarReadingJSON {
  return {
    value,
    unit,
    per_layer: { "0": value - 0.08, "1": value },
    depth: { center: [1], spread: [0.45], basis: "layer" },
  };
}

function stableId(value: string): number {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function fixtureTokenAlternatives(
  originalText: string,
  index: number,
  count: number,
): Array<{ id: number; text: string; logprob: number }> {
  const replacements = ["Alternative ", "Different ", "Another ", "Fresh "];
  return [
    { id: stableId(originalText), text: originalText, logprob: -0.1 },
    ...replacements.map((text, replacementIndex) => {
      const tokenText = `${text.trim()}-${index} `;
      return {
        id: stableId(tokenText),
        text: tokenText,
        logprob: -0.6 - replacementIndex * 0.25,
      };
    }),
  ].slice(0, count);
}

function nextFixtureIdCounter(tree: LoomTreeJSON): number {
  return tree.nodes.reduce((maximum, node) => {
    const match = /^fake-node-(\d+)$/.exec(node.id);
    return match ? Math.max(maximum, Number(match[1])) : maximum;
  }, 0);
}

function fakeError(code: string, message: string): Error & { code: string } {
  return Object.assign(new Error(message), { code });
}

function readErrorCode(error: unknown): string {
  return typeof error === "object" && error !== null && "code" in error &&
      typeof (error as { code?: unknown }).code === "string"
    ? (error as { code: string }).code
    : "FIXTURE_GENERATION_FAILED";
}

function fakeFittingUnavailable(): Error & { code: string } {
  return fakeError(
    "FIXTURE_FITTING_DISABLED",
    "The deterministic development fixture does not run model fitting jobs",
  );
}

function fakeArtifactRoutingRequired(operation: string): Error & { code: string } {
  return fakeError(
    "FIXTURE_ARTIFACT_ROUTING_REQUIRED",
    `${operation} must be routed through the hosted worker artifact repository`,
  );
}

function clone<T>(value: T): T {
  return structuredClone(value);
}
