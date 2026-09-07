import type {
  InstalledInstrumentPackActivationResult,
  RuntimeProgressEvent,
  RuntimeServiceRequest,
} from "../../lib/runtime/contracts";
import type {
  InstrumentFamily,
  InstrumentFamilyBlock,
  InstrumentSourceJSON,
  MeasurementsEnvelopeJSON,
  PreparationStatusJSON,
  ProbeInfo,
  ProbeRequest,
} from "../../lib/types";
import type { BrowserFeasibilityCorePackCompiler } from "./browserCorePack";
import type { BrowserJlensRuntimeResolver } from "./browserInstrumentPacks";
import { BROWSER_EXACT_READOUT_TOP_K_CAPACITY } from "./webLlmGeneration";
import { loadSaeDescription } from "../../lib/saeDescriptions";

type TokenLookup = (
  nodeId: string,
  rawIndex: number,
) => MeasurementsEnvelopeJSON | null;

export interface BrowserTokenReplayRequest {
  family: InstrumentFamily;
  nodeId: string;
  rawIndex: number;
  options: TokenReadoutOptions;
}

type TokenReplay = (
  request: BrowserTokenReplayRequest,
  onProgress: (event: RuntimeProgressEvent) => void,
) => Promise<MeasurementsEnvelopeJSON>;

export class BrowserInstrumentRuntime {
  private readonly preparation = new Map<string, PreparationStatusJSON>();
  private tokenReplay: TokenReplay | null = null;

  constructor(
    private compiler: BrowserFeasibilityCorePackCompiler,
    private readonly jlensResolver: BrowserJlensRuntimeResolver | null = null,
    private readonly contextBindingSha256: string | null = null,
    enableLiveReadout = true,
  ) {
    const descriptor = this.compiler.instrumentDescriptor();
    if (
      enableLiveReadout &&
      this.jlensResolver !== null &&
      descriptor.jlens !== null
    ) {
      this.compiler.setInstrumentLive("lens", true);
    }
    if (enableLiveReadout && descriptor.sae !== null) {
      this.compiler.setInstrumentLive("sae", true);
    }
  }

  async replaceCompiler(compiler: BrowserFeasibilityCorePackCompiler): Promise<void> {
    if (compiler === this.compiler) return;
    const live = this.compiler.instrumentLiveState();
    const probes = this.compiler.listProbes();
    const lensWords = probes
      .filter((probe): probe is Extract<ProbeInfo, { family: "lens" }> =>
        probe.family === "lens"
      )
      .map((probe) => probe.word);
    if (lensWords.length > 0 && this.jlensResolver !== null) {
      await compiler.prepareJlensWords(lensWords, this.jlensResolver);
    }
    for (const probe of probes) compiler.attachProbe(probeRefreshRequest(probe));
    compiler.setInstrumentLive("geometry", live.geometry);
    compiler.setInstrumentLive("lens", live.lens, live.lensLayers ?? undefined);
    compiler.setInstrumentLive("sae", live.sae);
    this.compiler = compiler;
  }

  configureTokenReplay(replay: TokenReplay | null): void {
    this.tokenReplay = replay;
  }

  hasLiveReadout(): boolean {
    const live = this.compiler.instrumentLiveState();
    return live.lens || live.sae || this.compiler.listProbes().some((probe) =>
      probe.family !== "geometry" || live.geometry
    );
  }

  jlensFitted(): boolean {
    return this.compiler.instrumentDescriptor().jlens !== null;
  }

  profileNames(): string[] {
    return this.compiler.listProfiles().profiles.map((profile) => profile.name);
  }

  probeHashes(): Record<string, string> {
    return this.compiler.probeHashes();
  }

  blocks(): InstrumentFamilyBlock[] {
    const descriptor = this.compiler.instrumentDescriptor();
    const live = this.compiler.instrumentLiveState();
    const probes = this.compiler.listProbes();
    return [
      {
        family: "geometry",
        live: { enabled: live.geometry },
        source: null,
        probes: probes.filter((probe) => probe.family === "geometry").map((probe) => probe.name),
        capabilities: {
          sources: false,
          preparations: [],
          token_readout: this.tokenReplay !== null,
          source_switch: false,
        },
      },
      {
        family: "lens",
        live: {
          enabled: live.lens,
          layers: live.lens ? live.lensLayers : null,
        },
        source: descriptor.jlens?.source ?? null,
        probes: probes.filter((probe) => probe.family === "lens").map((probe) => probe.name),
        capabilities: {
          sources: descriptor.jlens !== null,
          preparations: [],
          token_readout: this.tokenReplay !== null && descriptor.jlens !== null,
          source_switch: this.lensSources().length > 1,
        },
      },
      {
        family: "sae",
        live: {
          enabled: live.sae,
          layer: descriptor.sae?.layer ?? null,
          source: descriptor.sae?.source ?? null,
        },
        source: descriptor.sae?.source ?? null,
        probes: probes.filter((probe) => probe.family === "sae").map((probe) => probe.name),
        capabilities: {
          sources: descriptor.sae !== null,
          preparations: [],
          token_readout: this.tokenReplay !== null && descriptor.sae !== null,
          source_switch: false,
        },
      },
    ];
  }

  request(
    request: RuntimeServiceRequest,
    tokenLookup: TokenLookup,
    onProgress: (event: RuntimeProgressEvent) => void = () => undefined,
  ): unknown {
    const { method, args } = request;
    if (request.service === "profiles") return this.profileRequest(method, args);
    if (request.service === "probes") return this.probeRequest(method, args);
    if (request.service === "manifolds") return this.manifoldRequest(method, args);
    if (method === "setLive") return this.setLive(args);
    if (method === "sources") return this.sources(args);
    if (method === "setLensSource") return this.setLensSource(args);
    if (method === "activateInstalledPack") return this.activateInstalledPack(args);
    if (method === "startPreparation") return this.startPreparation(args);
    if (method === "preparationStatus") return this.preparationStatus(args);
    if (method === "cancelPreparation") return this.cancelPreparation(args);
    if (method === "tokenReadout") return this.tokenReadout(args, tokenLookup, onProgress);
    if (method === "validateLensToken") return this.validateLensToken(args);
    if (method === "validateSaeFeature") return this.validateSaeFeature(args);
    if (method === "saeFeaturesMetadata") return this.saeFeaturesMetadata(args);
    throw instrumentError(
      "BROWSER_SERVICE_UNAVAILABLE",
      `Browser runtime service instruments.${method} is not implemented`,
    );
  }

  private profileRequest(method: string, args: unknown[]): unknown {
    if (method === "list") return this.compiler.listProfiles();
    if (method === "get") {
      return this.compiler.getProfile(nonemptyString(args[0], "profile name"));
    }
    if (method === "correlation") {
      const value = args[0];
      if (value === undefined || value === null) return this.compiler.profileCorrelation();
      if (
        !Array.isArray(value) ||
        value.some((name) => typeof name !== "string" || name.trim() === "")
      ) {
        throw instrumentError(
          "INVALID_INSTRUMENT_REQUEST",
          "Correlation profile names must be a list of non-empty strings",
        );
      }
      return this.compiler.profileCorrelation(value);
    }
    if (method === "pairwise") {
      return this.compiler.profilePairwise(
        nonemptyString(args[0], "first profile name"),
        nonemptyString(args[1], "second profile name"),
      );
    }
    throw instrumentError(
      "BROWSER_SERVICE_UNAVAILABLE",
      `Browser runtime service profiles.${method} is not implemented`,
    );
  }

  private probeRequest(method: string, args: unknown[]): unknown {
    if (method === "list") return { probes: this.compiler.listProbes() };
    if (method === "attach") {
      const request = record(args[0], "probe request");
      return this.attachProbe(request as unknown as ProbeRequest);
    }
    if (method === "detach") {
      this.compiler.detachProbe(nonemptyString(args[0], "probe name"));
      return undefined;
    }
    if (method === "geometry") {
      return this.compiler.probeGeometry(nonemptyString(args[0], "probe name"));
    }
    throw instrumentError(
      "BROWSER_SERVICE_UNAVAILABLE",
      `Browser runtime service probes.${method} is not implemented`,
    );
  }

  private async attachProbe(request: ProbeRequest): Promise<ReturnType<BrowserFeasibilityCorePackCompiler["attachProbe"]>> {
    const selector = request.selector?.trim();
    if (selector?.startsWith("jlens/") && this.jlensResolver !== null) {
      await this.compiler.prepareJlensWords(
        [selector.slice("jlens/".length)],
        this.jlensResolver,
      );
    }
    return this.compiler.attachProbe(request);
  }

  private manifoldRequest(method: string, args: unknown[]): unknown {
    if (method === "list") return { manifolds: this.compiler.listManifolds() };
    if (method === "get") {
      return this.compiler.getManifold(
        nonemptyString(args[0], "manifold namespace"),
        nonemptyString(args[1], "manifold name"),
      );
    }
    throw instrumentError(
      "BROWSER_SERVICE_UNAVAILABLE",
      `Browser runtime service manifolds.${method} is not implemented`,
    );
  }

  private setLive(args: unknown[]): unknown {
    const family = instrumentFamily(args[0]);
    const body = record(args[1], "instrument live settings");
    if (typeof body.enabled !== "boolean") {
      throw instrumentError("INVALID_INSTRUMENT_REQUEST", "Instrument live state must be a boolean");
    }
    const allowed = family === "lens" ? new Set(["enabled", "layers"]) : new Set(["enabled"]);
    const extras = Object.keys(body).filter((key) => !allowed.has(key));
    if (extras.length > 0) {
      throw instrumentError(
        "INVALID_INSTRUMENT_REQUEST",
        `${family} live does not accept ${extras.sort().join(", ")}`,
      );
    }
    if (family === "lens") {
      const hasLayers = body.layers !== undefined && body.layers !== null;
      if (!body.enabled && hasLayers) {
        throw instrumentError(
          "INVALID_INSTRUMENT_REQUEST",
          "J-lens live disable does not accept layers",
        );
      }
      const layers = liveLensLayers(body.layers);
      this.compiler.setInstrumentLive("lens", body.enabled, layers);
    } else {
      this.compiler.setInstrumentLive(family, body.enabled);
    }
    return this.blocks().find((block) => block.family === family)!.live;
  }

  private sources(args: unknown[]): { sources: InstrumentSourceJSON[] } {
    const family = instrumentFamily(args[0]);
    const descriptor = this.compiler.instrumentDescriptor();
    if (family === "geometry") return { sources: [] };
    if (family === "lens") {
      return {
        sources: this.lensSources().map((source) => ({
          source: source.source,
          name: source.displayName,
          kind: "catalog",
          provider: "catalog",
          active: source.active,
        })),
      };
    }
    if (family === "sae" && descriptor.sae !== null) {
      return {
        sources: [{
          source: descriptor.sae.source,
          name: descriptor.sae.displayName,
          kind: "catalog",
          provider: "catalog",
          active: true,
          layer: descriptor.sae.layer,
          features: descriptor.sae.features,
          ...(descriptor.sae.modelLayers === undefined
            ? {} : { model_layers: descriptor.sae.modelLayers }),
          ...(descriptor.sae.descriptionSource === undefined
            ? {} : { description_source: descriptor.sae.descriptionSource }),
        }],
      };
    }
    return { sources: [] };
  }

  private async setLensSource(args: unknown[]): Promise<{ source: string; live_layers: number[] }> {
    const source = nonemptyString(args[0], "J-lens source");
    if (this.jlensResolver === null || this.jlensResolver.setJlensDictionary === undefined) {
      throw instrumentError(
        "JLENS_SOURCE_SWITCH_UNAVAILABLE",
        "The browser model runtime cannot replace the active J-lens dictionary",
      );
    }
    const previousSource = this.compiler.instrumentDescriptor().jlens?.source ?? null;
    if (previousSource !== source) {
      try {
        const dictionary = await this.compiler.activateJlensSource(source);
        await this.jlensResolver.setJlensDictionary(dictionary);
        await this.compiler.refreshJlensProbes(this.jlensResolver);
      } catch (error) {
        if (previousSource !== null) {
          try {
            const previous = await this.compiler.activateJlensSource(previousSource);
            await this.jlensResolver.setJlensDictionary(previous);
            await this.compiler.refreshJlensProbes(this.jlensResolver);
          } catch (rollbackError) {
            throw instrumentError(
              "JLENS_SOURCE_SWITCH_ROLLBACK_FAILED",
              `The J-lens source change failed and the previous GPU dictionary could not be restored: ${
                rollbackError instanceof Error ? rollbackError.message : String(rollbackError)
              }`,
            );
          }
        }
        throw error;
      }
    }
    const descriptor = this.compiler.instrumentDescriptor().jlens!;
    this.compiler.setInstrumentLive("lens", true);
    return { source, live_layers: [...descriptor.layers] };
  }

  private lensSources(): ReturnType<BrowserFeasibilityCorePackCompiler["jlensSources"]> {
    const sourceReader = (this.compiler as BrowserFeasibilityCorePackCompiler & {
      jlensSources?: BrowserFeasibilityCorePackCompiler["jlensSources"];
    }).jlensSources;
    if (typeof sourceReader === "function") return sourceReader.call(this.compiler);
    const descriptor = this.compiler.instrumentDescriptor().jlens;
    return descriptor === null
      ? []
      : [{
          source: descriptor.source,
          displayName: descriptor.displayName,
          active: true,
        }];
  }

  private activateInstalledPack(args: unknown[]): InstalledInstrumentPackActivationResult {
    const family = instrumentFamily(args[0]);
    if (family === "geometry") {
      throw instrumentError(
        "INSTRUMENT_ACTIVATION_UNAVAILABLE",
        "Geometry is part of the required core pack and has no optional pack to activate",
      );
    }
    const body = record(args[1], "installed instrument activation");
    const source = nonemptyString(body.source, "installed instrument source");
    const requestedContext = nullableString(
      body.contextBindingSha256,
      "instrument context binding",
    );
    if (
      requestedContext !== null &&
      requestedContext !== this.contextBindingSha256
    ) {
      throw instrumentError(
        "INSTRUMENT_PACK_CONTEXT_MISMATCH",
        this.contextBindingSha256 === null
          ? "The loaded runtime cannot verify this pack's context binding; close and reopen the model"
          : "This model tool was made for a different conversation length; choose a compatible tool in Model settings",
      );
    }
    const descriptor = this.compiler.instrumentDescriptor();
    const installed = family === "lens" ? descriptor.jlens : descriptor.sae;
    if (installed === null) {
      throw instrumentError(
        "INSTRUMENT_PACK_NOT_LOADED",
        `No compatible ${family === "lens" ? "J-lens" : "SAE"} pack is loaded. Download it in Model settings, then close and reopen the model`,
      );
    }
    if (source !== installed.source) {
      throw instrumentError(
        "INSTRUMENT_PACK_SOURCE_MISMATCH",
        `The loaded ${family === "lens" ? "J-lens" : "SAE"} pack is ${installed.source}; close the model before changing model tools`,
      );
    }
    if (family === "sae") {
      if (!("layer" in installed)) {
        throw instrumentError(
          "INSTRUMENT_PACK_BINDING_INVALID",
          "The loaded optional pack does not expose an SAE layer binding",
        );
      }
      const layer = nullableNonnegativeInteger(body.layer, "SAE layer");
      if (layer !== null && layer !== installed.layer) {
        throw instrumentError(
          "INSTRUMENT_PACK_LAYER_MISMATCH",
          `The loaded SAE pack is for layer ${installed.layer}, not layer ${layer}`,
        );
      }
    }
    this.compiler.setInstrumentLive(family, true);
    return {
      state: "active",
      family,
      source: installed.source,
      live: this.blocks().find((block) => block.family === family)!.live,
      contextBindingSha256: this.contextBindingSha256,
      reloadRequired: false,
    };
  }

  private startPreparation(args: unknown[]): never {
    const family = instrumentFamily(args[0]);
    if (family === "geometry") {
      throw instrumentError(
        "INSTRUMENT_PREPARATION_UNAVAILABLE",
        "Geometry has no optional preparation job",
      );
    }
    const body = record(args[1], "instrument preparation");
    const operation = nonemptyString(body.operation, "instrument preparation operation");
    if (family === "lens" && operation === "fit") {
      throw instrumentError(
        "HOSTED_JLENS_FITTING_EXCLUDED",
        "J-lens fitting is available in the Python runtime; install a compatible browser pack instead",
      );
    }
    if (family === "sae" && operation === "train") {
      throw instrumentError(
        "HOSTED_SAE_TRAINING_EXCLUDED",
        "SAE training is available in the Python runtime; install a compatible browser pack instead",
      );
    }
    if (operation === "fetch") {
      throw instrumentError(
        "HOSTED_PACK_DOWNLOAD_MANAGED",
        "Browser tool downloads are managed in Model settings. Activate an already loaded compatible pack instead",
      );
    }
    throw instrumentError(
      "INVALID_INSTRUMENT_REQUEST",
      `Operation ${JSON.stringify(operation)} is not valid for the browser ${family} runtime`,
    );
  }

  private preparationStatus(args: unknown[]): PreparationStatusJSON {
    const family = instrumentFamily(args[0]);
    if (family === "geometry") {
      throw instrumentError(
        "INSTRUMENT_PREPARATION_UNAVAILABLE",
        "Geometry has no optional preparation job",
      );
    }
    return this.preparation.get(family) ?? {
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

  private cancelPreparation(args: unknown[]): PreparationStatusJSON {
    const family = instrumentFamily(args[0]);
    if (family === "geometry") {
      throw instrumentError(
        "INSTRUMENT_PREPARATION_UNAVAILABLE",
        "Geometry has no optional preparation to cancel",
      );
    }
    return this.preparation.get(family) ?? {
      state: "idle",
      operation: null,
      progress: null,
      message: "No preparation is running. Pause an active download from Model settings",
      error: null,
      started_at: null,
      finished_at: null,
      cancellable: false,
    };
  }

  private async tokenReadout(
    args: unknown[],
    lookup: TokenLookup,
    onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<{ measurements: MeasurementsEnvelopeJSON }> {
    const family = instrumentFamily(args[0]);
    const nodeId = nonemptyString(args[1], "node id");
    const rawIndex = nonnegativeInteger(args[2], "raw token index");
    const options = tokenReadoutOptions(args[3]);
    const selectedLensLayers = replayLayersForFamily(
      family,
      options,
      this.compiler.instrumentDescriptor().jlens?.layers ?? [],
    );
    const captured = lookup(nodeId, rawIndex);
    const changesCapture = options.steered === false ||
      options.topK !== undefined ||
      options.raw !== undefined ||
      options.layers !== undefined;
    if (!changesCapture && captured !== null && captured.instruments[family] !== undefined) {
      return { measurements: captured };
    }
    if (this.tokenReplay !== null) {
      const live = this.compiler.instrumentLiveState();
      const wasLive = live[family];
      if (family === "lens") {
        this.compiler.setInstrumentLive("lens", true, selectedLensLayers ?? undefined);
      } else if (!wasLive) {
        this.compiler.setInstrumentLive(family, true);
      }
      try {
        const replayed = await this.tokenReplay(
          { family, nodeId, rawIndex, options },
          onProgress,
        );
        return {
          measurements: replayEnvelope(replayed, family, options, selectedLensLayers),
        };
      } finally {
        if (family === "lens") {
          this.compiler.setInstrumentLive(
            "lens",
            wasLive,
            live.lensLayers ?? undefined,
          );
        } else if (!wasLive) {
          this.compiler.setInstrumentLive(family, false);
        }
      }
    }
    throw instrumentError(
      "TOKEN_REPLAY_UNAVAILABLE",
      changesCapture
        ? "The loaded browser runtime cannot recompute this token with the requested replay settings"
        : `No captured ${family} reading exists for this token, and browser token replay is not available`,
    );
  }

  private async validateLensToken(args: unknown[]): Promise<{ word: string; token_id: number }> {
    const word = nonemptyString(args[0], "J-lens token");
    if (this.jlensResolver !== null) {
      await this.compiler.prepareJlensWords([word], this.jlensResolver);
    }
    return this.compiler.validateLensToken(word);
  }

  private validateSaeFeature(args: unknown[]): ReturnType<BrowserFeasibilityCorePackCompiler["validateSaeFeature"]> {
    return this.compiler.validateSaeFeature(nonnegativeInteger(args[0], "SAE feature"));
  }

  private async saeFeaturesMetadata(args: unknown[]): Promise<{
    features: Record<string, { label: string | null; max_act: number | null }>;
  }> {
    if (!Array.isArray(args[0])) {
      throw instrumentError("INVALID_INSTRUMENT_REQUEST", "SAE feature IDs must be a list");
    }
    const features: Record<string, { label: string | null; max_act: number | null }> = {};
    for (const value of args[0]) {
      const feature = this.compiler.validateSaeFeature(nonnegativeInteger(value, "SAE feature"));
      features[String(feature.id)] = {
        label: feature.label,
        max_act: feature.max_act,
      };
    }
    const binding = this.compiler.instrumentDescriptor().sae?.descriptionSource;
    if (binding) {
      const ids = Object.keys(features).filter(id => !features[id].label?.trim());
      const signal = AbortSignal.timeout(20_000);
      for (let start = 0; start < ids.length; start += 4) {
        await Promise.all(ids.slice(start, start + 4).map(async id => {
          features[id].label = (await loadSaeDescription(binding, Number(id), signal)).label;
        }));
      }
    }
    return { features };
  }
}

function probeRefreshRequest(probe: ProbeInfo): ProbeRequest {
  if (probe.family === "geometry") {
    return { selector: probe.manifold, name: probe.name, top_n: probe.top_n };
  }
  if (probe.family === "lens") {
    return { selector: `jlens/${probe.word}`, name: probe.name };
  }
  return { selector: `sae/${probe.feature_id}`, name: probe.name };
}

export interface TokenReadoutOptions {
  topK?: number;
  steered?: boolean;
  raw?: boolean;
  layers?: string;
}

function replayEnvelope(
  value: MeasurementsEnvelopeJSON,
  family: InstrumentFamily,
  options: TokenReadoutOptions,
  selectedLensLayers: readonly number[] | null,
): MeasurementsEnvelopeJSON {
  const instrument = value.instruments[family];
  if (instrument === undefined) {
    throw instrumentError(
      "TOKEN_REPLAY_UNAVAILABLE",
      `The replay produced no ${family} reading for this token`,
    );
  }
  const cloned = structuredClone(instrument);
  if (family === "lens") {
    const lens = cloned as NonNullable<MeasurementsEnvelopeJSON["instruments"]["lens"]>;
    if (!lens.readout) {
      throw instrumentError(
        "TOKEN_REPLAY_UNAVAILABLE",
        "The browser replay produced no J-lens readout for this token",
      );
    }
    const actualLayers = lens.readout.layers.map((row) => row.layer);
    if (
      selectedLensLayers !== null &&
      (actualLayers.length !== selectedLensLayers.length ||
        actualLayers.some((layer, index) => layer !== selectedLensLayers[index]))
    ) {
      throw instrumentError(
        "TOKEN_REPLAY_UNAVAILABLE",
        "The browser replay did not honor the requested J-lens layer set",
      );
    }
    const width = options.topK ?? Number.POSITIVE_INFINITY;
    if (
      options.topK !== undefined &&
      (lens.readout.aggregate.length !== options.topK ||
        lens.readout.layers.some((row) => row.tokens.length !== options.topK))
    ) {
      throw instrumentError(
        "TOKEN_REPLAY_UNAVAILABLE",
        `The browser replay did not honor the requested exact top-${options.topK} J-lens width`,
      );
    }
    lens.readout.layers = lens.readout.layers.map((row) => ({
      ...row,
      tokens: row.tokens.slice(0, width),
    }));
    lens.readout.aggregate = lens.readout.aggregate.slice(0, width);
  } else if (family === "sae") {
    const sae = cloned as NonNullable<MeasurementsEnvelopeJSON["instruments"]["sae"]>;
    if (options.layers !== undefined) {
      throw instrumentError(
        "INVALID_INSTRUMENT_REQUEST",
        "SAE token replay reads its installed layer and does not accept a layer selection",
      );
    }
    if (
      sae.readout && options.topK !== undefined &&
      sae.readout.features.length > options.topK
    ) {
      throw instrumentError(
        "TOKEN_REPLAY_UNAVAILABLE",
        `The browser replay exceeded the requested exact top-${options.topK} SAE width`,
      );
    }
  } else if (options.topK !== undefined || options.layers !== undefined) {
    throw instrumentError(
      "INVALID_INSTRUMENT_REQUEST",
      "Geometry token replay does not accept topK or layer selection",
    );
  }
  const readingNames = new Set(Object.keys(cloned.readings ?? {}));
  const scores = filterScores(value.scores, readingNames);
  const perLayerScores = Object.fromEntries(
    Object.entries(value.per_layer_scores ?? {}).flatMap(([layer, row]) => {
      const filtered = filterScores(row, readingNames);
      return Object.keys(filtered).length > 0 ? [[layer, filtered]] : [];
    }),
  );
  return {
    version: value.version,
    scope: "replay",
    provenance: "replayed",
    instruments: { [family]: cloned },
    ...(Object.keys(scores).length > 0 ? { scores } : {}),
    ...(Object.keys(perLayerScores).length > 0 ? { per_layer_scores: perLayerScores } : {}),
  };
}

function filterScores(
  value: Readonly<Record<string, number>> | undefined,
  names: ReadonlySet<string>,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(value ?? {}).filter(([name]) => names.has(name)),
  );
}

function parseReplayLayers(value: string | undefined): Set<number> | null {
  if (value === undefined || value.trim().toLowerCase() === "all") return null;
  const parts = value.split(",").map((part) => part.trim());
  if (parts.some((part) => !/^(0|[1-9]\d*)$/u.test(part))) {
    throw instrumentError(
      "INVALID_INSTRUMENT_REQUEST",
      "Token replay layers must be 'all' or a comma-separated list of layer numbers",
    );
  }
  return new Set(parts.map(Number));
}

function replayLayersForFamily(
  family: InstrumentFamily,
  options: TokenReadoutOptions,
  fittedLensLayers: readonly number[],
): number[] | null {
  if (family === "geometry") {
    if (options.topK !== undefined || options.layers !== undefined) {
      throw instrumentError(
        "INVALID_INSTRUMENT_REQUEST",
        "Geometry token replay does not accept topK or layer selection",
      );
    }
    return null;
  }
  if (family === "sae") {
    if (options.layers !== undefined) {
      throw instrumentError(
        "INVALID_INSTRUMENT_REQUEST",
        "SAE token replay reads its installed layer and does not accept a layer selection",
      );
    }
    return null;
  }
  const requested = parseReplayLayers(options.layers);
  const layers = requested === null
    ? [...fittedLensLayers]
    : [...requested].sort((left, right) => left - right);
  const missing = layers.filter((layer) => !fittedLensLayers.includes(layer));
  if (layers.length === 0 || missing.length > 0) {
    throw instrumentError(
      "INVALID_INSTRUMENT_REQUEST",
      missing.length > 0
        ? `J-lens replay layers ${missing.join(", ")} are not fitted in the installed pack`
        : "J-lens replay needs at least one fitted layer",
    );
  }
  return layers;
}

function liveLensLayers(value: unknown): number[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", "J-lens live layers must be a list");
  }
  const layers = value.map((layer) => nonnegativeInteger(layer, "J-lens live layer"));
  if (layers.length === 0) {
    throw instrumentError(
      "INVALID_INSTRUMENT_REQUEST",
      "J-lens live readout needs at least one fitted layer",
    );
  }
  return [...new Set(layers)].sort((left, right) => left - right);
}

function tokenReadoutOptions(value: unknown): TokenReadoutOptions {
  if (value === undefined || value === null) return {};
  if (typeof value !== "object" || Array.isArray(value)) {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", "Token replay options must be an object");
  }
  const input = value as Record<string, unknown>;
  for (const key of Object.keys(input)) {
    if (!["topK", "steered", "raw", "layers"].includes(key)) {
      throw instrumentError(
        "INVALID_INSTRUMENT_REQUEST",
        `Unknown token replay option ${JSON.stringify(key)}`,
      );
    }
  }
  if (input.topK !== undefined && (!Number.isSafeInteger(input.topK) || Number(input.topK) < 1)) {
    throw instrumentError(
      "INVALID_INSTRUMENT_REQUEST",
      "Hosted token replay topK must be a positive integer",
    );
  }
  if (
    typeof input.topK === "number" &&
    input.topK > BROWSER_EXACT_READOUT_TOP_K_CAPACITY
  ) {
    throw instrumentError(
      "EXACT_READOUT_TOP_K_EXCEEDS_CAPACITY",
      `Requested exact browser readout top-${input.topK}, but the verified GPU ABI supports at most top-${BROWSER_EXACT_READOUT_TOP_K_CAPACITY}`,
    );
  }
  if (input.steered !== undefined && typeof input.steered !== "boolean") {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", "Token replay steered must be boolean");
  }
  if (input.raw !== undefined && typeof input.raw !== "boolean") {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", "Token replay raw must be boolean");
  }
  if (input.layers !== undefined && (typeof input.layers !== "string" || input.layers.trim() === "")) {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", "Token replay layers must be a non-empty string");
  }
  return input as TokenReadoutOptions;
}

function instrumentFamily(value: unknown): "geometry" | "lens" | "sae" {
  if (value !== "geometry" && value !== "lens" && value !== "sae") {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", "Instrument family is invalid");
  }
  return value;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", `${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function nonemptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", `${label} must be a non-empty string`);
  }
  return value;
}

function nonnegativeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw instrumentError("INVALID_INSTRUMENT_REQUEST", `${label} must be a non-negative integer`);
  }
  return value as number;
}

function nullableNonnegativeInteger(value: unknown, label: string): number | null {
  if (value === undefined || value === null) return null;
  return nonnegativeInteger(value, label);
}

function nullableString(value: unknown, label: string): string | null {
  if (value === undefined || value === null) return null;
  return nonemptyString(value, label);
}

function instrumentError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}
