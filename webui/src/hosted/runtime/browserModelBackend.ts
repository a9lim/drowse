import {
  type BrowserModelLoadRequest,
  type BrowserModelBackend,
} from "./modelBackend";
import type {
  BrowserManifoldArtifactPort,
  BrowserManifoldFittingDependencies,
} from "./browserManifoldFitting";
import { BrowserManifoldFitting } from "./browserManifoldFitting";
import { BrowserFeasibilityCorePackCompiler } from "./browserCorePack";
import { DrowseWebLlmBackend } from "./drowseWebLlmBackend";
import { DrowseWebLlmRuntime, type DrowseWebLlmModule } from "./webLlmEngine";
import { BrowserInstrumentRuntime } from "./browserInstrumentRuntime";
import { browserArtifactProducer } from "./buildProvenance";

export const BROWSER_MODEL_BACKEND_INTEGRATED = true;

export function compilerBackedFittingDependencies(
  compilerFor: (
    loadRequest: BrowserModelLoadRequest,
  ) => Promise<BrowserFeasibilityCorePackCompiler>,
): Pick<BrowserManifoldFittingDependencies, "loadWhiteners" | "loadExactSae" | "validateFittedPack"> {
  return {
    async loadWhiteners(loadRequest) {
      return (await compilerFor(loadRequest)).fittingWhiteners();
    },
    async loadExactSae(loadRequest, selector) {
      return (await compilerFor(loadRequest)).exactSaeFitting(selector);
    },
    async validateFittedPack(loadRequest, archive) {
      await (await compilerFor(loadRequest)).withInstalledManifolds([archive], loadRequest.signal);
    },
  };
}

export async function prepareBrowserInstrumentDictionaries(
  request: BrowserModelLoadRequest,
  compiler: Pick<
    BrowserFeasibilityCorePackCompiler,
    "saeGpuDictionary" | "jlensGpuDictionary"
  >,
  runtime: Pick<
    DrowseWebLlmRuntime,
    | "setSaeDictionary"
    | "setJlensDictionary"
    | "warmSaeDictionary"
    | "warmJlensDictionary"
  >,
): Promise<void> {
  request.signal.throwIfAborted();
  const saeDictionary = compiler.saeGpuDictionary();
  const jlensDictionary = compiler.jlensGpuDictionary();
  const hasDictionaries = saeDictionary !== null || jlensDictionary !== null;
  if (hasDictionaries) {
    reportLoadProgress(
      request,
      "instrument_dictionaries_staging",
      "Preparing insight data for the GPU",
    );
  }
  await runtime.setSaeDictionary(saeDictionary);
  request.signal.throwIfAborted();
  await runtime.setJlensDictionary(jlensDictionary);
  request.signal.throwIfAborted();
  if (hasDictionaries) {
    reportLoadProgress(
      request,
      "instrument_dictionaries_staged",
      "Insight data prepared",
    );
  }
  if (saeDictionary !== null) {
    reportLoadProgress(
      request,
      "sae_dictionary_uploading",
      "Loading SAE features onto the GPU",
    );
    await runtime.warmSaeDictionary();
    request.signal.throwIfAborted();
    reportLoadProgress(request, "sae_dictionary_ready", "SAE features ready");
  }
  if (jlensDictionary !== null) {
    reportLoadProgress(
      request,
      "jlens_dictionary_uploading",
      "Loading J-lens insights onto the GPU",
    );
    await runtime.warmJlensDictionary();
    request.signal.throwIfAborted();
    reportLoadProgress(request, "jlens_dictionary_ready", "J-lens insights ready");
  }
}

export function createBrowserModelBackend(
  module: DrowseWebLlmModule,
  artifacts: BrowserManifoldArtifactPort,
): BrowserModelBackend {
  return createDrowseBrowserModelBackend(module, artifacts);
}

export function createLazyBrowserModelBackend(
  loadModule: () => Promise<DrowseWebLlmModule>,
  artifacts: BrowserManifoldArtifactPort,
): BrowserModelBackend {
  let backend: BrowserModelBackend | null = null;
  let pending: Promise<BrowserModelBackend> | null = null;
  const getBackend = (): Promise<BrowserModelBackend> => {
    if (backend !== null) return Promise.resolve(backend);
    if (pending !== null) return pending;
    pending = loadModule()
      .then((module) => {
        backend = createDrowseBrowserModelBackend(module, artifacts);
        return backend;
      })
      .catch((error) => {
        pending = null;
        throw error;
      });
    return pending;
  };
  return {
    async load(request) {
      return (await getBackend()).load(request);
    },
    async unload() {
      if (backend !== null) await backend.unload();
      else if (pending !== null) await (await pending).unload();
    },
    async refreshManifolds(signal) {
      await (await getBackend()).refreshManifolds(signal);
    },
    async request(request, onProgress, signal) {
      return (await getBackend()).request(request, onProgress, signal);
    },
    async generate(request, emit, onAdmitted) {
      return (await getBackend()).generate(request, emit, onAdmitted);
    },
    async stop(reason) {
      if (backend !== null) await backend.stop(reason);
      else if (pending !== null) await (await pending).stop(reason);
    },
  };
}

export function createDrowseBrowserModelBackend(
  module: DrowseWebLlmModule,
  artifacts: BrowserManifoldArtifactPort,
): BrowserModelBackend {
  const producer = browserArtifactProducer();
  const runtime = new DrowseWebLlmRuntime(module);
  type CompilerEntry = {
    artifactSignature: string;
    compiler: BrowserFeasibilityCorePackCompiler;
  };
  const compilers = new WeakMap<BrowserModelLoadRequest, CompilerEntry>();
  let activeLoadRequest: BrowserModelLoadRequest | null = null;
  let activeCompiler: CompilerEntry | null = null;
  let activeInstruments: BrowserInstrumentRuntime | null = null;
  const loadCompiler = async (
    loadRequest: BrowserModelLoadRequest,
    signal: AbortSignal,
  ): Promise<CompilerEntry> => {
    const installed = await installedManifoldIndex(artifacts, signal);
    const request = signal === loadRequest.signal ? loadRequest : { ...loadRequest, signal };
    return withInstalledManifoldArchives(artifacts, installed.ids, signal, async (archives) => ({
      artifactSignature: installed.signature,
      compiler: await BrowserFeasibilityCorePackCompiler.load(request, {
        installedManifoldArchives: archives,
      }),
    }));
  };
  const compilerEntryFor = async (
    loadRequest: BrowserModelLoadRequest,
  ): Promise<CompilerEntry> => {
    if (loadRequest === activeLoadRequest && activeCompiler !== null) {
      return activeCompiler;
    }
    let cached = compilers.get(loadRequest);
    if (cached === undefined) {
      cached = await loadCompiler(loadRequest, loadRequest.signal);
      compilers.set(loadRequest, cached);
    }
    return cached;
  };
  const compilerFor = async (
    loadRequest: BrowserModelLoadRequest,
  ): Promise<BrowserFeasibilityCorePackCompiler> => {
    return (await compilerEntryFor(loadRequest)).compiler;
  };
  const refreshCompiler = async (
    loadRequest: BrowserModelLoadRequest,
    signal: AbortSignal,
  ): Promise<void> => {
    if (
      loadRequest !== activeLoadRequest || activeCompiler === null ||
      activeInstruments === null
    ) {
      throw backendError(
        "STEERING_COMPILER_UNAVAILABLE",
        "The verified core pack has not been loaded for this browser session",
      );
    }
    const installed = await installedManifoldIndex(artifacts, signal);
    const compiler = activeCompiler.compiler;
    const next = await withInstalledManifoldArchives(artifacts, installed.ids, signal, async (archives) => ({
      artifactSignature: installed.signature,
      compiler: await compiler.withInstalledManifolds(archives, signal),
    }));
    signal.throwIfAborted();
    await activeInstruments.replaceCompiler(next.compiler);
    activeCompiler = next;
    compilers.set(loadRequest, next);
  };
  const fitting = new BrowserManifoldFitting(
    artifacts,
    producer,
    {
      ...compilerBackedFittingDependencies(compilerFor),
      allowAutomaticTopologyDiscovery: false,
    },
  );
  return new DrowseWebLlmBackend(module, {
    drowseVersion: producer.drowseVersion,
    runtime,
    async prepareLoad(loadRequest) {
      reportLoadProgress(
        loadRequest,
        "core_pack_preparing",
        "Preparing response controls and insight files",
      );
      const entry = await compilerEntryFor(loadRequest);
      loadRequest.signal.throwIfAborted();
      reportLoadProgress(loadRequest, "core_pack_ready", "Response controls ready");
      await prepareBrowserInstrumentDictionaries(loadRequest, entry.compiler, runtime);
      const instruments = new BrowserInstrumentRuntime(
        entry.compiler,
        runtime,
        loadRequest.variant.contextProfiles.find(
          (profile) => profile.contextTokens === loadRequest.contextTokens,
        )?.bindingSha256 ?? null,
        loadRequest.runtimeClass !== "apple-mobile-webkit" &&
          loadRequest.runtimeClass !== "desktop-webkit" &&
          loadRequest.runtimeClass !== "desktop-gecko",
      );
      activeLoadRequest = loadRequest;
      activeCompiler = entry;
      activeInstruments = instruments;
      return instruments;
    },
    onUnload() {
      fitting.dispose();
      activeLoadRequest = null;
      activeCompiler = null;
      activeInstruments = null;
    },
    async compileSteering({ expression, loadRequest, probeRequests }) {
      const compiler = await compilerFor(loadRequest);
      await compiler.prepareJlensSelectors(expression, runtime);
      if (probeRequests !== undefined) {
        const words = probeRequests.flatMap((request) => {
          const selector = request.selector.trim();
          return selector.startsWith("jlens/")
            ? [selector.slice("jlens/".length)]
            : [];
        });
        await compiler.prepareJlensWords(words, runtime);
      }
      return compiler.compile(expression, probeRequests);
    },
    steeringDelta({ parent, child, loadRequest }) {
      const cached = loadRequest === activeLoadRequest ? activeCompiler : compilers.get(loadRequest);
      if (!cached) {
        throw backendError(
          "STEERING_COMPILER_UNAVAILABLE",
          "The verified core pack has not been loaded for this browser session",
        );
      }
      return cached.compiler.steeringDelta(parent, child);
    },
    authoringRequest: (context) => fitting.request(context),
    async refreshAfterAuthoring(context) {
      if (!refreshesInstalledManifolds(context.request)) return;
      await refreshCompiler(context.loadRequest, context.signal);
    },
    refreshManifolds: refreshCompiler,
  });
}

function reportLoadProgress(
  request: BrowserModelLoadRequest,
  phase: string,
  message: string,
): void {
  request.onProgress?.({
    event: "progress",
    data: { kind: "model_load", phase, message },
  });
}

function refreshesInstalledManifolds(request: { service: string; method: string }): boolean {
  return request.service === "manifolds" && request.method === "fit" ||
    request.service === "profiles" && request.method === "extract";
}

async function installedManifoldIndex(
  artifacts: BrowserManifoldArtifactPort,
  signal: AbortSignal,
): Promise<{ signature: string; ids: string[] }> {
  signal.throwIfAborted();
  const result = await artifacts.request(
    { service: "manifolds", method: "drowseArchiveList", args: [] },
    () => undefined,
  );
  if (!isRecord(result) || !Array.isArray(result.packs)) {
    throw backendError("ARTIFACT_REPOSITORY_INVALID", "The local manifold repository returned an invalid index");
  }
  const packs = result.packs.map((value) => {
    if (
      !isRecord(value) || typeof value.id !== "string" ||
      !Number.isSafeInteger(value.installedAt) || Number(value.installedAt) < 0
    ) {
      throw backendError("ARTIFACT_REPOSITORY_INVALID", "The local manifold repository returned an invalid entry");
    }
    return { id: value.id, installedAt: Number(value.installedAt) };
  }).sort((left, right) => left.id.localeCompare(right.id));
  return { signature: JSON.stringify(packs), ids: packs.map((pack) => pack.id) };
}

async function installedManifoldArchives(
  artifacts: BrowserManifoldArtifactPort,
  ids: readonly string[],
  signal: AbortSignal,
): Promise<Blob[]> {
  const archives: Blob[] = [];
  for (const id of ids) {
    signal.throwIfAborted();
    const archive = await artifacts.request(
      { service: "manifolds", method: "drowseArchiveExport", args: [id] },
      () => undefined,
    );
    if (!(archive instanceof Blob)) {
      throw backendError(
        "ARTIFACT_REPOSITORY_INVALID",
        `The local manifold repository could not read ${id}`,
      );
    }
    archives.push(archive);
  }
  return archives;
}

export async function withInstalledManifoldArchives<T>(
  artifacts: BrowserManifoldArtifactPort,
  ids: readonly string[],
  signal: AbortSignal,
  consumer: (archives: readonly Blob[]) => Promise<T>,
): Promise<T> {
  if (artifacts.withDrowseArchiveArchives !== undefined) {
    return artifacts.withDrowseArchiveArchives(ids, signal, consumer);
  }
  return consumer(await installedManifoldArchives(artifacts, ids, signal));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function backendError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}
