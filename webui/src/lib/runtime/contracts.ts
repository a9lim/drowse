import type {
  CorrelationData,
  CreateDiscoverManifoldRequest,
  CreateManifoldFromTemplateRequest,
  CreateManifoldRequest,
  CreateTemplateRequest,
  FilterMatchesJSON,
  FitManifoldRequest,
  GenerateManifoldRequest,
  InstallManifoldRequest,
  InstrumentFamily,
  InstrumentLiveState,
  LensTokenValidationJSON,
  LoomNodeJSON,
  LoomTreeJSON,
  ManifoldInfo,
  ManifoldListResponse,
  MeasurementsEnvelopeJSON,
  MergeManifoldRequest,
  NodeDiffJSON,
  PairwiseCompareResponse,
  PreparationOp,
  PreparationStatusJSON,
  ProbeGeometryResponse,
  ProbeInfo,
  ProbeListResponse,
  ProbeRequest,
  ProfileListResponse,
  RemoteManifoldInfo,
  SaeFeatureMetaResponse,
  ScoreTemplateResponse,
  SessionInfo,
  SourcesResponse,
  TemplateDetail,
  TemplateSummary,
  TranscriptLoadResponseJSON,
  VectorInfo,
  WSClientMessage,
  WSGenerateRequest,
  WSServerMessage,
  WSSubmitRequest,
} from "../types";

export const RUNTIME_PROTOCOL_VERSION = 1 as const;
export const MEBIBYTE = 1024 * 1024;

export type RuntimeMode = "http" | "browser" | "fake";
export type RuntimeLifecycle =
  | "uninitialized"
  | "checking"
  | "unloaded"
  | "loading"
  | "ready"
  | "failed";
export type RuntimeOperationPhase =
  | "idle"
  | "running"
  | "paused"
  | "cancelling"
  | "failed"
  | "complete";
export type RuntimeOperation =
  | "download"
  | "generation"
  | "fitting";
export type RuntimeCapabilityOperation =
  | RuntimeOperation
  | "manifold_artifacts"
  | "probe_subspace_trails"
  | "jlens_fitting"
  | "sae_training"
  | "session_admin"
  | "server_endpoints";

export interface RuntimeFailure {
  code: string;
  message: string;
  recoverable: boolean;
  status: number;
  detail?: unknown;
}

export interface RuntimeOperationState {
  phase: RuntimeOperationPhase;
  startedAt: number | null;
  finishedAt: number | null;
  error: RuntimeFailure | null;
}

export interface RuntimeSnapshot {
  modelDefaults?: SessionInfo;
  lifecycle: RuntimeLifecycle;
  modelVariantId: string | null;
  contextTokens: number | null;
  selectedModelVariantId: string | null;
  installedModelVariantIds: string[];
  installedPackIds: string[];
  loadRecords: DeviceLoadRecord[];
  download: RuntimeOperationState;
  generation: RuntimeOperationState;
  fitting: RuntimeOperationState;
  error: RuntimeFailure | null;
}

export type CapabilitySeverity = "hard" | "advisory";

export interface CapabilityIssue {
  code: string;
  message: string;
  severity: CapabilitySeverity;
}

export interface OperationAvailability {
  available: boolean;
  reasons: CapabilityIssue[];
}

export type FallbackAdapterStatus = "hardware" | "fallback" | "unknown";
export type BrowserRuntimeClass =
  | "apple-mobile-webkit"
  | "desktop-webkit"
  | "desktop-gecko"
  | "android-chromium"
  | "desktop-chromium"
  | "other";

export interface WebGpuCapabilityFacts {
  available: boolean;
  fallback: FallbackAdapterStatus;
  features: string[];
  limits: Record<string, number>;
  adapterInfo: {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
  } | null;
}

export interface RuntimeCapabilities {
  checkedAt: number;
  supported: boolean;
  secureContext: boolean;
  crossOriginIsolated: boolean;
  dedicatedWorker: boolean;
  indexedDb: boolean;
  opfs: boolean;
  webLocks: boolean;
  broadcastChannel: boolean;
  webAssembly: boolean;
  webGpu: WebGpuCapabilityFacts;
  storage: {
    quotaBytes: number | null;
    usageBytes: number | null;
    availableBytes: number | null;
    persisted: boolean | null;
  };
  signals: {
    deviceMemoryGiB: number | null;
    logicalCpuCount: number | null;
    mobile: boolean;
    runtimeClass?: BrowserRuntimeClass;
    appleMobile?: boolean;
    language: string;
    calibrationScore: number | null;
  };
  deviceSignature: string;
  runtimeVersion: string;
  issues: CapabilityIssue[];
  operations: Record<RuntimeCapabilityOperation, OperationAvailability>;
  limits: {
    manifoldFitMaxIntrinsicDim: number | null;
  };
}

export type ModelTier = "fastest" | "balanced" | "quality";
export type ModelPreference = "speed" | "balanced" | "quality";
export type StructuredHookProfileId =
  | "standard-v1"
  | "standard-v2"
  | "standard-v3";

export interface RuntimeIdentity {
  sourceModel: string;
  sourceRevision: string;
  convertedManifestSha256: string;
  quantization: string;
  tokenizerSha256: string;
  chatTemplateSha256: string;
  modelLibrarySha256: string;
  runtimeAbi: string;
  hookAbi: string;
  hiddenSize: number;
  layerMap: number[];
}

export type CatalogFileRole =
  | "weight"
  | "tokenizer"
  | "configuration"
  | "converted_manifest"
  | "chat_template"
  | "model_library"
  | "core_pack"
  | "instrument";

export interface CatalogFile {
  path: string;
  role: CatalogFileRole;
  url: string;
  revision: string;
  bytes: number;
  sha256: string;
}

export interface ModelContextProfile {
  contextTokens: number;
  bindingSha256: string;
  minimumCalibrationScore: number | null;
  minimumDeviceMemoryGiB: number | null;
  expectedPrefillTokensPerSecond: [number, number] | null;
  expectedDecodeTokensPerSecond: [number, number] | null;
  measuredDevices: number;
}

export interface ModelVariantRequirements {
  features: string[];
  limits: Record<string, number>;
}

export interface ModelThinkingProfile {
  start: string;
  end: string;
  startsInThinking: boolean;
  startTokenIds: number[];
  endTokenIds: number[];
}

export interface CatalogInstrumentPack {
  id: string;
  kind: "core" | "jlens" | "sae";
  displayName: string;
  license: string;
  sourceRepository: string;
  sourceRevision: string;
  bytes: number;
  required: boolean;
  runtimeIdentitySha256: string;
  compatibleContextBindingSha256: string[];
  files: CatalogFile[];
}

export interface ModelVariant {
  id: string;
  tier: ModelTier;
  structuredHookProfile: StructuredHookProfileId;
  thinkingProfile: ModelThinkingProfile | null;
  contextProfiles: ModelContextProfile[];
  downloadBytes: number;
  requiredCorePackBytes: number;
  requirements: ModelVariantRequirements;
  runtimeIdentity: RuntimeIdentity;
  runtimeIdentitySha256: string;
  files: CatalogFile[];
  packs: CatalogInstrumentPack[];
}

export interface CatalogModel {
  id: string;
  modelType?: "chat" | "base";
  displayName: string;
  description: string;
  sourceUrl: string;
  license: string;
  languages: string[];
  variants: ModelVariant[];
}

export interface CatalogDocument {
  schemaVersion: 1;
  sequence: number;
  issuedAt: string;
  expiresAt: string;
  runtimeAbi: string;
  models: CatalogModel[];
}

export interface DetachedCatalogSignature {
  schemaVersion: 1;
  algorithm: "Ed25519";
  keyId: string;
  signature: string;
}

export interface VerifiedCatalog {
  document: CatalogDocument;
  exactBytes: Uint8Array;
  signature: DetachedCatalogSignature;
  allowDownloads: boolean;
  stale: boolean;
}

export interface CatalogWorkerRequest {
  refresh?: boolean;
  offline?: boolean;
  preferCached?: boolean;
}

export interface DeviceLoadRecord {
  modelVariantId: string;
  runtimeIdentitySha256: string;
  deviceSignature: string;
  contextTokens: number;
  result: "success" | "oom" | "device_lost" | "failed";
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
  recordedAt: number;
}

export interface ModelAssessment {
  model: CatalogModel;
  variant: ModelVariant;
  context: ModelContextProfile | null;
  installed: boolean;
  eligible: boolean;
  proven: boolean;
  hardFailures: CapabilityIssue[];
  advisories: CapabilityIssue[];
  requiredStorageBytes: number;
}

export interface ModelRecommendation extends ModelAssessment {
  recommended: boolean;
  reason: string;
}

export type DownloadVerificationState =
  | "pending"
  | "hashing"
  | "verified"
  | "failed";

export interface DownloadFileProgress {
  path: string;
  bytesReceived: number;
  bytesTotal: number;
  resumable: boolean;
  verification: DownloadVerificationState;
}

export interface DownloadProgress {
  modelVariantId: string;
  packId?: string;
  files: DownloadFileProgress[];
  bytesReceived: number;
  bytesTotal: number;
  throughputBytesPerSecond: number | null;
  etaSeconds: [number, number] | null;
  calculatingEta: boolean;
  stalled: boolean;
  offline: boolean;
  resumable: boolean;
}

export interface DownloadWorkerResult {
  modelVariantId: string;
  installed: boolean;
  cancelled: boolean;
}

export interface PackDownloadWorkerResult extends DownloadWorkerResult {
  packId: string;
}

export interface RuntimeProgressEvent {
  event: string;
  data: unknown;
}

export type RuntimeEventChannelState =
  | { state: "open" }
  | {
      state: "closed";
      expected: boolean;
      reason: string | null;
    };

export interface RuntimeEventChannel {
  readonly isOpen: boolean;
  open(): Promise<void>;
  subscribe(listener: (message: WSServerMessage) => void): () => void;
  subscribeState(listener: (state: RuntimeEventChannelState) => void): () => void;
  send(message: WSClientMessage): void;
  stop(): Promise<void>;
  acknowledgeSnapshot(): void;
  close(): void;
}

export interface RuntimeSessionsService {
  list(): Promise<{ sessions: SessionInfo[] }>;
  get(id?: string): Promise<SessionInfo>;
  patch(
    body: Partial<{
      temperature: number;
      top_p: number;
      top_k: number | null;
      max_tokens: number;
      system_prompt: string;
      thinking: boolean;
    }>,
    id?: string,
  ): Promise<SessionInfo>;
  validateSteering(
    expression: string,
    id?: string,
    options?: SteeringValidationOptions,
  ): Promise<{ valid: boolean; expression: string; error: string | null }>;
}

export interface SteeringValidationOptions {
  /** Prospective probe roster used by an import.  Browser validation stages
   * this roster only for compilation; the live attached set is unchanged. */
  probeRequests?: readonly ProbeRequest[];
  /** Prospective tree used by an import.  Browser validation runs the same
   * structural/session checks as restore without replacing the live tree. */
  tree?: LoomTreeJSON;
}

export interface RuntimeProfilesService {
  list(id?: string): Promise<ProfileListResponse>;
  get(name: string, id?: string): Promise<VectorInfo>;
  correlation(names?: string[] | null, id?: string): Promise<CorrelationData>;
  pairwise(a: string, b: string, id?: string): Promise<PairwiseCompareResponse>;
  extract(
    request: import("../types").ExtractRequest,
    onEvent: (event: RuntimeProgressEvent) => void,
    id?: string,
  ): Promise<{ canonical: string; profile: VectorInfo }>;
}

export interface RuntimeProbesService {
  list(id?: string): Promise<ProbeListResponse>;
  attach(request: ProbeRequest, id?: string): Promise<ProbeInfo>;
  detach(name: string, id?: string): Promise<void>;
  geometry(name: string, id?: string): Promise<ProbeGeometryResponse>;
}

export interface RuntimeManifoldsService {
  inspectSurface?(points: number[][]): Promise<import("../types.gen").SurfaceEvidence>;
  list(): Promise<ManifoldListResponse>;
  get(namespace: string, name: string): Promise<ManifoldInfo>;
  create(request: CreateManifoldRequest): Promise<ManifoldInfo & { advisories: string[] }>;
  createDiscover(request: CreateDiscoverManifoldRequest): Promise<ManifoldInfo>;
  createFromTemplate(request: CreateManifoldFromTemplateRequest): Promise<ManifoldInfo>;
  delete(namespace: string, name: string): Promise<{ namespace: string; name: string; removed: boolean }>;
  search(query: string, limit?: number): Promise<{ results: RemoteManifoldInfo[] }>;
  install(
    request: InstallManifoldRequest,
    onEvent?: (event: RuntimeProgressEvent) => void,
  ): Promise<ManifoldInfo>;
  merge(request: MergeManifoldRequest): Promise<ManifoldInfo>;
  fit(
    namespace: string,
    name: string,
    request: FitManifoldRequest,
    onEvent: (event: RuntimeProgressEvent) => void,
  ): Promise<ManifoldInfo>;
  generate(
    request: GenerateManifoldRequest,
    onEvent: (event: RuntimeProgressEvent) => void,
  ): Promise<ManifoldInfo>;
  drowseArchiveList(): Promise<{ packs: HostedManifoldPackInfo[] }>;
  drowseArchiveInstall(
    source: Blob,
    options: { force?: boolean },
    onEvent: (event: RuntimeProgressEvent) => void,
  ): Promise<HostedManifoldPackInfo>;
  drowseArchiveExport(primary: string): Promise<Blob>;
  drowseArchiveDelete(primary: string): Promise<{ id: string; removed: boolean }>;
}

export interface HostedManifoldPackInfo {
  id: string;
  namespace: string;
  name: string;
  template: string | null;
  installedAt: number;
  producerVersion: string;
  source: {
    uri: string;
    repository: string | null;
    revision: string | null;
  };
}

export interface RuntimeTemplatesService {
  list(): Promise<{ templates: TemplateSummary[] }>;
  get(namespace: string, name: string): Promise<TemplateDetail>;
  create(request: CreateTemplateRequest): Promise<TemplateDetail>;
  delete(namespace: string, name: string): Promise<{ namespace: string; name: string; removed: boolean }>;
  score(namespace: string, name: string, steering: string | null): Promise<ScoreTemplateResponse>;
}

export interface RuntimeTreeService {
  replayCapabilities(): Promise<RuntimeTreeReplayCapabilities>;
  get(id?: string): Promise<LoomTreeJSON>;
  reset(id?: string): Promise<void>;
  restore(tree: LoomTreeJSON, id?: string): Promise<{ rev: number; root_id: string; active_node_id: string; nodes: number }>;
  active(id?: string): Promise<{ active_node_id: string; rev: number; messages: { role: string; content: string }[]; node_ids: string[] }>;
  navigate(nodeId: string, id?: string): Promise<{ active_node_id: string; rev: number; messages: { role: string; content: string }[]; node_ids: string[] }>;
  edit(nodeId: string, text: string, id?: string): Promise<LoomNodeJSON>;
  branch(nodeId: string, text: string, id?: string, role?: "user" | "assistant" | null): Promise<{ node_id: string; node: LoomNodeJSON; active_path: { active_node_id: string; rev: number; messages: { role: string; content: string }[]; node_ids: string[] } }>;
  delete(nodeId: string, id?: string): Promise<{ removed: number }>;
  star(nodeId: string, on: boolean, id?: string): Promise<LoomNodeJSON>;
  note(nodeId: string, text: string, id?: string): Promise<LoomNodeJSON>;
  edgeLabel(parentId: string, childId: string, id?: string): Promise<{ label: string }>;
  filter(expression: string, id?: string): Promise<FilterMatchesJSON>;
  diff(aId: string, bId: string, id?: string): Promise<NodeDiffJSON>;
  transcriptExport(nodeId: string | null, id?: string): Promise<{ yaml: string; node_id: string }>;
  transcriptLoad(yaml: string, mode: "default" | "here" | "merge", strict: boolean, id?: string): Promise<TranscriptLoadResponseJSON>;
  jointLogprobs(aId: string, bId: string, id?: string): Promise<import("../types").JointLogprobsJSON>;
  cast(id?: string): Promise<{ cast: Record<string, import("../types").CastMemberJSON> }>;
  castPut(label: string, body: { steering?: string | null; thinking?: boolean | null; seed?: number | null; notes?: string }, id?: string): Promise<{ label: string; member: import("../types").CastMemberJSON }>;
  castDelete(label: string, id?: string): Promise<void>;
}

export interface RuntimeTreeReplayCapabilities {
  jointLogprobs: {
    available: boolean;
    reason: string | null;
  };
}

export interface RuntimeInstrumentsService {
  setLive(family: InstrumentFamily, body: { enabled: boolean; layers?: number[] | null }, id?: string): Promise<InstrumentLiveState>;
  sources(family: InstrumentFamily, id?: string): Promise<SourcesResponse>;
  setLensSource(source: string, id?: string): Promise<{ source: string; live_layers: number[] }>;
  /**
   * Activate an exact instrument pack that is already resident in the loaded
   * runtime. Hosted mode never downloads through this service: optional pack
   * installation belongs to HostedController, and a newly installed/deleted
   * pack takes effect on the next model load.
   */
  activateInstalledPack(
    family: Exclude<InstrumentFamily, "geometry">,
    body: InstalledInstrumentPackActivationRequest,
    id?: string,
  ): Promise<InstalledInstrumentPackActivationResult>;
  startPreparation(family: InstrumentFamily, body: { operation: PreparationOp } & Record<string, unknown>, id?: string): Promise<PreparationStatusJSON>;
  preparationStatus(family: InstrumentFamily, id?: string): Promise<PreparationStatusJSON>;
  cancelPreparation(family: InstrumentFamily, id?: string): Promise<PreparationStatusJSON>;
  tokenReadout(
    family: InstrumentFamily,
    nodeId: string,
    rawIndex: number,
    options?: { topK?: number; steered?: boolean; raw?: boolean; layers?: string },
    id?: string,
    onProgress?: (event: RuntimeProgressEvent) => void,
  ): Promise<{ measurements: MeasurementsEnvelopeJSON }>;
  validateLensToken(word: string, id?: string): Promise<LensTokenValidationJSON>;
  validateSaeFeature(featureId: number, id?: string): Promise<{ id: number; label?: string | null; layer: number; max_act?: number | null }>;
  saeFeaturesMetadata(ids: number[], id?: string): Promise<SaeFeatureMetaResponse>;
}

export interface InstalledInstrumentPackActivationRequest {
  source: string;
  layer?: number | null;
  contextBindingSha256?: string | null;
}

export interface InstalledInstrumentPackActivationResult {
  state: "active";
  family: "lens" | "sae";
  source: string;
  live: InstrumentLiveState;
  contextBindingSha256: string | null;
  reloadRequired: false;
}

export interface RuntimeClient {
  readonly mode: RuntimeMode;
  readonly sessions: RuntimeSessionsService;
  readonly profiles: RuntimeProfilesService;
  readonly probes: RuntimeProbesService;
  readonly manifolds: RuntimeManifoldsService;
  readonly templates: RuntimeTemplatesService;
  readonly tree: RuntimeTreeService;
  readonly instruments: RuntimeInstrumentsService;
  readonly events: RuntimeEventChannel;
  dispose(): Promise<void>;
}

export interface RuntimeServiceMap {
  sessions: RuntimeSessionsService;
  profiles: RuntimeProfilesService;
  probes: RuntimeProbesService;
  manifolds: RuntimeManifoldsService;
  templates: RuntimeTemplatesService;
  tree: RuntimeTreeService;
  instruments: RuntimeInstrumentsService;
}

export type RuntimeServiceName = keyof RuntimeServiceMap;

export const RUNTIME_SERVICE_METHODS = Object.freeze({
  sessions: Object.freeze({
    list: true,
    get: true,
    patch: true,
    validateSteering: true,
  }),
  profiles: Object.freeze({
    list: true,
    get: true,
    correlation: true,
    pairwise: true,
    extract: true,
  }),
  probes: Object.freeze({
    list: true,
    attach: true,
    detach: true,
    geometry: true,
  }),
  manifolds: Object.freeze({
    inspectSurface: true,
    list: true,
    get: true,
    create: true,
    createDiscover: true,
    createFromTemplate: true,
    delete: true,
    search: true,
    install: true,
    merge: true,
    fit: true,
    generate: true,
    drowseArchiveList: true,
    drowseArchiveInstall: true,
    drowseArchiveExport: true,
    drowseArchiveDelete: true,
  }),
  templates: Object.freeze({
    list: true,
    get: true,
    create: true,
    delete: true,
    score: true,
  }),
  tree: Object.freeze({
    replayCapabilities: true,
    get: true,
    reset: true,
    restore: true,
    active: true,
    navigate: true,
    edit: true,
    branch: true,
    delete: true,
    star: true,
    note: true,
    edgeLabel: true,
    filter: true,
    diff: true,
    transcriptExport: true,
    transcriptLoad: true,
    jointLogprobs: true,
    cast: true,
    castPut: true,
    castDelete: true,
  }),
  instruments: Object.freeze({
    setLive: true,
    sources: true,
    setLensSource: true,
    activateInstalledPack: true,
    startPreparation: true,
    preparationStatus: true,
    cancelPreparation: true,
    tokenReadout: true,
    validateLensToken: true,
    validateSaeFeature: true,
    saeFeaturesMetadata: true,
  }),
}) satisfies {
  [Service in RuntimeServiceName]: Readonly<
    Record<Extract<keyof RuntimeServiceMap[Service], string>, true>
  >;
};

export function isRuntimeServiceMethod(
  service: RuntimeServiceName,
  method: string,
): method is Extract<keyof RuntimeServiceMap[typeof service], string> {
  return Object.hasOwn(RUNTIME_SERVICE_METHODS[service], method);
}

export interface HostedController {
  readonly snapshot: RuntimeSnapshot;
  subscribe(listener: (snapshot: RuntimeSnapshot) => void): () => void;
  check(): Promise<RuntimeCapabilities>;
  catalog(request?: CatalogWorkerRequest): Promise<VerifiedCatalog>;
  recommend(
    catalog: VerifiedCatalog,
    preference: ModelPreference,
    contextTokens?: number,
    language?: string,
    options?: { explicitOomRetry?: boolean },
  ): ModelRecommendation[];
  download(
    modelVariantId: string,
    onProgress?: (progress: DownloadProgress) => void,
    options?: { contextTokens?: number; explicitUnsafeOverride?: boolean },
  ): Promise<DownloadWorkerResult>;
  downloadPack(
    modelVariantId: string,
    packId: string,
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<PackDownloadWorkerResult>;
  cancelDownload(): Promise<void>;
  cancelFitting(): Promise<void>;
  deleteModel(modelVariantId: string): Promise<void>;
  deletePack(packId: string): Promise<void>;
  clearAll(): Promise<void>;
  requestPersistence(onLateGranted?: () => void): Promise<boolean>;
  refreshStorage(): Promise<RuntimeCapabilities["storage"]>;
  load(
    modelVariantId: string,
    contextTokens: number,
    options?: { explicitUnsafeOverride?: boolean; resetSession?: boolean },
  ): Promise<void>;
  unload(): Promise<void>;
  dispose(): Promise<void>;
}

export type WorkerCommandName =
  | "check"
  | "catalog"
  | "download"
  | "download_pack"
  | "cancel"
  | "cancel_fitting"
  | "delete"
  | "clear"
  | "persist"
  | "storage"
  | "load"
  | "unload"
  | "takeover"
  | "request"
  | "submit"
  | "generate"
  | "stop";

export interface RuntimeServiceRequest {
  service: RuntimeServiceName;
  method: string;
  args: unknown[];
}

type WorkerCommandPayloads = {
  check: undefined;
  catalog: CatalogWorkerRequest;
  download: {
    modelVariantId: string;
    contextTokens: number;
    explicitUnsafeOverride?: boolean;
  };
  download_pack: {
    modelVariantId: string;
    packId: string;
  };
  cancel: undefined;
  cancel_fitting: undefined;
  delete: { kind: "model" | "pack"; id: string };
  clear: undefined;
  persist: undefined;
  storage: undefined;
  load: {
    modelVariantId: string;
    contextTokens: number;
    explicitUnsafeOverride?: boolean;
    resetSession?: boolean;
  };
  unload: undefined;
  takeover: undefined;
  request: RuntimeServiceRequest;
  submit: WSSubmitRequest;
  generate: WSGenerateRequest;
  stop: undefined;
};

export type WorkerRequest = {
  [K in WorkerCommandName]: {
    protocolVersion: typeof RUNTIME_PROTOCOL_VERSION;
    requestId: string;
    command: K;
    payload: WorkerCommandPayloads[K];
  };
}[WorkerCommandName];

export interface WorkerSuccessResponse {
  protocolVersion: typeof RUNTIME_PROTOCOL_VERSION;
  kind: "response";
  requestId: string;
  ok: true;
  result: unknown;
}

export interface WorkerErrorResponse {
  protocolVersion: typeof RUNTIME_PROTOCOL_VERSION;
  kind: "response";
  requestId: string;
  ok: false;
  error: RuntimeFailure;
}

export type WorkerEventName =
  | "status"
  | "progress"
  | "started"
  | "generation_progress"
  | "token"
  | "done"
  | "tree_mutated"
  | "device_lost"
  | "error";

export type WorkerFailureScope =
  | "lifecycle"
  | "download"
  | "generation"
  | "fitting"
  | "transport";

export interface WorkerScopedFailure {
  scope: WorkerFailureScope;
  failure: RuntimeFailure;
}

export interface WorkerEventEnvelope {
  protocolVersion: typeof RUNTIME_PROTOCOL_VERSION;
  kind: "event";
  requestId: string | null;
  sequence: number;
  generationId: string | null;
  event: WorkerEventName;
  payload:
    | RuntimeSnapshot
    | DownloadProgress
    | WSServerMessage
    | RuntimeFailure
    | WorkerScopedFailure;
}

export type WorkerMessage =
  | WorkerSuccessResponse
  | WorkerErrorResponse
  | WorkerEventEnvelope;
