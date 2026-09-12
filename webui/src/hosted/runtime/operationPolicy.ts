import {
  isRuntimeServiceMethod,
  type RuntimeCapabilityOperation,
  type RuntimeServiceMap,
  type RuntimeServiceName,
  type RuntimeServiceRequest,
} from "../../lib/runtime/contracts";

export type RuntimeServiceRoute =
  | "artifact"
  | "hybrid_manifold_read"
  | "fitting"
  | "model";

export interface RuntimeServiceOperationPolicy {
  lifecycle: "ordinary" | "fitting";
  requiredCapabilities: readonly RuntimeCapabilityOperation[];
}

export const RUNTIME_SERVICE_ROUTES = Object.freeze({
  sessions: Object.freeze({
    operationStatus: "model",
    list: "model",
    get: "model",
    patch: "model",
    validateSteering: "model",
  }),
  profiles: Object.freeze({
    list: "model",
    get: "model",
    correlation: "model",
    pairwise: "model",
    extract: "fitting",
  }),
  probes: Object.freeze({
    list: "model",
    attach: "model",
    detach: "model",
    geometry: "model",
  }),
  manifolds: Object.freeze({
    inspectSurface: "model",
    list: "hybrid_manifold_read",
    get: "hybrid_manifold_read",
    create: "artifact",
    createDiscover: "artifact",
    createFromTemplate: "artifact",
    delete: "artifact",
    search: "artifact",
    install: "artifact",
    merge: "artifact",
    fit: "fitting",
    generate: "fitting",
    drowseArchiveList: "artifact",
    drowseArchiveInstall: "artifact",
    drowseArchiveExport: "artifact",
    drowseArchiveDelete: "artifact",
  }),
  templates: Object.freeze({
    list: "artifact",
    get: "artifact",
    create: "artifact",
    delete: "artifact",
    score: "fitting",
  }),
  tree: Object.freeze({
    replayCapabilities: "model",
    get: "model",
    reset: "model",
    restore: "model",
    active: "model",
    navigate: "model",
    edit: "model",
    branch: "model",
    delete: "model",
    star: "model",
    note: "model",
    edgeLabel: "model",
    filter: "model",
    diff: "model",
    transcriptExport: "model",
    transcriptLoad: "model",
    jointLogprobs: "model",
    cast: "model",
    castPut: "model",
    castDelete: "model",
  }),
  instruments: Object.freeze({
    setLive: "model",
    sources: "model",
    setLensSource: "model",
    activateInstalledPack: "model",
    startPreparation: "model",
    preparationStatus: "model",
    cancelPreparation: "model",
    tokenReadout: "model",
    validateLensToken: "model",
    validateSaeFeature: "model",
    saeFeaturesMetadata: "model",
  }),
}) satisfies {
  [Service in RuntimeServiceName]: Readonly<
    Record<
      Extract<keyof RuntimeServiceMap[Service], string>,
      RuntimeServiceRoute
    >
  >;
};

const ORDINARY: RuntimeServiceOperationPolicy = Object.freeze({
  lifecycle: "ordinary",
  requiredCapabilities: Object.freeze([]),
});

export function runtimeServiceRoute(
  request: RuntimeServiceRequest,
): RuntimeServiceRoute | null {
  if (!isRuntimeServiceMethod(request.service, request.method)) return null;
  return (RUNTIME_SERVICE_ROUTES[request.service] as Readonly<
    Record<string, RuntimeServiceRoute>
  >)[request.method] ?? null;
}

export function runtimeServiceOperationPolicy(
  request: RuntimeServiceRequest,
): RuntimeServiceOperationPolicy {
  const route = runtimeServiceRoute(request);
  if (route === "artifact") return policy("ordinary", "manifold_artifacts");
  if (route === "fitting") {
    return policy("fitting", "fitting", "manifold_artifacts");
  }
  if (request.service === "instruments" && request.method === "startPreparation") {
    const [family, body] = request.args;
    if (isRecord(body)) {
      if (family === "lens" && body.operation === "fit") {
        return policy("ordinary", "jlens_fitting");
      }
      if (family === "sae" && body.operation === "train") {
        return policy("ordinary", "sae_training");
      }
    }
  }
  return ORDINARY;
}

export function isFittingLifecycleRequest(request: RuntimeServiceRequest): boolean {
  return runtimeServiceRoute(request) === "fitting";
}

function policy(
  lifecycle: RuntimeServiceOperationPolicy["lifecycle"],
  ...requiredCapabilities: RuntimeCapabilityOperation[]
): RuntimeServiceOperationPolicy {
  return Object.freeze({
    lifecycle,
    requiredCapabilities: Object.freeze(requiredCapabilities),
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
