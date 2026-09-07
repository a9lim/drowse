export { BrowserFittingWorkerClient } from "../fitting/fittingWorkerClient";
export {
  BrowserFittingCoordinator,
  anchorBrowserNeutralLayout,
  finalizeBrowserAffineLayers,
  fitBrowserCurvedSigmaFields,
  fitBrowserCurvedSurfaceLayers,
  selectBrowserDlsAxes,
} from "../fitting/coordinator";
export type {
  ActivationCaptureSink,
  BrowserAffineFittingPlan,
  BrowserActivationCaptureSource,
  BrowserFittingAffineLayers,
  BrowserFittingCapturePlan,
  BrowserFittingCentroids,
  BrowserFittingCoordinatorProgress,
  BrowserFittingLayerCentroids,
  BrowserFittingTopologyFoundation,
  BrowserSigmaFieldSummary,
  BrowserCurvedMeanLayer,
  BrowserTopologyFittingPlan,
  FinalizedBrowserAffineLayer,
  FinalizedBrowserCurvedLayer,
} from "../fitting/coordinator";
export type {
  AffineFisherResult,
  FittingMatrixSource,
  FittingWorkerJob,
  FittingWorkerResult,
  SerializedMahalanobisWhitener,
  TopologyResult,
} from "../fitting/workerContracts";

export const BROWSER_ARTIFACT_AUTHORING_INTEGRATED = true;
export const BROWSER_AUTHORING_IMPLEMENTATION_INTEGRATED = true;
export const BROWSER_AUTHORING_RUNTIME_INTEGRATED = true;
export const BROWSER_AUTHORING_RELEASE_VERIFIED =
  BROWSER_AUTHORING_RUNTIME_INTEGRATED;

export {
  WebLlmActivationCaptureSource,
  type WebLlmActivationCapturePlan,
  type WebLlmCaptureRuntimePort,
} from "./webLlmActivationCapture";
export {
  BrowserManifoldFitting,
  type BrowserManifoldArtifactPort,
  type BrowserManifoldFittingContext,
} from "./browserManifoldFitting";

export {
  browserFittedCurvedDiscoverPack,
  browserFittedFlatDiscoverPack,
  type BrowserFittedCurvedDiscoverPackInput,
  type BrowserFittedFlatDiscoverPackInput,
  type BrowserFlatFitIdentity,
} from "../artifacts/fittedAuthoring";
