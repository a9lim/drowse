import type {
  ExtractRequest,
  FitManifoldRequest,
  GenerateManifoldRequest,
  InstallManifoldRequest,
} from "../types";
import { ApiError, describeError } from "./errors";
import { runtimeClient } from "./client";
import { beginManifoldJob, updateManifoldJob, finishManifoldJob, failManifoldJob } from "../stores/manifoldJobs.svelte";

export const apiSessions = runtimeClient.sessions;
export const apiProfiles = runtimeClient.profiles;
export const apiProbes = runtimeClient.probes;
export const apiManifolds = runtimeClient.manifolds;
export const apiTemplates = runtimeClient.templates;
export const apiTree = runtimeClient.tree;
export const apiInstruments = runtimeClient.instruments;

export async function apiManifoldFitStream(
  namespace: string,
  name: string,
  request: FitManifoldRequest,
  onEvent: (event: { event: string; data: unknown }) => void,
) {
  const job = beginManifoldJob(namespace, name, "fitting");
  try {
    const result = await runtimeClient.manifolds.fit(namespace, name, request, event => {
      updateManifoldJob(job, event);
      onEvent(event);
    });
    finishManifoldJob(job);
    return result;
  } catch (error) {
    failManifoldJob(job, error);
    throw error;
  }
}

export function apiManifoldInstallStream(
  request: InstallManifoldRequest,
  onEvent: (event: { event: string; data: unknown }) => void,
) {
  return runtimeClient.manifolds.install(request, onEvent);
}

export async function apiManifoldGenerateStream(
  request: GenerateManifoldRequest,
  onEvent: (event: { event: string; data: unknown }) => void,
  fitAfterwards = false,
) {
  const job = beginManifoldJob(request.namespace ?? "local", request.name, "generating", request.concepts, fitAfterwards);
  try {
    const result = await runtimeClient.manifolds.generate(request, event => {
      updateManifoldJob(job, event);
      onEvent(event);
    });
    finishManifoldJob(job);
    return result;
  } catch (error) {
    failManifoldJob(job, error);
    throw error;
  }
}

export function apiExtractStream(
  request: ExtractRequest,
  onEvent: (event: { event: string; data: unknown }) => void,
  id?: string,
) {
  return runtimeClient.profiles.extract(request, onEvent, id);
}

export { ApiError, describeError };
export type * from "../types";
