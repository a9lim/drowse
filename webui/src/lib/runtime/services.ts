import type {
  ExtractRequest,
  FitManifoldRequest,
  GenerateManifoldRequest,
  InstallManifoldRequest,
} from "../types";
import { ApiError, describeError } from "./errors";
import { runtimeClient } from "./client";

export const apiSessions = runtimeClient.sessions;
export const apiProfiles = runtimeClient.profiles;
export const apiProbes = runtimeClient.probes;
export const apiManifolds = runtimeClient.manifolds;
export const apiTemplates = runtimeClient.templates;
export const apiTree = runtimeClient.tree;
export const apiInstruments = runtimeClient.instruments;

export function apiManifoldFitStream(
  namespace: string,
  name: string,
  request: FitManifoldRequest,
  onEvent: (event: { event: string; data: unknown }) => void,
) {
  return runtimeClient.manifolds.fit(namespace, name, request, onEvent);
}

export function apiManifoldInstallStream(
  request: InstallManifoldRequest,
  onEvent: (event: { event: string; data: unknown }) => void,
) {
  return runtimeClient.manifolds.install(request, onEvent);
}

export function apiManifoldGenerateStream(
  request: GenerateManifoldRequest,
  onEvent: (event: { event: string; data: unknown }) => void,
) {
  return runtimeClient.manifolds.generate(request, onEvent);
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
