import type {
  HostedController,
  RuntimeCapabilities,
  RuntimeClient,
} from "./contracts";

let installedRuntime: RuntimeClient | null = null;
let installedCapabilities: RuntimeCapabilities | null = null;
let installedHostedController: HostedController | null = null;

export function installRuntimeClient(runtime: RuntimeClient): void {
  if (installedRuntime && installedRuntime !== runtime) {
    throw new Error("A Drowse runtime is already installed");
  }
  installedRuntime = runtime;
}

export function getRuntimeClient(): RuntimeClient {
  if (!installedRuntime) {
    throw new Error("Drowse runtime has not been installed");
  }
  return installedRuntime;
}

export function installRuntimeCapabilities(
  capabilities: RuntimeCapabilities,
): void {
  if (installedCapabilities && installedCapabilities !== capabilities) {
    throw new Error("Drowse runtime capabilities are already installed");
  }
  installedCapabilities = capabilities;
}

export function getRuntimeCapabilities(): RuntimeCapabilities | null {
  return installedCapabilities;
}

export function getRuntimeManifoldFitMaxIntrinsicDim(): number | null {
  return installedCapabilities?.limits?.manifoldFitMaxIntrinsicDim ?? null;
}

export function installHostedController(controller: HostedController): void {
  if (installedHostedController && installedHostedController !== controller) {
    throw new Error("A Drowse hosted controller is already installed");
  }
  installedHostedController = controller;
}

export function getHostedController(): HostedController | null {
  return installedHostedController;
}
