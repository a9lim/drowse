import { BrowserRuntimeClient, WorkerRpcTransport } from "./browserRuntimeClient";
import {
  HostedControllerImpl,
  type HostedControllerOptions,
  type HostedRuntimeBundle,
} from "./hostedController";

export function createWorkerFixtureRuntime(
  options: HostedControllerOptions & {
    saeAvailable?: boolean;
    slowGeneration?: boolean;
  } = {},
): HostedRuntimeBundle {
  const {
    saeAvailable = true,
    slowGeneration = false,
    ...controllerOptions
  } = options;
  const fixtureName = [
    "drowse-runtime-fixture",
    ...(saeAvailable ? [] : ["without-sae"]),
    ...(slowGeneration ? ["slow-generation"] : []),
  ].join("-");
  const worker = new Worker(new URL("./fixtureBrowser.worker.ts", import.meta.url), {
    type: "module",
    name: fixtureName,
  });
  const transport = new WorkerRpcTransport(worker, controllerOptions);
  const controller = new HostedControllerImpl(transport, {
    ...controllerOptions,
    ownsTransport: false,
  });
  const runtime = new BrowserRuntimeClient(transport, { ownsTransport: false });
  let disposed = false;
  return {
    controller,
    runtime,
    async dispose() {
      if (disposed) return;
      disposed = true;
      try {
        await controller.dispose();
        await runtime.dispose();
      } finally {
        transport.dispose();
      }
    },
  };
}
