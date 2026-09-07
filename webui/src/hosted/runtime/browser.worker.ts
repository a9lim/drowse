import { BrowserDrowseArchiveService } from "../artifacts/service";
import { HostedRuntimeWorker } from "./worker";
import { createLazyBrowserModelBackend } from "./browserModelBackend";

const artifacts = new BrowserDrowseArchiveService();

new HostedRuntimeWorker(
  globalThis as unknown as import("./worker").WorkerScopeLike,
  {
    artifactService: artifacts,
    modelBackend: createLazyBrowserModelBackend(
      () => import("@drowse/web-llm"),
      artifacts,
    ),
  },
);
