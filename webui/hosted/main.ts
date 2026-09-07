/// <reference types="vite/client" />
import { mount } from "svelte";
import {
  chunkLoadFailureMessage,
  completeChunkRecovery,
  isChunkLoadError,
  recoverFromChunkLoadError,
} from "../src/hosted/runtime/chunkRecovery";
import { initializeTheme } from "../src/lib/theme";
import "../src/lib/style/fonts.css";
import "../src/lib/style/tokens.css";
import "../src/lib/style/global.css";

const target = document.getElementById("app");
if (!target) throw new Error("drowse hosted: #app element missing in index.html");
initializeTheme();
const path = window.location.pathname.replace(/\/+$/, "") || "/";
if (path !== "/") {
  const robots = document.createElement("meta");
  robots.name = "robots";
  robots.content = "noindex, follow";
  document.head.append(robots);
}

let controller: import("../src/hosted/ui/types").HostedShellController | undefined;
let component = null;
let props = {};
let bootstrapError: unknown = null;
try {
  const searchParams = new URLSearchParams(window.location.search);
  const layoutFixture = import.meta.env.DEV && ["1", "instruments", "base"].includes(searchParams.get("layoutFixture") ?? "");
  if (
    import.meta.env.DEV &&
    searchParams.get("layoutFixture") === "setup"
  ) {
    const [{ createFixtureHostedRuntime }, { createShellController }] = await Promise.all([
      import("../src/hosted/runtime/fixtureHostedRuntime"),
      import("./shell-controller"),
    ]);
    controller = createShellController(createFixtureHostedRuntime());
  }
  if (
    import.meta.env.DEV &&
    searchParams.get("fixture") === "1"
  ) {
    const fixtureSaeAvailable =
      searchParams.get("fixtureSae") !== "0";
    const fixtureSlowGeneration = searchParams.get("fixtureSlow") === "1";
    const [{ createWorkerFixtureRuntime }, { createShellController }] = await Promise.all([
      import("../src/hosted/runtime/workerFixtureRuntime"),
      import("./shell-controller"),
    ]);
    controller = createShellController(undefined, {
      runtimeFactory: (options) => createWorkerFixtureRuntime({
        ...options,
        saeAvailable: fixtureSaeAvailable,
        slowGeneration: fixtureSlowGeneration,
      }),
    });
  }

  if (layoutFixture) {
    const [
      { createFixtureHostedRuntime },
      { installHostedController, installRuntimeCapabilities, installRuntimeClient },
      { installTooltipLayer },
    ] = await Promise.all([
      import("../src/hosted/runtime/fixtureHostedRuntime"),
      import("../src/lib/runtime/registry"),
      import("../src/lib/tooltips"),
    ]);
    const fixture = createFixtureHostedRuntime(searchParams.get("layoutFixture") === "instruments", searchParams.get("layoutFixture") === "base");
    const capabilities = await fixture.controller.check();
    installRuntimeClient(fixture.runtime);
    installRuntimeCapabilities(capabilities);
    installHostedController(fixture.controller);
    installTooltipLayer();
    component = (await import("../src/App.svelte")).default;
  } else if (path === "/") {
    component = (await import("../src/hosted/ui/LandingRoot.svelte")).default;
  } else if (path === "/credits") {
    component = (await import("../src/hosted/ui/Credits.svelte")).default;
  } else if (path === "/app" || path.startsWith("/app/")) {
    const [{ default: HostedRoot }, { startHostedNetworkStateTracking }] = await Promise.all([
      import("../src/hosted/ui/HostedRoot.svelte"),
      import("../src/hosted/runtime/networkState"),
    ]);
    startHostedNetworkStateTracking();
    component = HostedRoot;
    props = { controller };
  } else {
    component = (await import("../src/hosted/ui/NotFound.svelte")).default;
  }
} catch (error) {
  if (!await recoverFromChunkLoadError(error, "bootstrap")) bootstrapError = error;
}

let app = null;
if (component) {
  completeChunkRecovery("bootstrap");
  app = mount(component, { target, props });
} else if (bootstrapError) {
  const message = document.createElement("main");
  const heading = document.createElement("h1");
  const detail = document.createElement("p");
  const reload = document.createElement("button");
  heading.textContent = "Drowse could not open";
  detail.textContent = isChunkLoadError(bootstrapError)
    ? chunkLoadFailureMessage
    : "Reload the page and try again.";
  reload.type = "button";
  reload.textContent = "Reload Drowse";
  reload.addEventListener("click", () => window.location.reload());
  message.className = "bootstrap-error";
  message.append(heading, detail, reload);
  target.append(message);
}
export default app;
