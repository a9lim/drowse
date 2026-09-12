/// <reference types="vite/client" />
import { mount } from "svelte";
import {
  chunkLoadFailureMessage,
  completeChunkRecovery,
  isChunkLoadError,
  recoverFromChunkLoadError,
  reloadHostedApp,
  reloadInstructions,
  appRefreshSafetyMessage,
} from "../src/hosted/runtime/chunkRecovery";
import { initializeTheme } from "../src/lib/theme";
import { initializeInputModality } from "../src/lib/inputModality";
import { startWebMcp } from "../src/lib/webmcp";
import "../src/lib/style/fonts.css";
import "../src/lib/style/tokens.css";
import "../src/lib/style/global.css";

const target = document.getElementById("app");
if (!target) throw new Error("drowse hosted: #app element missing in index.html");
initializeTheme();
initializeInputModality();
startWebMcp(true);
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
    controller = createShellController(createFixtureHostedRuntime(searchParams.get("fixtureInstruments") === "1"));
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
    ] = await Promise.all([
      import("../src/hosted/runtime/fixtureHostedRuntime"),
      import("../src/lib/runtime/registry"),
    ]);
    const fixture = createFixtureHostedRuntime(searchParams.get("layoutFixture") === "instruments", searchParams.get("layoutFixture") === "base");
    const capabilities = await fixture.controller.check();
    installRuntimeClient(fixture.runtime);
    installRuntimeCapabilities(capabilities);
    installHostedController(fixture.controller);
    component = (await import("../src/App.svelte")).default;
  } else if (path === "/") {
    component = (await import("../src/hosted/ui/LandingRoot.svelte")).default;
  } else if (path === "/credits") {
    component = (await import("../src/hosted/ui/Credits.svelte")).default;
  } else if (path === "/contact") {
    component = (await import("../src/hosted/ui/Contact.svelte")).default;
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
  if (target.hasAttribute("data-prerendered")) target.replaceChildren();
  app = mount(component, { target, props });
} else if (bootstrapError) {
  if (target.hasAttribute("data-prerendered")) target.replaceChildren();
  const message = document.createElement("main");
  const heading = document.createElement("h1");
  const detail = document.createElement("p");
  const actions = document.createElement("div");
  const guidance = document.createElement("p");
  const safety = document.createElement("p");
  const status = document.createElement("p");
  const reload = document.createElement("button");
  heading.textContent = "Drowse could not open";
  detail.textContent = isChunkLoadError(bootstrapError)
    ? chunkLoadFailureMessage
    : "Reload the page and try again.";
  reload.type = "button";
  reload.textContent = "Reload Drowse";
  reload.addEventListener("click", async () => {
    reload.disabled = true;
    reload.textContent = "Refreshing app…";
    status.textContent = "Checking for current app files…";
    try {
      await reloadHostedApp("bootstrap");
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "The app could not refresh. Please try again.";
      reload.disabled = false;
      reload.textContent = "Try reloading again";
    }
  });
  actions.className = "bootstrap-error-actions";
  guidance.className = "bootstrap-error-guidance";
  guidance.textContent = reloadInstructions();
  safety.className = "bootstrap-error-safety";
  safety.textContent = appRefreshSafetyMessage;
  status.setAttribute("role", "status");
  actions.append(reload, safety);
  message.className = "bootstrap-error";
  heading.id = "bootstrap-error-title";
  message.setAttribute("aria-labelledby", heading.id);
  message.append(heading, detail, actions, guidance, status);
  target.append(message);
}
export default app;
