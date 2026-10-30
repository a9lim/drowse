/// <reference types="vite/client" />
import { hydrate, mount } from "svelte";
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
import "../src/lib/style/fonts.css";
import "../src/lib/style/tokens.css";
import "../src/lib/style/global.css";

const target = document.getElementById("app");
if (!target) throw new Error("drowse hosted: #app element missing in index.html");
initializeTheme();
initializeInputModality();
// WebMCP is ~50 KB and inert without the browser's experimental model context, so load it on demand.
// It must start before any route mounts, because app routes attach their tools to the running adapter.
const webMcpReady = "modelContext" in document || "modelContext" in navigator
  ? import("../src/lib/webmcp").then(({ startWebMcp }) => startWebMcp(true)).catch(() => undefined)
  : undefined;
const path = window.location.pathname.replace(/\/+$/, "") || "/";
// Indexing policy lives in the static HTML: prerendered pages are indexable, while the app shell
// and the not-found page already carry `noindex, follow`.
const learnGuides = new Set(["/learn/activation-steering", "/learn/interpretability", "/learn/browser-models", "/learn/experiments"]);

let controller: import("../src/hosted/ui/types").HostedShellController | undefined;
let component = null;
let props = {};
// Matches the `data-prerendered` key written by scripts/prerender-hosted.mjs for this route.
let prerenderedKey: string | null = null;
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
    prerenderedKey = "landing";
  } else if (path === "/credits") {
    component = (await import("../src/hosted/ui/Credits.svelte")).default;
    prerenderedKey = "credits";
  } else if (path === "/contact") {
    component = (await import("../src/hosted/ui/Contact.svelte")).default;
    prerenderedKey = "contact";
  } else if (path === "/privacy") {
    component = (await import("../src/hosted/ui/pages/Privacy.svelte")).default;
    prerenderedKey = "privacy";
  } else if (path === "/about") {
    component = (await import("../src/hosted/ui/pages/About.svelte")).default;
    prerenderedKey = "about";
  } else if (path === "/learn") {
    component = (await import("../src/hosted/ui/pages/Learn.svelte")).default;
    prerenderedKey = "learn";
  } else if (learnGuides.has(path)) {
    component = (await import("../src/hosted/ui/pages/LearnGuide.svelte")).default;
    props = { path };
    prerenderedKey = `learn-guide:${path}`;
  } else if (path === "/contribute") {
    component = (await import("../src/hosted/ui/pages/Contribute.svelte")).default;
    prerenderedKey = "contribute";
  } else if (path === "/neurons") {
    component = (await import("../src/hosted/ui/pages/Neurons.svelte")).default;
    prerenderedKey = "neurons";
  } else if (path === "/third-party-licenses") {
    component = (await import("../src/hosted/ui/pages/ThirdPartyLicenses.svelte")).default;
    prerenderedKey = "third-party-licenses";
  } else if (path === "/app") {
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

function mountReplacingPrerender(component: Parameters<typeof mount>[0], props: Record<string, unknown>) {
  // Remounting discards the static DOM, so carry keyboard focus and the URL fragment across.
  const active = document.activeElement instanceof HTMLElement && target!.contains(document.activeElement)
    ? document.activeElement : null;
  const focusId = active?.id || null;
  const focusHref = !focusId && active instanceof HTMLAnchorElement ? active.getAttribute("href") : null;
  target!.replaceChildren();
  const mounted = mount(component, { target: target!, props });
  queueMicrotask(() => {
    const focus = focusId
      ? document.getElementById(focusId)
      : focusHref ? target!.querySelector<HTMLElement>(`a[href="${CSS.escape(focusHref)}"]`) : null;
    focus?.focus({ preventScroll: true });
    const fragment = decodeURIComponent(window.location.hash.slice(1));
    if (fragment) document.getElementById(fragment)?.scrollIntoView();
  });
  return mounted;
}

await webMcpReady;
let app = null;
if (component) {
  completeChunkRecovery("bootstrap");
  const prerendered = target.getAttribute("data-prerendered");
  if (prerendered !== null && prerendered === prerenderedKey) {
    // The static markup was rendered from this exact component and props: adopt it in place.
    app = hydrate(component, { target, props });
  } else if (prerendered !== null) {
    app = mountReplacingPrerender(component, props);
  } else {
    app = mount(component, { target, props });
  }
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
