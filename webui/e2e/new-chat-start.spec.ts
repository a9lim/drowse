import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

async function chats(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/outside-the-workbench");
  await page.evaluate(async ({ rootUrl, fixtureUrl }) => {
    const [{ default: Root }, { createFixtureHostedRuntime }, { mount }] = await Promise.all([
      import(rootUrl), import(fixtureUrl), import("/e2e/svelte-runtime.ts"),
    ]);
    const fixture = createFixtureHostedRuntime();
    Object.defineProperty(fixture.runtime, "mode", { value: "browser" });
    const capabilities = await fixture.controller.check();
    const model = {
      id: "qwen3-1.7b-fixture", modelId: "qwen3-1.7b-fixture", name: "Qwen3 1.7B",
      modelType: "chat", catalogAvailable: true, tier: "fastest", fit: "recommended",
      installed: true, setupComplete: true, contextTokens: 2048, context: "Short conversations",
      reason: "Ready", language: "English", sourceUrl: "https://example.com", license: "Apache-2.0",
      size: "1 GB", firstRunBytes: 100, modelDownloadBytes: 100, remainingDownloadBytes: 0, firstRunPacks: [],
    };
    let snapshot = {
      phase: "supported", headline: "Ready", detail: "Ready", checks: [], models: [model],
      selectedModelVariantId: model.id,
      runtime: { available: true, phase: "unloaded", reason: "Ready", modelVariantId: model.id },
      download: { available: true, phase: "installed", reason: "Ready" }, storage: { persisted: true },
    };
    const listeners = new Set<(snapshot: unknown) => void>();
    const calls: Array<{ id: string; options: unknown }> = [];
    let finish: () => void;
    let fail: (error: Error) => void;
    let failWorkbench = false;
    const publish = (phase: string, reason: string) => {
      snapshot = { ...snapshot, runtime: { ...snapshot.runtime, phase, reason } };
      listeners.forEach(listener => listener(snapshot));
    };
    const controller = {
      current: () => snapshot,
      subscribe: (listener: (snapshot: unknown) => void) => { listeners.add(listener); listener(snapshot); return () => listeners.delete(listener); },
      check: async () => {}, capabilities: () => capabilities,
      runtimeClient: () => {
        if (failWorkbench) { failWorkbench = false; throw new Error("The chat interface is temporarily unavailable."); }
        return fixture.runtime;
      },
      hostedController: () => fixture.controller,
      retryPersistence: async () => true, setOptionalPackSelected() {},
      prepareForReload: async () => { publish("unloaded", "Ready"); },
      dispose: () => { void fixture.dispose(); },
      open: (id: string, options: unknown) => {
        calls.push({ id, options });
        publish("loading", "Loading the installed model");
        return new Promise<void>((resolve, reject) => { finish = resolve; fail = reject; });
      },
    };
    (window as any).chatStart = {
      calls,
      fixture,
      failWorkbench: () => { failWorkbench = true; },
      finish: () => { publish("ready", "Ready"); finish(); },
      fail: () => { publish("failed", "Could not load the model. Try again."); fail(new Error("Could not load the model. Try again.")); },
    };
    localStorage.setItem("drowse.entry.v1", JSON.stringify({ version: 1, completedAt: Date.now(), lastModelVariantId: model.id }));
    window.history.replaceState(null, "", `/app?choose=1&model=${model.id}`);
    document.body.replaceChildren();
    const target = document.createElement("div"); document.body.append(target);
    mount(Root, { target, props: { controller } });
  }, {
    rootUrl: `/@fs${resolve("src/hosted/ui/HostedRoot.svelte")}`,
    fixtureUrl: `/@fs${resolve("src/hosted/runtime/fixtureHostedRuntime.ts")}`,
  });
  await page.getByRole("link", { name: "Chats", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "New Instance", exact: true }).click();
  await page.locator('[data-model-id="qwen3-1.7b-fixture"] .model-choice-button').click();
  await expect(page.getByRole("heading", { name: "Start a new chat with Qwen3 1.7B?", exact: true })).toBeVisible();
  await page.evaluate(() => {
    const routes: string[] = [];
    (window as any).chatStartRoutes = routes;
    new MutationObserver(() => {
      for (const route of document.querySelectorAll<HTMLElement>("[data-route]")) routes.push(route.dataset.route!);
    }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-route"] });
  });
}

test("new chat loads on Your chats and enters the ready conversation without Models", async ({ page }, testInfo) => {
  await chats(page);
  await page.getByRole("button", { name: "Start new chat", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Loading Qwen3 1.7B…", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Loading…", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Cancel", exact: true })).toBeDisabled();
  expect(new URL(page.url()).searchParams.has("choose")).toBe(false);
  expect(await page.evaluate(() => (window as any).chatStart.calls)).toEqual([{ id: "qwen3-1.7b-fixture", options: { resetSession: true } }]);
  await page.screenshot({ path: testInfo.outputPath("loading-from-chats.png") });
  await page.evaluate(() => (window as any).chatStart.finish());
  await expect(page.locator(".shell")).toBeVisible();
  expect(await page.evaluate(() => (window as any).chatStartRoutes)).not.toContain("models");
  expect(new URL(page.url()).searchParams.get("reopen")).toBe("1");
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeEnabled();
});

test("a fresh same-model chat discards the previous tree, status, and hidden generation settings", async ({ page }) => {
  await chats(page);
  await page.evaluate(async (sourceRoot) => {
    const fixture = (window as any).chatStart.fixture;
    const { installRuntimeClient } = await import(`${sourceRoot}/lib/runtime/registry.ts`);
    installRuntimeClient(fixture.runtime);
    const [{ applyTreeSnapshot }, { samplingState }, { steerRack }, { genStatus }] = await Promise.all([
      import(`${sourceRoot}/lib/stores/loom.svelte.ts`),
      import(`${sourceRoot}/lib/stores/sampling.svelte.ts`),
      import(`${sourceRoot}/lib/stores/steering.svelte.ts`),
      import(`${sourceRoot}/lib/stores/chat.svelte.ts`),
    ]);
    const tree = await fixture.runtime.tree.get();
    const root = { ...tree.nodes[0], id: "previous-root" };
    const user = { ...root, id: "previous-user", parent_id: root.id, role: "user", text: "Previous chat content" };
    applyTreeSnapshot({ ...tree, root_id: root.id, active_node_id: user.id, rev: 12,
      nodes: [root, user], children_of: { [root.id]: [user.id], [user.id]: [] } });
    Object.assign(samplingState, { stop_sequences: "Hello", seed: 42, assistant_role: "wrongrole" });
    steerRack.customExpression = "0.5 jlens/fake";
    Object.assign(genStatus, { startedAt: 1, tokensSoFar: 17, finishReason: "stop" });
  }, `/@fs${resolve("src")}`);
  await page.getByRole("button", { name: "Start new chat", exact: true }).click();
  await page.evaluate(() => (window as any).chatStart.finish());
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeEnabled();
  await expect(page.getByText("Previous chat content", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Generation status", { exact: true })).toContainText("Ready");
  const fresh = await page.evaluate(async (sourceRoot) => {
    const [{ samplingState }, { steerRack }, { sessionState }] = await Promise.all([
      import(`${sourceRoot}/lib/stores/sampling.svelte.ts`),
      import(`${sourceRoot}/lib/stores/steering.svelte.ts`),
      import(`${sourceRoot}/lib/stores/session.svelte.ts`),
    ]);
    return { stop: samplingState.stop_sequences, seed: samplingState.seed,
      role: samplingState.assistant_role, defaultRole: sessionState.info.default_assistant_role ?? "assistant",
      steering: steerRack.customExpression, defaultSteering: sessionState.info.default_steering };
  }, `/@fs${resolve("src")}`);
  expect(fresh.stop).toBe("");
  expect(fresh.seed).toBeNull();
  expect(fresh.role).toBe(fresh.defaultRole);
  expect(fresh.steering).toBe(fresh.defaultSteering);
});

test("a failed new-chat load stays on Your chats and can be retried", async ({ page }) => {
  await chats(page);
  await page.getByRole("button", { name: "Start new chat", exact: true }).click();
  await page.evaluate(() => (window as any).chatStart.fail());
  await expect(page.getByRole("alert")).toContainText("Could not load the model. Try again.");
  await expect(page.locator(".model-opening")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Start new chat", exact: true }).click();
  expect(await page.evaluate(() => (window as any).chatStart.calls.length)).toBe(2);
  await page.evaluate(() => (window as any).chatStart.finish());
  await expect(page.locator(".shell")).toBeVisible();
  expect(await page.evaluate(() => (window as any).chatStartRoutes)).not.toContain("models");
});

test("a chat-interface failure can be retried on Your chats without resetting the model again", async ({ page }) => {
  await chats(page);
  await page.getByRole("button", { name: "Start new chat", exact: true }).click();
  await page.evaluate(() => {
    (window as any).chatStart.failWorkbench();
    (window as any).chatStart.finish();
  });
  await expect(page.getByRole("heading", { name: "Your chat could not open", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(page.locator(".model-opening")).toHaveCount(0);
  await page.getByRole("button", { name: "Try opening chat again", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  expect(await page.evaluate(() => (window as any).chatStart.calls.length)).toBe(1);
  expect(await page.evaluate(() => (window as any).chatStartRoutes)).not.toContain("models");
});
