import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const moduleUrl = (path: string) => `/@fs${resolve(path)}`;
type Scenario = "granted" | "denied" | "rejected" | "unsupported" | "late";

test("native browser storage requests settle without blocking the page", async ({ page }, testInfo) => {
  await page.goto(`${devUrl}/outside-the-workbench`);
  await page.evaluate(async url => {
    const { requestPersistentStorage } = await import(url);
    const button = document.createElement("button");
    button.textContent = "Request native protection";
    button.onclick = async () => {
      (window as any).__nativeStorageResult = await requestPersistentStorage(navigator.storage, 250);
    };
    document.body.replaceChildren(button);
  }, moduleUrl("src/lib/runtime/storagePersistence.ts"));
  await page.getByRole("button", { name: "Request native protection" }).click();
  await expect.poll(() => page.evaluate(() => typeof (window as any).__nativeStorageResult)).toBe("boolean");
  const result = await page.evaluate(async () => ({
    granted: (window as any).__nativeStorageResult,
    persisted: await navigator.storage?.persisted?.() ?? null,
  }));
  if (result.granted) expect(result.persisted).toBe(true);
  testInfo.annotations.push({ type: "native-storage-decision", description: JSON.stringify(result) });
});

async function mountNotice(page: Page, surface: "models" | "chats", scenario: Scenario) {
  await page.goto(`${devUrl}/outside-the-workbench`);
  await page.evaluate(async ({ surface, scenario, appUrl, homeUrl, persistenceUrl, libraryUrl }) => {
    const [{ default: Component }, { mount, SvelteMap }, { requestPersistentStorage }, { conversationLibrary }] = await Promise.all([
      import(surface === "models" ? appUrl : homeUrl), import("/e2e/svelte-runtime.ts"),
      import(persistenceUrl), import(libraryUrl),
    ]);
    let resolvePermission: (value: boolean) => void = () => {};
    const calls: boolean[] = [];
    // Keep WebKit's patched StorageManager wrapper alive until the click.
    const storage = navigator.storage ?? {};
    Object.defineProperty(navigator, "storage", { configurable: true, value: storage });
    Object.defineProperty(storage, "persist", {
      configurable: true,
      value: scenario === "unsupported" ? undefined : () => {
        calls.push(navigator.userActivation.isActive);
        if (scenario === "late") return new Promise<boolean>(resolve => { resolvePermission = resolve; });
        if (scenario === "rejected") return Promise.reject(new TypeError("Storage disabled"));
        return Promise.resolve(scenario === "granted");
      },
    });
    Object.defineProperty(storage, "persisted", { configurable: true, value: async () => false });
    conversationLibrary.listSummaries = async () => ({ conversations: [], issues: [] });
    const model = {
      id: "storage-model", modelId: "storage-model", name: "Storage model", tier: "fastest",
      fit: "recommended", installed: true, setupComplete: true, reason: "Ready", size: "1 GB",
      context: "2K", contextTokens: 2048, language: "English", license: "Apache-2.0",
      sourceUrl: "https://huggingface.co", modelDownloadBytes: 100, firstRunBytes: 100,
      remainingDownloadBytes: 0, firstRunPacks: [],
    };
    const state = new SvelteMap([["snapshot", {
      phase: "supported", headline: "Ready", detail: "Ready", checks: [], models: [model],
      selectedModelVariantId: model.id,
      download: { available: true, phase: "installed", reason: "Ready" },
      runtime: { available: true, phase: "unloaded", reason: "Ready" },
      storage: { availableBytes: 1e9, persisted: scenario === "unsupported" ? undefined : false },
    }]]);
    const listeners = new Set<(value: unknown) => void>();
    const current = () => state.get("snapshot");
    const protect = () => {
      state.set("snapshot", { ...current(), storage: { ...current().storage, persisted: true } });
      for (const listener of listeners) listener(current());
    };
    const controller = {
      current,
      subscribe(listener: (value: unknown) => void) { listeners.add(listener); listener(current()); return () => listeners.delete(listener); },
      capabilities: () => ({ signals: { appleMobile: false } }),
      async check() {},
      async retryPersistence() {
        const granted = await requestPersistentStorage(storage, 100, protect);
        if (granted) protect();
        return granted;
      },
    };
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(Component, { target, props: { controller, get snapshot() { return current(); }, onChooseModels() {} } });
    (window as any).__storageTest = { calls, grant: () => resolvePermission(true) };
  }, {
    surface, scenario, appUrl: moduleUrl("src/hosted/ui/HostedApp.svelte"),
    homeUrl: moduleUrl("src/hosted/ui/HostedHome.svelte"),
    persistenceUrl: moduleUrl("src/lib/runtime/storagePersistence.ts"),
    libraryUrl: moduleUrl("src/lib/stores/savedConversations.svelte.ts"),
  });
}

for (const surface of ["models", "chats"] as const) {
  for (const scenario of ["granted", "denied", "rejected", "unsupported", "late"] as const) {
    test(`${surface} storage protection handles ${scenario}`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on("pageerror", error => errors.push(error.message));
      await mountNotice(page, surface, scenario);
      const notice = page.locator(surface === "models" ? ".persistence-notice" : ".storage-notice");
      await expect(notice).toBeVisible();
      const button = notice.getByRole("button", { name: "Protect storage", exact: true });
      if (scenario === "unsupported") {
        await expect(button).toHaveCount(0);
        await expect(notice).toContainText(/Back up your chats/);
      } else {
        await page.evaluate(() => {
          (window as any).__storageAnimations = 0;
          const animate = Element.prototype.animate;
          Element.prototype.animate = function(...args) {
            if (this.closest(".storage-notice, .persistence-notice")) (window as any).__storageAnimations++;
            return animate.apply(this, args);
          };
        });
        await button.click();
        await expect.poll(() => page.evaluate(() => (window as any).__storageTest.calls)).toEqual([true]);
        if (scenario === "granted") {
          await expect(notice).toHaveCount(0);
        } else {
          await expect(button).toBeEnabled();
          await expect(notice).not.toContainText("didn't grant");
          await expect(notice).toContainText(/chats (still )?save/);
          expect(await page.evaluate(() => (window as any).__storageAnimations)).toBe(0);
          await expect(notice.locator(".loading-pulse")).toHaveCount(0);
          if (scenario === "late") {
            await page.evaluate(() => (window as any).__storageTest.grant());
            await expect(notice).toHaveCount(0);
          }
        }
      }
      if (["denied", "rejected", "unsupported"].includes(scenario)) {
        for (const width of [320, 1440]) {
          await page.setViewportSize({ width, height: 900 });
          expect(await notice.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
        }
        await notice.screenshot({ path: testInfo.outputPath(`${surface}-${scenario}.png`) });
      }
      expect(errors).toEqual([]);
    });
  }
}

for (const [state, animation] of [["loading-pulse", "loading-breathe"], ["generation-active", "generation-breathe"]]) {
  test(`${state} retains a full glowing pulse and reduced-motion fallback`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto(`${devUrl}/outside-the-workbench`);
    await page.evaluate(state => {
      const surface = document.createElement("p");
      surface.className = `${state} loading-placeholder`;
      surface.textContent = state === "loading-pulse" ? "Opening your local workspace…" : "Generating…";
      document.body.append(surface);
    }, state);
    const surface = page.locator(`.${state}`);
    const decoration = await surface.evaluate(el => {
      const style = getComputedStyle(el, "::before");
      return { width: parseFloat(style.width), hostWidth: el.getBoundingClientRect().width, shadow: style.boxShadow, animation: style.animationName };
    });
    expect(decoration.width).toBeCloseTo(decoration.hostWidth, 0);
    expect(decoration.shadow).toContain("inset");
    expect(decoration.animation).toBe(animation);
    await page.emulateMedia({ reducedMotion: "reduce" });
    expect(await surface.evaluate(el => getComputedStyle(el, "::before").animationName)).toBe("none");
  });
}

test("chats save and reload without persistent storage permission", async ({ page }) => {
  await page.addInitScript(() => {
    const storage = navigator.storage ?? {};
    Object.defineProperty(navigator, "storage", { configurable: true, value: storage });
    Object.defineProperty(storage, "persist", { configurable: true, value: async () => false });
    Object.defineProperty(storage, "persisted", { configurable: true, value: async () => false });
  });
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await expect(page.locator(".shell")).toBeVisible();
  const record = await page.evaluate(async ({ libraryUrl, snapshotUrl }) => {
    const [{ conversationLibrary, requestPersistentConversationStorage }, { captureConversationSnapshot }] = await Promise.all([
      import(libraryUrl), import(snapshotUrl),
    ]);
    const protectedStorage = await requestPersistentConversationStorage();
    const record = await conversationLibrary.create({ name: "Unprotected saved chat", snapshot: captureConversationSnapshot() });
    return { id: record.id, protectedStorage };
  }, {
    libraryUrl: moduleUrl("src/lib/stores/savedConversations.svelte.ts"),
    snapshotUrl: moduleUrl("src/lib/conversationWorkspace.ts"),
  });
  expect(record.protectedStorage).toBe(false);
  await page.reload();
  await expect(page.locator(".shell")).toBeVisible();
  expect(await page.evaluate(async ({ url, id }) => {
    const { conversationLibrary } = await import(url);
    return (await conversationLibrary.get(id)).name;
  }, { url: moduleUrl("src/lib/stores/savedConversations.svelte.ts"), id: record.id })).toBe("Unprotected saved chat");
});
