import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

async function home(page: Page, installed = true) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/outside-the-workbench");
  await page.evaluate(async ({ moduleUrl, installed }) => {
    const [{ default: Harness }, { mount }] = await Promise.all([import(moduleUrl), import("/@id/svelte")]);
    const models = ["Gemma 3 1B", "Gemma 3 4B", "Qwen3 0.6B", "Qwen3 1.7B", "GPT-2", "Pythia 410M", "Pythia 1.4B", "Qwen3 0.6B Base"].map((name, index) => ({
      id: `model-${index}`, modelId: name.toLowerCase().replaceAll(" ", "-"), modelType: index < 4 ? "chat" : "base",
      name, catalogAvailable: true, tier: "fastest", sourceUrl: "https://example.com", license: "Apache-2.0",
      size: "2.28 GB", context: "Short conversations", contextTokens: 2048, fit: "eligible", reason: "Device memory is uncertain.", language: "English",
      installed: installed && [0, 3, 4].includes(index), setupComplete: installed && [0, 3, 4].includes(index),
      modelDownloadBytes: 2_000_000_000, firstRunBytes: 2_280_000_000, remainingDownloadBytes: installed && [0, 3, 4].includes(index) ? 0 : 2_280_000_000,
      firstRunPacks: [],
    }));
    let snapshot = {
      phase: "supported", headline: "Ready", detail: "Ready", checks: [], models,
      download: { available: true, phase: "idle", reason: "Ready" },
      runtime: { available: true, phase: "unloaded", reason: "Ready" }, storage: { persisted: true },
    };
    const listeners = new Set<(snapshot: unknown) => void>();
    let settle: () => void;
    let reject: (error: Error) => void;
    const calls: Array<{ id: string; options: unknown }> = [];
    const opens: Array<{ id: string; options: unknown }> = [];
    let routes = 0;
    const emit = (patch: object) => { snapshot = { ...snapshot, ...patch }; listeners.forEach(listener => listener(snapshot)); };
    const fixture = {
      calls, opens, get routes() { return routes; }, get snapshot() { return snapshot; }, emit,
      model: (index: number, patch: object) => emit({ models: snapshot.models.map((model, i) => i === index ? { ...model, ...patch } : model) }),
      progress: (bytesReceived: number, extra = {}) => emit({ download: { ...snapshot.download, phase: "downloading", progress: {
        modelVariantId: calls.at(-1)!.id, bytesTotal: 2_280_000_000, bytesReceived,
        files: [], throughputBytesPerSecond: 10_000_000, etaSeconds: null, calculatingEta: false,
        stalled: false, offline: false, resumable: true, ...extra,
      } } }),
      finish: () => {
        fixture.model(Number(calls.at(-1)!.id.split("-")[1]), { installed: true, setupComplete: true, remainingDownloadBytes: 0 });
        emit({ download: { ...snapshot.download, phase: "installed" } }); settle();
      },
      fail: () => { emit({ download: { ...snapshot.download, phase: "failed", reason: "Connection lost. Try again." } }); reject(new Error("Connection lost. Try again.")); },
    };
    (window as any).downloads = fixture;
    const controller = {
      current: () => snapshot,
      subscribe: (listener: (snapshot: unknown) => void) => { listeners.add(listener); return () => listeners.delete(listener); },
      capabilities: () => ({ signals: { appleMobile: false } }), check: async () => {}, retryPersistence: async () => true,
      download: (id: string, options: unknown) => {
        calls.push({ id, options });
        emit({ download: { available: true, phase: "requesting_persistence", reason: "Preparing", modelVariantId: id } });
        return new Promise<void>((resolve, fail) => { settle = resolve; reject = fail; });
      },
      cancelDownload: async () => { emit({ download: { ...snapshot.download, phase: "paused" } }); settle(); },
      open: async (id: string, options: unknown) => { opens.push({ id, options }); },
    };
    document.body.replaceChildren();
    const target = document.createElement("div"); document.body.append(target);
    mount(Harness, { target, props: { controller, onChooseModels: () => { routes++; } } });
  }, { moduleUrl: `/@fs/${resolve("e2e/fixtures/ModelDownloadHarness.svelte")}`, installed });
  await page.getByRole("button", { name: installed ? "New Instance" : "Download a model", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Choose a model" })).toBeVisible();
}

const card = (page: Page, index: number) => page.locator(`[data-model-id="model-${index}"]`);
const action = (page: Page, index: number) => card(page, index).locator(".model-choice-button");

test("two clicks download inline, verify before ready, and preserve the new-chat confirmation", async ({ page }, testInfo) => {
  const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  await home(page);
  await action(page, 1).click();
  await expect(action(page, 1)).toContainText("Click again to download - 2.28 GB");
  expect(await page.evaluate(() => (window as any).downloads.calls)).toEqual([]);
  await action(page, 1).click();
  await expect(card(page, 1).getByRole("progressbar")).toBeVisible();
  await expect(card(page, 1)).toContainText("Preparing download…");
  await expect(action(page, 1)).toBeDisabled();
  await expect(action(page, 2)).toBeDisabled();
  await page.evaluate(() => (window as any).downloads.progress(570_000_000));
  await expect(card(page, 1).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  await card(page, 1).screenshot({ path: testInfo.outputPath("inline-progress.png") });
  await page.evaluate(() => (window as any).downloads.progress(570_000_000, { offline: true }));
  await expect(card(page, 1)).toContainText("Offline. Waiting for a connection");
  await page.evaluate(() => (window as any).downloads.progress(2_280_000_000));
  await expect(card(page, 1)).toContainText("Verifying files…");
  await expect(card(page, 1)).not.toHaveClass(/installed/);
  await page.evaluate(() => (window as any).downloads.finish());
  await expect(card(page, 1)).toHaveClass(/installed/);
  await expect(action(page, 1)).toContainText("Ready to use");
  await expect(card(page, 1).getByRole("progressbar")).toHaveCount(0);
  expect(await page.evaluate(() => ({ calls: (window as any).downloads.calls, opens: (window as any).downloads.opens, routes: (window as any).downloads.routes }))).toEqual({ calls: [{ id: "model-1", options: { explicitUnsafeOverride: false } }], opens: [], routes: 0 });
  await action(page, 1).click();
  await expect(page.getByRole("heading", { name: "Start a new chat with Gemma 3 4B?" })).toBeVisible();
  expect(await page.evaluate(() => (window as any).downloads.opens)).toEqual([]);
  await page.getByRole("button", { name: "Start new chat", exact: true }).click();
  expect(await page.evaluate(() => (window as any).downloads.opens)).toEqual([{ id: "model-1", options: { resetSession: true } }]);
  expect(errors).toEqual([]);
});

test("pause, resume, retry, and chooser reopening retain download state", async ({ page }) => {
  await home(page);
  await action(page, 1).click(); await action(page, 1).click();
  await page.evaluate(() => (window as any).downloads.progress(570_000_000));
  await page.locator(".chooser-heading").getByRole("button", { name: "Cancel" }).click();
  await page.getByRole("button", { name: "New Instance", exact: true }).click();
  await expect(card(page, 1).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  await card(page, 1).getByRole("button", { name: "Pause", exact: true }).click();
  await expect(action(page, 1)).toContainText("Download paused. Click to resume");
  await action(page, 1).click(); await action(page, 1).click();
  await page.evaluate(() => (window as any).downloads.fail());
  await expect(card(page, 1).getByRole("alert")).toHaveText("Connection lost. Try again.");
  await expect(action(page, 1)).toContainText("Download failed. Click to retry");
  await action(page, 1).click(); await action(page, 1).click();
  await page.evaluate(() => (window as any).downloads.finish());
  await expect(action(page, 1)).toContainText("Ready to use");
  expect(await page.evaluate(() => (window as any).downloads.calls.length)).toBe(3);
});

test("keyboard confirmation, device warnings, unavailable models, and first download", async ({ page }) => {
  await home(page, false);
  await action(page, 1).focus(); await page.keyboard.press("Enter");
  await expect(action(page, 1)).toContainText("Click again to download");
  await page.keyboard.press("Escape");
  await expect(action(page, 1)).toContainText("Click to download");
  await action(page, 1).click(); await action(page, 2).click();
  await expect(action(page, 1)).not.toContainText("Click again");
  await page.evaluate(() => {
    (window as any).downloads.model(1, { fit: "blocked" });
    (window as any).downloads.model(3, { catalogAvailable: false });
    (window as any).downloads.model(2, { fit: "uncertain", requiresOomRetry: true });
  });
  await expect(action(page, 1)).toBeDisabled(); await expect(action(page, 3)).toBeDisabled();
  await expect(card(page, 2).getByRole("alert")).toContainText("The browser may run out of memory");
  expect(await page.evaluate(() => (window as any).downloads.calls)).toEqual([]);
  await action(page, 2).focus(); await page.keyboard.press("Enter");
  expect(await page.evaluate(() => (window as any).downloads.calls)).toEqual([{ id: "model-2", options: { explicitUnsafeOverride: true } }]);
  await page.evaluate(() => (window as any).downloads.finish());
  await expect(action(page, 2)).toContainText("Ready to use");
});

test("two padded 2x2 groups, purple installed cards, readable themes and narrow layouts", async ({ page }, testInfo) => {
  await home(page);
  for (const theme of ["light", "dark"]) {
    await page.getByRole("button", { name: theme === "light" ? "Light" : "Dark", exact: true }).click();
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await expect(page.locator("#new-chat-title")).toHaveCSS("color", theme === "light" ? "rgb(20, 24, 34)" : "rgb(241, 243, 250)");
      await expect(page.locator(".model-choice-group")).toHaveCount(2);
      await expect(page.locator(".model-choice.installed")).toHaveCount(3);
      for (const group of await page.locator(".model-choices").all()) {
        await expect(group.locator(".model-choice")).toHaveCount(4);
        const boxes = await group.locator(".model-choice").evaluateAll(els => els.map(el => { const rect = el.getBoundingClientRect(); return { x: rect.x, y: rect.y, height: rect.height }; }));
        if (width >= 768) {
          expect(boxes[0].y).toBe(boxes[1].y); expect(boxes[2].y).toBe(boxes[3].y);
          expect(boxes[0].x).toBe(boxes[2].x); expect(boxes[1].x).toBeGreaterThan(boxes[0].x);
        } else expect(boxes.every(box => box.x === boxes[0].x)).toBe(true);
        expect(boxes.every(box => box.height >= 100)).toBe(true);
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      expect((await new AxeBuilder({ page }).include(".new-chat-panel").analyze()).violations).toEqual([]);
      await page.locator(".new-chat-panel").screenshot({ path: testInfo.outputPath(`model-groups-${theme}-${width}.png`) });
    }
  }
  await action(page, 1).click();
  expect((await new AxeBuilder({ page }).include(".new-chat-panel").analyze()).violations).toEqual([]);
  expect(await page.locator(".new-chat-panel").innerText()).not.toContain("\u2014");
});
