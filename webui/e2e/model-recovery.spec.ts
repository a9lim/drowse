import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { resolve } from "node:path";

const homeModule = `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`;
const libraryModule = `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`;

async function mountRecovery(page: Page, fits = ["recommended"]) {
  await page.goto("http://127.0.0.1:4176/outside-the-workbench");
  await page.evaluate(async ({ homeModule, libraryModule, fits }) => {
    const [{ default: Home }, { mount }, { SvelteMap }, { conversationLibrary }] = await Promise.all([
      import(homeModule), import("/e2e/svelte-runtime.ts"), import("/e2e/svelte-runtime.ts"), import(libraryModule),
    ]);
    localStorage.setItem("drowse.entry.v1", JSON.stringify({ version: 1, completedAt: Date.now(), lastModelVariantId: "model-1" }));
    const models = fits.map((fit, index) => ({
      id: `model-${index + 1}`, modelId: `model-${index + 1}`, name: `Model ${index + 1}`,
      tier: "fastest", fit, installed: false, setupComplete: false,
      reason: fit === "blocked" ? "Not supported on this device" : "This model may exceed available memory",
      size: "1 GB", context: "2K", contextTokens: 2048, language: "English",
      sourceUrl: "https://huggingface.co/a9lim/polythetic", license: "Apache-2.0",
      modelDownloadBytes: 100, firstRunBytes: 100, remainingDownloadBytes: 100, firstRunPacks: [],
    }));
    const records = models.map((model, index) => ({
      id: `chat-${index + 1}`, name: `Saved chat ${index + 1}`, modelId: model.id,
      avatarSeed: model.id, messageCount: 4, threadCount: 2, updatedAt: Date.now(), createdAt: Date.now(),
    }));
    conversationLibrary.listSummaries = async () => ({ conversations: records, issues: [] });
    const state = new SvelteMap([["snapshot", {
      phase: "supported", headline: "Ready", detail: "Ready", models, checks: [],
      download: { available: true, phase: "idle", reason: "Ready" },
      runtime: { available: true, phase: "unloaded", reason: "Ready" },
      storage: { availableBytes: 10_000_000_000, persisted: true },
    }]]);
    const current = () => state.get("snapshot");
    const patch = (next: any) => state.set("snapshot", { ...current(), ...next });
    let resolveDownload: (() => void) | null = null;
    let rejectDownload: ((error: Error) => void) | null = null;
    const calls: Array<{ id: string; options: unknown }> = [];
    const opened: string[] = [];
    let redirects = 0;
    const controller = {
      current,
      capabilities: () => ({ signals: { appleMobile: false } }),
      async check() {},
      async retryPersistence() { return true; },
      async open(id: string) { opened.push(id); },
      async download(id: string, options: unknown) {
        calls.push({ id, options });
        patch({ download: { available: true, phase: "downloading", modelVariantId: id, reason: "Downloading" } });
        await new Promise<void>((resolve, reject) => { resolveDownload = resolve; rejectDownload = reject; });
      },
      async cancelDownload() {
        patch({ download: { ...current().download, phase: "paused" } });
        resolveDownload?.();
      },
    };
    (window as any).__recovery = {
      calls, opened, records,
      redirects: () => redirects,
      progress(percent: number) {
        patch({ download: { ...current().download, progress: {
          bytesReceived: percent, bytesTotal: 100, files: [], etaSeconds: [10, 20], calculatingEta: false,
        } } });
      },
      finish() {
        const id = current().download.modelVariantId;
        patch({
          models: current().models.map((model: any) => model.id === id ? { ...model, installed: true, setupComplete: true } : model),
          download: { ...current().download, phase: "installed" },
        });
        resolveDownload?.();
      },
      fail() {
        patch({ download: { ...current().download, phase: "failed", reason: "Connection interrupted" } });
        rejectDownload?.(new Error("Connection interrupted"));
      },
      lock() { patch({ download: { available: false, phase: "locked", reason: "Connect to download models" } }); },
    };
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(Home, { target, props: { controller, get snapshot() { return current(); }, onChooseModels() { redirects++; } } });
  }, { homeModule, libraryModule, fits });
  await expect(page.locator(".model-recovery")).toBeVisible();
  await expect(page.locator(".chat-card")).toHaveCount(fits.length);
}

const calls = (page: Page) => page.evaluate(() => (window as any).__recovery.calls);

test("saved chat action rows match the card inset at every layout size", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mountRecovery(page);
  const card = page.locator(".chat-card").first();
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      const spacing = await card.evaluate(element => {
        const primary = element.querySelector(".chat-footer > .primary")!.getBoundingClientRect();
        const download = element.querySelector(".backup-action")!.getBoundingClientRect();
        const remove = element.querySelector(".delete-control")!.getBoundingClientRect();
        const color = element.querySelector(".chat-color")!.getBoundingClientRect();
        const box = element.getBoundingClientRect();
        const css = getComputedStyle(element);
        const footer = getComputedStyle(element.querySelector(".chat-footer")!);
        return {
          inset: parseFloat(css.paddingBottom),
          row: Math.abs(download.top + download.height / 2 - primary.top - primary.height / 2),
          rowGap: download.top - primary.bottom,
          gap: parseFloat(footer.gap),
          bottom: box.bottom - Math.max(remove.bottom, download.bottom, color.bottom),
          buttons: Math.max(remove.left - download.right, remove.top - download.bottom),
          overflow: element.scrollWidth > element.clientWidth,
        };
      });
      expect(spacing.inset).toBe(16);
      if (spacing.row > 1) expect(spacing.rowGap).toBeGreaterThanOrEqual(spacing.gap - 1);
      expect(spacing.bottom).toBeCloseTo(spacing.inset, 0);
      expect(spacing.buttons).toBeGreaterThanOrEqual(spacing.gap - 1);
      expect(spacing.overflow).toBe(false);
      for (const button of await card.locator(".primary, .backup-action, .delete-control").all()) {
        await expect(button).toHaveCSS("background-image", /^none(?:, none)*$/);
      }
      await card.screenshot({ path: testInfo.outputPath(`card-spacing-${theme}-${width}.png`) });
    }
  }
  await card.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(card.locator(".chat-actions")).toHaveCSS("gap", "16px");
  expect(await card.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
});

test("a single missing model downloads in the notice with visible progress", async ({ page }, testInfo) => {
  await mountRecovery(page);
  const notice = page.locator(".model-recovery");
  await expect(notice.getByRole("checkbox")).toHaveCount(0);
  await expect(page.getByText("Backups include", { exact: false })).toHaveCount(0);
  await notice.getByRole("button", { name: "Download model", exact: true }).click();
  await expect.poll(() => calls(page)).toEqual([{ id: "model-1", options: { explicitUnsafeOverride: false } }]);
  await expect(notice.getByRole("progressbar")).toBeVisible();
  await page.evaluate(() => (window as any).__recovery.progress(50));
  await expect(notice.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const width of [1440, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      expect(await notice.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
      await notice.screenshot({ path: testInfo.outputPath(`recovery-${theme}-${width}.png`) });
    }
    expect((await new AxeBuilder({ page }).include(".model-recovery").analyze()).violations).toEqual([]);
  }
  await page.evaluate(() => (window as any).__recovery.finish());
  await expect(notice).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open chat", exact: true })).toBeVisible();
  expect(await page.evaluate(() => (window as any).__recovery.redirects())).toBe(0);
  expect(await page.evaluate(() => (window as any).__recovery.opened)).toEqual([]);
});

test("multiple missing models download only the selected queue and resume the current model", async ({ page }, testInfo) => {
  await mountRecovery(page, ["recommended", "eligible", "eligible", "blocked"]);
  const notice = page.locator(".model-recovery");
  await expect(notice.getByRole("checkbox")).toHaveCount(4);
  await expect(notice.getByRole("checkbox", { name: /Model 4/ })).toBeDisabled();
  await expect(notice.getByRole("button", { name: "Download selected" })).toBeDisabled();
  await notice.getByRole("checkbox", { name: /Model 1/ }).check();
  await notice.getByRole("checkbox", { name: /Model 3/ }).check();
  await notice.screenshot({ path: testInfo.outputPath("recovery-choices.png") });
  await notice.getByRole("button", { name: "Download selected" }).click();
  await expect.poll(async () => (await calls(page)).map((call: any) => call.id)).toEqual(["model-1"]);
  await expect(notice).toContainText("1 queued");
  await page.evaluate(() => (window as any).__recovery.finish());
  await expect.poll(async () => (await calls(page)).map((call: any) => call.id)).toEqual(["model-1", "model-3"]);
  await notice.getByRole("button", { name: "Pause download" }).click();
  await expect(notice.getByText("Download paused")).toBeVisible();
  await notice.getByRole("button", { name: "Resume download" }).click();
  await expect.poll(async () => (await calls(page)).map((call: any) => call.id)).toEqual(["model-1", "model-3", "model-3"]);
  await page.evaluate(() => (window as any).__recovery.finish());
  await expect(notice.getByRole("checkbox")).toHaveCount(2);
  await expect(page.locator('[data-saved-conversation="chat-2"]')).toContainText("Download required");
  await expect(page.locator('[data-saved-conversation="chat-3"]')).not.toContainText("Download required");
  expect(await page.evaluate(() => (window as any).__recovery.redirects())).toBe(0);
});

test("saved chat recovery retries inline and opens the original chat only after success", async ({ page }) => {
  await mountRecovery(page);
  await page.locator('[data-saved-conversation="chat-1"]').getByRole("button", { name: "Download model", exact: true }).click();
  await expect.poll(() => calls(page)).toHaveLength(1);
  await page.evaluate(() => (window as any).__recovery.fail());
  await expect(page.locator(".model-recovery").getByRole("alert")).toHaveText("Connection interrupted");
  expect(await page.evaluate(() => (window as any).__recovery.opened)).toEqual([]);
  await page.getByRole("button", { name: "Retry download" }).click();
  await expect.poll(() => calls(page)).toHaveLength(2);
  await page.evaluate(() => (window as any).__recovery.finish());
  await expect.poll(() => page.evaluate(() => (window as any).__recovery.opened)).toEqual(["model-1"]);
  expect(await page.evaluate(() => (window as any).__recovery.redirects())).toBe(0);
  expect(await page.evaluate(() => (window as any).__recovery.records.map((record: any) => record.id))).toEqual(["chat-1"]);
  expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem("drowse.pending-conversation.v1") ?? "null"))).toMatchObject({ id: "chat-1", modelId: "model-1" });
});

test("recovery preserves memory warnings and download availability gates", async ({ page }) => {
  await mountRecovery(page, ["uncertain"]);
  const notice = page.locator(".model-recovery");
  await notice.getByRole("button", { name: "Download model", exact: true }).click();
  await expect(notice.getByRole("alert")).toContainText("may run out of memory");
  expect(await calls(page)).toEqual([]);
  await notice.getByRole("button", { name: "Cancel", exact: true }).click();
  expect(await calls(page)).toEqual([]);
  await notice.getByRole("button", { name: "Download model", exact: true }).click();
  await notice.getByRole("button", { name: "Download anyway" }).click();
  await expect.poll(() => calls(page)).toEqual([{ id: "model-1", options: { explicitUnsafeOverride: true } }]);
  await notice.getByRole("button", { name: "Pause download" }).click();
  await notice.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.evaluate(() => (window as any).__recovery.lock());
  await expect(notice).toContainText("Connect to download models");
  await expect(notice.getByRole("button", { name: "Download model", exact: true })).toBeDisabled();
});
