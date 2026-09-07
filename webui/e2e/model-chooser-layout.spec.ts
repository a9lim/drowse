import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

test("new chat separates all model types without clipping and keeps downloads inline", async ({ page }, testInfo) => {
  test.setTimeout(60_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/outside-the-workbench");
  await expect(page.getByRole("heading", { name: "Page not found", exact: true })).toBeVisible();
  await page.evaluate(async moduleUrl => {
    const [{ default: HostedHome }, { mount }] = await Promise.all([
      import(moduleUrl), import("/e2e/svelte-runtime.ts"),
    ]);
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    const models = [
      ["gemma-chat", "Gemma 3 1B", "chat", false],
      ["gemma-large", "Gemma 3 4B", "chat", false],
      ["qwen-chat", "Qwen3 1.7B", "chat", false],
      ["qwen-large", "Qwen3 4B", "chat", false],
      ["gemma-base", "Gemma 3 1B PT", "base", true],
      ["pythia-base", "Pythia 70M Deduped Base", "base", true],
      ["gpt-base", "GPT-2 Base", "base", true],
      ["qwen-base", "Qwen 3.5 2B Base", "base", true],
    ].map(([id, name, modelType, setupComplete]) => ({ id, modelId: id, name, modelType, setupComplete, fit: "supported" }));
    mount(HostedHome, { target, props: {
      controller: {
        capabilities: () => ({ signals: { appleMobile: false } }),
        retryPersistence: async () => false,
        check: async () => {},
        download: async id => { document.body.dataset.downloaded = id; },
        open: async (id, options) => { document.body.dataset.opened = JSON.stringify({ id, options }); },
      },
      snapshot: {
        phase: "supported", headline: "Ready", detail: "Ready", checks: [], models,
        download: { available: true, phase: "idle", reason: "Ready" },
        runtime: { available: true, phase: "unloaded", reason: "Ready" },
        storage: { availableBytes: 2_000_000_000, persisted: false },
      },
      onChooseModels(id) { document.body.dataset.chosen = id; },
    } });
  }, `/@fs/${resolve("src/hosted/ui/HostedHome.svelte")}`);
  await page.getByRole("button", { name: "New Instance", exact: true }).click();
  const chooser = page.getByRole("region", { name: "Choose a model", exact: true });
  await expect(chooser.getByRole("region", { name: "Chat models", exact: true }).getByRole("button")).toHaveCount(4);
  await expect(chooser.getByRole("region", { name: "Base models", exact: true }).locator(".model-choices button")).toHaveCount(4);
  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const width of [320, 390, 760, 960, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await expect.poll(() => page.evaluate(() => {
        const clipped = Array.from(document.querySelectorAll(".model-choices button, .model-choice-copy, .notice, .chooser-heading, .hero-actions"))
          .filter(el => el.scrollWidth > el.clientWidth + 1)
          .map(el => el.className);
        const heading = document.querySelector("#new-chat-title").getBoundingClientRect();
        const firstGroup = document.querySelector(".model-choice-group").getBoundingClientRect();
        return { clipped, overflow: document.documentElement.scrollWidth > innerWidth + 1, headingAboveChoices: heading.bottom <= firstGroup.top };
      })).toEqual({ clipped: [], overflow: false, headingAboveChoices: true });
      if (width === 320 || width === 1440) await chooser.screenshot({ path: testInfo.outputPath(`chooser-${theme}-${width}.png`) });
    }
  }
  const help = chooser.getByRole("button", { name: "What are base models?" });
  const tooltip = page.getByRole("tooltip").filter({ hasText: "Base models predict what comes next" });
  await help.hover();
  await expect(tooltip).toBeVisible();
  await help.click();
  await expect(tooltip).toBeVisible();
  await page.keyboard.press("Escape");
  await page.mouse.move(0, 0);
  await expect(tooltip).toBeHidden();
  await chooser.getByRole("button", { name: "Qwen3 4B Click to download", exact: true }).focus();
  await page.keyboard.press("Tab");
  await expect(help).toBeFocused();
  await expect(tooltip).toBeVisible();
  await page.keyboard.press("Enter");
  await expect(tooltip).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(tooltip).toBeHidden();
  const download = chooser.locator('[data-model-id="gemma-chat"] .model-choice-button');
  await download.click();
  await expect(download).toContainText("Click again to download");
  expect(await page.locator("body").getAttribute("data-downloaded")).toBeNull();
  await download.click();
  await expect(page.locator("body")).toHaveAttribute("data-downloaded", "gemma-chat");
  expect(await page.locator("body").getAttribute("data-chosen")).toBeNull();
  await chooser.getByRole("button", { name: "Gemma 3 1B PT Ready to use", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Start a new chat with Gemma 3 1B PT?" })).toBeVisible();
  await page.getByRole("button", { name: "Start new chat", exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute("data-opened", JSON.stringify({ id: "gemma-base", options: { resetSession: true } }));
});
