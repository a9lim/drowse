import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { resolve } from "node:path";

const registryUrl = `/@fs${resolve("src/lib/runtime/registry.ts")}`;
const drawersUrl = `/@fs${resolve("src/lib/stores/drawers.svelte.ts")}`;
const themeUrl = `/@fs${resolve("src/lib/theme.ts")}`;
const creatorName = "Create a concept or scale";

async function prepare(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.getByRole("navigation", { name: "Workspace", exact: true })).toBeVisible({ timeout: 30_000 });
  await page.evaluate(async url => {
    const registry = await import(url);
    registry.getRuntimeCapabilities().operations.fitting = { available: true, reasons: [] };
    const runtime = registry.getRuntimeClient();
    const harness: any = { requests: [], fits: [] };
    (window as any).creationHarness = harness;
    runtime.manifolds.generate = (request: unknown, emit: unknown) => {
      harness.requests.push(request);
      harness.emit = emit;
      return new Promise((resolve, reject) => { harness.finishGeneration = resolve; harness.reject = reject; });
    };
    runtime.manifolds.fit = (namespace: string, name: string, request: unknown, emit: unknown) => {
      harness.fits.push({ namespace, name, request });
      harness.emit = emit;
      return new Promise((resolve, reject) => { harness.finishFit = resolve; harness.reject = reject; });
    };
    registry.getHostedController().cancelFitting = async () => {
      harness.reject(Object.assign(new Error("Creation cancelled"), { code: "FITTING_CANCELLED" }));
    };
  }, registryUrl);
  await page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button", { name: /^Tools/ }).click();
  const search = page.getByRole("searchbox", { name: "Find a tool" });
  await search.fill("no such tool");
  await expect(page.getByText("No tools match", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Clear search", exact: true }).click();
  await page.getByRole("region", { name: "Frequently used tools" }).getByRole("button", { name: /^Create a concept or scale/ }).click();
  await expect(page.getByRole("complementary", { name: creatorName, exact: true })).toBeVisible();
}

async function fillConcepts(page: Page) {
  await page.getByRole("textbox", { name: "Name (required)", exact: true }).fill("pirate");
  await page.getByRole("textbox", { name: "Concepts (at least 2)", exact: false }).fill("pirate, assistant");
}

async function emit(page: Page, data: unknown) {
  await page.evaluate(data => (window as any).creationHarness.emit({ event: "progress", data }), data);
}

test("creator budgets, mode changes, and presentations preserve the draft", async ({ page }) => {
  await prepare(page);
  await page.getByRole("button", { name: "Generate and fit", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Name (required)", exact: true })).toBeFocused();
  await fillConcepts(page);
  await expect(page.getByText("96 examples total", { exact: true })).toBeVisible();
  await page.getByRole("radio", { name: /^More examples/ }).check();
  await expect(page.getByText("192 examples total", { exact: true })).toBeVisible();
  await page.getByRole("radio", { name: /^Thorough/ }).check();
  await expect(page.getByText("384 examples total", { exact: true })).toBeVisible();
  await page.getByRole("radio", { name: /^Custom Choose/ }).check();
  await page.getByRole("spinbutton", { name: "Responses per prompt" }).fill("3");
  await expect(page.getByText("288 examples total", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Use your examples", exact: true }).click();
  await page.getByRole("tab", { name: "Generate examples", exact: true }).click();
  await expect(page.getByRole("spinbutton", { name: "Responses per prompt" })).toHaveValue("3");
  await page.getByRole("button", { name: "Dialog", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: creatorName, exact: true });
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAttribute("aria-modal", "true");
  await expect(page.locator(".app-sidebar")).toHaveAttribute("inert", "");
  await expect(dialog.getByRole("textbox", { name: "Name (required)", exact: true })).toHaveValue("pirate");
  const last = dialog.getByRole("button", { name: "Generate and fit", exact: true });
  await last.focus();
  await last.press("Tab");
  await expect(dialog.getByRole("button", { name: "Side panel", exact: true })).toBeFocused();
  await dialog.getByRole("button", { name: "Side panel", exact: true }).click();
  await page.getByRole("button", { name: "Close drawer", exact: true }).click();
  await page.evaluate(async url => (await import(url)).openToolInSidebar("advanced_sampling"), drawersUrl);
  await expect(page.getByRole("complementary", { name: "Sampling settings", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Close drawer", exact: true }).click();
  await page.evaluate(async url => (await import(url)).openToolInSidebar("manifold_builder"), drawersUrl);
  await expect(page.getByRole("textbox", { name: "Name (required)", exact: true })).toHaveValue("pirate");
  await expect(page.getByRole("spinbutton", { name: "Responses per prompt" })).toHaveValue("3");
  expect(await page.evaluate(() => localStorage.getItem("drowse.tools.presentation"))).toBe("sidebar");
  await page.getByRole("button", { name: "Generate and fit", exact: true }).click();
  expect(await page.evaluate(() => (window as any).creationHarness.requests[0].samples_per_prompt)).toBe(3);
  expect(await page.evaluate(() => (window as any).creationHarness.requests[0].fit_mode)).toBe("pca");
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
});

test("generation progress survives closing, shows a measured ETA, and chains into fitting", async ({ page }, testInfo) => {
  await prepare(page);
  await fillConcepts(page);
  await page.clock.install();
  await page.getByRole("radio", { name: /^More examples/ }).check();
  await page.getByRole("button", { name: "Generate and fit", exact: true }).click();
  expect(await page.evaluate(() => (window as any).creationHarness.requests[0].samples_per_prompt)).toBe(2);
  await emit(page, { message: 'Generating "pirate" response 1/96...' });
  await expect(page.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
  await expect(page.getByText("Estimating this step…", { exact: true })).toBeVisible();
  await page.clock.runFor(3000);
  await emit(page, { message: 'Generating "pirate" response 4/96...' });
  await expect(page.getByText(/About .* left in this step/)).toBeVisible();
  await page.getByRole("button", { name: "Close drawer", exact: true }).click();
  await expect(page.getByRole("progressbar")).toBeVisible();
  await expect(page.getByRole("progressbar")).toHaveAttribute("aria-valuetext", "3 of 192 completed in this step");
  await page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button", { name: /^Controls/ }).click();
  await expect(page.getByRole("progressbar")).toBeVisible();
  await page.clock.runFor(500);
  await page.screenshot({ path: testInfo.outputPath("creation-progress.png") });
  await page.evaluate(() => (window as any).creationHarness.finishGeneration({}));
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.fits.length)).toBe(1);
  await emit(page, { stage: "capturing", completed: 24, total: 192 });
  await expect(page.getByText("Reading examples", { exact: true })).toBeVisible();
  await expect(page.getByRole("progressbar")).toHaveAttribute("aria-valuetext", "24 of 192 completed in this step");
  await page.evaluate(() => (window as any).creationHarness.finishFit({}));
  await expect(page.getByText("Saved and ready to steer or probe.", { exact: true })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button", { name: /^Controls/ })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.evaluate(async url => (await import(url)).openToolInSidebar("manifold_builder"), drawersUrl);
  await expect(page.getByRole("region", { name: "Creation result" }).getByRole("button", { name: "Add probe" })).toBeVisible();
});

test("failed fits retry saved examples and cancellation restores usable controls", async ({ page }) => {
  await prepare(page);
  await fillConcepts(page);
  await page.getByRole("button", { name: "Generate and fit", exact: true }).click();
  await page.evaluate(() => (window as any).creationHarness.finishGeneration({}));
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.fits.length)).toBe(1);
  await page.evaluate(() => (window as any).creationHarness.reject(new Error("Test fit failed")));
  await expect(page.getByRole("alert").filter({ hasText: "Examples were saved, but fitting failed" })).toBeVisible();
  await page.getByRole("button", { name: "Fit saved examples", exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).creationHarness.fits.length)).toBe(2);
  expect(await page.evaluate(() => (window as any).creationHarness.requests.length)).toBe(1);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByText("Creation cancelled", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Fit saved examples", exact: true })).toBeEnabled();
  await expect(page.getByRole("textbox", { name: "Name (required)", exact: true })).toBeEnabled();
});

for (const group of [0, 1, 2, 3]) test(`every available directory tool opens in the side panel and can become a dialog (group ${group + 1}/4)`, async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await prepare(page);
  await page.getByRole("button", { name: "Close drawer", exact: true }).click();
  const directory = page.locator(".tool-directory");
  const labels = (await directory.locator("button:not(:disabled)").allTextContents()).sort();
  for (const label of labels.filter((_, index) => index % 4 === group)) {
    await directory.locator("button:not(:disabled)").filter({ hasText: label.trim() }).first().click();
    const panel = page.locator(".drawer.docked:not(.token-details):not([hidden])");
    await expect(panel, label).toBeVisible();
    await expect(panel).toHaveAttribute("role", "complementary");
    await panel.getByRole("button", { name: "Expand", exact: true }).click();
    await expect(panel).toHaveClass(/workspace-tool/);
    await panel.getByRole("button", { name: "Restore width", exact: true }).click();
    await panel.getByRole("button", { name: "Dialog", exact: true }).click();
    const dialog = page.locator('.drawer[role="dialog"]');
    await expect(dialog, label).toBeVisible();
    await dialog.getByRole("button", { name: "Side panel", exact: true }).click();
    await page.evaluate(async url => (await import(url)).closeDrawer(), drawersUrl);
  }
  expect(labels).toEqual(expect.arrayContaining([expect.stringContaining("Concept library"), expect.stringContaining("Scale library")]));
});

test("creator reflows in both themes and respects keyboard, contrast, and motion settings", async ({ page }, testInfo) => {
  await prepare(page);
  await fillConcepts(page);
  for (const theme of ["dark", "light"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: themeUrl, theme });
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await expect(page.locator("html")).toHaveCSS("color-scheme", theme);
    for (const width of [1440, 900, 320, 312]) {
      await page.setViewportSize({ width, height: 1000 });
      const panel = page.getByRole("complementary", { name: creatorName, exact: true });
      await expect(panel).toBeVisible();
      if (width <= 1100) await expect(page.locator(".workspace-frame")).toHaveCSS("opacity", "0");
      expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth + 1), `${theme} ${width} panel`).toBe(true);
      const budget = page.getByRole("radiogroup", { name: "Example budget" });
      await budget.scrollIntoViewIfNeeded();
      const foreground = await page.locator("body").evaluate(el => getComputedStyle(el).color);
      for (const heading of await budget.locator("strong").all()) await expect(heading).toHaveCSS("color", foreground);
      expect(await budget.evaluate(el => el.scrollWidth <= el.clientWidth + 1), `${theme} ${width} budget`).toBe(true);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${theme} ${width} page`).toBe(true);
      const overflowingLabels = await page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button").evaluateAll(buttons => buttons
        .filter(button => button.scrollWidth > button.clientWidth + 1)
        .map(button => ({ label: button.textContent, contentWidth: button.scrollWidth, width: button.clientWidth })));
      expect(overflowingLabels, `${theme} ${width} navigation labels`).toEqual([]);
      await page.screenshot({ path: testInfo.outputPath(`creator-${theme}-${width}.png`) });
      const axe = await new AxeBuilder({ page }).include(".drawer.docked").withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
      expect(axe.violations, `${theme} ${width}`).toEqual([]);
    }
  }
  await page.getByRole("button", { name: "Generate and fit", exact: true }).click();
  await emit(page, { message: 'Generating "pirate" response 4/48...' });
  expect(await page.locator(".meter-fill").evaluate(el => parseFloat(getComputedStyle(el).transitionDuration))).toBeLessThan(0.001);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(page.locator(".meter-fill")).toHaveCSS("transition-duration", "0.6s");
  await page.emulateMedia({ forcedColors: "active" });
  await expect(page.locator(".meter")).toHaveCSS("border-top-width", "1px");
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
});
