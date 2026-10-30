import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { readFile } from "node:fs/promises";

const storesUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;

test("max tokens follows the model limit and rejects repeated oversized edits", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.getByRole("button", { name: /^Controls/ })).toBeVisible();
  const registryUrl = `/@fs${resolve("src/lib/runtime/registry.ts")}`;
  async function setModelLimit(limit: number) {
    await page.evaluate(async ({ storesUrl, registryUrl, limit }) => {
      const registry = await import(registryUrl);
      registry.getHostedController().snapshot.contextTokens = limit;
      const stores = await import(storesUrl);
      stores.sessionState.info = { ...stores.sessionState.info };
      stores.hydrateSamplingFromInfo();
    }, { storesUrl, registryUrl, limit });
  }
  await setModelLimit(2048);
  await page.getByRole("button", { name: /^Controls/ }).click();
  const input = page.getByRole("spinbutton", { name: "Max tokens", exact: true });
  await expect(input).toHaveAttribute("max", "2048");
  for (const key of ["Enter", "Tab", "Enter"]) {
    await input.fill("999999");
    await expect(input).toHaveValue("2048");
    await input.press(key);
    await expect(input).toHaveValue("2048");
  }
  await setModelLimit(1024);
  await expect(input).toHaveAttribute("max", "1024");
  await expect(input).toHaveValue("1024");
  expect(await page.evaluate(async url => {
    const stores = await import(url);
    stores.setSampling("max_tokens", 999999);
    const stored = stores.samplingState.max_tokens;
    stores.samplingState.max_tokens = 999999;
    const sent = stores.buildSamplingPayload().max_tokens;
    await stores.patchSessionDefaults({ max_tokens: 999999 });
    return { stored, sent, persisted: stores.sessionState.info.config.max_tokens };
  }, storesUrl)).toEqual({ stored: 1024, sent: 1024, persisted: 1024 });
});

test("sampling controls retain large supported values on desktop and phone layouts", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.getByRole("button", { name: /^Controls/ })).toBeVisible();
  await page.evaluate(async url => (await import(url)).openDrawer("advanced_sampling"), storesUrl);
  const drawer = page.getByRole("dialog", { name: "Sampling settings", exact: true });
  for (const [label, value] of [["Top K", "262144"], ["Temperature value", "3"], ["Seed", "9007199254740991"]]) {
    const input = drawer.getByRole("spinbutton", { name: label, exact: true });
    await input.fill(value);
    await input.press("Tab");
    await expect(input).toHaveValue(value);
  }
  expect(await page.evaluate(async url => {
    const { samplingState } = await import(url);
    return { top_k: samplingState.top_k, temperature: samplingState.temperature, seed: samplingState.seed };
  }, storesUrl)).toEqual({ top_k: 262144, temperature: 3, seed: Number.MAX_SAFE_INTEGER });
  await expect.poll(() => page.evaluate(async url => {
    const { sessionState } = await import(url);
    return sessionState.info.config.temperature;
  }, storesUrl)).toBe(3);
  expect(await drawer.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  await drawer.screenshot({ path: testInfo.outputPath("sampling-limits.png") });
});

test("technical controls preserve full names, slider behavior, and translucent side panels", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await page.getByRole("button", { name: /^Controls/ }).click();
  const tabs = page.locator(".instrument-head .tabs");
  for (const label of ["Subspace", "Manifold", "SAE", "J-lens"]) {
    const tab = tabs.getByRole("button", { name: label, exact: true });
    await expect(tab).not.toHaveAttribute("title");
    await tab.click();
    await expect(tab).toHaveAttribute("aria-pressed", "true");
  }
  await tabs.getByRole("button", { name: "Subspace", exact: true }).click();
  await page.evaluate(async url => {
    const { steerRack, addSubspaceToRack } = await import(url);
    steerRack.catalog = [{
      namespace: "local", name: "welcoming.reserved", description: "", source: "local", tags: [],
      template_ref: null, fit_mode: "pca", resolved_fit_mode: "pca", is_discover: true,
      domain: { type: "custom" }, domain_label: "", intrinsic_dim: 1, min_nodes: null,
      node_count: 2, node_labels: ["welcoming", "reserved"], node_coords: [[5.72], [-5.72]],
      node_roles: [null, null], node_kinds: [null, null], hyperparams: {}, fitted_models: [],
      tensor_variants: {}, fitted_for_session: true, stale: false,
    }];
    addSubspaceToRack("local/welcoming.reserved");
  }, storesUrl);
  const rack = page.getByRole("region", { name: "Steering rack", exact: true });
  const card = rack.locator(".card");
  await expect(card.locator(".name")).toHaveText("welcoming.reserved");
  await expect(card.getByRole("slider", { name: "c0 coordinate" })).toBeDisabled();
  const strength = rack.getByRole("slider", { name: "Subspace steering strength" });
  const before = Number(await strength.inputValue());
  await strength.focus();
  await strength.press("ArrowRight");
  expect(Number(await strength.inputValue())).toBeCloseTo(before + .05);
  const bounds = (await strength.boundingBox())!;
  await page.mouse.move(bounds.x + bounds.width * .75, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * .9, bounds.y + bounds.height / 2);
  await page.mouse.up();
  expect(Number(await strength.inputValue())).toBeGreaterThan(1.5);
  await card.getByRole("button", { name: "ablate", exact: true }).click();
  await expect(card.getByRole("slider")).toHaveCount(0);
  await card.getByRole("button", { name: "push", exact: true }).click();
  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    for (const width of [1440, 900, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await card.scrollIntoViewIfNeeded();
      expect(await card.evaluate(el => {
        const name = el.querySelector(".name")!;
        const stat = el.querySelector(".statline")!.getBoundingClientRect();
        const trigger = el.querySelector(".trigger-row")!.getBoundingClientRect();
        return name.scrollWidth <= name.clientWidth + 1 && el.scrollWidth <= el.clientWidth + 1 && trigger.top >= stat.bottom;
      }), `${theme} ${width}`).toBe(true);
      expect(await card.locator(".name").evaluate(el => {
        const box = el.getBoundingClientRect();
        return el.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
      }), `name is not covered at ${width}`).toBe(true);
      await card.screenshot({ path: testInfo.outputPath(`steer-card-${theme}-${width}.png`) });
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(page.locator(".app-sidebar")).toHaveCSS("background-color", /\/ 0\.9\)$/);
  await tabs.getByRole("button", { name: "Manifold", exact: true }).click();
  await page.getByRole("button", { name: "Add manifold", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Add manifold", exact: true })).toHaveCSS("background-color", /\/ 0\.9\)$/);
  await page.screenshot({ path: testInfo.outputPath("translucent-drawer.png") });
  await page.emulateMedia({ forcedColors: "active" });
  await expect(page.getByRole("dialog", { name: "Add manifold", exact: true })).toHaveCSS("backdrop-filter", "none");
  expect(await page.getByRole("dialog", { name: "Add manifold", exact: true }).evaluate(el => getComputedStyle(el).backgroundColor)).not.toContain("0.9");
  const sliderSource = await readFile(resolve("src/lib/Slider.svelte"), "utf8");
  expect(sliderSource).toMatch(/::-webkit-slider-thumb\s*\{[^}]*margin-top:\s*-8px;/);
});

test("help supports hover and keyboard focus while Escape keeps the parent drawer open", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.getByRole("button", { name: /^Controls/ })).toBeVisible();
  await expect(page.locator(".boot-loading")).toHaveCount(0);
  await page.evaluate(async url => (await import(url)).openDrawer("advanced_sampling"), storesUrl);
  const drawer = page.getByRole("dialog", { name: "Sampling settings" });
  await expect(drawer).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  await expect.poll(() => drawer.evaluate(el => el.getAnimations({ subtree: true }).filter(animation => animation.playState === "running").length)).toBe(0);
  const help = drawer.getByRole("button", { name: "About Top K", exact: true });
  const tip = drawer.getByRole("tooltip").filter({ hasText: "Top K" });
  await help.hover();
  await expect(tip).toBeVisible();
  await page.mouse.move(0, 0);
  await expect(tip).toBeHidden();
  await help.focus();
  await expect(tip).toBeVisible();
  await help.press("Escape");
  await expect(tip).toBeHidden();
  await expect(drawer).toBeVisible();
  await help.click();
  await expect(tip).toBeVisible();
  await drawer.getByRole("heading").first().click();
  await expect(tip).toBeHidden();
});
