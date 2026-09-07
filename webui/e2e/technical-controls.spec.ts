import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { readFile } from "node:fs/promises";

const storesUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;

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
  expect(await page.locator(".drawer").evaluate(el => getComputedStyle(el).backgroundColor)).not.toContain("0.9");
  const sliderSource = await readFile(resolve("src/lib/Slider.svelte"), "utf8");
  expect(sliderSource).toMatch(/::-webkit-slider-thumb\s*\{[^}]*margin-top:\s*-8px;/);
});

test("help supports hover and keyboard focus while Escape keeps the parent drawer open", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await page.evaluate(async url => (await import(url)).openDrawer("advanced_sampling"), storesUrl);
  const drawer = page.getByRole("dialog", { name: "Sampling settings" });
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
