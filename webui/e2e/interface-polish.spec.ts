import { expect, test, type Page, type Locator } from "@playwright/test";
import { resolve } from "node:path";

const dev = "http://127.0.0.1:4176";
const moduleUrl = (path: string) => `/@fs${resolve(`src/${path}`)}`;
const layoutFrame = (page: Page) => page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));

async function workbench(page: Page, fixture = "1") {
  await page.goto(`${dev}/app?layoutFixture=${fixture}`);
  await expect(page.locator(".shell")).toBeVisible();
}

async function openDrawer(page: Page, name: string) {
  await page.evaluate(async ({ url, name }) => {
    const stores = await import(url);
    if (["template_lab", "manifold_builder"].includes(name)) {
      // Exercise native authoring UI without claiming the fixture can fit a model.
      stores.drawerState.open = name;
      stores.drawerState.params = null;
    } else stores.openDrawer(name);
  }, {
    url: moduleUrl("lib/stores/drawers.svelte.ts"), name,
  });
  await expect(page.locator('.drawer[role="dialog"]'), name).toBeVisible();
}

async function targets(root: Locator) {
  return root.locator('button, summary, input:not([type="hidden"]), select, [role="slider"]').evaluateAll(elements => elements.flatMap(element => {
    if (!element.checkVisibility({ visibilityProperty: true, opacityProperty: true }) || element.closest("[inert]")) return [];
    const target = element.matches('input[type="radio"],input[type="checkbox"]') ? element.closest("label") ?? element : element;
    const rect = target.getBoundingClientRect();
    return rect.width < 39.9 || rect.height < 39.9
      ? [{ name: element.getAttribute("aria-label") ?? element.textContent?.slice(0, 60), class: element.className, width: rect.width, height: rect.height }] : [];
  }));
}

test("public headers keep visible branding aligned and navigation reachable", async ({ page }, testInfo) => {
  test.slow();
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const path of ["/", "/credits", "/not-a-page", "/app?layoutFixture=setup"]) {
    await page.goto(`${dev}${path}`);
    await expect(page.locator(".page-header")).toBeVisible();
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await layoutFrame(page);
      const header = page.locator(".page-header");
      expect(await targets(header)).toEqual([]);
      await expect(async () => {
        const result = await header.evaluate(el => {
          const brand = el.querySelector(".page-brand")!.getBoundingClientRect();
          const nav = el.querySelector("nav")!;
          const theme = el.querySelector(".appearance")!.getBoundingClientRect();
          const links = [...nav.querySelectorAll("a")].map(a => a.getBoundingClientRect().toJSON());
          return { brand: brand.toJSON(), brandVisible: getComputedStyle(el.querySelector(".page-brand")!).visibility !== "hidden", theme: theme.toJSON(), links, scrollable: nav.scrollWidth > nav.clientWidth, overflow: document.documentElement.scrollWidth - innerWidth };
        });
        expect(result.overflow).toBeLessThanOrEqual(1);
        for (const link of result.links) {
          expect(link.height).toBeGreaterThanOrEqual(40);
          expect(link.width).toBeGreaterThanOrEqual(40);
          if (result.brandVisible) expect(link.y + link.height / 2).toBeCloseTo(result.brand.y + result.brand.height / 2, 0);
        }
        if (result.brandVisible) expect(result.theme.y + result.theme.height / 2).toBeCloseTo(result.brand.y + result.brand.height / 2, 0);
        if (width === 320) {
          expect(result.brandVisible).toBe(false);
          expect(result.scrollable).toBe(false);
        }
      }).toPass({ timeout: 10_000 });
      if (width === 320) {
        for (const name of ["Contribute", "Chats"]) {
          const link = header.getByRole("link", { name, exact: true });
          await link.focus();
          await layoutFrame(page);
          expect(await link.evaluate(el => {
            const r = el.getBoundingClientRect();
            return document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2) === el;
          })).toBe(true);
        }
      }
    }
    if (path === "/credits") await page.screenshot({ path: testInfo.outputPath("credits-desktop.png") });
  }
  await page.setViewportSize({ width: 320, height: 900 });
  await layoutFrame(page);
  await page.screenshot({ path: testInfo.outputPath("models-phone.png") });
});

test("workbench and drawer controls retain minimum targets at desktop and phone widths", async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.getByRole("textbox", { name: /^Compose as / }).focus();
    await layoutFrame(page);
    expect(await targets(page.locator(".shell"))).toEqual([]);
    expect(await page.getByRole("button", { name: "Generate reply", exact: true }).evaluate(el => {
      const text = el.querySelector(".morph-source")!.firstChild!;
      const range = document.createRange();
      const start = text.textContent!.indexOf("Generate");
      range.setStart(text, start);
      range.setEnd(text, start + "Generate".length);
      return range.getClientRects().length;
    })).toBe(1);
    for (const name of ["advanced_sampling", "local_runtime", "save_conversation", "load_conversation", "appearance", "help", "cast", "correlation", "compare", "system_prompt", "template_lab", "transcript", "health", "subspace", "manifolds", "surface_geometry"]) {
      await openDrawer(page, name);
      const drawer = page.locator('.drawer[role="dialog"]');
      expect(await targets(drawer), `${name}, ${width}`).toEqual([]);
      expect(await drawer.evaluate(el => el.scrollWidth - el.clientWidth), `${name}, ${width}`).toBeLessThanOrEqual(1);
      await page.keyboard.press("Escape");
      await expect(drawer).toHaveCount(0);
    }
    await page.screenshot({ path: testInfo.outputPath(`chat-${width}.png`) });
  }
});

test("help popovers retain their exit and reverse a dismissal without losing Escape priority", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await workbench(page);
  await openDrawer(page, "advanced_sampling");
  await expect.poll(() => page.locator('.drawer[role="dialog"]').evaluate(el => el.getAnimations({ subtree: true }).filter(animation => animation.playState === "running").length)).toBe(0);
  const trigger = page.getByRole("button", { name: "About Top K", exact: true });
  const tip = page.locator(".info-popover").filter({ hasText: /^Top K / });
  await trigger.click();
  await expect(tip).toHaveCSS("opacity", "1");
  const closing = await trigger.evaluate(async el => {
    const { tick } = await import("/e2e/svelte-runtime.ts");
    const tip = document.getElementById(el.getAttribute("aria-describedby")!)!;
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await tick();
    const closing = { open: tip.matches(":popover-open"), inert: tip.inert, duration: getComputedStyle(tip).transitionDuration };
    (el as HTMLButtonElement).click();
    await tick();
    return closing;
  });
  expect(closing).toEqual({ open: true, inert: true, duration: "0.1s" });
  await expect(tip).toHaveCSS("opacity", "1");
  await expect(tip).toHaveJSProperty("inert", false);
  await trigger.press("Escape");
  await expect.poll(() => tip.evaluate(el => el.matches(":popover-open"))).toBe(false);
  await expect(page.locator('.drawer[role="dialog"]')).toBeVisible();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await trigger.click();
  await expect(tip).toHaveCSS("transition-duration", "0s");
  await trigger.press("Escape");
  expect(await tip.evaluate(el => el.matches(":popover-open"))).toBe(false);
});

test("composer state icons crossfade in place and settle after rapid changes", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await workbench(page);
  const icon = page.locator(".input-actions .state-icon");
  await expect(icon.locator(".variant")).toHaveCount(3);
  await expect(icon.locator(".active svg")).toHaveAttribute("data-icon", "conversation");
  expect(await icon.evaluate(el => el.getAnimations({ subtree: true }).length)).toBe(0);
  const input = page.getByRole("textbox", { name: /^Compose as / });
  await input.fill("Hello");
  await expect(icon.locator(".active svg")).toHaveAttribute("data-icon", "send");
  await expect(icon.locator(".variant:not(.active)").first()).toHaveCSS("scale", "0.25");
  await expect(icon.locator(".variant:not(.active)").first()).toHaveCSS("filter", "blur(4px)");
  await expect(icon.locator(".active")).toHaveCSS("transition-duration", "0.3s, 0.3s, 0.3s");
  await input.fill("");
  await input.fill("Again");
  await expect(icon.locator(".active")).toHaveCSS("opacity", "1");
  await expect(icon.locator(".variant")).toHaveCount(3);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await input.fill("");
  expect(await icon.locator(".active").evaluate(el => parseFloat(getComputedStyle(el).transitionDuration))).toBeLessThan(0.001);
  await expect(icon.locator(".active svg")).toHaveAttribute("data-icon", "conversation");
});

test("dismissed drawer controls become inert before their visual exit completes", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await workbench(page);
  await openDrawer(page, "appearance");
  const drawer = page.locator('.drawer[role="dialog"]');
  await expect.poll(() => drawer.evaluate(el => el.getAnimations().length)).toBe(0);
  const inert = await page.evaluate(async url => {
    const drawer = document.querySelector<HTMLElement>('.drawer[role="dialog"]')!;
    (await import(url)).closeDrawer();
    await (await import("/e2e/svelte-runtime.ts")).tick();
    await new Promise(requestAnimationFrame);
    return { inert: drawer.inert, connected: drawer.isConnected };
  }, moduleUrl("lib/stores/drawers.svelte.ts"));
  expect(inert).toEqual({ inert: true, connected: true });
  await expect(drawer).toHaveCount(0);
});

test("token probability popovers fade out through their parent conditional and release focus", async ({ page }) => {
  test.slow();
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await workbench(page, "base");
  await page.getByRole("textbox", { name: "Editable completion buffer" }).fill("A field note: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  await page.getByRole("button", { name: "Inspect tokens", exact: true }).click();
  const token = page.locator(".inspect .tok").first();
  await token.click();
  const popup = page.locator(".token-logits-popover");
  await expect(popup).toHaveCSS("opacity", "1");
  const closing = await popup.getByRole("button", { name: "Close token probabilities" }).evaluate(async el => {
    const { tick } = await import("/e2e/svelte-runtime.ts");
    const popup = el.closest<HTMLElement>(".token-logits-popover")!;
    (el as HTMLButtonElement).click();
    await tick();
    return { inert: popup.inert, open: popup.matches(":popover-open") };
  });
  expect(closing).toEqual({ inert: true, open: true });
  await expect(popup).toHaveCount(0);
  await expect(token).toBeFocused();
  await token.click();
  await expect(popup).toHaveCSS("opacity", "1");
  const next = page.locator(".inspect .tok").nth(1);
  for (const selected of [next, token, next]) {
    await selected.click();
    await expect(popup).toHaveCount(1);
    await expect(popup).toHaveCSS("opacity", "1");
    await expect(popup).toHaveJSProperty("inert", false);
  }
  await page.keyboard.press("Escape");
  await expect(popup).toHaveCount(0);
  await expect(next).toBeFocused();
  await next.click();
  await popup.getByRole("button", { name: "Full token details", exact: true }).click();
  const ribbon = page.getByRole("group", { name: "Token context sequence", exact: true });
  await ribbon.focus();
  await expect(ribbon).toBeFocused();
  const current = await ribbon.locator('[aria-current="true"]').getAttribute("aria-description");
  await ribbon.press("ArrowRight");
  await expect(ribbon.locator('[aria-current="true"]')).not.toHaveAttribute("aria-description", current!);
  expect(errors).toEqual([]);
});

test("page entrances stagger semantic groups only after initial load and survive reversals", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("drowse.entry.v1", JSON.stringify({ version: 1, completedAt: 1, lastModelVariantId: "qwen3-1.7b-fixture" })));
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto(`${dev}/app?layoutFixture=setup&choose=1`);
  await expect(page.locator(".app-shell")).toBeVisible();
  await expect(page.locator(".page-route")).not.toHaveClass(/page-entering/);
  const models = page.locator('.page-route[data-route="models"]');
  await models.getByRole("link", { name: "Chats", exact: true }).evaluate(el => (el as HTMLAnchorElement).click());
  const home = page.locator('.page-route[data-route="home"]');
  await expect(home).toHaveClass(/page-entering/);
  await expect(models).toHaveAttribute("inert", "");
  const groups = await home.locator("[data-page-group]").evaluateAll(elements => elements.map(el => ({ group: el.getAttribute("data-page-group"), delay: getComputedStyle(el).animationDelay, name: getComputedStyle(el).animationName })));
  expect(groups).toEqual([
    { group: "0", delay: "0s", name: "page-group-enter" },
    { group: "2", delay: "0.08s", name: "page-group-enter" },
  ]);
  await home.getByRole("link", { name: "Models", exact: true }).evaluate(el => (el as HTMLAnchorElement).click());
  await expect(models.locator('[data-page-group="1"]')).toHaveCSS("animation-delay", "0.04s");
  await expect(page.locator(".page-route")).toHaveCount(1);
  await expect(models).not.toHaveClass(/page-entering/);
  await expect(models).toHaveJSProperty("inert", false);
  await expect.poll(() => models.evaluate(el => el.getAnimations({ subtree: true }).filter(a => a.playState === "running").length)).toBe(0);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await models.getByRole("link", { name: "Chats", exact: true }).click();
  await expect(page.locator(".page-route")).toHaveCount(1);
  await expect(home.locator("h1")).toBeFocused();
  expect(await home.locator("[data-page-group]").evaluateAll(elements => elements.every(el => getComputedStyle(el).animationName === "none"))).toBe(true);
});
