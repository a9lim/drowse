import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

async function expectGradients(page: Page) {
  const flat = await page.locator("button:visible").evaluateAll(buttons => buttons
    .filter(button => !getComputedStyle(button).backgroundImage.includes("linear-gradient("))
    .map(button => button.getAttribute("aria-label") || button.textContent?.trim()));
  expect(flat).toEqual([]);
}

test("filled actions have visible but restrained top-down shading", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/credits");
  const action = page.getByRole("link", { name: "Contribute on GitHub" });
  for (const theme of ["light", "dark"]) {
    await page.getByRole("button", { name: theme === "light" ? "Light" : "Dark", exact: true }).click();
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.mouse.move(0, 0);
      const screenshot = await action.screenshot({ path: testInfo.outputPath(`filled-action-${theme}-${width}.png`) });
      const shading = await page.evaluate(async bytes => {
        const bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: "image/png" }));
        const canvas = document.createElement("canvas");
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(bitmap, 0, 0);
        const top = context.getImageData(Math.floor(canvas.width / 2), 6, 1, 1).data;
        const bottom = context.getImageData(Math.floor(canvas.width / 2), canvas.height - 7, 1, 1).data;
        bitmap.close();
        return Math.max(...[0, 1, 2].map(channel => top[channel] - bottom[channel]));
      }, [...screenshot]);
      expect(shading).toBeGreaterThanOrEqual(10);
      expect(shading).toBeLessThanOrEqual(28);
    }
  }
});

test("compact controls keep one gentle size-aware gradient", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/credits");
  const toggle = page.getByRole("group", { name: "Appearance" });
  for (const theme of ["Light", "Dark"]) {
    await toggle.getByRole("button", { name: theme, exact: true }).click();
    await expect(toggle).toHaveCSS("background-image", "none");
    await expect(toggle.locator(".selection-indicator")).toHaveCSS("background-image", "none");
    for (const button of await toggle.getByRole("button").all()) {
      const gradient = await button.evaluate(element => getComputedStyle(element).backgroundImage);
      expect(gradient.match(/linear-gradient/g)).toHaveLength(1);
      expect(gradient).toContain("-48px");
      expect(gradient).toContain("96px");
    }
    await page.mouse.move(0, 0);
    await toggle.screenshot({ path: testInfo.outputPath(`compact-gradient-${theme}.png`) });
  }
});

test("all button variants retain the shared gradient through interaction states", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/credits");
  await page.evaluate(async url => {
    const [{ mount, createRawSnippet }, { default: Button }] = await Promise.all([
      import("/@id/svelte"), import(url),
    ]);
    document.body.replaceChildren();
    const target = document.createElement("main");
    target.style.cssText = "display:flex;flex-wrap:wrap;gap:16px;padding:24px";
    document.body.append(target);
    for (const variant of ["solid", "ghost", "flat", "danger"]) {
      mount(Button, { target, props: { variant, children: createRawSnippet(() => ({ render: () => `<span>${variant}</span>` })) } });
    }
    mount(Button, { target, props: { disabled: true, children: createRawSnippet(() => ({ render: () => "<span>Disabled</span>" })) } });
  }, `/@fs${resolve("src/lib/ui/Button.svelte")}`);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    await expectGradients(page);
    for (const button of await page.locator("button:not(:disabled)").all()) {
      await button.hover();
      await expect(button).toHaveCSS("background-image", /linear-gradient/);
      await page.mouse.down();
      await expect(button).toHaveCSS("background-image", /linear-gradient/);
      await page.mouse.up();
      await button.focus();
      await expect(button).toHaveCSS("background-image", /linear-gradient/);
    }
    await page.mouse.move(0, 0);
    await page.locator("main").screenshot({ path: testInfo.outputPath(`button-gradients-${theme}.png`) });
  }
  await page.emulateMedia({ forcedColors: "active" });
  for (const button of await page.locator("button").all()) await expect(button).toHaveCSS("background-image", "none");
});

test("workbench and dialog buttons use gradients in both appearances", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.locator(".shell")).toBeVisible();
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await expectGradients(page);
      for (const name of ["download_chat", "system_prompt", "save_conversation"]) {
        await page.evaluate(async ({ url, name }) => { (await import(url)).openDrawer(name); }, { url: `/@fs${resolve("src/lib/stores.svelte.ts")}`, name });
        await expect(page.locator(".drawer")).toBeVisible();
        await expectGradients(page);
        await page.locator(".drawer").screenshot({ path: testInfo.outputPath(`${name}-${theme}-${width}.png`) });
        await page.evaluate(async url => { (await import(url)).closeDrawer(); }, `/@fs${resolve("src/lib/stores.svelte.ts")}`);
        await expect(page.locator(".drawer")).toHaveCount(0);
      }
    }
  }
});
