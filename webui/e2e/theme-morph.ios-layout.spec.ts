import { expect, test, type Locator } from "@playwright/test";
import { openWorkspaceMenu } from "./workbench-navigation";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";

for (const motion of ["reduce", "no-preference"] as const) {
test(`theme changes update inherited recorded-demo text colors (${motion})`, async ({ page }) => {
  await page.emulateMedia({ reducedMotion: motion });
  await page.goto(`${devUrl}/`);
  await expect(page.locator(".demo-panel .result .morph-source").first()).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  for (const width of [390, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    if (motion === "no-preference") {
      await page.locator(".demo-panel").scrollIntoViewIfNeeded();
      await expect(page.locator(".demo-panel .morph-text[data-morph-active]").first()).toBeVisible();
    }
    for (const theme of ["light", "dark", "light", "dark"]) {
      await page.evaluate(async ({ url, theme }) => {
        (await import(url)).setTheme(theme);
        await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
      }, { url: `/@fs${resolve("src/lib/theme.ts")}`, theme });
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      await expect(page.locator("html")).not.toHaveAttribute("data-theme-transition", /.+/);
      const mismatches = await page.locator(".demo-panel").evaluate(panel => {
        const expected = getComputedStyle(panel).color;
        return [...panel.querySelectorAll(".alpha, .alpha .morph-text, .alpha .morph-source, .result, .result .morph-text, .result .morph-source")]
          .map(element => ({
            text: element.textContent?.slice(0, 40), actual: getComputedStyle(element).color,
            expected: element.matches(".morph-text[data-morph-active] > .morph-source") ? "rgba(0, 0, 0, 0)" : expected,
          }))
          .filter(item => item.actual !== item.expected);
      });
      expect(mismatches, `${theme} ${width}`).toEqual([]);
    }
  }
});
}

async function expectIcon(button: Locator, theme: "light" | "dark") {
  await expect(button).toHaveAccessibleName(`Switch to ${theme === "dark" ? "light" : "dark"} theme`);
  await expect(button.locator("svg")).toHaveCount(1);
  await expect(button.locator("svg")).toHaveAttribute("data-theme-icon", theme === "dark" ? "moon" : "sunny");
  await expect(button.locator("svg")).toHaveCSS("width", "21px");
  await expect(button.locator("svg")).toHaveCSS("height", "21px");
  await expect(button.locator("clipPath circle")).toHaveAttribute("r", theme === "dark" ? "8" : "5");
  await expect(button.locator(".rays")).toHaveAttribute("opacity", theme === "dark" ? "0" : "1");
  await expect(button).toHaveCSS("background-image", "none");
  const rect = await button.boundingBox();
  expect(rect!.width).toBeGreaterThanOrEqual(44);
  expect(rect!.height).toBeGreaterThanOrEqual(44);
}

test("one theme button works on every public surface and screen size", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const path of ["/", "/credits", "/contact", "/missing-page", "/app?layoutFixture=setup"]) {
    await page.goto(`${devUrl}${path}`);
    const button = page.locator(".theme-toggle button");
    await expect(button).toHaveCount(1);
    await expect(page.locator(".theme-toggle .selection-indicator")).toHaveCount(0);
    for (const width of [320, 390, 820, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await expectIcon(button, "dark");
      await button.click({ trial: true });
      const before = await button.boundingBox();
      await button.click();
      await expectIcon(button, "light");
      expect(await button.boundingBox(), `${path} at ${width}px`).toEqual(before);
      await button.press("Space");
      await expectIcon(button, "dark");
    }
  }
  await page.locator(".page-header").screenshot({ path: testInfo.outputPath("theme-button-header.png") });
});

test("workbench menu and appearance drawer use the same theme control", async ({ page, browserName }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?layoutFixture=base`);
  await openWorkspaceMenu(page);
  const menu = page.getByRole("dialog", { name: "Workspace menu", exact: true });
  const menuButton = menu.locator(".theme-toggle button");
  await expectIcon(menuButton, "dark");
  if (browserName === "firefox") {
    for (let i = 0; i < 20 && !await menuButton.evaluate(button => button === document.activeElement); i++) {
      await page.keyboard.press("Tab");
    }
  } else {
    await menuButton.focus();
  }
  await expect(menuButton).toBeFocused();
  await page.keyboard.press("Enter");
  await expectIcon(menuButton, "light");
  await expect(menuButton).toBeFocused();
  await expect(menuButton).toHaveCSS("outline-style", "solid");
  await menu.getByRole("button", { name: "Appearance", exact: true }).click();
  const drawerButton = page.locator('.drawer .theme-toggle button');
  await expectIcon(drawerButton, "light");
  await drawerButton.click();
  await expectIcon(drawerButton, "dark");
  await page.keyboard.press("Escape");
  await openWorkspaceMenu(page);
  await expectIcon(menuButton, "dark");
});

for (const fallback of [false, true]) {
  test(`SVG geometry morphs through intermediate states${fallback ? " without view transitions" : " during page transitions"}`, async ({ page }, testInfo) => {
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.clock.install();
    await page.goto(`${devUrl}/`);
    if (fallback) await page.evaluate(() => Object.defineProperty(document, "startViewTransition", { value: undefined }));
    const button = page.locator(".theme-toggle button");
    await expectIcon(button, "dark");
    await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
    await button.click();
    await expect(button.locator("svg")).toHaveAttribute("data-theme-icon", "sunny");
    await page.clock.runFor(160);
    const sunRadius = Number(await button.locator("clipPath circle").getAttribute("r"));
    expect(sunRadius).toBeGreaterThan(5);
    expect(sunRadius).toBeLessThan(8);
    await page.clock.runFor(400);
    await expectIcon(button, "light");
    await button.screenshot({ path: testInfo.outputPath(`sun-${fallback}.png`) });
    await button.click();
    await expect(button.locator("svg")).toHaveAttribute("data-theme-icon", "moon");
    await page.clock.runFor(160);
    const moonRadius = Number(await button.locator("clipPath circle").getAttribute("r"));
    expect(moonRadius).toBeGreaterThan(5);
    expect(moonRadius).toBeLessThan(8);
    await page.clock.runFor(400);
    await expectIcon(button, "dark");
    await button.screenshot({ path: testInfo.outputPath(`moon-${fallback}.png`) });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await button.click();
    await expectIcon(button, "light");
  });
}
