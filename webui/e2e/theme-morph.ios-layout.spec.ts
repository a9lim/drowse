import { expect, test, type Locator } from "@playwright/test";
import { openWorkspaceMenu } from "./workbench-navigation";

const devUrl = "http://127.0.0.1:4176";

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
    await page.goto(`${devUrl}/`);
    if (fallback) await page.evaluate(() => Object.defineProperty(document, "startViewTransition", { value: undefined }));
    const button = page.locator(".theme-toggle button");
    await expectIcon(button, "dark");
    await button.evaluate(button => {
      const frames: number[] = [];
      (window as any).themeMorphFrames = frames;
      const circle = button.querySelector("clipPath circle")!;
      const observer = new MutationObserver(() => frames.push(Number(circle.getAttribute("r"))));
      observer.observe(circle, { attributes: true, attributeFilter: ["r"] });
      (window as any).stopThemeMorphObservation = () => observer.disconnect();
    });
    await button.click();
    await expectIcon(button, "light");
    expect(await page.evaluate(() => (window as any).themeMorphFrames.some((r: number) => r > 5 && r < 8))).toBe(true);
    await button.screenshot({ path: testInfo.outputPath(`sun-${fallback}.png`) });
    await button.click();
    await expectIcon(button, "dark");
    await button.screenshot({ path: testInfo.outputPath(`moon-${fallback}.png`) });
    await page.evaluate(() => (window as any).stopThemeMorphObservation());
    await page.emulateMedia({ reducedMotion: "reduce" });
    await button.click();
    await expectIcon(button, "light");
  });
}
