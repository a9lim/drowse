import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

test("landing keeps its full-page shader and quiet button sheen in both themes", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("http://127.0.0.1:4176/");
  await expect(page.locator(".landing-shell")).toBeVisible();
  for (const theme of ["dark", "light"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), {
      url: `/@fs${resolve("src/lib/theme.ts")}`, theme,
    });
    await expect(page.locator(".primary-action").first()).toHaveCSS("background-image", /linear-gradient/);
    const cards = page.locator(".capabilities li, .model-group");
    await expect(cards).toHaveCount(5);
    for (const card of await cards.all()) {
      await expect(card).toHaveCSS("background-color", theme === "dark" ? "rgba(7, 10, 18, 0.24)" : "rgba(255, 255, 255, 0.24)");
      await expect(card).not.toHaveCSS("backdrop-filter", "none");
    }
    await expect(page.locator(".hero-visual")).toBeVisible();
    await page.locator(".capabilities ol").scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath(`landing-material-${theme}.png`) });
  }
});
