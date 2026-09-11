import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
});

test("fresh pages default to dark even on a light system", async ({ page }) => {
  for (const path of ["/", "/credits", "/app?layoutFixture=base"]) {
    await page.goto(`${devUrl}${path}`);
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    expect(await page.evaluate(() => localStorage.getItem("drowse.theme"))).toBeNull();
  }
  await page.emulateMedia({ colorScheme: "dark" });
  await page.emulateMedia({ colorScheme: "light" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("an explicit light choice survives navigation and reload", async ({ page }) => {
  await page.goto(devUrl);
  await page.getByRole("button", { name: "Light", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(page.getByRole("button", { name: "Light", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.goto(`${devUrl}/app?layoutFixture=base`);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});

test("the runtime also defaults to dark without the bootstrap or available storage", async ({ page }) => {
  await page.goto(`${devUrl}/outside-the-workbench`);
  const theme = await page.evaluate(async url => {
    delete document.documentElement.dataset.theme;
    Object.defineProperty(window, "localStorage", { configurable: true, get() { throw new Error("Storage unavailable"); } });
    const { initializeTheme } = await import(url);
    return initializeTheme();
  }, `/@fs${resolve("src/lib/theme.ts")}?bootstrap-test`);
  expect(theme).toBe("dark");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("cross-tab choices sync and removing a choice restores dark", async ({ page, context }) => {
  await page.goto(devUrl);
  const second = await context.newPage();
  await second.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await second.goto(devUrl);
  await page.getByRole("button", { name: "Light", exact: true }).click();
  await expect(second.locator("html")).toHaveAttribute("data-theme", "light");
  await page.evaluate(() => localStorage.removeItem("drowse.theme"));
  await expect(second.locator("html")).toHaveAttribute("data-theme", "dark");
  await second.close();
});
