import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

test("purple accents connect conversation, controls, and Loom in both themes", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("http://127.0.0.1:4176/app?fixture=1");
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Tell me about marmots.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const navigation = page.locator(".workspace-nav");

  for (const theme of ["light", "dark"] as const) {
    await page.evaluate(async ({ url, theme }) => { (await import(url)).setTheme(theme); }, {
      url: `/@fs/${resolve("src/lib/theme.ts")}`, theme,
    });
    const accent = theme === "dark" ? "rgb(197, 179, 255)" : "rgb(91, 63, 191)";
    const wash = theme === "dark" ? "rgba(197, 179, 255, 0.16)" : "rgba(91, 63, 191, 0.12)";
    await navigation.getByRole("button", { name: "Conversation", exact: true }).click();
    await composer.fill("Another question");
    await expect(page.getByRole("button", { name: "Send", exact: true })).toHaveCSS("background-color", accent);
    await expect(navigation.getByRole("button", { name: "Conversation", exact: true })).toHaveCSS("background-color", wash);
    await page.screenshot({ path: testInfo.outputPath(`conversation-${theme}.png`) });

    await navigation.getByRole("button", { name: "Controls", exact: true }).click();
    await expect(page.getByRole("group", { name: "Controls section", exact: true }).locator(".tab.on")).toHaveCSS("background-color", wash);
    await page.screenshot({ path: testInfo.outputPath(`controls-${theme}.png`) });
    await navigation.getByRole("button", { name: "Loom", exact: true }).click();
    await page.getByRole("button", { name: "Map All branches", exact: true }).click();
    await expect(page.locator(".current-label").first()).toHaveCSS("color", accent);
    await page.screenshot({ path: testInfo.outputPath(`loom-${theme}.png`) });
    await expect(page.locator('link[rel="icon"]')).toHaveAttribute("href", new RegExp(`-${theme}\\.png$`));
  }
});
