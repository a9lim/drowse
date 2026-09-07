import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("cards and buttons share the credits material in both appearances", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/credits");
  for (const theme of ["light", "dark"]) {
    await page.getByRole("button", { name: theme === "light" ? "Light" : "Dark", exact: true }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    const material = await page.evaluate(() => {
      const card = getComputedStyle(document.querySelector(".team-member")!);
      const contributor = getComputedStyle(document.querySelector(".thanks-list a")!);
      const action = getComputedStyle(document.querySelector(".contribute-action")!);
      return { card: card.backgroundImage, contributor: contributor.backgroundImage, action: action.backgroundImage,
        edge: card.getPropertyValue("--surface-edge"), border: card.getPropertyValue("--glass-line") };
    });
    expect(material.card).toMatch(/^linear-gradient\((?:180deg, )?rgba/);
    expect(material.card).not.toContain("135deg");
    expect(material.contributor).toBe(material.card);
    expect(material.action).toMatch(/^linear-gradient\((?:180deg, )?rgba/);
    expect(material.card).toContain(theme === "light" ? "rgba(156, 126, 230, 0.07)" : "rgba(255, 255, 255,");
    expect(material.action).toContain("rgba(255, 255, 255, 0.16)");
    expect(material.edge).not.toBe(material.border);
    for (const width of [1440, 320]) {
      await page.setViewportSize({ width, height: 1050 });
      await page.locator(".team-section").scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`material-${theme}-${width}.png`), fullPage: true });
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    }
  }
  await page.emulateMedia({ contrast: "more" });
  await expect(page.locator(".team-member").first()).toHaveCSS("background-image", /^none(?:, none)*$/);
  await expect(page.locator(".contribute-action")).toHaveCSS("background-image", /^none(?:, none)*$/);
});

test("workbench uses the same material without changing control behavior", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?fixture=1");
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  for (const theme of ["light", "dark"]) {
    await page.getByRole("button", { name: theme === "light" ? "Light" : "Dark", exact: true }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    for (const width of [1440, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      const send = page.getByRole("button", { name: "Generate reply", exact: true });
      await expect(send).toBeVisible();
      await expect(send).toHaveCSS("background-image", /^linear-gradient\((?:180deg, )?rgba/);
      await expect(page.locator(".chat-zone")).toHaveCSS("background-image", "none");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`workbench-material-${theme}-${width}.png`) });
    }
  }
  await page.getByRole("button", { name: "Controls", exact: true }).click();
  await expect(page.locator(".controls")).toBeVisible();
  await page.getByRole("button", { name: "Loom", exact: true }).click();
  await expect(page.locator(".loom-zone")).toBeVisible();
});
