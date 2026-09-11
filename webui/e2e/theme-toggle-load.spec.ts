import { expect, test } from "@playwright/test";

for (const theme of ["light", "dark"]) {
  test(`theme toggle is static on load in ${theme} and animates only after a change`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.addInitScript(theme => {
      localStorage.setItem("drowse.theme", theme);
      document.addEventListener("transitionrun", event => {
        if (event.target instanceof Element && event.target.closest(".theme-toggle")) {
          document.documentElement.dataset.toggleTransitions = String(Number(document.documentElement.dataset.toggleTransitions ?? 0) + 1);
        }
      }, true);
    }, theme);
    await page.goto("http://127.0.0.1:4176/");
    const toggle = page.locator(".theme-toggle");
    const button = toggle.getByRole("button");
    await expect(button).toHaveCount(1);
    await expect(toggle.locator(".selection-indicator")).toHaveCount(0);
    await page.evaluate(() => document.fonts.ready);
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.locator("html")).not.toHaveAttribute("data-toggle-transitions");
    const before = await button.boundingBox();
    expect(before!.width).toBeGreaterThanOrEqual(44);
    expect(before!.height).toBeGreaterThanOrEqual(44);
    await button.focus();
    await page.keyboard.press("Enter");
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme === "light" ? "dark" : "light");
    await expect(toggle.locator("svg")).toHaveCount(1);
    await expect(toggle.locator("clipPath circle")).toHaveAttribute("r", theme === "light" ? "8" : "5");
    expect(await button.boundingBox()).toEqual(before);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.keyboard.press("Space");
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await expect(toggle.locator("clipPath circle")).toHaveAttribute("r", theme === "light" ? "5" : "8");
  });
}
