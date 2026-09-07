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
    const indicator = toggle.locator(".selection-indicator");
    await expect(indicator).toBeVisible();
    await page.evaluate(() => document.fonts.ready);
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(toggle).not.toHaveAttribute("data-selection-ready");
    await expect(indicator).toHaveCSS("transition-duration", "0s");
    await expect(toggle.locator("button").first()).toHaveCSS("transition-duration", "0s");
    await expect(page.locator("html")).not.toHaveAttribute("data-toggle-transitions");
    const alignment = () => toggle.evaluate(el => {
      const selected = el.querySelector('[aria-pressed="true"]').getBoundingClientRect();
      const indicator = el.querySelector(".selection-indicator").getBoundingClientRect();
      return Math.abs(selected.x - indicator.x) + Math.abs(selected.y - indicator.y) + Math.abs(selected.width - indicator.width);
    });
    await expect.poll(alignment).toBeLessThan(1);
    await toggle.getByRole("button", { name: theme === "dark" ? "Light" : "Dark", exact: true }).click();
    await expect(toggle).toHaveAttribute("data-selection-ready", "");
    await expect(indicator).not.toHaveCSS("transition-duration", "0s");
    await expect.poll(alignment).toBeLessThan(1);
  });
}
