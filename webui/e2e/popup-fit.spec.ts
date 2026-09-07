import { expect, test } from "@playwright/test";

test("sampling popup stays compact and inside the viewport across themes and sizes", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await page.getByRole("button", { name: "Controls", exact: true }).click();
  for (const theme of ["Light", "Dark"]) {
    await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
    await page.getByRole("button", { name: theme, exact: true }).click();
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Sampling settings", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Sampling settings", exact: true });
    for (const viewport of [
      { width: 320, height: 780 },
      { width: 390, height: 844 },
      { width: 667, height: 375 },
      { width: 900, height: 900 },
      { width: 1440, height: 900 },
    ]) {
      await page.setViewportSize(viewport);
      await expect(dialog).toBeVisible();
      expect(await dialog.evaluate(element => {
        const rect = element.getBoundingClientRect();
        const close = element.querySelector(".drawer-close")!;
        const closeRect = close.getBoundingClientRect();
        const body = element.querySelector(".body")!;
        return {
          insideViewport: rect.top >= 0 && rect.left >= 0 && rect.bottom <= innerHeight && rect.right <= innerWidth,
          popupRadius: getComputedStyle(element).borderRadius,
          closeRadius: getComputedStyle(close).borderRadius,
          closeVisible: closeRect.top >= rect.top && closeRect.bottom <= rect.bottom,
          fieldsFit: [...element.querySelectorAll(".setting")].every(field => field.scrollWidth <= field.clientWidth),
          bodyFits: body.scrollWidth <= body.clientWidth,
        };
      })).toEqual({ insideViewport: true, popupRadius: "12px", closeRadius: "8px", closeVisible: true, fieldsFit: true, bodyFits: true });
    }
    await page.setViewportSize({ width: 320, height: 780 });
    await page.getByText("Additional parameters", { exact: true }).click();
    await page.getByRole("textbox", { name: "Logit bias", exact: true }).scrollIntoViewIfNeeded();
    await expect(dialog.getByRole("button", { name: "Close drawer", exact: true })).toBeInViewport();
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
    await page.setViewportSize({ width: 1440, height: 900 });
  }
});
