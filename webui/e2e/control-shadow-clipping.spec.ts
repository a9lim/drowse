import { expect, test } from "@playwright/test";

test("toolbar shadows and focus rings stay outside their control bounds", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
  await page.getByRole("textbox", { name: "Editable completion buffer" }).fill("A field note: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  await page.getByRole("button", { name: "Show chat tools", exact: true }).click();
  const control = page.getByRole("button", { name: "Color completion tokens by", exact: true });
  await control.click();
  await page.getByRole("option", { name: "Token surprisal", exact: true }).click();
  const header = page.locator("#chat-tools-header");
  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    for (const viewport of [
      { width: 320, height: 780 },
      { width: 667, height: 375 },
      { width: 1100, height: 600 },
      { width: 1440, height: 900 },
    ]) {
      await page.setViewportSize(viewport);
      await control.focus();
      await expect(header).toHaveCSS("overflow", "visible");
      expect(await control.evaluate(element => {
        const rect = element.getBoundingClientRect();
        const clipped: string[] = [];
        for (let parent = element.parentElement; parent; parent = parent.parentElement) {
          const style = getComputedStyle(parent);
          const bounds = parent.getBoundingClientRect();
          const clipsX = /auto|hidden|clip|scroll/.test(style.overflowX);
          const clipsY = /auto|hidden|clip|scroll/.test(style.overflowY);
          if ((clipsX && (rect.left - 8 < bounds.left || rect.right + 8 > bounds.right)) ||
              (clipsY && (rect.top - 8 < bounds.top || rect.bottom + 8 > bounds.bottom))) clipped.push(parent.className);
        }
        return clipped;
      }), `${theme} ${viewport.width}x${viewport.height}`).toEqual([]);
      await expect(control).not.toHaveCSS("box-shadow", "none");
      await page.screenshot({ path: testInfo.outputPath(`control-focus-${theme}-${viewport.width}.png`) });
      await control.click();
      await expect(page.getByRole("listbox")).toBeInViewport();
      await page.keyboard.press("Escape");
      await page.getByRole("button", { name: "Compare colors", exact: true }).click();
      await expect(page.getByRole("button", { name: "Second word color", exact: true })).toBeInViewport();
      expect(await header.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
      await page.getByRole("button", { name: "Use one color", exact: true }).click();
    }
  }
});
