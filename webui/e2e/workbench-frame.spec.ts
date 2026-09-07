import { expect, test } from "@playwright/test";

const devUrl = "http://127.0.0.1:4176";

for (const fixture of ["base", "1"]) {
  test(`flat workbench retains its sidebar and active model: ${fixture}`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(`${devUrl}/app?layoutFixture=${fixture}`);
    const model = page.getByLabel("Active model", { exact: true });
    await expect(model).toBeVisible();
    for (const theme of ["Light", "Dark"]) {
      await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
      await page.getByRole("button", { name: theme, exact: true }).click();
      await page.keyboard.press("Escape");
      for (const width of [1440, 900]) {
        await page.setViewportSize({ width, height: 900 });
        expect(await page.evaluate(() => {
          const sidebar = document.querySelector(".app-sidebar")!;
          const workspace = document.querySelector(".workspace-frame")!;
          const surface = document.querySelector(".chat-zone")!;
          return {
            sidebar: Math.abs(sidebar.getBoundingClientRect().right - workspace.getBoundingClientRect().left) < 1,
            compactSidebar: sidebar.getBoundingClientRect().width < 200,
            openSurface: getComputedStyle(surface).backgroundColor === "rgba(0, 0, 0, 0)",
            squareSurface: getComputedStyle(surface).borderRadius === "0px",
            compactButtons: getComputedStyle(document.querySelector(".workspace-parent")!).borderRadius === "4px",
            noFooterClutter: !sidebar.querySelector(".page-footer"),
            fullHeight: workspace.getBoundingClientRect().bottom === innerHeight,
            noOverflow: document.documentElement.scrollWidth <= innerWidth,
          };
        })).toEqual({ sidebar: true, compactSidebar: true, openSurface: true, squareSurface: true, compactButtons: true, noFooterClutter: true, fullHeight: true, noOverflow: true });
      }
    }
    await page.setViewportSize({ width: 320, height: 780 });
    await expect(page.getByRole("button", { name: "Workspace menu", exact: true })).toBeVisible();
    await expect(model).toBeVisible();
    expect(await model.evaluate(element => element.getBoundingClientRect().bottom <= innerHeight)).toBe(true);
    expect(await page.locator(".app-sidebar").evaluate(element => element.getBoundingClientRect().height < 130)).toBe(true);
    await page.getByRole("button", { name: "Controls", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Generation settings", exact: true })).toBeVisible();
    expect(await page.locator(".sampling-strip .control").evaluateAll(elements => elements.every(element => getComputedStyle(element).backgroundColor === "rgba(0, 0, 0, 0)"))).toBe(true);
    await expect(page.getByRole("button", { name: "Loom", exact: true })).toBeVisible();
  });
}
