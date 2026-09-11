import { selectWorkspaceView } from "./workbench-navigation";
import { setAppearance } from "./workbench-navigation";
import { expect, test } from "@playwright/test";

const devUrl = "http://127.0.0.1:4176";

for (const fixture of ["base", "1"]) {
  test(`flat workbench keeps model identity in the sidebar: ${fixture}`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(`${devUrl}/app?layoutFixture=${fixture}`);
    const model = page.locator(".model-summary");
    await expect(model).toBeAttached();
    await expect(page.getByLabel("Active model", { exact: true })).toHaveCount(0);
    for (const theme of ["Light", "Dark"]) {
      await setAppearance(page, theme);
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
    await expect(model).toBeAttached();
    expect(await page.locator(".app-sidebar").evaluate(element => element.getBoundingClientRect().height < 130)).toBe(true);
    await selectWorkspaceView(page, "Controls");
    await expect(page.getByRole("heading", { name: "Generation settings", exact: true })).toBeVisible();
    expect(await page.locator(".sampling-strip .control").evaluateAll(elements => elements.every(element => getComputedStyle(element).backgroundColor === "rgba(0, 0, 0, 0)"))).toBe(true);
    await expect(page.getByRole("button", { name: "Loom", exact: true })).toBeVisible();
  });
}
