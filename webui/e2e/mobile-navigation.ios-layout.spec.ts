import { expect, test, type Page } from "@playwright/test";
import { selectWorkspaceView, setAppearance } from "./workbench-navigation";

test.use({ hasTouch: true });

async function open(page: Page) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
}

test("mobile navigation fits, matches surface radii, and restores screen space", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 440, height: 796 });
  await open(page);
  for (const theme of ["Dark", "Light"]) {
    await setAppearance(page, theme);
    for (const width of [320, 375, 440, 680, 760]) {
      await page.setViewportSize({ width, height: 796 });
      const navigation = page.getByRole("navigation", { name: "Workspace", exact: true });
      await expect(navigation).toBeVisible();
      const buttons = page.locator(".workspace-navigation > .mobile-navigation-action, .workspace-nav button");
      await expect(buttons).toHaveCount(5);
      const boxes = await buttons.evaluateAll(elements => elements.map(element => {
        const rect = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, height: rect.height,
          radius: style.borderRadius, expected: style.getPropertyValue("--radius").trim() };
      }));
      for (const box of boxes) {
        expect(box.left).toBeGreaterThanOrEqual(0);
        expect(box.right).toBeLessThanOrEqual(width);
        expect(box.height).toBeGreaterThanOrEqual(44);
        expect(box.radius).toBe(box.expected);
      }
      for (let i = 1; i < boxes.length; i++) expect(boxes[i].left).toBeGreaterThanOrEqual(boxes[i - 1].right);
      const before = (await page.locator(".workspace-frame").boundingBox())!.height;
      await page.getByRole("button", { name: "Hide navigation bar", exact: true }).click();
      await expect(navigation).not.toBeVisible();
      const restore = page.getByRole("button", { name: "Show navigation bar", exact: true });
      await expect(restore).toBeFocused();
      const after = (await page.locator(".workspace-frame").boundingBox())!.height;
      expect(after - before).toBeGreaterThanOrEqual(44);
      const sidebar = (await page.getByRole("button", { name: "Show left sidebar", exact: true }).boundingBox())!;
      const restoreBox = (await restore.boundingBox())!;
      expect(restoreBox.x).toBeGreaterThanOrEqual(sidebar.x + sidebar.width);
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await restore.click();
      await expect(navigation).toBeVisible();
      await expect(page.getByRole("button", { name: "Hide navigation bar", exact: true })).toBeFocused();
    }
    await page.setViewportSize({ width: 440, height: 796 });
    await page.screenshot({ path: testInfo.outputPath(`navigation-${theme.toLowerCase()}.png`) });
  }
  await selectWorkspaceView(page, "Controls");
  await page.getByRole("button", { name: "Hide navigation bar", exact: true }).click();
  await expect(page.locator('.controls-page[aria-hidden="false"]')).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.getByRole("button", { name: "Show navigation bar", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
  await expect(page.locator(".workspace-nav")).toBeVisible();
  await expect(page.locator(".mobile-navigation-action")).toHaveCount(0);
});

test("navigation collapse is interruptible and Back to Chats opens saved conversations", async ({ page }) => {
  await page.setViewportSize({ width: 440, height: 796 });
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await open(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep this conversation.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator(".log .tok").first()).toBeVisible();
  const hide = page.getByRole("button", { name: "Hide navigation bar", exact: true });
  await hide.click();
  const animation = await page.locator(".app-sidebar").evaluate(element => {
    const style = getComputedStyle(element);
    return { duration: style.transitionDuration, transform: style.transform };
  });
  expect(animation.duration).toContain("0.3s");
  await page.getByRole("button", { name: "Show navigation bar", exact: true }).click();
  await expect(page.locator(".app-sidebar")).toHaveCSS("opacity", "1");
  await expect(hide).toBeFocused();
  await page.locator(".mobile-navigation-action").first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("dialog")).toContainText("Saved");
});
