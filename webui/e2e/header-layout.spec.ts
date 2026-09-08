import { expect, test } from "@playwright/test";

const devUrl = "http://127.0.0.1:4176";

test("sidebar leads with chats and compact navigation stays centered", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${devUrl}/app?layoutFixture=base`);
  const sidebar = page.getByRole("complementary", { name: "Workspace sidebar" });
  const menu = page.getByRole("button", { name: "Workspace menu", exact: true });
  await expect(sidebar.getByRole("button").first()).toHaveText("Back to Chats");
  await expect(sidebar.getByRole("separator")).toBeVisible();
  expect(await sidebar.getByRole("button").first().evaluate(button => button.nextElementSibling?.tagName)).toBe("HR");
  await expect(menu).toHaveCount(0);
  await sidebar.getByRole("button", { name: "Show chat tools", exact: true }).click();
  await expect(page.locator(".chat-header")).toBeVisible();
  await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
  await expect(menu).toBeVisible();
  await menu.click();
  await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
  await expect(menu).toHaveCount(0);
  await expect(page.getByRole("dialog", { name: "Workspace menu", exact: true })).toHaveCount(0);
  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    for (const width of [1440, 760, 700, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      if (width <= 760) {
        await expect(menu).toBeVisible();
        expect(await page.locator(".workspace-nav").evaluate(nav => {
          const rect = nav.getBoundingClientRect();
          return Math.abs(rect.left + rect.width / 2 - innerWidth / 2);
        })).toBeLessThanOrEqual(1);
        await expect(sidebar.getByRole("button", { name: "Back to Chats", exact: true })).toBeHidden();
      } else {
        await expect(menu).toHaveCount(0);
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`navigation-${theme}-${width}.png`) });
    }
  }
});

for (const viewport of [
  { width: 320, height: 780 },
  { width: 390, height: 844 },
  { width: 667, height: 375 },
  { width: 900, height: 740 },
  { width: 1440, height: 900 },
]) {
  test(`contextual header and menu fit at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(`${devUrl}/app?layoutFixture=base`);
    const trigger = page.getByRole("button", { name: "Workspace menu", exact: true });
    const menu = page.getByRole("dialog", { name: "Workspace menu", exact: true });
    if (viewport.width > 760) await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
    async function selectView(name: RegExp) {
      if (viewport.width > 760) await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
      await page.getByRole("button", { name }).click();
      if (viewport.width > 760) await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
    }
    await expect(trigger).toBeVisible();
    await expect(page.locator(".chat-header")).toHaveCount(0);
    await expect(page.locator(".header-toggle, .kbd-hint")).toHaveCount(0);
    await expect(page.locator(".app-header nav, .app-header .theme-toggle")).toHaveCount(0);

    for (const key of ["Meta+K", "Control+K"]) {
      await page.getByRole("textbox", { name: "Editable completion buffer" }).focus();
      await page.keyboard.press(key);
      await expect(page.getByRole("dialog", { name: "Command palette" })).toHaveCount(0);
    }

    for (const theme of ["Light", "Dark"]) {
      await trigger.click();
      const backToChats = menu.getByRole("button", { name: "Back to Chats", exact: true });
      await expect(backToChats).toBeFocused();
      await expect(menu.getByRole("button").first()).toHaveText("Back to Chats");
      await expect(menu.getByRole("separator")).toBeVisible();
      expect(await backToChats.evaluate(button => button.nextElementSibling?.tagName)).toBe("HR");
      await expect(menu.getByRole("button", { name: "Download chat", exact: true })).toHaveCount(0);
      await menu.getByRole("button", { name: theme, exact: true }).click();
      expect(await menu.evaluate(element => {
        const rect = element.getBoundingClientRect();
        return { fits: rect.left >= 0 && rect.right <= innerWidth && rect.top >= 0 && rect.bottom <= innerHeight,
          noOverflow: element.scrollWidth <= element.clientWidth, radius: getComputedStyle(element).borderRadius };
      })).toEqual({ fits: true, noOverflow: true, radius: "16px" });
      await page.keyboard.press("Escape");
      await expect(menu).toHaveCount(0);
      await expect(trigger).toBeFocused();
    }

    await trigger.click();
    await menu.getByRole("button", { name: "Show chat tools", exact: true }).click();
    await expect(page.locator(".chat-panel > .chat-header")).toBeInViewport();
    await expect(trigger).toBeFocused();
    await selectView(/^Loom\b/);
    await expect(page.locator(".loom-header")).toHaveCount(0);
    await trigger.click();
    await menu.getByRole("button", { name: "Show Loom tools", exact: true }).click();
    await expect(page.locator(".loom-header")).toBeVisible();
    await selectView(/^(Completion|Text)\b/);
    await expect(page.locator(".chat-header")).toBeVisible();
    await trigger.click();
    await menu.getByRole("button", { name: "Hide chat tools", exact: true }).click();
    await expect(page.locator(".chat-header")).toHaveCount(0);
    await selectView(/^Controls\b/);
    await trigger.click();
    await expect(menu.locator(".view-tools")).toHaveCount(0);
    await menu.getByRole("button", { name: "All tools", exact: true }).click();
    await expect(page.getByRole("combobox", { name: "Filter commands" })).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(trigger).toBeFocused();
    expect(await page.evaluate(() => {
      const header = document.querySelector(".app-header")!.getBoundingClientRect();
      const sidebar = document.querySelector(".app-sidebar")!.getBoundingClientRect();
      const workspace = document.querySelector(".workspace-frame")!.getBoundingClientRect();
      return { smallHeader: header.height <= 72, aligned: header.bottom <= workspace.top,
        compactNav: innerWidth > 760 || sidebar.height <= 72,
        usable: workspace.height >= 150, fits: document.documentElement.scrollWidth <= innerWidth };
    })).toEqual({ smallHeader: true, aligned: true, compactNav: true, usable: true, fits: true });
  });
}
