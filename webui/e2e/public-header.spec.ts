import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";

for (const path of ["/", "/credits", "/app?layoutFixture=setup", "compact"]) {
  test(`navigation takes priority over the wordmark on ${path}`, async ({ page, browserName }, testInfo) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(`${devUrl}${path === "compact" ? "/outside-the-workbench" : path}`);
    if (path === "compact") await page.evaluate(async url => {
      const [{ default: Header }, { mount }] = await Promise.all([import(url), import("/e2e/svelte-runtime.ts")]);
      document.body.replaceChildren();
      mount(Header, { target: document.body, props: { current: "chats", compact: true } });
    }, `/@fs${resolve("src/hosted/ui/PageHeader.svelte")}`);
    await page.evaluate(() => document.fonts.ready);
    const header = page.locator(".page-header");
    const brand = header.locator(".page-brand");
    const nav = header.getByRole("navigation", { name: "Primary navigation" });
    for (const width of [1440, 800, 761, 760, 600, 481, 480, 468, 390, 375, 320, 390, 600, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      if (width === 320) await expect(brand).toBeHidden();
      if (width >= 600) await expect(brand).toBeVisible();
      await expect.poll(() => header.evaluate(h => {
        const nav = h.querySelector("nav")!;
        const box = nav.getBoundingClientRect();
        const appearance = h.querySelector(".appearance")!.getBoundingClientRect();
        return h.scrollWidth <= h.clientWidth + 1 && nav.scrollWidth <= nav.clientWidth + 1
          && [...nav.children].every(link => {
            const bounds = link.getBoundingClientRect();
            return bounds.left >= box.left - 1 && bounds.right <= box.right + 1
              && bounds.right <= appearance.left + 1;
          });
      })).toBe(true);
      await expect(nav.getByRole("link")).toHaveText(["Chats", "Models", "Credits", "Contribute"]);
      if (width === 320) {
        await expect(header.getByRole("link", { name: "Drowse home" })).toHaveCount(0);
        await expect(brand).toHaveAttribute("tabindex", "-1");
        await nav.getByRole("link", { name: "Chats", exact: true }).focus();
        for (const name of ["Models", "Credits", "Contribute"]) {
          await page.keyboard.press(browserName === "webkit" ? "Alt+Tab" : "Tab");
          await expect(nav.getByRole("link", { name, exact: true })).toBeFocused();
        }
      }
      if ([320, 390, 1440].includes(width)) await header.screenshot({ path: testInfo.outputPath(`header-${width}.png`) });
    }
    expect(errors).toEqual([]);
  });
}

test("workbench keeps its wordmark without the public navigation row", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 });
  await page.goto(`${devUrl}/app?layoutFixture=base`);
  await expect(page.locator(".page-brand")).toBeVisible();
  await expect(page.getByRole("button", { name: "Workspace menu", exact: true })).toBeVisible();
});
