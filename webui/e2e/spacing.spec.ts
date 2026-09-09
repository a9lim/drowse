import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const storesModule = `/@fs${resolve("src/lib/stores.svelte.ts")}`;

test("composer and dialog actions follow their container spacing", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.locator(".shell")).toBeVisible();
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 320, height: 640 }, { width: 844, height: 390 }]) {
    await page.setViewportSize(viewport);
    const composer = page.locator(".chat-zone");
    const compactDesktop = viewport.width > 760;
    await composer.locator("textarea.input").focus();
    await expect.poll(() => composer.evaluate(element => {
      const inset = parseFloat(getComputedStyle(element).paddingBottom);
      const input = element.querySelector("textarea.input")!;
      const actions = element.querySelector(".input-actions")!;
      const horizontal = getComputedStyle(input.parentElement!).gridTemplateColumns.split(" ").length === 2;
      return {
        inset, row: Math.round(horizontal
          ? actions.getBoundingClientRect().left - input.getBoundingClientRect().right
          : actions.getBoundingClientRect().top - input.getBoundingClientRect().bottom),
        buttons: parseFloat(getComputedStyle(actions).gap),
        overflow: element.scrollWidth > element.clientWidth,
      };
    })).toEqual({
      row: viewport.width > 620 && viewport.height <= 600 ? 0 : compactDesktop ? 8 : 16,
      buttons: viewport.width > 760 && viewport.height <= 600 ? 4 : 8,
      inset: viewport.width > 620 && viewport.height <= 600 ? 0 : 12,
      overflow: false,
    });

    for (const name of ["download_chat", "save_conversation", "system_prompt", "transcript"]) {
      await page.evaluate(async ({ storesModule, name }) => { (await import(storesModule)).openDrawer(name); }, { storesModule, name });
      const content = page.locator(name === "download_chat" ? ".download-chat" : ".drawer-shell").last();
      await expect(content).toBeVisible();
      const footer = content.locator("footer");
      await expect(footer).toBeVisible();
      const dimensions = await footer.evaluate(element => {
        const css = getComputedStyle(element);
        const parent = element.parentElement!;
        const inset = parseFloat(getComputedStyle(parent).paddingBottom) || parseFloat(css.paddingBottom);
        return { gap: parseFloat(css.gap), inset, overflow: parent.scrollWidth > parent.clientWidth };
      });
      expect(dimensions.gap).toBe(name === "download_chat" ? (compactDesktop ? 16 : 24) : dimensions.inset);
      expect(dimensions.overflow).toBe(false);
      if (name === "download_chat") {
        await expect(content).toHaveCSS("row-gap", `${compactDesktop ? 16 : 24}px`);
        await expect(content.getByRole("button", { name: "Download", exact: true })).toBeEnabled();
      }
      if (name === "save_conversation") {
        await expect(footer.locator(".footer-actions")).toHaveCSS("gap", `${dimensions.inset}px`);
      }
      await content.screenshot({ path: testInfo.outputPath(`${name}-${viewport.width}x${viewport.height}.png`) });
      await page.evaluate(async url => { (await import(url)).closeDrawer(); }, storesModule);
      await expect(content).toHaveCount(0);
    }
  }
});
