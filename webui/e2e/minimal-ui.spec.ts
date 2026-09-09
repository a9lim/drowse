import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;

test("main workspaces and dialogs keep a clean responsive layout", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain how a language model chooses its next word.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();

  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const name of ["Conversation", "Loom", "Controls"]) {
      await page.getByRole("button", { name, exact: true }).click();
      await expect(page.locator(".shell")).toBeVisible();
      await expect(page.locator(".layout")).toHaveCSS("background-image", "none");
      expect(await page.locator(".layout").evaluate(element => getComputedStyle(element, "::before").content)).toBe("none");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`${name}-${theme}.png`), animations: "disabled" });
    }
    for (const name of ["Response", "Model", "Chat"]) {
      await page.getByRole("group", { name: "Controls section" }).getByRole("button", { name, exact: true }).click();
      await page.screenshot({ path: testInfo.outputPath(`controls-${name}-${theme}.png`), animations: "disabled" });
    }
    for (const name of ["help", "advanced_sampling", "local_runtime", "save_conversation", "download_chat", "token_drilldown"]) {
      await page.evaluate(async ({ storesUrl, name }) => { (await import(storesUrl)).openDrawer(name, name === "token_drilldown" ? { turnIdx: 1, tokenIdx: 0 } : undefined); }, { storesUrl, name });
      const dialog = page.getByRole("dialog").last();
      await expect(dialog).toBeVisible();
      await expect(dialog).toHaveCSS("background-image", /^(none)(, none)*$/);
      await dialog.screenshot({ path: testInfo.outputPath(`${name}-${theme}.png`), animations: "disabled" });
      await page.evaluate(async url => { (await import(url)).closeDrawer(); }, storesUrl);
      await expect(dialog).toHaveCount(0);
    }
  }
});

test("public pages keep flat canvases separate from top-lit cards", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  for (const route of ["/", "/credits", "/app?choose=1"]) {
    await page.goto(`${devUrl}${route}`);
    await expect(page.locator("h1").first()).toBeVisible();
    for (const theme of ["dark", "light"]) {
      await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
      await expect(page.locator("body")).toHaveCSS("background-image", "none");
      if (route === "/app?choose=1") {
        await expect(page.locator(".app-shell")).toHaveCSS("background-image", "none");
        expect(await page.locator(".app-shell").evaluate(element => ["::before", "::after"].map(pseudo => getComputedStyle(element, pseudo).content))).toEqual(["none", "none"]);
      }
      await page.screenshot({ path: testInfo.outputPath(`public-${route.replace(/\W/g, "_")}-${theme}.png`), animations: "disabled" });
    }
  }
});
