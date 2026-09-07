import { test, expect } from "@playwright/test";
import { resolve } from "node:path";

test("favicon animation maintains a smooth cadence without the app renderer", async ({ page }) => {
  await page.route("**/icon-motion-test", route => route.fulfill({
    contentType: "text/html",
    body: "<!doctype html><html><head></head><body></body></html>",
  }));
  await page.goto("http://127.0.0.1:4176/icon-motion-test");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  const result = await page.evaluate(async url => {
    const { createTabIcon } = await import(url);
    const controller = createTabIcon();
    controller.update("loading");
    const link = document.querySelector<HTMLLinkElement>('link[rel="icon"]')!;
    let updates = 0;
    const hrefs = new Set<string>();
    const observer = new MutationObserver(() => { updates++; hrefs.add(link.href); });
    observer.observe(link, { attributes: true, attributeFilter: ["href"] });
    await new Promise(resolve => setTimeout(resolve, 2100));
    observer.disconnect();
    controller.dispose();
    return { updates, unique: hrefs.size };
  }, `/@fs${resolve("src/lib/tabIdentity.ts")}`);
  expect(result.updates).toBeGreaterThan(36);
  expect(result.unique).toBeGreaterThan(18);
});
