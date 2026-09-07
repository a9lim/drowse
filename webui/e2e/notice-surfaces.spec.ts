import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

test("humanized public copy stays readable without em dashes", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator(".lede")).toContainText("Run a language model on your device");
  await expect(page.getByRole("heading", { name: "Test what shapes a reply." })).toBeVisible();
  await expect(page.locator(".capabilities li")).toHaveCount(3);
  for (const theme of ["Light", "Dark"]) {
    await page.getByRole("button", { name: theme, exact: true }).click();
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await page.locator("body").innerText()).not.toContain("\u2014");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`public-copy-${theme}-${width}.png`) });
    }
  }
  for (const path of ["/credits", "/missing-page"]) {
    await page.goto(path);
    expect(await page.title()).not.toContain("\u2014");
    expect(await page.locator("body").innerText()).not.toContain("\u2014");
  }
});

test("storage guidance has no decorative side stripe in either theme", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/outside-the-workbench");
  await page.evaluate(async moduleUrl => {
    const [{ default: HostedHome }, { mount }] = await Promise.all([
      import(moduleUrl), import("/@id/svelte"),
    ]);
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(HostedHome, { target, props: {
      controller: {
        capabilities: () => ({ signals: { appleMobile: false } }),
        retryPersistence: async () => false,
        check: async () => {}, open: async () => {},
      },
      snapshot: {
        phase: "supported", headline: "Ready", detail: "Ready", checks: [], models: [],
        download: { available: true, phase: "idle", reason: "Ready" },
        runtime: { available: true, phase: "unloaded", reason: "Ready" },
        storage: { availableBytes: 2_000_000_000, persisted: false },
      },
      onChooseModels() {},
    } });
  }, `/@fs/${resolve("src/hosted/ui/HostedHome.svelte")}`);
  const notice = page.locator(".storage-notice");
  await expect(notice).toBeVisible();
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const width of [390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await expect.poll(() => notice.evaluate(el => {
        const css = getComputedStyle(el);
        const probe = document.createElement("div");
        probe.style.boxShadow = "var(--shadow-card)";
        el.append(probe);
        const expectedShadow = getComputedStyle(probe).boxShadow;
        probe.remove();
        return { shadowMatches: css.boxShadow === expectedShadow, balancedBorders: css.borderLeftWidth === css.borderRightWidth, image: css.backgroundImage };
      })).toEqual({ shadowMatches: true, balancedBorders: true, image: "none" });
      await notice.screenshot({ path: testInfo.outputPath(`storage-${theme}-${width}.png`) });
    }
  }
  await notice.getByRole("button", { name: "Protect storage", exact: true }).click();
  await expect(notice.getByRole("button", { name: "Protect storage", exact: true })).toBeEnabled();
  await expect(notice).toContainText("Your browser didn't grant storage protection");
  await expect(notice).toContainText("Your existing chats and models won't transfer to it");
  await expect(notice.getByRole("link", { name: "Chrome site data settings", exact: true }))
    .toHaveAttribute("href", "https://support.google.com/chrome/answer/14114868?hl=en");
  await expect(notice.getByRole("link", { name: "Safari web apps on Mac", exact: true }))
    .toHaveAttribute("href", "https://support.apple.com/en-us/104996");
  await expect(notice.locator('a[href*="web.dev"], a[href*="webkit.org"]')).toHaveCount(0);
  await expect(notice.locator("strong")).toHaveCount(0);
  expect(await notice.innerText()).not.toContain("\u2014");
  for (const width of [320, 390, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await notice.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    await notice.screenshot({ path: testInfo.outputPath(`storage-guide-${width}.png`) });
  }
});
