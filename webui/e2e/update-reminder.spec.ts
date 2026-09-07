import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const key = "drowse.pwa-update-reminder.v1";
const updateNotice = (page: Page) => page.locator(".pwa-notice").filter({ hasText: "A Drowse update is ready." });

async function installUpdate(context: BrowserContext, page: Page) {
  let version = 1;
  await context.route("**/sw.js", route => route.fulfill({
    contentType: "text/javascript",
    headers: { "Cache-Control": "no-store" },
    body: `
      const version = ${version};
      self.addEventListener("install", event => {
        if (version === 1) event.waitUntil(self.skipWaiting());
      });
      self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));
      self.addEventListener("message", event => {
        if (event.data?.type === "SKIP_WAITING") event.waitUntil(self.skipWaiting());
      });
    `,
  }));
  await page.goto("/");
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.reload();
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);
  version = 2;
  await page.evaluate(async () => { await (await navigator.serviceWorker.ready).update(); });
  await expect(updateNotice(page)).toBeVisible();
}

test("update entrance and fading glow play only once, with comfortable button spacing", async ({ context, page }, testInfo) => {
  test.setTimeout(60_000);
  await page.emulateMedia({ colorScheme: "dark" });
  await page.addInitScript(() => localStorage.setItem("drowse.theme", "dark"));
  await installUpdate(context, page);
  const notice = updateNotice(page);
  await expect(notice).toHaveClass(/first-update/);
  await expect(notice).toHaveCSS("animation-name", /update-pop.*update-glow/);
  const animation = await notice.evaluate(element => {
    const card = element as HTMLElement;
    card.style.animation = "none";
    void card.offsetHeight;
    card.style.removeProperty("animation");
    const animations = element.getAnimations();
    for (const animation of animations) {
      animation.playbackRate = 0.1;
      animation.pause(); animation.currentTime = 120;
    }
    return animations.map(animation => (animation as CSSAnimation).animationName);
  });
  expect(animation.some(name => name.endsWith("update-pop"))).toBe(true);
  expect(animation.some(name => name.endsWith("update-glow"))).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("update-entrance.png") });
  await notice.evaluate(element => {
    for (const animation of element.getAnimations()) animation.currentTime = 1500;
  });
  const glow = await notice.evaluate(element => getComputedStyle(element).boxShadow);
  await notice.evaluate(element => {
    for (const animation of element.getAnimations()) animation.finish();
  });
  expect(await notice.evaluate(element => getComputedStyle(element).boxShadow)).not.toBe(glow);
  await expect(notice.locator(".pwa-actions")).toHaveCSS("gap", "16px");
  const gradient = await page.locator(".hero-action-row .primary-action").evaluate(element => getComputedStyle(element).backgroundImage.replace(/, none$/, ""));
  expect(gradient).toContain("linear-gradient(");
  for (const button of await notice.getByRole("button").all()) {
    await expect(button).toHaveCSS("background-image", gradient);
    await button.hover();
    await expect(button).toHaveCSS("background-image", gradient);
  }
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await notice.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    await notice.screenshot({ path: testInfo.outputPath(`update-rest-${width}.png`), animations: "disabled" });
  }
  expect((await new AxeBuilder({ page }).include(".pwa-notice").analyze()).violations).toEqual([]);
  await page.reload();
  await expect(notice).toBeVisible();
  await expect(notice).not.toHaveClass(/first-update/);
  await page.goto("/credits/");
  await expect(notice).toBeVisible();
  await expect(notice).not.toHaveClass(/first-update/);
  expect(await notice.evaluate(element => element.getAnimations().length)).toBe(0);
});

test("snoozes survive navigation and reload and escalate through 1h, 6h, and daily", async ({ context, page }, testInfo) => {
  test.setTimeout(90_000);
  await page.clock.install();
  await installUpdate(context, page);
  const notice = updateNotice(page);
  for (const [index, hours] of [1, 6, 24, 24].entries()) {
    const before = await page.evaluate(() => Date.now());
    await notice.getByRole("button", { name: "Update later" }).click();
    await expect(notice).toHaveCount(0);
    const confirmation = page.locator(".pwa-confirmation");
    await expect(confirmation).toHaveText(`Okay! We'll remind you in ${hours === 24 ? "1 day" : `${hours} ${hours === 1 ? "hour" : "hours"}`}.`);
    const saved = await page.evaluate(key => JSON.parse(localStorage.getItem(key)!), key);
    expect(saved.remindAt - before).toBeGreaterThanOrEqual(hours * 3_600_000);
    expect(saved.deferrals).toBe(Math.min(index + 1, 3));
    if (index === 0) {
      await confirmation.locator(".countdown-remaining").evaluate(element => {
        const animation = element.getAnimations()[0];
        animation.pause(); animation.currentTime = 2500;
      });
      await expect(confirmation.locator(".countdown-remaining")).toHaveCSS("stroke-dashoffset", "0.5px");
      await confirmation.screenshot({ path: testInfo.outputPath("reminder-confirmation.png") });
    }
    await page.clock.fastForward(4_000);
    await expect(confirmation).toBeVisible();
    await page.clock.fastForward(1_200);
    await expect(confirmation).toHaveCount(0);
    await page.goto(index % 2 === 0 ? "/credits/" : "/");
    await expect(notice).toHaveCount(0);
    await page.reload();
    await expect(notice).toHaveCount(0);
    expect(await page.evaluate(key => JSON.parse(localStorage.getItem(key)!).remindAt, key)).toBe(saved.remindAt);
    const now = await page.evaluate(() => Date.now());
    await page.clock.fastForward(saved.remindAt - now - 2_000);
    await expect(notice).toHaveCount(0);
    await page.clock.fastForward(2_001);
    await expect(notice).toBeVisible();
    await expect(notice).not.toHaveClass(/first-update/);
  }
});

test("another tab shares the snooze and applying the update resets the next reminder cycle", async ({ context, page }) => {
  await installUpdate(context, page);
  const other = await context.newPage();
  await other.goto("/credits/");
  await expect(updateNotice(other)).toBeVisible();
  await expect(updateNotice(other)).not.toHaveClass(/first-update/);
  await updateNotice(page).getByRole("button", { name: "Update later" }).click();
  await expect(updateNotice(other)).toHaveCount(0);
  await other.reload();
  await expect(updateNotice(other)).toHaveCount(0);
  await other.evaluate(key => {
    const reminder = JSON.parse(localStorage.getItem(key)!);
    localStorage.setItem(key, JSON.stringify({ ...reminder, remindAt: Date.now() - 1 }));
    window.dispatchEvent(new Event("focus"));
  }, key);
  await expect(updateNotice(other)).toBeVisible();
  await Promise.all([
    other.waitForEvent("domcontentloaded"),
    updateNotice(other).getByRole("button", { name: "Update and reload" }).click(),
  ]);
  await expect(updateNotice(page)).toHaveCount(0);
  expect(await other.evaluate(key => localStorage.getItem(key), key)).toBeNull();
});

test("reduced motion removes the entrance and glow but retains the confirmation deadline", async ({ context, page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.clock.install();
  await installUpdate(context, page);
  const notice = updateNotice(page);
  await expect(notice).toHaveCSS("animation-name", "none");
  await notice.getByRole("button", { name: "Update later" }).click();
  await expect(page.locator(".countdown-remaining")).toHaveCSS("animation-timing-function", "steps(5)");
  await page.clock.fastForward(5_001);
  await expect(page.locator(".pwa-confirmation")).toHaveCount(0);
});
