import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;
const compose = (page: Page) => page.getByRole("textbox", { name: /^Compose as / });
const send = (page: Page) => page.getByRole("button", { name: /^(Send|Generate reply)$/ });
const bottomGap = (page: Page) => page.locator(".log").evaluate(el =>
  el.scrollHeight - el.clientHeight - el.scrollTop);

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
});

async function submit(page: Page, text: string, turns: number) {
  await compose(page).fill(text);
  await send(page).click();
  await expect(page.locator(".msg")).toHaveCount(turns * 2);
  await expect(page.locator(".msg .response-body").last())
    .toHaveText("This is a deterministic local Drowse runtime fixture.");
  await expect.poll(() => bottomGap(page)).toBeLessThanOrEqual(2);
}

test("idle actions collapse and editing controls share one row", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await submit(page, "Show the compact composer.", 1);
  await compose(page).blur();
  await page.locator(".input-actions").evaluate(el => (el.querySelector(":focus") as HTMLElement | null)?.blur());
  await expect(page.locator(".input-actions")).toBeHidden();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toHaveCount(0);
  const idleHeight = await page.locator(".log").evaluate(el => el.clientHeight);
  await page.screenshot({ path: testInfo.outputPath("chat-idle.png") });

  await compose(page).focus();
  await expect(send(page)).toBeVisible();
  await expect.poll(() => page.locator(".log").evaluate(el => el.clientHeight)).toBeLessThan(idleHeight - 40);
  for (const viewport of [
    { width: 320, height: 568 }, { width: 390, height: 844 },
    { width: 430, height: 932 }, { width: 844, height: 390 },
    { width: 1440, height: 900 },
  ]) {
    await page.setViewportSize(viewport);
    await compose(page).focus();
    for (const active of [false, true]) {
      await page.evaluate(async ({ url, active }) => {
        (await import(url)).genStatus.active = active;
      }, { url: storesUrl, active });
      const stop = page.getByRole("button", { name: "Stop", exact: true });
      if (active) await expect(stop).toBeEnabled();
      else await expect(stop).toBeDisabled();
      const buttons = page.locator(".input-actions button:visible");
      await expect(buttons).toHaveCount(3);
      const boxes = await buttons.evaluateAll(elements => elements.map(el => {
        const box = el.getBoundingClientRect();
        return { top: box.top, bottom: box.bottom, left: box.left, right: box.right, height: box.height };
      }));
      expect(Math.max(...boxes.map(box => box.top)) - Math.min(...boxes.map(box => box.top))).toBeLessThanOrEqual(1);
      for (const box of boxes) {
        expect(box.left).toBeGreaterThanOrEqual(0);
        expect(box.right).toBeLessThanOrEqual(viewport.width);
        expect(box.height).toBeGreaterThanOrEqual(viewport.width <= 760 ? 44 : 24);
      }
      const clippedLabels = await buttons.evaluateAll(elements => elements.flatMap(el => {
        const button = el.getBoundingClientRect();
        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
        const clipped: string[] = [];
        while (walker.nextNode()) {
          if (!walker.currentNode.textContent?.trim()) continue;
          const range = document.createRange();
          range.selectNodeContents(walker.currentNode);
          for (const rect of range.getClientRects()) {
            if (rect.width && (rect.left < button.left || rect.right > button.right)) {
              clipped.push(walker.currentNode.textContent);
            }
          }
        }
        return clipped;
      }));
      expect(clippedLabels).toEqual([]);
      await page.screenshot({ path: testInfo.outputPath(`chat-controls-${viewport.width}-${active}.png`) });
    }
  }
  await page.evaluate(async url => { (await import(url)).genStatus.active = false; }, storesUrl);
});

test("only Stop remains while an unfocused empty composer is generating", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await submit(page, "Keep the stop action reachable.", 1);
  await page.evaluate(async url => {
    (document.activeElement as HTMLElement)?.blur();
    (await import(url)).genStatus.active = true;
  }, storesUrl);
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeVisible();
  await expect(send(page)).toBeHidden();
  await expect(page.getByRole("button", { name: "Clear conversation", exact: true })).toBeHidden();
  await page.evaluate(async url => { (await import(url)).genStatus.active = false; }, storesUrl);
  await expect(page.locator(".input-actions")).toBeHidden();
});

test("sending and resizing follow the latest message without interrupting history reading", async ({ page, browserName, isMobile }, testInfo) => {
  for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
    await page.setViewportSize(viewport);
    await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
    await expect(page.locator(".shell")).toBeVisible();
    for (let index = 1; index <= 8; index++) {
      await submit(page, `Message ${index}. ` + "Keep every line available while following the newest reply. ".repeat(16), index);
    }
    await compose(page).focus();
    await page.setViewportSize({ width: viewport.width, height: Math.round(viewport.height / 2) });
    await expect.poll(() => bottomGap(page)).toBeLessThanOrEqual(2);
    await compose(page).blur();
    await page.setViewportSize(viewport);
    await expect.poll(() => bottomGap(page)).toBeLessThanOrEqual(2);

    const log = page.locator(".log");
    if (browserName === "webkit" && isMobile) {
      await log.evaluate(el => {
        for (const [type, clientY] of [["touchstart", 100], ["touchmove", 200]] as const) {
          const event = new Event(type, { bubbles: true });
          Object.defineProperty(event, "touches", { value: [{ clientY }] });
          el.dispatchEvent(event);
        }
        el.scrollTop = 0;
        el.dispatchEvent(new Event("touchend", { bubbles: true }));
      });
    } else {
      await log.hover();
      await page.mouse.wheel(0, -100_000);
    }
    await expect.poll(() => bottomGap(page)).toBeGreaterThan(200);
    const readingPosition = await log.evaluate(el => el.scrollTop);
    await page.evaluate(async url => {
      const turn = (await import(url)).chatLog.turns.at(-1);
      turn.tokens = [];
      turn.text += " More streamed text at the bottom.".repeat(30);
    }, storesUrl);
    await expect(page.locator(".response-body").last()).toContainText("More streamed text");
    await expect.poll(() => log.evaluate((el, position) => Math.abs(el.scrollTop - position), readingPosition)).toBeLessThanOrEqual(2);
    await submit(page, "Return to the latest reply. ".repeat(40), 9);
    for (let index = 0; index < 8; index++) {
      await page.evaluate(async url => {
        const turn = (await import(url)).chatLog.turns.at(-1);
        turn.tokens = [];
        turn.text += " A further paragraph arrives during generation.".repeat(8);
      }, storesUrl);
      await expect.poll(() => bottomGap(page)).toBeLessThanOrEqual(2);
    }
    await log.focus();
    await page.keyboard.press("Home");
    await expect.poll(() => bottomGap(page)).toBeGreaterThan(200);
    await page.keyboard.press("End");
    await expect.poll(() => bottomGap(page)).toBeLessThanOrEqual(2);
    await page.screenshot({ path: testInfo.outputPath(`latest-message-${viewport.width}.png`) });
  }
});
