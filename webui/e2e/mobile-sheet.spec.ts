import { expect, test, type Locator, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
const sheet = (page: Page) => page.locator('.sheet-host[data-mobile-sheet="true"].present');
const handle = (page: Page) => sheet(page).getByRole("button", { name: /^Resize word details:/ });

async function drag(target: Locator, delta: number, duration: number, cancel = false) {
  await target.click({ trial: true });
  await target.evaluate(async (element, { delta, duration, cancel }) => {
    const rect = element.getBoundingClientRect();
    const x = rect.x + Math.min(rect.width / 2, 100), y = rect.y + Math.min(rect.height / 2, 24);
    const started = performance.now();
    const send = (type: string, at: number, elapsed: number, active = true) => {
      const touch = { identifier: 1, target: element, clientX: x, clientY: at };
      const event = new Event(type, { bubbles: true, cancelable: true });
      Object.defineProperties(event, { timeStamp: { value: started + elapsed }, touches: { value: active ? [touch] : [] }, targetTouches: { value: active ? [touch] : [] }, changedTouches: { value: [touch] } });
      element.dispatchEvent(event);
    };
    send("touchstart", y, 0);
    for (let step = 1; step <= 10; step++) {
      await new Promise(resolve => setTimeout(resolve, duration / 10));
      send("touchmove", y + delta * step / 10, duration * step / 10);
    }
    send(cancel ? "touchcancel" : "touchend", y + delta, duration + 1, false);
  }, { delta, duration, cancel });
}

async function generate(page: Page) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain language models.");
  await page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toHaveText("This is a deterministic local Drowse runtime fixture.");
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
}

test("sheet overlays the unchanged workspace with stepped swipes, focus containment, and dismissal", async ({ page }, info) => {
  test.slow();
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  const workspace = page.locator(".workspace-frame");
  const before = await workspace.boundingBox();
  await page.getByRole("button", { name: "Show right sidebar", exact: true }).click();
  await expect(sheet(page)).toHaveAttribute("data-detent", "half");
  await expect(handle(page)).toBeFocused();
  await handle(page).click({ trial: true });
  expect(await workspace.boundingBox()).toEqual(before);
  await expect(page.locator("#workbench-header")).toHaveAttribute("inert", "");
  await expect(sheet(page).getByRole("button", { name: /Dock token|Undock token/ })).toHaveCount(0);
  await handle(page).press("End");
  await expect(sheet(page)).toHaveAttribute("data-detent", "peek");
  await handle(page).click({ trial: true });
  await drag(handle(page), -100, 650);
  await expect(sheet(page)).toHaveAttribute("data-detent", "half");
  await handle(page).click({ trial: true });
  await drag(handle(page), -100, 650);
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await handle(page).press("End");
  await expect(sheet(page)).toHaveAttribute("data-detent", "peek");
  await handle(page).click({ trial: true });
  await drag(handle(page), -100, 70);
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await handle(page).click({ trial: true });
  await drag(handle(page), 40, 120, true);
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  for (const width of [320, 430, 760]) {
    await page.setViewportSize({ width, height: 844 });
    await expect(sheet(page)).toHaveAttribute("data-detent", "full");
    expect(await sheet(page).evaluate(el => { const r = el.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1; })).toBe(true);
  }
  await handle(page).press("ArrowDown");
  await expect(sheet(page)).toHaveAttribute("data-detent", "half");
  await handle(page).click({ trial: true });
  await page.screenshot({ path: info.outputPath("sheet-middle.png") });
  expect((await new AxeBuilder({ page }).include('.sheet-host.present').analyze()).violations).toEqual([]);
  await handle(page).press("Shift+Tab");
  expect(await sheet(page).evaluate(el => el.contains(document.activeElement))).toBe(true);
  await handle(page).focus();
  await page.keyboard.press("Escape");
  await expect(sheet(page)).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Show right sidebar", exact: true })).toBeFocused();
  await page.getByRole("button", { name: "Show right sidebar", exact: true }).click();
  await page.locator(".sheet-backdrop").click({ position: { x: 10, y: 20 } });
  await expect(sheet(page)).toHaveCount(0);
  await page.getByRole("button", { name: "Show right sidebar", exact: true }).click();
  await expect(sheet(page)).toBeVisible();
  await handle(page).click({ trial: true });
  await sheet(page).getByRole("button", { name: "Close drawer", exact: true }).evaluate(el => (el as HTMLButtonElement).click());
  await page.setViewportSize({ width: 320, height: 640 });
  await expect(sheet(page)).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("both inspector modes preserve token and all analysis views across resizing", async ({ page }, info) => {
  test.slow();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await generate(page);
  await page.locator(".msg .tok").first().click();
  await page.getByRole("button", { name: "Full token details", exact: true }).click();
  const details = page.locator("[data-token-details-scroll]");
  await details.press("ArrowRight");
  const selected = await details.locator(".tok-text").textContent();
  for (const mode of ["floating", "docked"]) {
    if (mode === "docked") await page.getByRole("button", { name: "Dock token details", exact: true }).click();
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(sheet(page)).toBeVisible();
    await expect(details.locator(".tok-text")).toHaveText(selected!);
    await handle(page).press("Home");
    for (const tab of ["geometry", "logits", "sae", "j-lens"]) {
      await details.getByRole("group", { name: "Token detail view", exact: true }).getByRole("button", { name: tab, exact: true }).click();
      expect(await details.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
      await details.evaluate(el => el.scrollTop = el.scrollHeight);
      expect(await details.evaluate(el => el.scrollTop + el.clientHeight >= el.scrollHeight - 1)).toBe(true);
      await details.evaluate(el => el.scrollTop = 0);
      await page.screenshot({ path: info.outputPath(`${mode}-${tab}.png`) });
    }
    await page.setViewportSize({ width: 1440, height: 900 });
    await expect(sheet(page)).toHaveCount(0);
    await expect(details.locator(".tok-text")).toHaveText(selected!);
    await expect(page.getByRole("button", { name: mode === "docked" ? "Undock token details" : "Dock token details", exact: true })).toBeVisible();
  }
});

test("touch scrolling yields at content edges; landscape and downward dismissal work", async ({ page, isMobile }) => {
  test.skip(!isMobile, "Phone touch and landscape behavior");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await generate(page);
  await page.locator(".msg .tok").first().tap();
  await page.getByRole("button", { name: "Full token details", exact: true }).tap();
  const details = page.locator("[data-token-details-scroll]");
  await details.getByRole("group", { name: "Token detail view", exact: true }).getByRole("button", { name: "logits", exact: true }).click();
  await details.evaluate(el => el.scrollTop = 80);
  await drag(details, 35, 400);
  await expect(sheet(page)).toHaveAttribute("data-detent", "half");
  await details.evaluate(el => el.scrollTop = el.scrollHeight);
  await drag(details, -100, 650);
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await page.setViewportSize({ width: 844, height: 390 });
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await expect.poll(() => sheet(page).evaluate(el => el.getBoundingClientRect().bottom <= innerHeight + 1)).toBe(true);
  await handle(page).press("End");
  await expect(sheet(page)).toHaveAttribute("data-detent", "peek");
  await drag(handle(page), 110, 200);
  await expect(sheet(page)).toHaveCount(0);
});

test("mouse dragging, interrupted gestures, editing, and reduced motion remain usable", async ({ page }, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await generate(page);
  await page.locator(".msg .tok").first().click();
  await page.getByRole("button", { name: "Full token details", exact: true }).click();
  await handle(page).press("End");
  await expect(sheet(page)).toHaveAttribute("data-detent", "peek");
  await handle(page).click({ trial: true });
  const box = (await handle(page).boundingBox())!;
  await page.mouse.move(box.x + 100, box.y + 20);
  await page.mouse.down();
  for (let i = 1; i <= 10; i++) {
    await page.mouse.move(box.x + 100, box.y + 20 - i * 10);
    await page.waitForTimeout(65);
  }
  await page.mouse.up();
  await expect(sheet(page)).toHaveAttribute("data-detent", "half");
  await handle(page).press("Home");
  await sheet(page).getByRole("button", { name: "Replace token…", exact: true }).click();
  const input = sheet(page).getByRole("textbox");
  await input.fill("replacement");
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await page.setViewportSize({ width: 390, height: 390 });
  await expect(input).toBeFocused();
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await expect.poll(() => sheet(page).evaluate(el => el.getBoundingClientRect().bottom <= innerHeight + 1)).toBe(true);
  await page.setViewportSize({ width: 320, height: 844 });
  await page.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  await page.screenshot({ path: info.outputPath("light-editing-320.png") });
  await handle(page).press("End");
  await input.focus();
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await expect(input).toHaveValue("replacement");
  await expect.poll(() => sheet(page).evaluate(el => new DOMMatrixReadOnly(getComputedStyle(el).transform).m42)).toBe(12);
  await sheet(page).getByRole("button", { name: "Close drawer", exact: true }).click();
  await expect(sheet(page)).toHaveCount(0);
});

test("native touch input scrolls content without moving the sheet until an edge", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "Native touch injection uses Chromium's input protocol");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  const session = await page.context().newCDPSession(page);
  await session.send("Emulation.setTouchEmulationEnabled", { enabled: true });
  await generate(page);
  await page.locator(".msg .tok").first().click();
  await page.getByRole("button", { name: "Full token details", exact: true }).click();
  const details = page.locator("[data-token-details-scroll]");
  const swipe = async (target: Locator, delta: number, duration: number) => {
    await target.click({ trial: true });
    const r = (await target.boundingBox())!;
    const x = r.x + 8, y = r.y + r.height / 2;
    const steps = duration < 150 ? 3 : 10;
    const timestamp = Date.now() / 1000;
    await session.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x, y }], timestamp });
    for (let step = 1; step <= steps; step++) {
      await page.waitForTimeout(duration / steps);
      await session.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x, y: y + delta * step / steps }], timestamp: timestamp + duration * step / steps / 1000 });
    }
    await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [], timestamp: timestamp + duration / 1000 + 0.001 });
  };
  await details.evaluate(el => el.scrollTop = 120);
  const before = await details.evaluate(el => el.scrollTop);
  await swipe(details, -100, 300);
  await expect.poll(() => details.evaluate(el => el.scrollTop)).toBeGreaterThan(before + 30);
  await expect(sheet(page)).toHaveAttribute("data-detent", "half");
  await page.waitForTimeout(300);
  await details.evaluate(el => el.scrollTop = 0);
  await swipe(details, 100, 650);
  await expect(sheet(page)).toHaveAttribute("data-detent", "peek");
  await swipe(handle(page), -220, 60);
  await expect(sheet(page)).toHaveAttribute("data-detent", "full");
  await swipe(handle(page), 100, 650);
  await expect(sheet(page)).toHaveAttribute("data-detent", "half");
  await session.detach();
});

test("nested pickers keep their own keys and interrupted springs can reverse", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await generate(page);
  await page.locator(".msg .tok").first().click();
  await page.getByRole("button", { name: "Full token details", exact: true }).click();
  await handle(page).press("Home");
  await sheet(page).getByRole("group", { name: "Token detail view", exact: true }).getByRole("button", { name: "sae", exact: true }).click();
  await sheet(page).getByRole("button", { name: "Sort by", exact: true }).click();
  await expect(page.getByRole("listbox", { name: "Sort by", exact: true })).toBeVisible();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("listbox", { name: "Sort by", exact: true })).toBeHidden();
  await expect(sheet(page)).toBeVisible();
  await handle(page).press("End");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  const positions = await handle(page).evaluate(async element => {
    const surface = element.closest(".sheet-host")!;
    const position = () => new DOMMatrixReadOnly(getComputedStyle(surface).transform).m42;
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const initial = position();
    element.dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true, cancelable: true }));
    while (position() >= initial - 1) await new Promise(requestAnimationFrame);
    const before = position();
    element.dispatchEvent(new KeyboardEvent("keydown", { key: "End", bubbles: true, cancelable: true }));
    const after = position();
    while (position() <= before + 1) await new Promise(requestAnimationFrame);
    return { before, after, final: position() };
  });
  expect(Math.abs(positions.before - positions.after)).toBeLessThan(1);
  expect(positions.final).toBeGreaterThan(positions.before);
  await expect(sheet(page)).toHaveAttribute("data-detent", "peek");
});
