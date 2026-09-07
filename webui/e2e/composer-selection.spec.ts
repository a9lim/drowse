import { expect, test } from "@playwright/test";

test("composer handle leaves adjacent text selectable", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
  const handle = page.getByRole("slider", { name: "Resize writing area" });
  const composer = page.getByRole("textbox", { name: /^Compose as / });

  for (const width of [390, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    for (const theme of ["light", "dark"]) {
      await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
      await page.locator(".log").evaluate(log => {
        log.querySelector("[data-selection-check]")?.remove();
        (log as HTMLElement).style.position = "relative";
        const label = document.createElement("p");
        label.dataset.selectionCheck = "";
        label.textContent = "Selectable text beside the handle";
        label.style.cssText = "position:absolute;bottom:0;inset-inline:0;margin:0;text-align:center;font:16px/24px sans-serif;user-select:text";
        log.append(label);
      });
      const label = page.locator("[data-selection-check]");
      await label.click({ trial: true });
      const text = await label.evaluate(label => {
        const range = document.createRange();
        range.selectNodeContents(label);
        const rect = range.getBoundingClientRect();
        return { x: rect.x, y: rect.y, width: rect.width, height: rect.height, bottom: label.parentElement!.getBoundingClientRect().bottom };
      });
      const bounds = (await handle.boundingBox())!;
      expect(bounds.y).toBeGreaterThanOrEqual(text.bottom);
      expect(bounds.width).toBeLessThanOrEqual(100);
      expect(bounds.height).toBeGreaterThanOrEqual(width <= 760 ? 44 : 24);
      const before = await composer.evaluate(el => el.getBoundingClientRect().height);
      await page.mouse.move(text.x, text.y + text.height / 2);
      await page.mouse.down();
      await page.mouse.move(text.x + text.width, text.y + text.height / 2, { steps: 12 });
      await page.mouse.up();
      await expect.poll(() => page.evaluate(() => getSelection()?.toString())).toBe("Selectable text beside the handle");
      expect(await composer.evaluate(el => el.getBoundingClientRect().height)).toBe(before);
      await page.screenshot({ path: testInfo.outputPath(`selection-${theme}-${width}.png`) });
      await page.evaluate(() => getSelection()?.removeAllRanges());
    }
  }

  await handle.focus();
  await page.keyboard.press("Home");
  const minimum = await composer.evaluate(el => el.getBoundingClientRect().height);
  await page.keyboard.press("ArrowUp");
  await expect.poll(() => composer.evaluate(el => el.getBoundingClientRect().height)).toBeGreaterThan(minimum);
  const beforeDrag = await composer.evaluate(el => el.getBoundingClientRect().height);
  const bounds = (await handle.boundingBox())!;
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y - 48, { steps: 8 });
  await page.mouse.up();
  await expect.poll(() => composer.evaluate(el => el.getBoundingClientRect().height)).toBeGreaterThan(beforeDrag);
  await expect(page.locator(".composer-resizer-shell")).not.toHaveClass(/dragging/);
});
