import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const sourceRoot = `/@fs${resolve("src")}`;

for (const width of [390, 1440]) {
  for (const focused of [false, true]) {
    test(`one Stop click ends generation at ${width}px with composer focused=${focused}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
      await expect(page.locator(".shell")).toBeVisible();
      await page.evaluate(async root => {
        const { getRuntimeClient } = await import(`${root}/lib/runtime/registry.ts`);
        getRuntimeClient().generation.tokenDelayMs = 1_000;
      }, sourceRoot);
      const composer = page.getByRole("textbox", { name: /^Compose as / });
      await composer.fill("Stop this reply after its first token.");
      await page.getByRole("button", { name: "Send", exact: true }).click();
      await expect.poll(() => page.evaluate(async root =>
        (await import(`${root}/lib/stores.svelte.ts`)).genStatus.tokensSoFar,
      sourceRoot)).toBeGreaterThan(0);
      if (focused) await composer.focus();
      else await page.evaluate(() => (document.activeElement as HTMLElement)?.blur());

      const stop = page.getByRole("button", { name: "Stop", exact: true });
      await expect(stop).toBeEnabled();
      const before = (await stop.boundingBox())!;
      await page.mouse.move(before.x + before.width / 2, before.y + before.height / 2);
      await page.mouse.down();
      const pressed = (await stop.boundingBox())!;
      await page.mouse.up();
      expect(Math.abs(pressed.x - before.x)).toBeLessThanOrEqual(1);
      expect(Math.abs(pressed.y - before.y)).toBeLessThanOrEqual(1);
      await expect.poll(() => page.evaluate(async root =>
        (await import(`${root}/lib/stores.svelte.ts`)).genStatus.active,
      sourceRoot), { timeout: 2_000 }).toBe(false);
      const count = await page.evaluate(async root =>
        (await import(`${root}/lib/stores.svelte.ts`)).genStatus.tokensSoFar,
      sourceRoot);
      expect(count).toBeLessThan(10);
      await page.waitForTimeout(1_100);
      expect(await page.evaluate(async root =>
        (await import(`${root}/lib/stores.svelte.ts`)).genStatus.tokensSoFar,
      sourceRoot)).toBe(count);
    });
  }
}
