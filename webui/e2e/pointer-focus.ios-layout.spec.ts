import { expect, test } from "@playwright/test";

test("pointer controls stay quiet and keyboard navigation retains its focus ring", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/");
  const demo = page.locator(".capability-demo");
  const slider = demo.getByRole("slider");
  await slider.scrollIntoViewIfNeeded();
  await slider.focus();
  await slider.press("ArrowRight");
  await expect(slider).toHaveCSS("outline-style", "solid");
  const box = (await slider.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 10, box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  await expect(slider).toHaveCSS("outline-style", "none");
  expect(await page.evaluate(() => getSelection()?.toString())).toBe("");
  await slider.press("Home");
  await expect(slider).toHaveCSS("outline-style", "solid");
  await expect(slider).toHaveValue("0");
  const inspect = demo.getByRole("button", { name: "Inspect", exact: true });
  await inspect.focus();
  await inspect.press(" ");
  await expect(inspect).toHaveCSS("outline-style", "solid");
  await demo.getByRole("button", { name: "Compare", exact: true }).click();
  await expect(inspect).toHaveCSS("outline-style", "none");
  expect(await demo.locator(".context").evaluate(el => getComputedStyle(el).userSelect)).not.toBe("none");
});

test("forced colors preserves keyboard focus", async ({ page }) => {
  await page.emulateMedia({ forcedColors: "active", reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/");
  const button = page.locator(".capability-demo").getByRole("button", { name: "Inspect", exact: true });
  await button.focus();
  await button.press(" ");
  await expect(button).toHaveCSS("outline-style", "solid");
  await expect(button).toHaveCSS("outline-width", "2px");
});
