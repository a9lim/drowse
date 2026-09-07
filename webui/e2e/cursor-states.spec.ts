import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";

test("workbench cursors describe editing, inspecting, adjusting and navigating", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.locator(".page-brand")).toHaveCSS("cursor", "pointer");
  await expect(page.getByRole("button", { name: "Dark", exact: true }).locator("svg")).toHaveCSS("cursor", "pointer");
  await expect(page.getByRole("button", { name: "About word colors", exact: true })).toHaveCSS("cursor", "help");
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await expect(composer).toHaveCSS("cursor", "text");
  await expect(page.getByRole("slider", { name: "Resize writing area" })).toHaveCSS("cursor", "row-resize");
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toHaveCSS("cursor", "not-allowed");

  await page.getByRole("button", { name: "Controls", exact: true }).click();
  const slider = page.getByRole("slider", { name: "Temperature", exact: true });
  await expect(slider).toHaveCSS("cursor", "ew-resize");
  await page.evaluate(async url => { (await import(url)).steerRack.customExpression = "0.3 jlens/orange"; }, `/@fs${resolve("src/lib/stores.svelte.ts")}`);
  await expect(page.getByRole("button", { name: "Copy response recipe", exact: true })).toHaveCSS("cursor", "copy");
  await page.evaluate(async url => { (await import(url)).steerRack.customExpression = null; }, `/@fs${resolve("src/lib/stores.svelte.ts")}`);
  await page.getByRole("button", { name: "Conversation", exact: true }).click();
  await composer.fill("Describe the shape of a thought.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await expect(page.locator(".response-body .tok").first()).toHaveCSS("cursor", "crosshair");

  await page.getByRole("button", { name: "Loom", exact: true }).click();
  await page.getByRole("button", { name: /^Map/ }).click();
  await expect(page.getByRole("button", { name: "Zoom in", exact: true })).toHaveCSS("cursor", "zoom-in");
  await expect(page.getByRole("button", { name: "Zoom out", exact: true })).toHaveCSS("cursor", "zoom-out");
  const node = page.locator('.node[data-cursor="context-menu"]').first();
  await expect(node).toHaveCSS("cursor", "context-menu");
  const viewport = page.locator(".loom-viewport");
  await expect(viewport).toHaveCSS("cursor", "grab");
  const box = (await viewport.boundingBox())!;
  await page.mouse.move(box.x + box.width - 15, box.y + box.height - 15);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 90, box.y + box.height - 90, { steps: 4 });
  await expect(viewport).toHaveCSS("cursor", "grabbing");
  await expect(node).toHaveCSS("cursor", "grabbing");
  await page.mouse.up();
  await expect(viewport).toHaveCSS("cursor", "grab");
  await expect(node).toHaveCSS("cursor", "context-menu");
});

test("busy and unavailable states override the normal action cursor", async ({ page }) => {
  await page.goto(`${devUrl}/credits`);
  await page.evaluate(async ({ buttonUrl, sliderUrl }) => {
    const [{ mount, createRawSnippet }, { default: Button }, { default: Slider }] = await Promise.all([
      import("/@id/svelte"), import(buttonUrl), import(sliderUrl),
    ]);
    const target = document.createElement("div");
    document.body.append(target);
    for (const [ariaLabel, busy, disabled] of [
      ["Available action", false, false], ["Working action", true, false],
      ["Waiting action", true, true], ["Unavailable action", false, true],
    ] as const) mount(Button, { target, props: { ariaLabel, busy, disabled, children: createRawSnippet(() => ({ render: () => `<span>${ariaLabel}</span>` })) } });
    mount(Slider, { target, props: { value: 0.5, disabled: true, ariaLabel: "Unavailable slider" } });
  }, { buttonUrl: `/@fs${resolve("src/lib/ui/Button.svelte")}`, sliderUrl: `/@fs${resolve("src/lib/Slider.svelte")}` });
  for (const [name, cursor] of [
    ["Available action", "pointer"], ["Working action", "progress"],
    ["Waiting action", "wait"], ["Unavailable action", "not-allowed"],
  ]) await expect(page.getByRole("button", { name, exact: true })).toHaveCSS("cursor", cursor);
  await expect(page.getByRole("slider", { name: "Unavailable slider" })).toHaveCSS("cursor", "not-allowed");
});
