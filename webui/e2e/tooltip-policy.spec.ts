import { expect, test } from "@playwright/test";
import { openWorkspaceMenu, showWorkspaceTools } from "./workbench-navigation";

test("ordinary controls stay quiet across chat, instruments, and loom", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeVisible();
  await showWorkspaceTools(page, "chat");
  await expect(page.locator("[title], svg title, #drowse-tooltip")).toHaveCount(0);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain language models.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  await page.getByRole("textbox", { name: /^Compose as / }).blur();
  await page.locator(".log").press("Home");
  await expect.poll(() => page.locator(".log").evaluate(el => el.scrollTop)).toBe(0);
  await page.locator(".msg").first().hover();
  await page.waitForTimeout(400);
  await expect(page.locator(".info-popover:popover-open")).toHaveCount(0);
  await page.locator(".workspace-nav").getByRole("button", { name: "Controls", exact: true }).click();
  for (const label of ["Subspace", "Manifold", "SAE", "J-lens"]) {
    const tab = page.locator(".instrument-head .tabs").getByRole("button", { name: label, exact: true });
    await tab.click();
    await tab.hover();
    await page.waitForTimeout(350);
    await expect(page.locator("[title], svg title, #drowse-tooltip")).toHaveCount(0);
    await expect(page.locator(".info-popover:popover-open")).toHaveCount(0);
  }
  await page.locator(".workspace-nav").getByRole("button", { name: "Loom", exact: true }).click();
  await showWorkspaceTools(page, "Loom");
  await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
  const overview = page.locator(".map-summary").first();
  if (await overview.count()) await overview.click();
  await openWorkspaceMenu(page);
  await page.getByRole("button", { name: "Hide Loom tools", exact: true }).click();
  const generate = page.locator('.node').getByRole("button", { name: /^Generate another path from user/ });
  await generate.first().hover();
  await page.waitForTimeout(400);
  await expect(page.locator("[title], svg title, #drowse-tooltip")).toHaveCount(0);
  await expect(page.locator(".info-popover:popover-open")).toHaveCount(0);
});

test("explicit help ignores passing pointers and remains compact at phone edges", async ({ page, isMobile }, info) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  const trigger = page.getByRole("button", { name: "About generation status", exact: true });
  await expect(trigger).toBeVisible();
  const tip = page.locator(`#${await trigger.getAttribute("aria-describedby")}`);
  await trigger.dispatchEvent("pointerenter", { pointerType: "mouse" });
  await trigger.dispatchEvent("pointerleave", { pointerType: "mouse" });
  await page.waitForTimeout(400);
  await expect(tip).toBeHidden();
  for (const width of [320, 390, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    if (isMobile) await trigger.tap(); else await trigger.click();
    await expect(tip).toBeVisible();
    await expect(tip).toHaveCSS("opacity", "1");
    expect(await tip.evaluate(el => {
      const r = el.getBoundingClientRect();
      return r.left >= 7 && r.right <= innerWidth - 7 && r.top >= 7 && r.bottom <= innerHeight - 7;
    })).toBe(true);
    const padding = await tip.evaluate(el => parseFloat(getComputedStyle(el).paddingTop));
    expect(padding).toBeGreaterThanOrEqual(4);
    expect(padding).toBeLessThanOrEqual(8);
    await page.screenshot({ path: info.outputPath(`explicit-help-${width}.png`) });
    await page.keyboard.press("Escape");
    await expect(tip).toBeHidden();
  }
});
