import { selectWorkspaceView } from "./workbench-navigation";
import { openWorkspaceMenu } from "./workbench-navigation";
import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("help icons open on hover, stay readable, and dismiss with Escape", async ({ page }, testInfo) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  const trigger = page.getByRole("button", { name: "About generation status", exact: true });
  const tip = page.locator(`#${await trigger.getAttribute("aria-describedby")}`);
  await expect(tip).toBeHidden();
  await trigger.hover();
  await expect(tip).toBeVisible();
  await expect(trigger).not.toBeFocused();
  await tip.hover();
  await page.waitForTimeout(200);
  await expect(tip).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("hover-help.png") });
  await page.keyboard.press("Escape");
  await expect(tip).toBeHidden();
  await page.mouse.move(0, 0);
  await trigger.hover();
  await expect(tip).toBeVisible();
  await page.mouse.move(0, 0);
  await expect(tip).toBeHidden();
});

test("help icons open on keyboard focus without trapping it", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  const trigger = page.getByRole("button", { name: "About generation status", exact: true });
  const tip = page.locator(`#${await trigger.getAttribute("aria-describedby")}`);
  await page.keyboard.press("Tab");
  await trigger.focus();
  await expect(tip).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(tip).toBeHidden();
  await expect(trigger).toBeFocused();
  await trigger.press("Enter");
  await expect(tip).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(trigger).not.toBeFocused();
  await expect(tip).toBeHidden();
});

test("help icons still toggle with touch and close on an outside tap", async ({ page, isMobile }) => {
  test.skip(!isMobile, "Touch interaction");
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  const trigger = page.getByRole("button", { name: "About generation status", exact: true });
  const tip = page.locator(`#${await trigger.getAttribute("aria-describedby")}`);
  await trigger.tap();
  await expect(tip).toBeVisible();
  await trigger.tap();
  await expect(tip).toBeHidden();
  await trigger.tap();
  await expect(tip).toBeVisible();
  await page.getByRole("region", { name: "Conversation", exact: true }).tap();
  await expect(tip).toBeHidden();
});

test("chat text has no hover tooltip and remains inspectable", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("What do marmots eat?");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  const token = page.locator(".msg .tok").first();
  await expect(token).toBeVisible();
  await expect(page.locator(".msg .tok[title]")).toHaveCount(0);
  await token.hover();
  await page.waitForTimeout(400);
  await expect(page.locator("#drowse-tooltip")).toBeHidden();
  await expect(page.locator(".token-logits-popover")).toHaveCount(0);
  await token.click();
  await expect(page.getByRole("button", { name: "Full token details", exact: true })).toBeVisible();
  const popup = page.locator(".token-logits-popover");
  for (let reversal = 0; reversal < 3; reversal++) {
    await expect(popup).toHaveCSS("opacity", "1");
    const state = await token.evaluate(async el => {
      const { tick } = await import("/e2e/svelte-runtime.ts");
      const panel = document.querySelector<HTMLElement>(".token-logits-popover")!;
      panel.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
      await tick();
      const closingInert = panel.inert;
      (el as HTMLElement).focus();
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
      await tick();
      return { closingInert, reused: document.querySelector(".token-logits-popover") === panel };
    });
    expect(state).toEqual({ closingInert: true, reused: true });
    await expect(popup).toHaveJSProperty("inert", false);
    await popup.getByRole("button", { name: "Full token details", exact: true }).focus();
    await expect(popup.getByRole("button", { name: "Full token details", exact: true })).toBeFocused();
  }
  await page.getByRole("button", { name: "Full token details", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Generated word details", exact: true })).toBeVisible();
});

test("mirrored sidebar toggle pins an empty inspector and both slides share reversible timing", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  const left = page.locator("#workspace-sidebar");
  const right = page.locator("#workspace-token-sidebar");
  const rightToggle = page.getByRole("button", { name: /^(Show|Hide) right sidebar$/ });
  const leftToggle = page.getByRole("button", { name: /^(Show|Hide) left sidebar$/ });
  await expect(leftToggle.locator("path")).toHaveAttribute("d", "M9 4v16");
  await expect(rightToggle.locator("path")).toHaveAttribute("d", "M15 4v16");
  await expect(right).toBeHidden();
  expect(await left.evaluate(el => el.getAnimations().length)).toBe(0);
  await rightToggle.click();
  await expect(right).toBeVisible();
  await expect(right).toContainText("Select a token");
  await expect(rightToggle).toHaveAttribute("aria-expanded", "true");
  await expect(right).toHaveCSS("transform", "matrix(1, 0, 0, 1, 0, 0)");
  expect(await right.evaluate(el => getComputedStyle(el).transitionDuration)).toBe(await left.evaluate(el => getComputedStyle(el).transitionDuration));
  expect(await right.evaluate(el => getComputedStyle(el).transitionTimingFunction)).toBe(await left.evaluate(el => getComputedStyle(el).transitionTimingFunction));
  expect(await right.evaluate(el => getComputedStyle(el).transitionDuration)).toBe("0.3s, 0.3s, 0.3s");
  await openWorkspaceMenu(page);
  await page.getByRole("button", { name: "Help", exact: true }).click();
  const help = page.getByRole("dialog", { name: "Help and shortcuts", exact: true });
  await expect(help).toBeVisible();
  await expect(right).toHaveAttribute("inert", "");
  await page.keyboard.press("Escape");
  await expect(help).toHaveCount(0);
  await expect(right).toBeVisible();
  await expect(right).not.toHaveAttribute("inert");
  await selectWorkspaceView(page, /^Controls\b/);
  await expect(right).toBeVisible();
  await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
  for (const side of ["left", "right"]) {
    const panel = side === "left" ? left : right;
    const toggle = side === "left" ? leftToggle : rightToggle;
    await expect(panel).toHaveCSS("opacity", "1");
    await expect(panel).toHaveCSS("transform", "matrix(1, 0, 0, 1, 0, 0)");
    const sample = await toggle.evaluate(async (el, side) => {
      const { tick } = await import("/e2e/svelte-runtime.ts");
      const panel = document.querySelector(`#workspace-${side === "left" ? "sidebar" : "token-sidebar"}`)!;
      (el as HTMLButtonElement).click();
      await tick();
      for (const animation of panel.getAnimations()) {
        animation.pause();
        animation.currentTime = 80;
      }
      const style = getComputedStyle(panel);
      const result = { x: new DOMMatrixReadOnly(style.transform).m41, opacity: Number(style.opacity), inert: (panel as HTMLElement).inert };
      (el as HTMLButtonElement).click();
      return result;
    }, side);
    expect(sample.inert).toBe(true);
    expect(sample.opacity).toBeGreaterThan(0);
    expect(sample.opacity).toBeLessThan(1);
    expect(side === "left" ? sample.x < 0 : sample.x > 0).toBe(true);
    await expect(panel).toHaveCSS("transform", "matrix(1, 0, 0, 1, 0, 0)");
    await expect(panel).not.toHaveAttribute("inert");
  }
  for (const width of [1440, 900, 760, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(() => right.evaluate(el => {
      const r = el.getBoundingClientRect();
      return r.x >= -1 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1 && r.height > 150;
    })).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`pinned-sidebar-${width}.png`) });
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator('.sheet-host[data-mobile-sheet="true"]')).toHaveCount(0);
  await expect(right).toHaveAttribute("role", "complementary");
  for (const panel of [left, right]) await expect(panel).toHaveCSS("transition-duration", "0s");
  await rightToggle.focus();
  await rightToggle.press("Enter");
  await expect(right).toBeHidden();
  await expect(rightToggle).toBeFocused();
  await rightToggle.click();
  await expect(right).toBeVisible();
  expect(errors).toEqual([]);
});

test("token details dock, follow selection, and leave the workspace interactive", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
  const editor = page.getByRole("textbox", { name: "Editable completion buffer" });
  await editor.fill("A field note: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  await page.getByRole("button", { name: "Inspect tokens", exact: true }).click();
  const tokens = page.locator(".inspect .tok");
  await tokens.first().click();
  await page.getByRole("button", { name: "Full token details", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Generated word details", exact: true });
  await expect(dialog).toBeVisible();
  await expect(page.locator("#workbench-header")).toHaveAttribute("inert", "");
  const sheet = page.locator("[data-token-details-scroll]");
  const firstText = await sheet.locator(".tok-text").textContent();
  await page.getByRole("button", { name: "Dock token details", exact: true }).click();
  const sidebar = page.getByRole("complementary", { name: "Generated word details", exact: true });
  await expect(sidebar).toBeVisible();
  await expect(sidebar.getByRole("button", { name: "Undock token details", exact: true }).locator('[data-icon="popout"]')).toBeVisible();
  await expect(dialog).toHaveCount(0);
  await expect(page.locator(".drawer-backdrop")).toHaveCount(0);
  await expect(page.locator("#workbench-header")).not.toHaveAttribute("inert");
  await expect(sheet.locator(".tok-text")).toHaveText(firstText!);
  await page.keyboard.press("Shift+Tab");
  expect(await sidebar.evaluate(element => element.contains(document.activeElement))).toBe(false);
  await tokens.nth(1).click();
  await expect(sheet.locator(".tok-text")).not.toHaveText(firstText!);
  await expect(tokens.nth(1)).toBeFocused();
  await expect(page.locator(".token-logits-popover")).toHaveCount(0);
  const selectedText = await sheet.locator(".tok-text").textContent();
  await page.getByRole("button", { name: "Edit text", exact: true }).click();
  await editor.focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Escape");
  await expect(editor).toBeFocused();
  await expect(sidebar).toBeVisible();
  await expect(sheet.locator(".tok-text")).toHaveText(selectedText!);

  const frame = page.locator(".workspace-frame");
  const expandedWidth = (await frame.boundingBox())!.width;
  await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
  await expect(page.locator("#workspace-sidebar")).toBeHidden();
  await expect(page.getByRole("button", { name: "Show left sidebar", exact: true })).toHaveAttribute("aria-expanded", "false");
  expect((await frame.boundingBox())!.width).toBeGreaterThan(expandedWidth + 100);

  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    for (const width of [1440, 900, 760, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await expect(page.locator("#workspace-token-sidebar")).toBeVisible();
      expect(await sheet.evaluate(element => element.scrollWidth <= element.clientWidth + 1), `${theme} ${width}`).toBe(true);
      expect(await page.locator("#workspace-token-sidebar").evaluate(element => {
        const box = element.getBoundingClientRect();
        return box.left >= 0 && box.right <= innerWidth + 1 && box.bottom <= innerHeight + 1;
      })).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`token-sidebar-${theme}-${width}.png`) });
    }
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
  await expect(page.locator("#workspace-sidebar")).toBeVisible();
  await sheet.evaluate(element => element.scrollTop = 0);
  await page.getByRole("button", { name: "Undock token details", exact: true }).click();
  await expect(dialog).toBeVisible();
  await expect(dialog.locator('[data-icon="popout"]')).toHaveCount(0);
  await expect(dialog.getByRole("button", { name: "Dock token details", exact: true }).locator('svg:has(path[d="M15 4v16"])')).toBeVisible();
  await expect(page.locator("#workbench-header")).toHaveAttribute("inert", "");
  await expect(sheet.locator(".tok-text")).toHaveText(selectedText!);
  await expect(page.getByRole("button", { name: "Dock token details", exact: true })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  expect(await dialog.evaluate(element => element.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Dock token details", exact: true })).toBeFocused();
  await page.getByRole("button", { name: "Dock token details", exact: true }).click();
  await page.keyboard.press("Escape");
  await expect(sidebar).toBeVisible();
  await page.getByRole("button", { name: "Hide right sidebar", exact: true }).click();
  await expect(sidebar).toHaveCount(0);
  await page.getByRole("button", { name: "Inspect tokens", exact: true }).click();
  await tokens.first().click();
  await expect(sidebar).toBeVisible();
  await expect(tokens.first()).toBeFocused();
});

test("chat selection and all analysis views fit the docked inspector", async ({ page }, testInfo) => {
  test.setTimeout(60_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await selectWorkspaceView(page, "Controls");
  await page.getByRole("group", { name: "Response guidance type" }).getByRole("button", { name: "Subspace", exact: true }).click();
  await page.getByRole("button", { name: "Add subspace probe", exact: true }).click();
  const picker = page.getByRole("dialog");
  await picker.getByText("attach selector", { exact: true }).click();
  await picker.getByRole("textbox", { name: "Selector" }).fill("fixture/calm.focused");
  await picker.getByRole("button", { name: "+ attach", exact: true }).click();
  await expect(page.getByText("probe fixture/calm.focused", { exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await selectWorkspaceView(page, "Conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain language models.");
  await page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  const tokens = page.locator(".msg .tok");
  await tokens.first().click();
  await page.getByRole("button", { name: "Full token details", exact: true }).click();
  await page.getByRole("button", { name: "Dock token details", exact: true }).click();
  const sidebar = page.getByRole("complementary", { name: "Generated word details", exact: true });
  const sheet = page.locator("[data-token-details-scroll]");
  await tokens.nth(1).click();
  await expect(sheet.locator(".tok-text")).toContainText((await tokens.nth(1).textContent())!.trim());
  await expect(tokens.nth(1)).toBeFocused();
  for (const width of [1440, 900, 320]) {
    await page.setViewportSize({ width, height: 900 });
    for (const tab of ["geometry", "logits", "sae", "j-lens"]) {
      await sheet.getByRole("group", { name: "Token detail view", exact: true }).getByRole("button", { name: tab, exact: true }).click();
      if (tab === "geometry") {
        const guide = sheet.locator(".geometry-guide");
        await guide.locator("summary").click();
        await expect(guide).toHaveAttribute("open", "");
        await expect(guide).toContainText("not the probability that a trait is true");
        await expect(sheet.locator('[role="listitem"][aria-description*="typical label spacings away"]').first()).toBeVisible();
        await expect(sheet.locator('[role="listitem"]').filter({ hasText: "membership" })).toHaveCount(0);
      }
      expect(await sheet.evaluate(element => element.scrollWidth <= element.clientWidth + 1), `${tab} ${width}`).toBe(true);
      await sheet.locator(":scope > .body").scrollIntoViewIfNeeded();
      await page.screenshot({ path: testInfo.outputPath(`chat-sidebar-${tab}-${width}.png`) });
    }
    if (width > 760) {
      await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
      await expect(page.locator("#workspace-sidebar")).toBeHidden();
      await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
      await expect(page.locator("#workspace-sidebar")).toBeVisible();
    }
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  expect((await new AxeBuilder({ page }).include(".drawer.docked").analyze()).violations).toEqual([]);
  await page.getByRole("button", { name: "Help and shortcuts", exact: true }).click();
  const help = page.getByRole("dialog", { name: "Help and shortcuts", exact: true });
  await expect(help).toBeVisible();
  await expect(page.locator("#workbench-header")).toHaveAttribute("inert", "");
  await page.keyboard.press("Escape");
  await expect(help).toHaveCount(0);
  await tokens.first().click();
  await expect(sidebar).toBeVisible();
  expect(errors).toEqual([]);
});
