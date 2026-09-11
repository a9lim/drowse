import { chooseFixtureModel } from "./workbench-navigation";
import { selectWorkspaceView } from "./workbench-navigation";
import { openWorkspaceMenu } from "./workbench-navigation";
import { setAppearance, showWorkspaceTools, openTokenDetails, selectLoomView } from "./workbench-navigation";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
test.use({ hasTouch: true });
const portraitViewports = [
  { width: 320, height: 568 },
  { width: 375, height: 667 },
  { width: 390, height: 844 },
  { width: 430, height: 932 },
] as const;
const landscapeViewports = [
  { width: 667, height: 375 },
  { width: 844, height: 390 },
  { width: 932, height: 430 },
] as const;

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

test("Weave keeps branch controls reachable on narrow phones and in landscape", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkbench(page);
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await selectLoomView(page, /^Weave\b/);
  const weave = page.locator(".weave");
  await expect(weave).toBeVisible();
  await weave.getByRole("textbox", { name: "Starting text" }).fill("Explore the possibilities. ".repeat(12));
  await weave.getByRole("button", { name: "Generate 3", exact: true }).click();
  await expect(weave.locator("[data-weave-choice]")).toHaveCount(3);
  await expect(weave.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  for (const theme of ["Light", "Dark"]) {
    await page.setViewportSize({ width: 390, height: 844 });
    if (await page.locator("html").getAttribute("data-theme") !== theme.toLowerCase()) {
      await setAppearance(page, theme);
    }
    for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }, { width: 844, height: 390 }]) {
      await page.setViewportSize(viewport);
      await expectNoPageOverflow(page);
      await expectNoHorizontalOverflow(weave);
      const generate = weave.getByRole("button", { name: "Generate 3", exact: true });
      await generate.scrollIntoViewIfNeeded();
      await expectInsideVisualViewport(generate);
      await expectTouchHeight(generate);
      const choose = weave.locator("[data-weave-choice]").first().getByRole("button", { name: "Choose", exact: true });
      await choose.scrollIntoViewIfNeeded();
      await expectInsideVisualViewport(choose);
      await expectTouchHeight(choose);
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await weave.locator("[data-weave-choice]").first().scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("weave-phone.png") });
});

async function openSetupFixture(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?layoutFixture=setup`);
  await chooseFixtureModel(page);
  await expect(page.getByRole("button", { name: "Download and open", exact: true }))
    .toBeVisible();
}

async function openWorkbench(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeVisible();
}

async function expectNoPageOverflow(page: Page): Promise<void> {
  await expect.poll(async () => page.evaluate(() => (
    document.documentElement.scrollWidth - document.documentElement.clientWidth <= 1 &&
    document.body.scrollWidth - document.documentElement.clientWidth <= 1
  ))).toBe(true);
}

async function expectInsideVisualViewport(locator: Locator): Promise<void> {
  await expect.poll(async () => locator.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0;
    const top = viewport?.offsetTop ?? 0;
    const right = left + (viewport?.width ?? window.innerWidth);
    const bottom = top + (viewport?.height ?? window.innerHeight);
    return (
      box.left >= left - 1 &&
      box.right <= right + 1 &&
      box.top >= top - 1 &&
      box.bottom <= bottom + 1
    );
  })).toBe(true);
}

async function expectTouchHeight(locator: Locator): Promise<void> {
  const height = await locator.evaluate((element) => element.getBoundingClientRect().height);
  expect(height).toBeGreaterThanOrEqual(44);
}

async function expectNoHorizontalOverflow(locator: Locator): Promise<void> {
  await expect.poll(async () => locator.evaluate((element) =>
    element.scrollWidth - element.clientWidth <= 1
  )).toBe(true);
}

async function dispatchTouchPointer(
  locator: Locator,
  type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel",
  pointerId: number,
  x: number,
  y: number,
): Promise<void> {
  await locator.evaluate((element, args) => {
    const bounds = element.getBoundingClientRect();
    element.dispatchEvent(new PointerEvent(args.type, {
      bubbles: true,
      cancelable: true,
      pointerId: args.pointerId,
      pointerType: "touch",
      isPrimary: args.pointerId === 1,
      button: 0,
      buttons: args.type === "pointerup" || args.type === "pointercancel" ? 0 : 1,
      clientX: bounds.left + args.x,
      clientY: bounds.top + args.y,
    }));
  }, { type, pointerId, x, y });
}

async function expectConversationUsable(page: Page, minimumLogHeight: number): Promise<void> {
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  const send = page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ });
  const stop = page.getByRole("button", { name: "Stop", exact: true });

  await composer.focus();
  await expect(page.locator(".turn-plan-summary")).toBeVisible();
  await expectInsideVisualViewport(composer);
  await expectInsideVisualViewport(send);
  await expectInsideVisualViewport(stop);
  await expectTouchHeight(send);
  await expectTouchHeight(stop);
  await expectTouchHeight(page.locator(".composer-resizer"));

  const metrics = await page.evaluate(() => {
    const log = document.querySelector<HTMLElement>(".log");
    const chat = document.querySelector<HTMLElement>(".chat");
    const composer = document.querySelector<HTMLElement>(".input");
    return {
      logHeight: log?.getBoundingClientRect().height ?? 0,
      chatHasHorizontalOverflow: chat
        ? chat.scrollWidth - chat.clientWidth > 1
        : true,
      composerFontSize: Number.parseFloat(getComputedStyle(composer!).fontSize),
    };
  });
  expect(metrics.logHeight).toBeGreaterThanOrEqual(minimumLogHeight);
  expect(metrics.chatHasHorizontalOverflow).toBe(false);
  expect(metrics.composerFontSize).toBeGreaterThanOrEqual(16);
  await expectNoPageOverflow(page);
}

test("hosted setup reflows across representative iPhone viewports", async ({ page }) => {
  for (const viewport of [...portraitViewports, ...landscapeViewports]) {
    await page.setViewportSize(viewport);
    await openSetupFixture(page);
    await chooseFixtureModel(page);
    const action = page.getByRole("button", { name: "Download and open", exact: true });
    await action.scrollIntoViewIfNeeded();
    await expectInsideVisualViewport(action);
    await expectTouchHeight(action);
    await expectNoPageOverflow(page);
  }
});

test("conversation composer remains reachable in portrait and landscape", async ({ page }) => {
  const resizeErrors: string[] = [];
  await page.addInitScript(() => {
    window.addEventListener("error", event => {
      if (event.message.includes("ResizeObserver")) console.error(`composer resize: ${event.message}`);
    });
  });
  page.on("console", message => {
    if (message.text().startsWith("composer resize:")) resizeErrors.push(message.text());
  });
  await page.setViewportSize(portraitViewports[2]);
  await openWorkbench(page);

  for (const viewport of portraitViewports) {
    await page.setViewportSize(viewport);
    await expectConversationUsable(page, 44);
  }

  for (const viewport of landscapeViewports) {
    await page.setViewportSize(viewport);
    await expectConversationUsable(page, 24);
  }
  expect(resizeErrors).toEqual([]);
});

test("keyboard-short visual viewports keep the composer actions reachable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).focus();

  for (const viewport of [
    { width: 390, height: 430 },
    { width: 320, height: 360 },
  ]) {
    await page.setViewportSize(viewport);
    const send = page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ });
    await send.scrollIntoViewIfNeeded();
    await expectInsideVisualViewport(page.getByRole("textbox", { name: /^Compose as / }));
    await expectInsideVisualViewport(send);
    await expectNoPageOverflow(page);
  }
});

test("Loom preserves a useful touch canvas on short screens", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkbench(page);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Show the mobile loom layout.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toBeVisible();
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await selectLoomView(page, /^Map\b/);

  for (const viewport of [
    { width: 320, height: 568 },
    { width: 844, height: 390 },
  ]) {
    await page.setViewportSize(viewport);
    await showWorkspaceTools(page, "Loom");
    const tree = page.locator(".tree-scroll");
    await expect(tree).toBeVisible();
    await expect(page.locator(".shell")).toHaveCSS("height", `${viewport.height}px`);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const height = await tree.evaluate((element) => element.getBoundingClientRect().height);
    expect(height).toBeGreaterThanOrEqual(84);
    await expectTouchHeight(page.getByRole("button", { name: "Zoom out" }));
    await expectTouchHeight(page.getByRole("button", { name: "Zoom in" }));
    await openWorkspaceMenu(page);
    await page.getByRole("button", { name: "Hide Loom tools", exact: true }).click();
    await expect(page.locator("#loom-tools-header")).toHaveCount(0);
    await expect.poll(() => tree.evaluate(element => element.getBoundingClientRect().height)).toBeGreaterThanOrEqual(Math.max(90, height));
    const visibleTree = (await tree.boundingBox())!;
    expect(visibleTree.y + visibleTree.height).toBeLessThanOrEqual(viewport.height + 1);
    await expectNoPageOverflow(page);
  }
});

test("Loom supports anchored pinch zoom, touch panning, and cancellation", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkbench(page);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Build a touch gesture test branch.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toBeVisible();
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await selectLoomView(page, /^Map\b/);

  const tree = page.locator(".tree-scroll");
  const canvas = page.locator(".loom-canvas");
  await expect(tree).toBeVisible();
  await expect(canvas).toHaveAttribute("data-loom-zoom", /\d/);

  const initialZoom = Number(await canvas.getAttribute("data-loom-zoom"));
  await dispatchTouchPointer(tree, "pointerdown", 1, 72, 170);
  await dispatchTouchPointer(tree, "pointerdown", 2, 172, 170);
  await dispatchTouchPointer(tree, "pointermove", 2, 272, 170);
  await expect.poll(async () => Number(await canvas.getAttribute("data-loom-zoom")))
    .toBeGreaterThan(initialZoom);
  await dispatchTouchPointer(tree, "pointerup", 1, 72, 170);
  await dispatchTouchPointer(tree, "pointerup", 2, 272, 170);

  const initialX = Number(await canvas.getAttribute("data-loom-camera-x"));
  const initialY = Number(await canvas.getAttribute("data-loom-camera-y"));
  await dispatchTouchPointer(tree, "pointerdown", 3, 86, 250);
  await dispatchTouchPointer(tree, "pointermove", 3, 126, 282);
  await expect.poll(async () => Number(await canvas.getAttribute("data-loom-camera-x"))).toBeCloseTo(initialX + 40, 2);
  await expect.poll(async () => Number(await canvas.getAttribute("data-loom-camera-y"))).toBeCloseTo(initialY + 32, 2);
  await dispatchTouchPointer(tree, "pointerup", 3, 126, 282);

  await dispatchTouchPointer(tree, "pointerdown", 4, 100, 220);
  await dispatchTouchPointer(tree, "pointermove", 4, 120, 230);
  await dispatchTouchPointer(tree, "pointercancel", 4, 120, 230);
  await expect(tree).not.toHaveClass(/dragging/);
  await expectNoPageOverflow(page);
});

test("3D reading details pinch-zoom and reflow without clipping", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await openWorkbench(page);
  await selectWorkspaceView(page, "Controls");
  const guidanceTabs = page.getByRole("group", { name: "Response guidance type" });
  await guidanceTabs.getByRole("button", { name: "Subspace", exact: true }).click();
  await page.getByRole("button", { name: "Add subspace probe", exact: true }).click();
  const attachDrawer = page.getByRole("dialog");
  await attachDrawer.getByText("attach selector", { exact: true }).click();
  await attachDrawer.getByRole("textbox", { name: "Selector" }).fill("fixture/mobile-3d");
  await attachDrawer.getByRole("button", { name: "+ attach", exact: true }).click();
  await page.keyboard.press("Escape");

  await page.getByRole("button", { name: "Inspect probe fixture/mobile-3d" }).click();
  const inspector = page.getByRole("dialog", { name: "Reading details" });
  const plot = inspector.getByLabel(/Whitened geometry for mobile-3d/);
  await expect(inspector).toBeVisible();
  await expect(plot).toBeVisible();
  await expectInsideVisualViewport(inspector);
  await expectNoHorizontalOverflow(inspector);

  const initialZoom = Number(await plot.getAttribute("data-orbit-zoom"));
  await dispatchTouchPointer(plot, "pointerdown", 1, 75, 120);
  await dispatchTouchPointer(plot, "pointerdown", 2, 155, 120);
  await dispatchTouchPointer(plot, "pointermove", 2, 235, 120);
  await expect.poll(async () => Number(await plot.getAttribute("data-orbit-zoom")))
    .toBeGreaterThan(initialZoom);
  await dispatchTouchPointer(plot, "pointerup", 1, 75, 120);
  await dispatchTouchPointer(plot, "pointerup", 2, 235, 120);

  for (const label of ["Rotate geometry left", "Zoom geometry out", "reset"]) {
    await expectTouchHeight(inspector.getByRole("button", { name: label, exact: true }));
  }
  await expectNoPageOverflow(page);
});

test("response, model, J-lens, and SAE controls remain reachable on a phone", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await openWorkbench(page);
  await selectWorkspaceView(page, "Controls");

  const controlsTabs = page.getByRole("group", { name: "Controls section" });
  await expectTouchHeight(controlsTabs.getByRole("button", { name: "Response" }));
  await expectTouchHeight(controlsTabs.getByRole("button", { name: "Model" }));
  const guidanceTabs = page.getByRole("group", { name: "Response guidance type" });

  await guidanceTabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  await expect(lens).toBeVisible();
  await expectNoPageOverflow(page);

  await guidanceTabs.getByRole("button", { name: "SAE", exact: true }).click();
  await expect(page.getByLabel("Model feature controls")).toBeVisible();
  await expectNoPageOverflow(page);

  await controlsTabs.getByRole("button", { name: "Model" }).click();
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  await expect(modelControls.getByRole("button", { name: "refresh" })).toBeVisible();
  await expectNoHorizontalOverflow(modelControls);
  await expectNoPageOverflow(page);
});

test("unmeasured reading cards fit narrow phone screens", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(page.locator(".shell")).toBeVisible();
  await selectWorkspaceView(page, "Controls");
  const tabs = page.getByRole("group", { name: "Response guidance type" });
  await tabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  await lens.getByRole("textbox", { name: "Watch a prediction word" }).fill("fixture");
  await lens.getByRole("button", { name: "Watch word", exact: true }).click();
  await expect(lens.getByText("Not measured", { exact: true })).toBeVisible();
  await expectNoHorizontalOverflow(lens.getByRole("list", { name: "J-lens probe tokens" }));
  await expectNoPageOverflow(page);
  await tabs.getByRole("button", { name: "SAE", exact: true }).click();
  const sae = page.getByLabel("Model feature controls");
  await sae.getByRole("textbox", { name: "Watch a model feature" }).fill("7");
  await sae.getByRole("button", { name: "Watch feature", exact: true }).click();
  await expect(sae.getByText("Not measured", { exact: true })).toBeVisible();
  await expectNoHorizontalOverflow(sae.getByRole("list", { name: "SAE feature probes" }));
  await expectNoPageOverflow(page);
});

test("probability and lens meters keep labels, values and bars separated", async ({ page }, testInfo) => {
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Inspect the next word.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body .tok").first()).toBeVisible();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  await page.evaluate(async url => {
    const { chatLog } = await import(url);
    chatLog.turns.at(-1).tokens[0].topAlts = [
      { id: 1, text: "fixture", logprob: -0.1 },
      { id: 2, text: "alternative", logprob: -4.2 },
      { id: 3, text: "unlikely", logprob: -20 },
    ];
  }, `/@fs/${resolve("src/lib/stores.svelte.ts")}`);
  await page.locator(".msg .response-body .tok").first().click();
  const drawer = await openTokenDetails(page);
  for (const name of [/^logits\b/i, /^j-lens\b/i]) {
    await drawer.getByRole("button", { name }).click();
    const rows = drawer.locator(".reading");
    await expect(rows.first()).toBeVisible();
    for (const width of [1440, 900, 640, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await expectNoPageOverflow(page);
      expect(await rows.evaluateAll(elements => elements.every(row => {
        const parts = Array.from(row.children) as HTMLElement[];
        const [label, bar, middle, value] = parts.map(part => part.getBoundingClientRect());
        return row.scrollWidth <= row.clientWidth + 1 &&
          parts.every(part => part.scrollWidth <= part.clientWidth + 1) &&
          label.right + 4 <= middle.left && middle.right + 4 <= value.left &&
          bar.top >= Math.max(label.bottom, middle.bottom, value.bottom) && bar.width > 100;
      }))).toBe(true);
    }
  }
  await page.screenshot({ path: testInfo.outputPath("meter-padding-phone.png") });
});

test("SAE readout labels preserve metadata and activation bars never overlap values", async ({ page }, testInfo) => {
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Inspect feature activations.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body .tok").first()).toBeVisible();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  const description = "References to opening delimiters in nested expressions, including brackets and parentheses.";
  await page.evaluate(async ({ url, description }) => {
    const { chatLog } = await import(url);
    const token = chatLog.turns.at(-1).tokens[0];
    token.measurements.instruments.sae = {
      binding: { source: "fixture-sae", steering: null, layer: 13 },
      readout: { features: [
        { id: 16190, activation: 773.84, label: null, max_act: null },
        { id: 2696, activation: 706.89, label: description, max_act: 100 },
        { id: 396, activation: 706.30, label: "   ", max_act: null },
        { id: 2286, activation: 657.59, label: "long-description-without-spaces-".repeat(6), max_act: null },
        { id: 485, activation: 0, label: null, max_act: null },
      ] },
    };
  }, { url: `/@fs/${resolve("src/lib/stores.svelte.ts")}`, description });
  await page.locator(".msg .response-body .tok").first().click();
  const drawer = await openTokenDetails(page);
  await drawer.getByRole("button", { name: /^sae\b/i }).click();
  const list = drawer.getByRole("list", { name: "Top SAE features" });
  const cards = list.getByRole("listitem", { name: /^SAE feature / });
  await expect(list.locator('[role="listitem"][aria-label^="SAE feature "][title]')).toHaveCount(0);
  await expect(cards).toHaveCount(5);
  await expect(list.getByText("No description included in this pack", { exact: true })).toHaveCount(3);
  await expect(list.getByText(description, { exact: true })).toBeVisible();
  await expect(list.getByText("Layer 13", { exact: true })).toHaveCount(5);
  await expect(list.locator(".sae-value")).toHaveText(["773.84", "7.069", "706.30", "657.59", "0.00"]);
  for (const theme of ["light", "dark"]) {
    await page.emulateMedia({ colorScheme: theme });
    for (const width of [1440, 900, 760, 640, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await expectNoPageOverflow(page);
      for (const card of await cards.all()) {
        await expectNoHorizontalOverflow(card);
        await expectNoHorizontalOverflow(card.locator(".feature-description"));
        const { value, bar } = await card.evaluate(element => ({
          value: element.querySelector(".sae-value")!.getBoundingClientRect().toJSON(),
          bar: element.querySelector('.bar[role="img"]')!.getBoundingClientRect().toJSON(),
        }));
        expect(bar.y).toBeGreaterThanOrEqual(value.y + value.height);
        expect(bar.width).toBeGreaterThan(100);
        expect(await card.locator(".sae-value").evaluate((element) => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1);
      }
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    await list.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath(`sae-activations-${theme}.png`) });
  }
});

test("saved chats and token details use contained, scrollable phone drawers", async ({ page }) => {
  for (const viewport of [
    { width: 320, height: 568 },
    { width: 667, height: 375 },
  ]) {
    await page.setViewportSize(viewport);
    await openWorkbench(page);

    await selectWorkspaceView(page, "Loom");
    await showWorkspaceTools(page, "Loom");
    await page.locator(".loom-sidebar").getByRole("button", { name: "Open", exact: true }).click();
    const library = page.getByRole("dialog", { name: "Saved chats" });
    await expect(library).toBeVisible();
    await expectInsideVisualViewport(library);
    await expectNoHorizontalOverflow(library);
    await library.getByRole("button", { name: "Close drawer" }).click();

    await page.getByRole("button", { name: /^(Conversation|Chat)$/ }).click();
    const composer = page.getByRole("textbox", { name: /^Compose as / });
    await composer.fill("Inspect this reply on a phone.");
    await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
    const response = page.locator(".msg .response-body").last();
    await expect(response).toBeVisible();
    await response.locator(".tok").first().click();

    const tokenDetails = await openTokenDetails(page);
    await expect(tokenDetails).toBeVisible();
    await expectInsideVisualViewport(tokenDetails);
    await expectNoHorizontalOverflow(tokenDetails);
    await tokenDetails.getByRole("button", { name: /^sae\b/i }).click();
    await expect(tokenDetails.getByText("No SAE is loaded", { exact: true })).toBeVisible();
    await expectNoPageOverflow(page);
    await tokenDetails.getByRole("button", { name: "Close drawer" }).click();
  }
});
