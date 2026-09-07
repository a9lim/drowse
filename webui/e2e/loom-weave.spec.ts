import { setAppearance } from "./workbench-navigation";
import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const storesUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;

test("Loom header glow follows the mouse without moving content and respects reduced motion", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A reply to explore.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Loom", exact: true }).click();
  await page.getByRole("button", { name: /^Map/ }).click();
  const card = page.locator('.node[aria-current="true"]');
  await card.hover();
  const bounds = (await card.boundingBox())!;
  const x = () => card.evaluate(element => parseFloat((element as HTMLElement).style.getPropertyValue("--glow-x")));
  await page.mouse.move(bounds.x + bounds.width * 0.15, bounds.y + 20);
  await expect.poll(x).toBeLessThan(-8);
  const before = await card.locator(".node-head").boundingBox();
  await page.mouse.move(bounds.x + bounds.width * 0.85, bounds.y + 20);
  await expect.poll(x).toBeGreaterThan(8);
  expect(await card.locator(".node-head").boundingBox()).toEqual(before);
  await page.screenshot({ path: testInfo.outputPath("loom-pointer-glow.png") });
  await page.mouse.move(0, 0);
  await expect.poll(x).toBe(0);
  await card.dispatchEvent("pointermove", { pointerType: "touch", clientX: bounds.x, clientY: bounds.y });
  expect(await x()).toBe(0);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.mouse.move(bounds.x + bounds.width * 0.85, bounds.y + 20);
  expect(await x()).toBe(0);
  await expect.poll(() => card.evaluate(element => getComputedStyle(element, "::after").transform)).toBe("none");
});

test("Loom sentences have complete rounded borders without side shadows", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A reply to explore.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Loom", exact: true }).click();
  await page.getByRole("button", { name: /^Map/ }).click();
  const sentence = page.locator(".sentence-node").first();
  for (const theme of ["Light", "Dark"]) {
    await setAppearance(page, theme);
    for (const width of [390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await expect(sentence).toBeVisible();
      const style = await sentence.evaluate(element => {
        const css = getComputedStyle(element);
        return {
          widths: [css.borderTopWidth, css.borderRightWidth, css.borderBottomWidth, css.borderLeftWidth],
          shadow: css.boxShadow, background: css.backgroundImage,
          radius: css.borderTopLeftRadius, left: css.paddingLeft, right: css.paddingRight,
          fits: element.scrollWidth <= element.clientWidth,
        };
      });
      expect(style.widths).toEqual(["1px", "1px", "1px", "1px"]);
      expect(style.shadow).toBe("none");
      expect(style.background).toBe("none");
      expect(style.radius).toBe("4px");
      expect(style.left).toBe(style.right);
      expect(style.fits).toBe(true);
    }
  }
  await sentence.locator(".token-node").first().focus();
  await expect(sentence).toHaveCSS("box-shadow", "none");
  await page.screenshot({ path: testInfo.outputPath("loom-sentence-borders.png") });
});

test("Weave creates alternatives, backtracks, extends exact tokens, and preserves originals", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("button", { name: "Loom", exact: true }).click();
  await expect(page.getByRole("button", { name: /^Map/ })).toHaveAttribute("aria-current", "page");
  await page.getByRole("button", { name: /^Weave/ }).click();
  await expect(page.getByRole("button", { name: /^Weave/ })).toHaveAttribute("aria-current", "page");
  await page.getByRole("textbox", { name: "Starting text" }).fill("A path with several futures.");
  await page.getByRole("button", { name: /^Map/ }).click();
  await page.getByRole("button", { name: /^Weave/ }).click();
  await expect(page.getByRole("textbox", { name: "Starting text" })).toHaveValue("A path with several futures.");
  await page.getByRole("button", { name: "Generate 3", exact: true }).click();
  const options = page.locator("[data-weave-choice]");
  await expect(options).toHaveCount(3);
  const weave = page.locator(".weave");
  await expect(weave.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const snapshot = () => page.evaluate(async url => {
    const { loomTree } = await import(url);
    return [...loomTree.nodes.values()].map((node: any) => ({
      id: node.id, parent: node.parent_id, text: node.text, tokens: node.raw_token_ids,
    }));
  }, storesUrl);
  const original = await snapshot();
  const sourceId = await options.first().getAttribute("data-weave-choice");
  await options.first().getByRole("button", { name: "Choose", exact: true }).click();
  await expect(options.first()).toHaveClass(/chosen/);
  await expect(options).toHaveCount(3);
  await expect(weave.locator(".tokens button")).toHaveCount(0);
  await options.first().getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(options).toHaveCount(4);
  await expect(weave.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const extended = await snapshot();
  for (const old of original) expect(extended.find((node: any) => node.id === old.id)).toEqual(old);
  const source = original.find((node: any) => node.id === sourceId)!;
  const fork = extended.find((node: any) => !original.some((old: any) => old.id === node.id))!;
  expect(fork.parent).toBe(source.parent);
  expect(fork.tokens.slice(0, source.tokens.length)).toEqual(source.tokens);

  await options.first().getByRole("button", { name: "Explore next", exact: true }).click();
  await expect(options).toHaveCount(0);
  await expect(weave.getByText("No continuations at this point yet. Generate some to explore.")).toBeVisible();
  await weave.getByRole("button", { name: "Alternatives", exact: true }).click();
  await page.getByRole("option", { name: "1", exact: true }).click();
  await weave.getByRole("button", { name: "Generate 1", exact: true }).click();
  await expect(options).toHaveCount(1);
  await expect(weave.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const nextTurnId = await options.first().getAttribute("data-weave-choice");
  expect((await snapshot()).find((node: any) => node.id === nextTurnId)?.parent).toBe(sourceId);
  const beforeReroll = await snapshot();
  await page.evaluate(async ({ url, id }) => {
    const { loomRegenerateNode } = await import(url);
    await loomRegenerateNode(id);
  }, { url: storesUrl, id: nextTurnId });
  await expect(options).toHaveCount(2);
  await expect(weave.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const afterReroll = await snapshot();
  for (const old of beforeReroll) expect(afterReroll.find((node: any) => node.id === old.id)).toEqual(old);
  await weave.getByRole("button", { name: "Back one point", exact: true }).click();
  await expect(options).toHaveCount(4);
  await weave.getByRole("button", { name: "Back one point", exact: true }).click();
  await expect(options).toHaveCount(1);
  await expect(options.first()).toContainText("A path with several futures.");
  await options.first().getByRole("button", { name: "Explore next", exact: true }).click();
  await expect(options).toHaveCount(4);
  await weave.getByRole("button", { name: "Alternatives", exact: true }).click();
  await page.getByRole("option", { name: "2", exact: true }).click();
  await weave.getByRole("button", { name: "Generate 2", exact: true }).click();
  await expect(options).toHaveCount(6);
  await expect(weave.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await options.first().getByRole("button", { name: "Write alternative", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Start a branch" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("textbox").fill("An authored possibility.\nWith whitespace intact.");
  await dialog.locator(".primary").click();
  await expect(options).toHaveCount(7);
  const afterWrite = await snapshot();
  for (const old of original) expect(afterWrite.find((node: any) => node.id === old.id)).toEqual(old);
  await expect(options.last()).toContainText("An authored possibility.");
  await options.first().getByRole("button", { name: "Star option 1", exact: true }).click();
  await expect(options.first().getByRole("button", { name: "Starred option 1", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: /^Map/ }).click();
  await expect(page.locator(".tree-node-wrap:not([data-loom-shared])")).toHaveCount(10);
  await expect(page.locator("[data-loom-shared]")).toHaveCount(2);
  expect(errors).toEqual([]);
});

test("Weave opens at a chat reply's branch point without hiding its alternatives", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Start in chat, then explore.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Loom", exact: true }).click();
  await expect(page.getByRole("button", { name: /^Map/ })).toHaveAttribute("aria-current", "page");
  await expect(page.locator('.node[aria-current="true"]')).toBeVisible();
  await page.getByRole("button", { name: /^Weave/ }).click();
  await expect(page.locator("[data-weave-choice]")).toHaveCount(1);
  await expect(page.locator("[data-weave-choice]").first()).toContainText("deterministic local Drowse");
  await expect(page.locator(".weave .branch-point")).toContainText("Start in chat, then explore.");
});
