import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";
import { selectWorkspaceView, showWorkspaceTools } from "./workbench-navigation";

const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;
test.use({ hasTouch: true });

async function seed(page: Page, count = 3) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async ({ url, count }) => {
    const { applyTreeSnapshot, loomTree, loomUiState } = await import(url);
    const make = (id: string, parent_id: string | null, text: string, role = "assistant") => ({
      id, parent_id, text, role, starred: false, role_label: null, recipe: null,
      finish_reason: "stop", aggregate_readings: {}, thinking_tokens: [],
      tokens: role === "assistant" ? (text.match(/\S+\s*/g) ?? []).map((text, raw_index) => ({ text, raw_index, token_id: raw_index + 1, logprob: -0.2 })) : [],
    });
    const replies = Array.from({ length: count }, (_, i) => make(`reply-${i}`, "prompt",
      `Alternative ${i}. ` + "Long replies should fit comfortably on a phone without losing branch context. ".repeat(8)));
    applyTreeSnapshot({
      tree_format: 2, drowse_version: "test", session_id: loomTree.session_id, model_id: loomTree.modelId,
      root_id: "root", active_node_id: "reply-0", rev: loomTree.rev + 100,
      nodes: [make("root", null, "", "system"), make("prompt", "root", "Explore the possibilities.", "user"), ...replies],
      children_of: { root: ["prompt"], prompt: replies.map(node => node.id), ...Object.fromEntries(replies.map(node => [node.id, []])) }, cast: {},
    });
    loomUiState.view = "map";
  }, { url: storesUrl, count });
  await selectWorkspaceView(page, "Loom");
  await expect(page.locator(".loom-viewport")).toBeVisible();
}

async function expectFitted(page: Page) {
  await expect.poll(() => page.evaluate(() => {
    const viewport = document.querySelector(".loom-viewport")!.getBoundingClientRect();
    return [...document.querySelectorAll(".tree-node-wrap")].every(node => {
      const box = node.getBoundingClientRect();
      return box.left >= viewport.left && box.right <= viewport.right
        && box.top >= viewport.top && box.bottom <= viewport.bottom;
    });
  })).toBe(true);
}

test("Loom opens fitted and follows measured cards and phone rotation", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 440, height: 796 });
  await seed(page);
  await expectFitted(page);
  await expect(page.locator(".loom-canvas")).not.toHaveAttribute("data-loom-zoom", "0.90");
  for (const viewport of [{ width: 320, height: 568 }, { width: 844, height: 390 }, { width: 440, height: 796 }]) {
    await page.setViewportSize(viewport);
    await expectFitted(page);
  }
  await page.screenshot({ path: testInfo.outputPath("loom-fitted-phone.png") });
  await selectWorkspaceView(page, "Chat");
  await selectWorkspaceView(page, "Loom");
  await expectFitted(page);
});

test("manual zoom survives resizing and Fit restores responsive framing", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 440, height: 796 });
  await seed(page, 1);
  await expectFitted(page);
  await showWorkspaceTools(page, "Loom");
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  const zoom = await page.locator(".loom-canvas").getAttribute("data-loom-zoom");
  await page.setViewportSize({ width: 844, height: 390 });
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-zoom", zoom!);
  await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
  await expectFitted(page);
  await page.setViewportSize({ width: 320, height: 568 });
  await expectFitted(page);
});

test("large Loom releases token cards while hidden and restores every branch", async ({ page }) => {
  test.setTimeout(90_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 440, height: 796 });
  await seed(page, 60);
  const cards = page.locator(".tree-node-wrap");
  await expect(cards).toHaveCount(61);
  await expect(page.locator(".token-node")).toHaveCount(0);
  await expect(page.locator(".loom-canvas")).toHaveCSS("will-change", "auto");
  await expect(page.locator(".loom-edges .edge-depth").first()).toHaveCSS("filter", "none");
  for (let i = 0; i < 4; i++) {
    await selectWorkspaceView(page, "Chat");
    await expect(cards).toHaveCount(0);
    await expect(page.locator(".loom-canvas")).toHaveCount(0);
    await selectWorkspaceView(page, "Loom");
    await expect(cards).toHaveCount(61);
  }
  await page.locator('[data-loom-node-id="reply-0"] .map-summary').click();
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-zoom", "1.00");
  await expect(page.locator('[data-loom-node-id="reply-0"] .token-node')).toHaveCount(98);
  expect(await cards.count()).toBeLessThan(10);
  const snapshot = await page.evaluate(async url => (await import(url)).currentLoomTreeSnapshot(), storesUrl);
  expect(snapshot.nodes).toHaveLength(62);
  expect(snapshot.nodes.find((node: { id: string }) => node.id === "reply-59").tokens).toHaveLength(98);
  expect(errors).toEqual([]);
});
