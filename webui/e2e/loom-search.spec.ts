import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;

async function seed(page: Page) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
  if (!await page.getByRole("button", { name: /^Loom(?: Explore|$)/ }).isVisible()) {
    await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
  }
  await page.getByRole("button", { name: /^Loom(?: Explore|$)/ }).click();
  await page.evaluate(async url => {
    const { applyTreeSnapshot, loomTree, loomUiState } = await import(url);
    const make = (id: string, parent_id: string | null, text: string, role = "assistant", starred = false) => ({
      id, parent_id, text, role, starred, role_label: null, recipe: null,
      finish_reason: "stop", aggregate_readings: {}, thinking_tokens: [],
      tokens: role === "assistant" ? (text.match(/\S+\s*/g) ?? []).map((text, raw_index) => ({ text, raw_index, token_id: raw_index + 1, logprob: -0.2 })) : [],
    });
    applyTreeSnapshot({
      tree_format: 2, drowse_version: "test", session_id: loomTree.session_id, model_id: loomTree.modelId,
      root_id: "search-root", active_node_id: "short", rev: loomTree.rev + 100,
      nodes: [make("search-root", null, "", "system"), make("prompt", "search-root", "Explore these paths", "user"),
        make("long", "prompt", "A long message has room to breathe and retains every token. ".repeat(20) + "The amber fox appears near the end.", "assistant", true),
        make("child", "long", "AMBER\nfox, moon"), make("short", "prompt", "A short reply.")],
      children_of: { "search-root": ["prompt"], prompt: ["long", "short"], long: ["child"], child: [], short: [] }, cast: {},
    });
    loomUiState.view = "map";
  }, storesUrl);
  await expect(page.locator(".tree-node-wrap")).toHaveCount(4);
}

test("long cards widen without crossing columns; search navigates all branches without switching paths", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await seed(page);
  const long = page.locator('[data-loom-node-id="long"]');
  const short = page.locator('[data-loom-node-id="short"]');
  await expect(long).toHaveCSS("width", "680px");
  await expect(short).toHaveCSS("width", "276px");
  const positions = await page.locator(".tree-node-wrap").evaluateAll(nodes => nodes.map(node => ({
    id: node.getAttribute("data-loom-node-id"), left: (node as HTMLElement).offsetLeft, width: (node as HTMLElement).offsetWidth,
  })));
  const parent = positions.find(node => node.id === "long")!;
  expect(positions.find(node => node.id === "child")!.left).toBeGreaterThan(parent.left + parent.width);
  await expect(page.locator('[data-loom-edge="long|child"]')).toHaveAttribute("d", new RegExp(`^M ${parent.left + parent.width} `));
  const search = page.getByRole("searchbox", { name: "Search messages" });
  await search.pressSequentially("amber fox", { delay: 20 });
  await expect(search).toHaveValue("amber fox");
  await expect(page.locator("#loom-search-status")).toHaveText("2 matching messages");
  await search.press("Shift+Enter");
  await expect(page.locator("#loom-search-status")).toHaveText("2 of 2 matching messages");
  await search.press("Enter");
  await expect(long).toHaveClass(/search-current/);
  await expect(page.locator("#loom-search-status")).toHaveText("1 of 2 matching messages");
  await expect.poll(() => page.evaluate(async url => (await import(url)).loomTree.active_node_id, storesUrl)).toBe("short");
  const sentence = long.locator(".sentence-node").last();
  await expect.poll(async () => {
    const box = (await sentence.boundingBox())!;
    const viewport = (await page.locator(".loom-viewport").boundingBox())!;
    return Math.abs(box.y + box.height / 2 - viewport.y - viewport.height / 2);
  }).toBeLessThan(5);
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const theme of ["dark", "light"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: `/@fs${resolve("src/lib/theme.ts")}`, theme });
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await page.screenshot({ path: testInfo.outputPath(`loom-search-${theme}.png`) });
  }
  await page.getByRole("button", { name: "Next match", exact: true }).click();
  await expect(page.locator('[data-loom-node-id="child"]')).toHaveClass(/search-current/);
  await search.fill("fox, moon");
  await expect(page.locator("#loom-search-status")).toHaveText("1 matching message");
  await search.fill("not present anywhere");
  await expect(page.locator("#loom-search-status")).toContainText("No messages match");
  await page.getByRole("button", { name: "Clear search", exact: true }).click();
  await expect(search).toBeFocused();
  await expect(page.locator(".tree-node-wrap.filtered-out")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("search reveals collapsed children, restores collapsed state, and filters starred messages", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await seed(page);
  const search = page.getByRole("searchbox", { name: "Search messages" });
  await search.fill("amber fox");
  await search.press("Enter");
  await search.press("Escape");
  await page.locator('[data-node-id="long"]').getByRole("button", { name: "Hide child paths", exact: true }).press("Enter");
  await expect(page.locator('[data-loom-node-id="child"]')).toHaveCount(0);
  await search.fill("fox, moon");
  await expect(page.locator('[data-loom-node-id="child"]')).toHaveCount(1);
  await expect(page.locator("#loom-search-status")).toHaveText("1 matching message");
  await search.press("Escape");
  await expect(page.locator('[data-loom-node-id="child"]')).toHaveCount(0);
  await page.getByRole("button", { name: /^Starred/ }).click();
  await search.fill("amber fox");
  await expect(page.locator("#loom-search-status")).toHaveText("1 matching message");
  await search.fill("unmatched");
  await expect(page.getByText("No starred messages match", { exact: true })).toBeVisible();
});

test("advanced filters stay explicit and the search bar fits a narrow viewport", async ({ page }) => {
  await seed(page);
  await page.setViewportSize({ width: 320, height: 900 });
  await page.getByRole("button", { name: "Search mode", exact: true }).click();
  await page.getByRole("option", { name: "Advanced", exact: true }).click();
  const search = page.getByRole("searchbox", { name: "Advanced filters" });
  await search.fill("starred, text:amber");
  await search.press("Enter");
  await expect(page.locator("#loom-search-status")).toHaveText("1 matching message");
  await search.fill("text:");
  await search.press("Enter");
  await expect(search).toHaveAttribute("aria-invalid", "true");
  const fits = await page.locator(".filter-bar").evaluate(element => element.scrollWidth <= element.clientWidth);
  expect(fits).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
