import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;

test("historical chat text remains selectable and its tokens keyboard accessible", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=1");
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async url => {
    const { applyTreeSnapshot, loomTree } = await import(url);
    const nodes: any[] = [{id: "history-root", parent_id: null, role: "system", text: "", tokens: []}];
    const children: Record<string, string[]> = {"history-root": []};
    for (let i = 0; i < 24; i++) {
      const id = `history-${i}`;
      const parent = nodes.at(-1).id;
      const role = i % 2 === 0 ? "user" : "assistant";
      const text = i === 1 ? "Amberneedle keeps every original token." : `Message ${i}. ` + "This older message keeps its full text and token metadata. ".repeat(8);
      nodes.push({id, parent_id: parent, role, text, role_label: null, recipe: null,
        finish_reason: "stop", aggregate_readings: {}, thinking_tokens: [],
        tokens: role === "assistant" ? text.match(/\S+\s*/g)!.map((text, raw_index) => ({text, raw_index, token_id: raw_index + 1, logprob: -0.2})) : []});
      children[parent] = [id]; children[id] = [];
    }
    applyTreeSnapshot({tree_format: 2, drowse_version: "test", session_id: loomTree.session_id,
      model_id: loomTree.modelId, root_id: "history-root", active_node_id: nodes.at(-1).id,
      rev: loomTree.rev + 100, nodes, children_of: children, cast: {}});
  }, storesUrl);
  await expect(page.locator(".msg")).toHaveCount(24);
  const old = page.locator(".msg.historical").filter({hasText: "Amberneedle"});
  await expect(old).toHaveCount(1);
  await old.scrollIntoViewIfNeeded();
  const body = old.locator(".response-body");
  expect(await body.evaluate(element => {
    const range = document.createRange(); range.selectNodeContents(element);
    const selection = getSelection()!; selection.removeAllRanges(); selection.addRange(range);
    const text = selection.toString(); selection.removeAllRanges(); return text;
  })).toBe("Amberneedle keeps every original token.");
  const token = old.getByRole("button", {name: "Inspect token Amberneedle", exact: true});
  await token.focus();
  await expect(token).toBeFocused();
  await token.press("Enter");
  await expect(page.getByRole("button", {name: "Full token details", exact: true})).toBeVisible();
  expect(await page.evaluate(async url => (await import(url)).currentLoomTreeSnapshot().nodes.find((node: any) => node.id === "history-1").tokens.length, storesUrl)).toBe(5);
});
