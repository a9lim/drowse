import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const modules = {
  stores: `/@fs${resolve("src/lib/stores.svelte.ts")}`,
  api: `/@fs${resolve("src/lib/runtime/services.ts")}`,
  saved: `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`,
  workspace: `/@fs${resolve("src/lib/conversationWorkspace.ts")}`,
  home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`,
};
const editor = (page: Page) => page.getByRole("textbox", { name: "Editable completion buffer" });
async function select(page: Page, start: number, end = start) {
  await editor(page).evaluate((element, range) => {
    const textarea = element as HTMLTextAreaElement;
    textarea.focus();
    textarea.setSelectionRange(range.start, range.end);
    textarea.dispatchEvent(new Event("select", { bubbles: true }));
  }, { start, end });
}
async function tree(page: Page) {
  return page.evaluate(async url => (await import(url)).currentLoomTreeSnapshot(), modules.stores);
}
async function generate(page: Page) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
  await editor(page).fill("Once beneath the moon: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect.poll(async () => (await tree(page))?.nodes.some((node: any) => node.tokens?.length > 0)).toBe(true);
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
}

test("selection completion branches at the cursor, selection end, and selection start without overwriting tokens", async ({ page }, testInfo) => {
  await generate(page);
  const original = await tree(page);
  const originalText = await editor(page).inputValue();
  const originalNodes = original.nodes;
  for (const action of [
    { start: 5, end: 12, cut: 12, label: "Continue after selection" },
    { start: 5, end: 12, cut: 5, label: "Re-complete from selection" },
    { start: 0, end: 0, cut: 0, label: "Continue from cursor" },
    { start: 22, end: 22, cut: 22, label: "Continue from cursor" },
    { start: 24, end: originalText.length - 2, cut: 24, label: "Re-complete from selection" },
    { start: 10, end: 26, cut: 26, label: "Continue after selection" },
    { start: originalText.length, end: originalText.length, cut: originalText.length, label: "Continue from cursor" },
  ]) {
    await page.evaluate(async ({ url, id }) => (await import(url)).loomNavigate(id), { url: modules.stores, id: original.active_node_id });
    await expect(editor(page)).toHaveValue(originalText);
    await select(page, action.start, action.end);
    const button = page.getByRole("button", { name: action.label, exact: true });
    await expect(button).toBeEnabled();
    await button.focus();
    await page.keyboard.press("Enter");
    await expect.poll(async () => (await tree(page)).active_node_id).not.toBe(original.active_node_id);
    await expect.poll(() => editor(page).inputValue()).not.toBe(originalText.slice(0, action.cut));
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    const result = await tree(page);
    const active = result.nodes.find((node: any) => node.id === result.active_node_id);
    expect(active.tokens.length).toBeGreaterThan(0);
    expect(await editor(page).inputValue()).toBe(originalText.slice(0, action.cut) + active.text);
    for (const node of originalNodes) expect(result.nodes.find((candidate: any) => candidate.id === node.id)).toEqual(node);
  }
  await select(page, 1, 4);
  for (const width of [1440, 450, 320]) {
    await page.setViewportSize({ width, height: 900 });
    const actions = page.getByRole("group", { name: "Complete from selected text" });
    await actions.scrollIntoViewIfNeeded();
    expect(await actions.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    await actions.screenshot({ path: testInfo.outputPath(`selection-${width}.png`) });
  }
});

test("unsaved selections require saving, and branch failures leave the original editable", async ({ page }) => {
  await generate(page);
  const original = await editor(page).inputValue();
  await editor(page).fill(original + " Extra words.");
  await select(page, 3, 7);
  await expect(page.getByRole("button", { name: "Re-complete from selection", exact: true })).toBeDisabled();
  await expect(page.locator("#selection-hint")).toHaveText("Save your edit first to complete from this point.");
  await page.getByRole("button", { name: "Save edit", exact: true }).click();
  await expect(page.getByRole("button", { name: "Save edit", exact: true })).toHaveCount(0);
  await expect(editor(page)).toHaveValue(original + " Extra words.");
  await select(page, 3, 7);
  await page.evaluate(async url => {
    const { apiTree } = await import(url);
    const branch = apiTree.branch;
    apiTree.branch = async () => { apiTree.branch = branch; throw new Error("Selection branch unavailable"); };
  }, modules.api);
  await page.getByRole("button", { name: "Re-complete from selection", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Selection branch unavailable" })).toBeVisible();
  await expect(editor(page)).toHaveValue(original + " Extra words.");
  await expect(editor(page)).not.toHaveAttribute("readonly");
});

test("duplicate chat UI preserves full snapshots and independent autosave identities", async ({ page }, testInfo) => {
  await generate(page);
  await page.evaluate(async modules => {
    const [{ default: HostedHome }, { mount }, { conversationLibrary, registerConversationAutosave }, { sessionState }, { captureConversationSnapshot }] = await Promise.all([
      import(modules.home), import("/@id/svelte"), import(modules.saved), import(modules.stores), import(modules.workspace),
    ]);
    await conversationLibrary.create({ name: "Moon notes", modelType: "base", snapshot: captureConversationSnapshot() });
    registerConversationAutosave(async () => {});
    const modelId = sessionState.info.model_id;
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(HostedHome, { target, props: {
      controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {} },
      snapshot: { phase: "supported", models: [{ id: modelId, modelId, name: "Pythia 70M", setupComplete: true, fit: "recommended" }], download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: true } },
      onChooseModels() {},
    } });
  }, modules);
  const cards = page.locator("[data-saved-conversation]");
  await expect(page.getByRole("button", { name: "Duplicate Moon notes", exact: true })).toBeVisible();
  const beforeCount = await cards.count();
  await page.getByRole("button", { name: "Duplicate Moon notes", exact: true }).click();
  await expect(cards).toHaveCount(beforeCount + 1);
  const copyCard = cards.filter({ hasText: "Moon notes (copy)" });
  await expect(copyCard.locator(".chat-counts")).toHaveText("2 messages · 1 Loom thread");
  for (const width of [1440, 600, 450, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await copyCard.scrollIntoViewIfNeeded();
    expect(await copyCard.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    expect(await copyCard.evaluate(el => {
      const color = el.querySelector(".chat-color")!.getBoundingClientRect();
      const download = el.querySelector(".backup-action")!.getBoundingClientRect();
      return download.top >= color.bottom || download.left >= color.right;
    })).toBe(true);
    await copyCard.screenshot({ path: testInfo.outputPath(`duplicate-${width}.png`) });
  }
  const result = await page.evaluate(async modules => {
    const { conversationLibrary } = await import(modules.saved);
    const { restoreConversationSnapshot, captureConversationSnapshot } = await import(modules.workspace);
    const records = (await conversationLibrary.list()).conversations;
    const original = records.find((record: any) => record.name === "Moon notes");
    const copy = records.find((record: any) => record.name === "Moon notes (copy)");
    await restoreConversationSnapshot(copy.snapshot);
    const live = captureConversationSnapshot();
    await conversationLibrary.autosave(live, null, "base");
    return {
      original, copy, live,
      originalAfter: await conversationLibrary.get(original.id),
      foundId: (await conversationLibrary.findForTree(live.model_id, live.tree.root_id))?.id,
    };
  }, modules);
  expect(result.originalAfter).toEqual(result.original);
  expect(result.copy.snapshot.tree.root_id).not.toBe(result.original.snapshot.tree.root_id);
  expect(result.live.tree.root_id).toBe(result.copy.snapshot.tree.root_id);
  expect(result.foundId).toBe(result.copy.id);
  expect(result.copy.snapshot.tree.nodes.map((node: any) => node.tokens)).toEqual(result.original.snapshot.tree.nodes.map((node: any) => node.tokens));
});

test("a duplicate opens from saved chats and subsequent generation autosaves only the copy", async ({ page }) => {
  await generate(page);
  const original = await page.evaluate(async modules => {
    const { conversationLibrary, savedConversationState, flushConversationAutosave } = await import(modules.saved);
    await flushConversationAutosave();
    const record = await conversationLibrary.update(savedConversationState.activeId, { name: "Source chat" });
    (await import(modules.stores)).openDrawer("load_conversation");
    return record;
  }, modules);
  await page.getByRole("button", { name: "Duplicate Source chat", exact: true }).click();
  const card = page.locator(".conversation-card").filter({ hasText: "Source chat (copy)" });
  await expect(card).toBeVisible();
  await card.getByRole("button", { name: "Open", exact: true }).click();
  await expect(card).toHaveCount(0);
  const copied = await tree(page);
  expect(copied.root_id).not.toBe(original.snapshot.tree.root_id);
  await editor(page).fill((await editor(page).inputValue()) + " Another night: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect.poll(async () => (await tree(page)).active_node_id).not.toBe(copied.active_node_id);
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const saved = await page.evaluate(async ({ url, originalId }) => {
    const { conversationLibrary, savedConversationState, flushConversationAutosave } = await import(url);
    await flushConversationAutosave();
    return { original: await conversationLibrary.get(originalId), copy: await conversationLibrary.get(savedConversationState.activeId) };
  }, { url: modules.saved, originalId: original.id });
  expect(saved.original).toEqual(original);
  expect(saved.copy.name).toBe("Source chat (copy)");
  expect(saved.copy.snapshot.tree.nodes.length).toBeGreaterThan(copied.nodes.length);
});
