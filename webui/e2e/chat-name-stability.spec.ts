import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const modules = {
  stores: `/@fs${resolve("src/lib/stores.svelte.ts")}`,
  saved: `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`,
  workspace: `/@fs${resolve("src/lib/conversationWorkspace.ts")}`,
  home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`,
};
async function seed(page: Page) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
  await expect(page.getByRole("textbox", { name: "Editable completion buffer" })).toBeVisible();
  return page.evaluate(async modules => {
    const { conversationLibrary } = await import(modules.saved);
    if (!(await conversationLibrary.hasAny())) {
      const snapshot = (await import(modules.workspace)).captureConversationSnapshot();
      for (const [name, timestamp] of [["Newest", 5_000], ["Middle", 4_000], ["Oldest", 4_000]] as const) {
        await conversationLibrary.create({ name, snapshot, createdAt: timestamp, updatedAt: timestamp });
      }
    }
    return (await conversationLibrary.listSummaries()).conversations;
  }, modules);
}
async function mountHome(page: Page) {
  await page.evaluate(async modules => {
    const [{ default: HostedHome }, { mount }, { registerConversationAutosave }, { sessionState }] = await Promise.all([
      import(modules.home), import("/@id/svelte"), import(modules.saved), import(modules.stores),
    ]);
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
}
const order = (page: Page) => page.locator("[data-saved-conversation]").evaluateAll(cards => cards.map(card => card.getAttribute("data-saved-conversation")));

test("chat names stay transparent and neutral, and renaming preserves order through reload", async ({ page }, testInfo) => {
  const records = await seed(page);
  await mountHome(page);
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  const before = await order(page);
  const target = records.at(-1);
  const card = page.locator(`[data-saved-conversation="${target.id}"]`);
  const timestamp = await card.locator("time").getAttribute("datetime");
  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    const name = card.locator(".chat-name");
    await name.hover();
    const styles = await page.locator(".chat-name").evaluateAll(names => names.map(name => {
      const style = getComputedStyle(name);
      return { appearance: style.appearance, background: style.backgroundColor, image: style.backgroundImage, shadow: style.boxShadow, color: style.color };
    }));
    for (const style of styles) expect(style).toEqual({ appearance: "none", background: "rgba(0, 0, 0, 0)", image: "none", shadow: "none", color: styles[0].color });
    await name.focus();
    await card.screenshot({ path: testInfo.outputPath(`chat-name-${theme}.png`) });
  }
  await card.locator(".chat-name").click();
  const input = card.getByRole("textbox", { name: "Chat name", exact: true });
  await input.fill("Alpha renamed");
  await input.press("Enter");
  await expect(card.locator(".chat-name")).toHaveText("Alpha renamed");
  expect(await order(page)).toEqual(before);
  await expect(card.locator("time")).toHaveAttribute("datetime", timestamp!);
  await seed(page);
  await mountHome(page);
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  expect(await order(page)).toEqual(before);
  await expect(card.locator(".chat-name")).toHaveText("Alpha renamed");
});

test("renaming in the saved-chat drawer preserves position and activity time", async ({ page }) => {
  const records = await seed(page);
  const open = () => page.evaluate(async url => (await import(url)).openDrawer("load_conversation"), modules.stores);
  await open();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  const before = await order(page);
  const target = records.at(-1);
  const card = page.locator(`[data-saved-conversation="${target.id}"]`);
  const timestamp = await card.locator("time").getAttribute("datetime");
  await card.getByRole("button", { name: "Rename", exact: true }).click();
  const input = card.getByRole("textbox", { name: "Conversation name", exact: true });
  await input.fill("Alpha renamed");
  await input.press("Enter");
  await expect(card.locator(".card-copy strong")).toHaveText("Alpha renamed");
  expect(await order(page)).toEqual(before);
  await expect(card.locator("time")).toHaveAttribute("datetime", timestamp!);
  await page.evaluate(async url => (await import(url)).closeDrawer(), modules.stores);
  await expect(card).toHaveCount(0);
  await open();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  expect(await order(page)).toEqual(before);
});
