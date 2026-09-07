import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { CHAT_ACCENTS } from "../src/lib/chatAccent";

const savedUrl = `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`;
const themeUrl = `/@fs${resolve("src/lib/theme.ts")}`;
const rgb = (hex: string) => `rgb(${[1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)).join(", ")})`;

test("chat accents are independent, persistent, accessible, and follow the active workspace", async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("http://127.0.0.1:4176/app?fixture=1");
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A chat with its own color.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator(".msg .response-body").last()).toContainText("deterministic");
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const ids = await page.evaluate(async url => {
    const { conversationLibrary, flushConversationAutosave, savedConversationState } = await import(url);
    await flushConversationAutosave();
    const first = await conversationLibrary.get(savedConversationState.activeId);
    await conversationLibrary.update(first.id, { name: "Field notes" });
    const second = await conversationLibrary.create({ name: "Another chat", snapshot: first.snapshot });
    return [first.id, second.id];
  }, savedUrl);
  await page.getByRole("button", { name: "Back to your chats", exact: true }).click();
  const card = page.locator(`[data-saved-conversation="${ids[0]}"]`);
  const other = page.locator(`[data-saved-conversation="${ids[1]}"]`);
  await expect(card).toBeVisible();
  await expect(card).toHaveAttribute("data-chat-accent", "purple");
  const timestamp = await card.locator("time").getAttribute("datetime");
  const order = await page.locator("[data-saved-conversation]").evaluateAll(els => els.map(el => el.getAttribute("data-saved-conversation")));
  await card.locator("summary").click();
  for (const theme of ["light", "dark"] as const) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: themeUrl, theme });
    for (const accent of CHAT_ACCENTS) {
      await card.locator(".color-option").filter({ hasText: accent.name }).click();
      await expect(card).toHaveAttribute("data-chat-accent", accent.id);
      await expect(card).toHaveCSS("background-image", /linear-gradient/);
      const material = await card.evaluate(el => getComputedStyle(el).backgroundImage);
      await card.locator(".chat-counts").hover();
      expect(await card.evaluate(el => getComputedStyle(el).backgroundImage)).toBe(material);
      await expect(card.getByRole("radio", { name: accent.name, exact: true })).toBeChecked();
      await expect(card.getByRole("button", { name: "Open chat", exact: true })).toHaveCSS("background-color", rgb(accent[theme]));
      await expect(other).toHaveAttribute("data-chat-accent", "purple");
      await expect(other.getByRole("button", { name: "Open chat", exact: true })).toHaveCSS("background-color", rgb(CHAT_ACCENTS[0][theme]));
    }
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 1000 });
      expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await card.screenshot({ path: testInfo.outputPath(`chat-color-${theme}-${width}.png`) });
    }
  }
  await page.evaluate(async url => {
    const { conversationLibrary } = await import(url);
    const update = conversationLibrary.update.bind(conversationLibrary);
    conversationLibrary.update = async () => {
      conversationLibrary.update = update;
      throw new Error("Storage is unavailable");
    };
  }, savedUrl);
  await card.locator(".color-option").filter({ hasText: "Mint" }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(card).toHaveAttribute("data-chat-accent", "periwinkle");
  await expect(card.getByRole("radio", { name: "Iris", exact: true })).toBeChecked();
  await card.getByRole("radio", { name: "Sky", exact: true }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(card).toHaveAttribute("data-chat-accent", "mint");
  await expect(card.locator("time")).toHaveAttribute("datetime", timestamp!);
  expect(await page.locator("[data-saved-conversation]").evaluateAll(els => els.map(el => el.getAttribute("data-saved-conversation")))).toEqual(order);

  await page.reload();
  await expect(card).toHaveAttribute("data-chat-accent", "mint");
  await card.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-chat-accent", "mint");
  await page.emulateMedia({ contrast: "more" });
  expect(await page.locator("html").evaluate(el => getComputedStyle(el).getPropertyValue("--focus-ring").trim())).toBe("#ffffff");
  await page.emulateMedia({ contrast: "no-preference" });
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Continue in mint.");
  await expect(page.getByRole("button", { name: "Send", exact: true })).toHaveCSS("background-color", rgb(CHAT_ACCENTS[2].dark));
  await page.getByRole("button", { name: "Loom", exact: true }).click();
  await expect(page.locator(".current-label").first()).toHaveCSS("color", rgb(CHAT_ACCENTS[2].dark));
  await page.getByRole("button", { name: "Controls", exact: true }).click();
  await page.getByRole("group", { name: "Controls section" }).getByRole("button", { name: "Chat", exact: true }).click();
  const controls = page.getByRole("tabpanel", { name: "Chat controls" });
  await controls.locator("summary").click();
  await expect(controls.getByRole("radio", { name: "Mint", exact: true })).toBeChecked();
  await controls.locator(".color-option").filter({ hasText: "Rose" }).click();
  await controls.getByRole("button", { name: "Update", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-chat-accent", "rose");
  await page.getByRole("button", { name: "Back to your chats", exact: true }).click();
  await expect(card).toHaveAttribute("data-chat-accent", "rose");
  await expect(page.locator("html")).not.toHaveAttribute("data-chat-accent");
  await other.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.locator("html")).not.toHaveAttribute("data-chat-accent");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Purple again.");
  await expect(page.getByRole("button", { name: "Send", exact: true })).toHaveCSS("background-color", rgb(CHAT_ACCENTS[0].dark));
  expect(errors).toEqual([]);
});
