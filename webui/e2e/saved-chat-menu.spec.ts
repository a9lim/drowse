import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { resolve } from "node:path";

const modules = {
  home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`,
  saved: `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`,
};

async function mountHome(page: Page) {
  await page.goto("http://127.0.0.1:4176/outside-the-workbench");
  await page.evaluate(async modules => {
    const [{ default: Home }, { mount }, { conversationLibrary, registerConversationAutosave }] = await Promise.all([
      import(modules.home), import("/e2e/svelte-runtime.ts"), import(modules.saved),
    ]);
    const records = ["Chat 9", "A longer research conversation with several descriptive words"].map((name, index) => ({
      id: `chat-${index}`, schemaVersion: 1, name, avatarSeed: `avatar-${index}`, accent: "purple",
      modelId: "test-model", messageCount: 2, threadCount: 1, updatedAt: Date.now(), createdAt: Date.now(),
      snapshot: { tree: { root_id: "root", nodes: [{ id: "root", role: "system", text: "" }, { id: "user", role: "user", text: "Hello" }, { id: "assistant", role: "assistant", text: "Hi" }], children_of: { root: ["user"], user: ["assistant"] } } },
    }));
    const state = { records, duplicates: [] as string[], deletions: [] as string[], failDuplicate: false, failDelete: false, holdDuplicate: false, release: () => {} };
    (window as any).__chatMenu = state;
    conversationLibrary.listSummaries = async () => ({ conversations: records, issues: [] });
    conversationLibrary.duplicate = async (id: string) => {
      state.duplicates.push(id);
      if (state.holdDuplicate) await new Promise<void>(resolve => { state.release = resolve; });
      if (state.failDuplicate) throw new Error("Duplicate unavailable");
      const record = records.find(record => record.id === id)!;
      return { ...record, id: `${id}-copy`, name: `${record.name} (copy)` };
    };
    conversationLibrary.delete = async (id: string) => {
      state.deletions.push(id);
      if (state.failDelete) throw new Error("Delete unavailable");
      return true;
    };
    registerConversationAutosave(async () => {});
    document.body.replaceChildren();
    mount(Home, { target: document.body, props: {
      controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {} },
      snapshot: { phase: "supported", models: [{ id: "test-model", modelId: "test-model", name: "Gemma 3 1B", setupComplete: true, fit: "recommended" }],
        download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: true } },
      onChooseModels() {},
    } });
  }, modules);
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(2);
}

const firstCard = (page: Page) => page.locator('[data-saved-conversation="chat-0"]');
const trigger = (page: Page) => firstCard(page).getByRole("button", { name: "More options for Chat 9", exact: true });
const menu = (page: Page) => page.getByRole("menu", { name: "Options for Chat 9", exact: true });

test("chat options preserve the palette and fit phone, desktop, and enlarged text layouts", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mountHome(page);
  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await trigger(page).scrollIntoViewIfNeeded();
      await expect(firstCard(page).getByRole("button", { name: "Delete", exact: true })).toHaveCount(0);
      await expect(firstCard(page).getByRole("button", { name: "Duplicate Chat 9" })).toHaveCount(0);
      const layout = await firstCard(page).evaluate(card => {
        const row = card.querySelector(".chat-title-row")!.getBoundingClientRect();
        const time = card.querySelector("time")!.getBoundingClientRect();
        const button = card.querySelector(".chat-menu-trigger")!.getBoundingClientRect();
        return { overflow: card.scrollWidth > card.clientWidth, timeLeft: time.right <= button.left, aligned: Math.abs(row.right - button.right) < 1, target: Math.min(button.width, button.height) };
      });
      expect(layout.overflow).toBe(false);
      expect(layout.timeLeft).toBe(true);
      expect(layout.aligned).toBe(true);
      expect(layout.target).toBeGreaterThanOrEqual(width <= 760 || await page.evaluate(() => matchMedia("(pointer: coarse)").matches) ? 44 : 40);
      await trigger(page).click();
      await expect(menu(page).getByRole("menuitem")).toHaveText(["Duplicate", "Delete"]);
      expect(await menu(page).evaluate(el => {
        const outer = getComputedStyle(el);
        const item = getComputedStyle(el.querySelector("button")!);
        return parseFloat(outer.borderTopLeftRadius) - parseFloat(item.borderTopLeftRadius) - parseFloat(outer.paddingTop);
      })).toBe(0);
      await expect.poll(async () => {
        const bounds = (await menu(page).boundingBox())!;
        return bounds.x >= 7 && bounds.x + bounds.width <= width - 7 && bounds.y >= 7 && bounds.y + bounds.height <= 893;
      }).toBe(true);
      const bounds = (await menu(page).boundingBox())!;
      expect(bounds.x).toBeGreaterThanOrEqual(7);
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(width - 7);
      expect(bounds.y).toBeGreaterThanOrEqual(7);
      expect(bounds.y + bounds.height).toBeLessThanOrEqual(893);
      const dangerColor = await menu(page).evaluate(el => {
        const probe = document.createElement("span");
        probe.style.color = "var(--accent-red)";
        el.append(probe);
        const color = getComputedStyle(probe).color;
        probe.remove();
        return color;
      });
      await expect(menu(page).getByRole("menuitem", { name: "Delete", exact: true })).toHaveCSS("color", dangerColor);
      await page.screenshot({ path: testInfo.outputPath(`chat-menu-${theme}-${width}.png`) });
      await page.keyboard.press("Escape");
      await expect(trigger(page)).toBeFocused();
    }
  }
  await page.setViewportSize({ width: 390, height: 900 });
  await page.evaluate(() => document.documentElement.style.fontSize = "200%");
  await trigger(page).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect(menu(page).getByRole("menuitem", { name: "Delete", exact: true })).toBeVisible();
  expect((await new AxeBuilder({ page }).include(".chat-card").analyze()).violations).toEqual([]);
  expect(errors).toEqual([]);
});

test("options support keyboard, outside dismissal, one open menu, and viewport-edge placement", async ({ page, isMobile, browserName }) => {
  await mountHome(page);
  await trigger(page).focus();
  await page.keyboard.press("ArrowDown");
  await expect(menu(page).getByRole("menuitem", { name: "Duplicate Chat 9" })).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(menu(page).getByRole("menuitem", { name: "Delete", exact: true })).toBeFocused();
  await page.keyboard.press("Home");
  await expect(menu(page).getByRole("menuitem", { name: "Duplicate Chat 9" })).toBeFocused();
  await page.keyboard.press("End");
  await expect(menu(page).getByRole("menuitem", { name: "Delete", exact: true })).toBeFocused();
  await page.keyboard.press("d");
  await expect(menu(page).getByRole("menuitem", { name: "Duplicate Chat 9" })).toBeFocused();
  expect(await menu(page).evaluate(el => el.dispatchEvent(new KeyboardEvent("keydown", { key: "d", ctrlKey: true, bubbles: true, cancelable: true })))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(trigger(page)).toBeFocused();
  await page.keyboard.press("ArrowUp");
  await expect(menu(page).getByRole("menuitem", { name: "Delete", exact: true })).toBeFocused();
  await page.keyboard.press(browserName === "webkit" ? "Alt+Tab" : "Tab");
  await expect(trigger(page)).toHaveAttribute("aria-expanded", "false");
  await expect(firstCard(page).getByRole("button", { name: "Open chat", exact: true })).toBeFocused();
  if (isMobile) await trigger(page).tap();
  else await trigger(page).click();
  await page.getByRole("heading", { name: "Your chats", exact: true }).click();
  await expect(trigger(page)).toHaveAttribute("aria-expanded", "false");
  await trigger(page).click();
  const second = page.locator('[data-saved-conversation="chat-1"] .chat-menu-trigger');
  await second.click();
  await expect(page.getByRole("menu")).toHaveCount(1);
  await expect(trigger(page)).toHaveAttribute("aria-expanded", "false");
  await page.keyboard.press("Escape");
  await page.setViewportSize({ width: 390, height: 900 });
  await trigger(page).evaluate(button => {
    button.style.position = "fixed";
    button.style.top = "calc(100vh - 56px)";
    button.style.right = "8px";
    button.style.zIndex = "10";
  });
  await trigger(page).click();
  await expect(menu(page)).toHaveAttribute("data-origin", "bottom-right");
  await page.setViewportSize({ width: 320, height: 640 });
  await expect.poll(async () => {
    const box = (await menu(page).boundingBox())!;
    return box.x >= 7 && box.x + box.width <= 313 && box.y + box.height <= 633;
  }).toBe(true);
});

test("duplicate stays single-flight and delete requires a separate confirmation", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mountHome(page);
  await page.evaluate(() => (window as any).__chatMenu.holdDuplicate = true);
  await trigger(page).click();
  await menu(page).getByRole("menuitem", { name: "Duplicate Chat 9" }).click();
  await expect(trigger(page)).toBeDisabled();
  await expect(trigger(page)).toHaveAttribute("aria-busy", "true");
  await expect(page.getByRole("menu")).toHaveCount(0);
  expect(await page.evaluate(() => (window as any).__chatMenu.duplicates)).toEqual(["chat-0"]);
  await page.evaluate(() => (window as any).__chatMenu.release());
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(3);
  await expect(page.getByRole("status").filter({ hasText: "original chat is unchanged" })).toBeVisible();
  await expect(trigger(page)).toBeEnabled();
  await trigger(page).click();
  await menu(page).getByRole("menuitem", { name: "Delete", exact: true }).click();
  await expect(firstCard(page).getByRole("button", { name: "Cancel", exact: true })).toBeFocused();
  expect(await page.evaluate(() => (window as any).__chatMenu.deletions)).toEqual([]);
  await firstCard(page).getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(trigger(page)).toBeFocused();
  await trigger(page).click();
  await menu(page).getByRole("menuitem", { name: "Delete", exact: true }).click();
  await firstCard(page).getByRole("button", { name: "Delete chat", exact: true }).click();
  await expect(firstCard(page)).toHaveCount(0);
  expect(await page.evaluate(() => (window as any).__chatMenu.deletions)).toEqual(["chat-0"]);
  await expect(page.locator('[data-saved-conversation="chat-1"] .chat-menu-trigger')).toBeFocused();
});

test("failed duplicate and delete retain the chat and allow retry", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mountHome(page);
  await page.evaluate(() => (window as any).__chatMenu.failDuplicate = true);
  await trigger(page).click();
  await menu(page).getByRole("menuitem", { name: "Duplicate Chat 9" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Duplicate unavailable" })).toBeVisible();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(2);
  await expect(trigger(page)).toBeEnabled();
  await page.evaluate(() => (window as any).__chatMenu.failDelete = true);
  await trigger(page).click();
  await menu(page).getByRole("menuitem", { name: "Delete", exact: true }).click();
  await firstCard(page).getByRole("button", { name: "Delete chat", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Delete unavailable" })).toBeVisible();
  await expect(firstCard(page).getByRole("button", { name: "Delete chat", exact: true })).toBeEnabled();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(2);
  await firstCard(page).getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(trigger(page)).toBeFocused();
});
