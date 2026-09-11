import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;
const savedUrl = `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`;
const workspaceUrl = `/@fs${resolve("src/lib/conversationWorkspace.ts")}`;

async function conversation(page: Page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("What do marmots eat?");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  await expect(page.locator(".msg .model-avatar")).toBeVisible();
  await page.evaluate(async url => { await (await import(url)).flushConversationAutosave(); }, savedUrl);
}

test("each speaker has a distinct container and its saved avatar or live accent marker", async ({ page }, testInfo) => {
  const errors: string[] = []; page.on("pageerror", error => errors.push(error.message));
  await conversation(page);
  const model = page.locator(".msg").filter({ has: page.locator(".model-avatar") });
  const user = page.locator(".msg").filter({ has: page.locator(".user-avatar") });
  await expect(model).toHaveCount(1); await expect(user).toHaveCount(1);
  await expect(user.locator(".role-label")).toHaveText("user");
  await expect(model.locator(".role-label")).toHaveText("assistant");
  await expect(page.locator(".role-chip b")).toHaveCount(0);
  await expect(user.locator(".user-avatar")).toHaveText("");
  const identity = await page.evaluate(async url => {
    const { conversationLibrary, savedConversationState } = await import(url);
    const record = await conversationLibrary.get(savedConversationState.activeId);
    return { saved: record.avatarSeed, current: savedConversationState.avatarSeed };
  }, savedUrl);
  expect(identity.current).toBe(identity.saved);
  expect(identity.current).toBeTruthy();
  await expect(model.locator("img")).toHaveAttribute("src", /^data:image\/svg\+xml/);
  expect(await model.locator("img").evaluate(img => (img as HTMLImageElement).complete && (img as HTMLImageElement).naturalWidth > 0)).toBe(true);

  for (const theme of ["Light", "Dark"]) {
    await page.evaluate(async ({ url, theme }) => { (await import(url)).setTheme(theme); }, { url: `/@fs${resolve("src/lib/theme.ts")}`, theme: theme.toLowerCase() });
    for (const accent of ["rose", "sky", "purple"]) {
      await page.evaluate(async ({ url, accent }) => { (await import(url)).savedConversationState.accent = accent; }, { url: savedUrl, accent });
      await expect.poll(() => user.locator(".user-avatar").evaluate(el => {
        const probe = document.createElement("span"); probe.style.color = "var(--accent)"; el.append(probe);
        const matches = getComputedStyle(el).backgroundColor === getComputedStyle(probe).color; probe.remove(); return matches;
      })).toBe(true);
    }
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const message of [model, user]) {
        expect(await message.evaluate(el => getComputedStyle(el).backgroundColor)).not.toBe("rgba(0, 0, 0, 0)");
        expect(await message.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
        const avatar = await message.locator(".model-avatar, .speaker-marker").boundingBox();
        expect(avatar!.width).toBeGreaterThanOrEqual(40); expect(avatar!.height).toBeGreaterThanOrEqual(40);
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      expect((await new AxeBuilder({ page }).include(".chat .log").analyze()).violations).toEqual([]);
      await page.locator(".chat").screenshot({ path: testInfo.outputPath(`message-containers-${theme}-${width}.png`) });
    }
  }
  expect(errors).toEqual([]);
});

test("clicking or keyboard-activating the model picture opens the name and avatar editor without changing messages", async ({ page }) => {
  await conversation(page);
  const before = await page.locator(".response-body").allTextContents();
  for (const activate of ["click", "Enter", "Space"]) {
    const picture = page.getByRole("button", { name: "Edit name and avatar", exact: true });
    if (activate === "click") await picture.click();
    else {
      await page.keyboard.press("Tab");
      await picture.focus();
      await expect(picture).toHaveCSS("outline-style", "solid");
      await expect(picture).toHaveCSS("outline-width", "2px");
      await page.keyboard.press(activate);
    }
    const profile = page.getByRole("tabpanel", { name: "Chat controls", exact: true });
    await expect(profile).toBeVisible();
    await expect(profile.getByRole("textbox", { name: "Name", exact: true })).toBeEnabled();
    await expect(profile.getByRole("button", { name: "Generate another avatar", exact: true })).toBeVisible();
    const showSidebar = page.getByRole("button", { name: "Show left sidebar", exact: true });
    if (await showSidebar.isVisible()) await showSidebar.click();
    await page.getByRole("navigation", { name: "Workspace", exact: true }).getByRole("button", { name: /^(Conversation|Chat)(?: |$)/ }).click();
    expect(await page.locator(".response-body").allTextContents()).toEqual(before);
  }
});

test("updating a saved avatar and opening another chat keeps model pictures in sync", async ({ page }) => {
  await conversation(page);
  const original = await page.locator(".model-avatar img").getAttribute("src");
  await page.evaluate(async url => { (await import(url)).openDrawer("save_conversation"); }, storesUrl);
  const save = page.getByRole("dialog", { name: "Save chat", exact: true });
  await expect(save.getByRole("button", { name: "Update", exact: true })).toBeEnabled();
  await save.getByRole("button", { name: "Generate another avatar", exact: true }).click();
  await save.getByRole("button", { name: "Update", exact: true }).click();
  await expect(save).toHaveCount(0);
  await expect(page.locator(".model-avatar img")).not.toHaveAttribute("src", original!);
  const changed = await page.locator(".model-avatar img").getAttribute("src");
  const another = await page.evaluate(async ({ savedUrl, workspaceUrl }) => {
    const { conversationLibrary } = await import(savedUrl);
    const snapshot = (await import(workspaceUrl)).captureConversationSnapshot();
    return conversationLibrary.create({ name: "Another saved identity", snapshot, avatarSeed: "other-speaker-fixture" });
  }, { savedUrl, workspaceUrl });
  await page.evaluate(async url => { (await import(url)).openDrawer("load_conversation"); }, storesUrl);
  const library = page.getByRole("dialog", { name: "Saved chats", exact: true });
  await library.locator(`[data-saved-conversation="${another.id}"]`).getByRole("button", { name: "Open", exact: true }).click();
  await expect(library).toHaveCount(0);
  await expect.poll(() => page.evaluate(async url => (await import(url)).savedConversationState.avatarSeed, savedUrl)).toBe(another.avatarSeed);
  await expect(page.locator(".model-avatar img")).not.toHaveAttribute("src", changed!);
  await page.evaluate(async url => { (await import(url)).openDrawer("load_conversation"); }, storesUrl);
  await library.getByRole("button", { name: "Generate another avatar for Another saved identity", exact: true }).click();
  await expect.poll(() => page.evaluate(async url => (await import(url)).savedConversationState.avatarSeed, savedUrl)).not.toBe(another.avatarSeed);
  await page.keyboard.press("Escape");
  const reopenedAvatar = await page.locator(".model-avatar img").getAttribute("src");
  await page.evaluate(async ({ url, another }) => {
    (await import(url)).queueConversationOpen(another.id, another.modelId);
  }, { url: `/@fs${resolve("src/hosted/runtime/entryExperience.ts")}`, another });
  await page.reload();
  await expect(page.locator(".model-avatar img")).toHaveAttribute("src", reopenedAvatar!);
});

test("custom roles and comparison messages retain their labels without letter avatars", async ({ page }) => {
  await conversation(page);
  await page.evaluate(async url => {
    const stores = await import(url);
    stores.chatLog.turns[0].roleLabel = "researcher";
    stores.chatLog.turns[1].roleLabel = "marmot_expert";
    stores.chatLog.turns[1].abPair = { role: "assistant", roleLabel: "marmot_expert", text: "Comparison reply", generated: true };
    stores.autoRegenState.enabled = true;
  }, storesUrl);
  await expect(page.locator(".ab-grid")).toBeVisible();
  await expect(page.locator(".ab-primary .role-label")).toHaveText(["researcher", "marmot_expert"]);
  await expect(page.locator(".ab-shadow .model-avatar")).toHaveCount(1);
  await expect(page.locator(".role-chip b")).toHaveCount(0);
  const images = await page.locator(".model-avatar img").evaluateAll(imgs => imgs.map(img => img.getAttribute("src")));
  expect(new Set(images).size).toBe(1);
});
