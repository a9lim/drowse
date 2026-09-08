import { selectWorkspaceView } from "./workbench-navigation";
import { clickWorkspaceAction } from "./workbench-navigation";
import { openWorkspaceMenu } from "./workbench-navigation";
import { returnToChats, setAppearance, showWorkspaceTools, openTokenDetails, selectLoomView } from "./workbench-navigation";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const fixtureResponse = "This is a deterministic local Drowse runtime fixture.";
const toastModuleUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;
const fixtureModelHashes = [
  "f85e180e7e8d61d1e225062424360dcc7b1997ce4c28233188af886d8511156f",
  "89ff00a0903817d42e9fddb23babfbc8d2956d5e9b044729074062d1dc41bb93",
] as const;
const fixturePackHashes = [
  "fb79e067d57e7def0b65653f5f6329db5db98144c88d61c59ff84c77f1696770",
  "e6386fbf77c663ac05ac0b296e441476ad8aa8f2cdc0c905b75f84a41e0bdb0c",
] as const;
const fixtureHashes = [...fixtureModelHashes, ...fixturePackHashes] as const;

const sendButton = (page: Page) => page.getByRole("button", {
  name: /^(Send|Generate reply|Add message)$/,
});

interface FixtureTokenReadoutRequest {
  service: string;
  method: string;
  args: unknown[];
}

async function installFixtureRpcProbe(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & {
      __drowseFixtureRpc?: {
        requests: Array<{ service: string; method: string; args: unknown[] }>;
        failNextFamily: "lens" | "sae" | null;
      };
    };
    root.__drowseFixtureRpc = { requests: [], failNextFamily: null };
    const originalPostMessage = Worker.prototype.postMessage;
    Object.defineProperty(Worker.prototype, "postMessage", {
      configurable: true,
      value: function(this: Worker, message: unknown, transfer?: unknown): void {
        const request = message && typeof message === "object"
          ? message as {
              protocolVersion?: unknown;
              requestId?: unknown;
              command?: unknown;
              payload?: unknown;
            }
          : null;
        const payload = request?.payload && typeof request.payload === "object"
          ? request.payload as { service?: unknown; method?: unknown; args?: unknown }
          : null;
        if (
          request?.command === "request" && payload?.service === "instruments" &&
          payload.method === "tokenReadout" && Array.isArray(payload.args)
        ) {
          const state = root.__drowseFixtureRpc!;
          const recorded = structuredClone(payload) as {
            service: string;
            method: string;
            args: unknown[];
          };
          state.requests.push(recorded);
          const family = recorded.args[0];
          if (family === state.failNextFamily) {
            state.failNextFamily = null;
            queueMicrotask(() => {
              this.dispatchEvent(new MessageEvent("message", {
                data: {
                  protocolVersion: request.protocolVersion,
                  kind: "response",
                  requestId: request.requestId,
                  ok: false,
                  error: {
                    code: "FIXTURE_REPLAY_FAILED",
                    message: "Fixture token replay failed. Reopen the model and try again.",
                    recoverable: true,
                    status: 409,
                  },
                },
              }));
            });
            return;
          }
        }
        Reflect.apply(
          originalPostMessage,
          this,
          transfer === undefined ? [message] : [message, transfer],
        );
      },
    });
  });
}

async function installFixtureTreeDeleteFailure(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & {
      __drowseFailNextTreeDelete?: boolean;
    };
    root.__drowseFailNextTreeDelete = false;
    const originalPostMessage = Worker.prototype.postMessage;
    Object.defineProperty(Worker.prototype, "postMessage", {
      configurable: true,
      value: function(this: Worker, message: unknown, transfer?: unknown): void {
        const request = message && typeof message === "object"
          ? message as {
              protocolVersion?: unknown;
              requestId?: unknown;
              command?: unknown;
              payload?: unknown;
            }
          : null;
        const payload = request?.payload && typeof request.payload === "object"
          ? request.payload as { service?: unknown; method?: unknown }
          : null;
        if (
          root.__drowseFailNextTreeDelete && request?.command === "request" &&
          payload?.service === "tree" && payload.method === "delete"
        ) {
          root.__drowseFailNextTreeDelete = false;
          queueMicrotask(() => {
            this.dispatchEvent(new MessageEvent("message", {
              data: {
                protocolVersion: request.protocolVersion,
                kind: "response",
                requestId: request.requestId,
                ok: false,
                error: {
                  code: "MUTATION_DURING_GENERATION",
                  message: "cannot delete_subtree on a node inside an in-flight generation's reservation",
                  recoverable: true,
                  status: 409,
                },
              },
            }));
          });
          return;
        }
        Reflect.apply(
          originalPostMessage,
          this,
          transfer === undefined ? [message] : [message, transfer],
        );
      },
    });
  });
}

async function openModelAndStorage(page: Page): Promise<Locator> {
  await clickWorkspaceAction(page, "All tools");
  const search = page.getByRole("combobox", { name: "Filter commands" });
  await search.fill("model settings");
  await page.keyboard.press("Enter");
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  await modelControls.getByRole("button", { name: "Storage and downloads" }).click();
  return modelControls;
}

function downloadedPackRow(drawer: Locator, packId: string): Locator {
  const section = drawer.getByRole("heading", { name: "Downloaded additions" })
    .locator("..");
  return section.locator(`li[data-pack-id="${packId}"]`);
}

async function attemptPackDeletion(row: Locator): Promise<void> {
  await row.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(row.getByRole("button", { name: "Cancel", exact: true })).toBeVisible();
  await row.getByRole("button", { name: "Delete tool", exact: true }).click();
}

async function openWorkspace(page: Page, name: "Conversation" | "Branches" | "Controls") {
  const accessibleName = name === "Conversation"
    ? /^(Conversation|Chat)$/
    : name === "Controls"
      ? /^Controls$/
      : /^(Loom|Branches)$/;
  await selectWorkspaceView(page, accessibleName);
  if (name === "Branches") {
    if (!(await page.getByRole("button", { name: /^Map/ }).isVisible())) {
      await openWorkspaceMenu(page);
      await page.getByRole("button", { name: "Show Loom tools", exact: true }).click();
    }
    await selectLoomView(page, /^Map/);
  }
}

async function openTranscriptDrawer(page: Page): Promise<Locator> {
  await clickWorkspaceAction(page, "All tools");
  const search = page.getByRole("combobox", { name: "Filter commands" });
  await search.fill("conversation transcript");
  await page.keyboard.press("Enter");
  const drawer = page.getByRole("dialog", { name: "Conversation transcript" });
  await expect(drawer).toBeVisible();
  return drawer;
}

async function openFixtureWorkbench(page: Page, extraQuery = ""): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1${extraQuery}`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Qwen3 1.7B fixture source" }),
  ).toHaveAttribute(
    "href",
    "https://huggingface.co/Qwen/Qwen3-1.7B",
  );
  await expect(page.getByText("Apache-2.0", { exact: true })).toBeVisible();

  const toolPicker = page.locator(".tool-picker");
  await expect(toolPicker.getByText("Response controls", { exact: true })).toBeVisible();
  await expect(toolPicker.locator(".required-tool", { hasText: "Word insights" }).last()).toBeVisible();
  await expect(toolPicker.getByRole("checkbox", { name: /Word insights/ })).toHaveCount(0);
  await expect(toolPicker.getByRole("checkbox", { name: /R-lens readouts/ })).not.toBeChecked();
  await expect(toolPicker.getByRole("checkbox", { name: /Feature explorer/ })).toBeChecked();
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page).toHaveTitle("Drowse");
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeVisible();
}

async function openInstalledFixtureWorkbench(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1&reopen=1`);
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page).toHaveTitle("Drowse");
  await expect(page.getByRole("heading", { name: "This device is ready" })).toHaveCount(0);
}

async function continueFromChatHome(page: Page): Promise<void> {
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
}

async function savedConversationAvatarSeed(page: Page, name: string): Promise<string | null> {
  return page.evaluate((conversationName) => new Promise<string | null>((resolveSeed, rejectSeed) => {
    const open = indexedDB.open("drowse-saved-conversations", 1);
    open.onerror = () => rejectSeed(open.error);
    open.onsuccess = () => {
      const database = open.result;
      const request = database.transaction("conversations", "readonly")
        .objectStore("conversations")
        .getAll();
      request.onerror = () => {
        database.close();
        rejectSeed(request.error);
      };
      request.onsuccess = () => {
        database.close();
        const record = (request.result ?? []).find(
          (candidate: { name?: string }) => candidate.name === conversationName,
        ) as { avatarSeed?: string } | undefined;
        resolveSeed(record?.avatarSeed ?? null);
      };
    };
  }), name);
}

test("chats autosave with Blobatar identities and safe Loom clearing and cuts", async ({ page }, testInfo) => {
  await openFixtureWorkbench(page);
  const libraryUrl = `/@fs/${resolve("src/lib/stores/savedConversations.svelte.ts")}`;
  const records = () => page.evaluate(async url => (await import(url)).conversationLibrary.list(), libraryUrl);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Automatic marmot notes");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  await expect.poll(async () => (await records()).conversations.length).toBe(1);
  const original = (await records()).conversations[0];
  expect(original.name).toMatch(/^New Chat - [A-Z][a-z]{2} \d{1,2} - \d{1,2}:\d{2} \S+$/);
  expect(original.avatarSeed).toBeTruthy();
  for (const width of [320, 390, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.locator(".app-header").evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await returnToChats(page);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  const card = page.locator(`[data-saved-conversation="${original.id}"]`);
  const avatar = card.getByRole("button", { name: /Generate another avatar/ });
  await expect.poll(() => avatar.locator("img").evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true);
  const oldImage = await avatar.locator("img").getAttribute("src");
  await avatar.click();
  await expect(avatar.locator("img")).not.toHaveAttribute("src", oldImage!);
  const chosenSeed = (await records()).conversations[0].avatarSeed;
  await card.getByRole("button", { name: /^Rename / }).click();
  await card.getByRole("textbox", { name: "Chat name", exact: true }).fill("My automatic chat");
  await card.getByRole("button", { name: "Save name", exact: true }).click();
  await card.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(page.locator("#workspace-main")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator(".chat[aria-label='Chat']").getByText("Automatic marmot notes", { exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Add a second turn");
  await sendButton(page).click();
  await expect(page.locator(".chat[aria-label='Chat']").getByText("Add a second turn", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  await expect.poll(async () => (await records()).conversations[0].snapshot.tree.nodes.length).toBeGreaterThan(original.snapshot.tree.nodes.length);
  const updated = (await records()).conversations[0];
  expect(updated.id).toBe(original.id);
  expect(updated.name).toBe("My automatic chat");
  expect(updated.avatarSeed).toBe(chosenSeed);
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await page.getByRole("button", { name: "Cut branch…", exact: true }).click();
  const cutDialog = page.getByRole("dialog", { name: "Delete branch", exact: true });
  await expect(cutDialog).toContainText("Earlier turns and sibling branches stay");
  await cutDialog.getByRole("button", { name: "cancel", exact: true }).click();
  expect((await records()).conversations[0].snapshot.tree.nodes.length).toBe(updated.snapshot.tree.nodes.length);
  await page.getByRole("button", { name: "Cut branch…", exact: true }).click();
  await cutDialog.getByRole("button", { name: "Delete branch", exact: true }).click();
  await expect(cutDialog).not.toBeVisible();
  await expect.poll(async () => (await records()).conversations[0].snapshot.tree.nodes.length).toBe(updated.snapshot.tree.nodes.length - 1);
  await page.getByRole("button", { name: "Clear loom…", exact: true }).click();
  const clearDialog = page.getByRole("dialog", { name: "Clear loom", exact: true });
  await clearDialog.getByRole("button", { name: "cancel", exact: true }).click();
  await expect(page.getByRole("button", { name: "Clear loom…", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Clear loom…", exact: true }).click();
  await clearDialog.getByRole("button", { name: "Save and clear loom", exact: true }).click();
  await expect(clearDialog).not.toBeVisible();
  await expect(page.getByRole("button", { name: "Clear loom…", exact: true })).toBeDisabled();
  expect((await records()).conversations).toHaveLength(1);
  await selectWorkspaceView(page, "Conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A separate chat");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  await expect.poll(async () => (await records()).conversations.length).toBe(2);
  const newChat = (await records()).conversations.find((record: any) => record.id !== original.id);
  expect(newChat.avatarSeed).not.toBe(chosenSeed);
  await returnToChats(page);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await page.reload();
  await expect(page.locator("[data-saved-conversation]")).toHaveCount(2);
  expect((await records()).conversations.find((record: any) => record.id === original.id).avatarSeed).toBe(chosenSeed);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  for (const image of await page.locator(".avatar img").all()) {
    expect(await image.evaluate(element => {
      const img = element.getBoundingClientRect();
      const button = element.parentElement!.getBoundingClientRect();
      return img.width <= button.width && img.height <= button.height && getComputedStyle(element).borderRadius === "50%";
    })).toBe(true);
  }
  await page.locator(".chat-list").screenshot({ path: testInfo.outputPath("autosaved-chat-avatars-phone.png") });
});

test("autosave failures are visible and retry preserves the same chat", async ({ page }) => {
  await openFixtureWorkbench(page);
  const libraryUrl = `/@fs/${resolve("src/lib/stores/savedConversations.svelte.ts")}`;
  const records = () => page.evaluate(async url => {
    const { conversationLibrary } = await import(url);
    const { conversations } = await conversationLibrary.list();
    return Promise.all(conversations.map((record: { id: string }) => conversationLibrary.get(record.id)));
  }, libraryUrl);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep this safe");
  await sendButton(page).click();
  await expect(page.getByRole("status").filter({ hasText: "Response complete." })).toBeVisible();
  await expect.poll(async () => (await records())[0]?.snapshot.tree.nodes.some((node: { role: string; text: string }) => node.role === "assistant" && node.text === fixtureResponse)).toBe(true);
  await expect(page.locator(".app-header")).not.toContainText(/Autosaved|Saving chat|Save chat/);
  const original = await page.evaluate(async url => {
    const { conversationLibrary } = await import(url);
    const record = await conversationLibrary.get((await conversationLibrary.list()).conversations[0].id);
    const originalPut = IDBObjectStore.prototype.put;
    (window as any).__restoreAutosave = () => { IDBObjectStore.prototype.put = originalPut; };
    IDBObjectStore.prototype.put = function(...args) {
      if (this.name === "conversations") throw new DOMException("Storage is full", "QuotaExceededError");
      return originalPut.apply(this, args);
    };
    return record;
  }, libraryUrl);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Save this next reply too");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: "Retry autosave", exact: true })).toBeVisible();
  const beforeRetry = await records();
  expect(beforeRetry).toHaveLength(1);
  expect(beforeRetry[0].snapshot.tree.nodes.length).toBe(original.snapshot.tree.nodes.length);
  await page.evaluate(() => (window as any).__restoreAutosave());
  await page.getByRole("button", { name: "Retry autosave", exact: true }).click();
  await expect(page.getByRole("button", { name: "Retry autosave", exact: true })).toHaveCount(0);
  await expect.poll(async () => (await records())[0].snapshot.tree.nodes.length).toBeGreaterThan(original.snapshot.tree.nodes.length);
  const recovered = await records();
  expect(recovered).toHaveLength(1);
  expect(recovered[0].id).toBe(original.id);
  expect(recovered[0].avatarSeed).toBe(original.avatarSeed);
  expect(recovered[0].snapshot.tree.nodes.length).toBeGreaterThan(original.snapshot.tree.nodes.length);
});

test("saved chats persist with stable Blobatar identities and explicit deletion", async ({ page }) => {
  await openFixtureWorkbench(page);
  const savedChatsButton = page.locator(".loom-sidebar").getByRole("button", { name: "Open", exact: true });
  await expect(page.getByRole("button", { name: "Back to Chats", exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("What do marmots do in winter?");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await savedChatsButton.click();
  let libraryDrawer = page.getByRole("dialog", { name: "Saved chats" });
  await expect(libraryDrawer).toBeVisible();
  await libraryDrawer.getByRole("button", { name: "Save current chat", exact: true }).click();
  const saveDrawer = page.getByRole("dialog", { name: "Save chat" });
  await expect(saveDrawer).toBeVisible();
  await expect(saveDrawer.getByText("Click for a new avatar", { exact: true })).toBeVisible();
  const saveAvatar = saveDrawer.getByRole("button", { name: "Generate another avatar" });
  const initialAvatar = await saveAvatar.locator("img").getAttribute("src");
  await saveAvatar.click();
  await expect(saveAvatar.locator("img")).not.toHaveAttribute("src", initialAvatar ?? "");
  await saveDrawer.getByRole("textbox", { name: "Name" }).fill("Marmot field notes");
  await saveDrawer.getByRole("button", { name: /^(Save|Update)$/ }).click();
  await expect(saveDrawer).not.toBeVisible();
  const chosenAvatarSeed = await savedConversationAvatarSeed(page, "Marmot field notes");
  expect(chosenAvatarSeed).not.toBeNull();

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  const savedHomeCard = page.locator("[data-saved-conversation]").filter({
    hasText: "Marmot field notes",
  });
  await expect(savedHomeCard).toContainText("Marmot field notes");
  await savedHomeCard.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.locator(".chat[aria-label='Chat']").getByText(
    "What do marmots do in winter?",
    { exact: true },
  )).toBeVisible();
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await savedChatsButton.click();
  libraryDrawer = page.getByRole("dialog", { name: "Saved chats" });
  const list = libraryDrawer.getByRole("list", { name: "Saved conversations" });
  let card = list.locator("[data-saved-conversation]").first();
  await expect(card).toBeVisible();
  expect(await savedConversationAvatarSeed(page, "Marmot field notes")).toBe(chosenAvatarSeed);

  const libraryAvatar = card.getByRole("button", { name: /Generate another avatar/ });
  await libraryAvatar.click();
  await expect.poll(() => savedConversationAvatarSeed(page, "Marmot field notes"))
    .not.toBe(chosenAvatarSeed);
  const regeneratedAvatarSeed = await savedConversationAvatarSeed(page, "Marmot field notes");
  await card.getByRole("button", { name: "Rename" }).click();
  await card.getByRole("textbox", { name: "Conversation name" }).fill("Hibernation notes");
  await card.getByRole("button", { name: "Save name" }).click();
  await expect(card.getByText("Hibernation notes", { exact: true })).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(libraryDrawer).not.toBeVisible();
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await savedChatsButton.click();
  libraryDrawer = page.getByRole("dialog", { name: "Saved chats" });
  card = libraryDrawer.locator("[data-saved-conversation]").first();
  expect(await savedConversationAvatarSeed(page, "Hibernation notes")).toBe(regeneratedAvatarSeed);
  await card.getByRole("button", { name: "Open", exact: true }).click();
  await expect(libraryDrawer).not.toBeVisible();

  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await savedChatsButton.click();
  libraryDrawer = page.getByRole("dialog", { name: "Saved chats" });
  card = libraryDrawer.locator("[data-saved-conversation]").first();
  await expect(card.getByText("Open now", { exact: true })).toBeVisible();
  await card.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(card.getByText(/This cannot be undone/)).toBeVisible();
  await card.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(card.getByText("Hibernation notes", { exact: true })).toBeVisible();
  await card.getByRole("button", { name: "Delete", exact: true }).click();
  await card.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(card).toHaveCount(0);

  await page.keyboard.press("Escape");
  await page.setViewportSize({ width: 390, height: 844 });
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  const compactSavedChatsButton = page.locator(".loom-sidebar").getByRole("button", { name: "Open", exact: true });
  await expect(compactSavedChatsButton).toBeVisible();
  await compactSavedChatsButton.click();
  await expect(page.getByRole("dialog", { name: "Saved chats" })).toBeVisible();
});

test("saved chat cards keep summaries and open or export the latest stored transcript", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Original saved prompt");
  await sendButton(page).click();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (url) => (await import(url)).genStatus.active, toastModuleUrl))
    .toBe(false);
  const recordId = await page.evaluate(async ({ storesUrl, workspaceUrl }) => {
    const { conversationLibrary } = await import(storesUrl);
    const { captureConversationSnapshot } = await import(workspaceUrl);
    const snapshot = structuredClone(captureConversationSnapshot());
    snapshot.tree.root_id = `saved-${snapshot.tree.root_id}`;
    snapshot.tree.active_node_id = `saved-${snapshot.tree.active_node_id}`;
    snapshot.tree.nodes = snapshot.tree.nodes.map((node: any) => ({ ...node, id: `saved-${node.id}`, parent_id: node.parent_id === null ? null : `saved-${node.parent_id}` }));
    snapshot.tree.children_of = Object.fromEntries(Object.entries(snapshot.tree.children_of).map(([id, children]) => [`saved-${id}`, (children as string[]).map(child => `saved-${child}`)]));
    const record = await conversationLibrary.create({ name: "Fresh transcript", snapshot });
    const originalGetAll = IDBObjectStore.prototype.getAll;
    IDBObjectStore.prototype.getAll = function(...args) {
      if (this.name === "conversations") throw new Error("Saved chat browsing must not bulk-load transcripts");
      return Reflect.apply(originalGetAll, this, args);
    };
    try {
      const summaries = await conversationLibrary.listSummaries();
      if (summaries.issues.length || !summaries.conversations.some((summary: any) => summary.id === record.id) ||
        summaries.conversations.some((summary: any) => "snapshot" in summary) || !await conversationLibrary.hasAny()) {
        throw new Error("Invalid saved chat summary");
      }
    } finally {
      IDBObjectStore.prototype.getAll = originalGetAll;
    }
    return record.id as string;
  }, { storesUrl: toastModuleUrl, workspaceUrl: `/@fs/${resolve("src/lib/conversationWorkspace.ts")}` });

  const changeStoredPrompt = async (text: string) => {
    await page.evaluate(async ({ storesUrl, id, text }) => {
      const { conversationLibrary } = await import(storesUrl);
      const record = await conversationLibrary.get(id);
      record.snapshot.tree.nodes.find((node: { role: string }) => node.role === "user").text = text;
      await conversationLibrary.update(id, { snapshot: record.snapshot });
    }, { storesUrl: toastModuleUrl, id: recordId, text });
  };
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await page.locator(".loom-sidebar").getByRole("button", { name: "Open", exact: true }).click();
  const drawer = page.getByRole("dialog", { name: "Saved chats" });
  const card = drawer.locator(`[data-saved-conversation="${recordId}"]`);
  await expect(card).toBeVisible();
  await changeStoredPrompt("Updated after opening the library");
  const downloading = page.waitForEvent("download");
  await card.getByRole("button", { name: "Download", exact: true }).click();
  const download = await downloading;
  const exported = JSON.parse(await readFile((await download.path())!, "utf8"));
  expect(exported.conversation.snapshot.tree.nodes.find((node: { role: string }) => node.role === "user").text)
    .toBe("Updated after opening the library");

  await changeStoredPrompt("Latest prompt when opening the chat");
  await card.getByRole("button", { name: "Open", exact: true }).click();
  await expect(drawer).not.toBeVisible();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("Latest prompt when opening the chat");
});

test("Loom defers hidden edge work and preserves labels through navigation and bookmarks", async ({ page }) => {
  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & { __edgeRequests: number };
    root.__edgeRequests = 0;
    const originalPostMessage = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message, ...rest) {
      if (message?.command === "request" && message.payload?.service === "tree" &&
        message.payload.method === "edgeLabel") root.__edgeRequests += 1;
      return Reflect.apply(originalPostMessage, this, [message, ...rest]);
    };
  });
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A map performance check");
  await sendButton(page).click();
  await expect(page.locator(".chat[aria-label='Chat']")).toContainText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (url) => (await import(url)).genStatus.active, toastModuleUrl))
    .toBe(false);
  const requestCount = () => page.evaluate(() =>
    (globalThis as typeof globalThis & { __edgeRequests: number }).__edgeRequests);
  expect(await requestCount()).toBe(0);
  await openWorkspace(page, "Branches");
  await expect.poll(requestCount).toBeGreaterThan(0);
  const firstCount = await requestCount();
  await expect.poll(() => page.evaluate(async (url) => (await import(url)).edgeLabelCache.size, toastModuleUrl))
    .toBe(firstCount);
  await page.evaluate(async (url) => {
    const { loomTree, loomNavigate, loomStar } = await import(url);
    const id = loomTree.active_node_id;
    await loomStar(id, true);
    await loomNavigate(id);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }, toastModuleUrl);
  expect(await requestCount()).toBe(firstCount);
});

async function fixtureInstallIds(page: Page): Promise<string[]> {
  return page.evaluate(() => new Promise<string[]>((resolveInstalls, rejectInstalls) => {
    const open = indexedDB.open("drowse-hosted-runtime", 5);
    open.onerror = () => rejectInstalls(open.error);
    open.onsuccess = () => {
      const database = open.result;
      const request = database.transaction("installs", "readonly")
        .objectStore("installs")
        .getAll();
      request.onerror = () => {
        database.close();
        rejectInstalls(request.error);
      };
      request.onsuccess = () => {
        database.close();
        resolveInstalls(
          (request.result ?? []).map((install: { id: string }) => install.id).sort(),
        );
      };
    };
  }));
}

async function fixtureStorageSnapshot(page: Page) {
  return page.evaluate(async (hashes) => {
    const installs = await new Promise<Array<{
      id: string;
      kind: string;
      objectHashes: string[];
    }>>((resolveInstalls, rejectInstalls) => {
      const open = indexedDB.open("drowse-hosted-runtime", 5);
      open.onerror = () => rejectInstalls(open.error);
      open.onsuccess = () => {
        const database = open.result;
        const request = database.transaction("installs", "readonly")
          .objectStore("installs")
          .getAll();
        request.onerror = () => {
          database.close();
          rejectInstalls(request.error);
        };
        request.onsuccess = () => {
          database.close();
          resolveInstalls(request.result ?? []);
        };
      };
    });
    const storage = navigator.storage as StorageManager & {
      getDirectory(): Promise<FileSystemDirectoryHandle>;
    };
    const root = await storage.getDirectory();
    const objects = await (await root.getDirectoryHandle("drowse"))
      .getDirectoryHandle("objects");
    const sizes: Record<string, number> = {};
    for (const hash of hashes) {
      const file = await (await objects.getFileHandle(`${hash}.data`)).getFile();
      sizes[hash] = file.size;
    }
    const names: string[] = [];
    const iterable = objects as FileSystemDirectoryHandle & {
      entries(): AsyncIterable<[string, FileSystemHandle]>;
    };
    for await (const [name, handle] of iterable.entries()) {
      if (handle.kind === "file") names.push(name);
    }
    names.sort();
    return {
      install: installs.find((candidate) => candidate.id === "qwen3-1.7b-fixture") ?? null,
      installs,
      sizes,
      names,
    };
  }, fixtureHashes);
}

async function fixtureDeletionSnapshot(page: Page) {
  return page.evaluate(async (hashes) => {
    const read = <T>(databaseName: string, version: number, storeName: string, key: string) =>
      new Promise<T | null>((resolveRecord, rejectRecord) => {
        const open = indexedDB.open(databaseName, version);
        open.onerror = () => rejectRecord(open.error);
        open.onsuccess = () => {
          const database = open.result;
          const request = database.transaction(storeName, "readonly")
            .objectStore(storeName)
            .get(key);
          request.onerror = () => {
            database.close();
            rejectRecord(request.error);
          };
          request.onsuccess = () => {
            database.close();
            resolveRecord(request.result ?? null);
          };
        };
      });
    const readAll = <T>(databaseName: string, version: number, storeName: string) =>
      new Promise<T[]>((resolveRecords, rejectRecords) => {
        const open = indexedDB.open(databaseName, version);
        open.onerror = () => rejectRecords(open.error);
        open.onsuccess = () => {
          const database = open.result;
          const request = database.transaction(storeName, "readonly")
            .objectStore(storeName)
            .getAll();
          request.onerror = () => {
            database.close();
            rejectRecords(request.error);
          };
          request.onsuccess = () => {
            database.close();
            resolveRecords(request.result ?? []);
          };
        };
      });
    const [install, session, installs] = await Promise.all([
      read("drowse-hosted-runtime", 5, "installs", "qwen3-1.7b-fixture"),
      read("drowse-hosted-sessions", 2, "sessions", "qwen3-1.7b-fixture"),
      readAll<{ id: string; kind: string }>("drowse-hosted-runtime", 5, "installs"),
    ]);
    const storage = navigator.storage as StorageManager & {
      getDirectory(): Promise<FileSystemDirectoryHandle>;
    };
    const root = await storage.getDirectory();
    const objects = await (await root.getDirectoryHandle("drowse"))
      .getDirectoryHandle("objects");
    const remaining: string[] = [];
    for (const hash of hashes) {
      try {
        await objects.getFileHandle(`${hash}.data`);
        remaining.push(hash);
      } catch (error) {
        if (!(error instanceof DOMException) || error.name !== "NotFoundError") throw error;
      }
    }
    return {
      install,
      session,
      packIds: installs.filter((candidate) => candidate.kind === "pack")
        .map((candidate) => candidate.id)
        .sort(),
      remaining,
    };
  }, fixtureHashes);
}

test("landing page routes into the hosted app", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto("/");

  await expect(page).toHaveTitle("Drowse");
  await expect(
    page.getByRole("heading", { name: /See inside your model/ }),
  ).toBeVisible();
  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toHaveAttribute(
    "href",
    "/app",
  );
  await expect(page.getByText("Open source. Inference and saved work stay on your device.", { exact: true })).toBeVisible();
  await expect(page.locator(".hero-action-row > span")).toHaveText("Chrome or Safari · compatible device required");
  await expect(page.getByRole("button", { name: "Pause background animation" })).toHaveCount(0);
  await page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" }).click();
  await expect(page).toHaveURL(/\/app$/);
  expect(pageErrors).toEqual([]);
});

test("model storage notice retries protection without repeating instructions or reinstalling", async ({ page }, testInfo) => {
  await page.addInitScript(() => {
    let calls = 0;
    Object.defineProperty(navigator.storage, "persist", { configurable: true, value: async () => ++calls >= 2 });
    Object.defineProperty(navigator.storage, "persisted", { configurable: true, value: async () => false });
  });
  await page.route("**/fixtureBrowser.worker.ts*", async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: (await response.text()).replace(/persisted:\s*true/g, "persisted: false") });
  });
  await openFixtureWorkbench(page);
  await page.goto(`${devUrl}/app?fixture=1&choose=1`);
  const notice = page.getByRole("status", { name: "Storage protection", exact: true });
  await expect(notice).toHaveCount(1);
  await notice.getByRole("button", { name: "Protect storage", exact: true }).click();
  await expect(notice.getByRole("button", { name: "Protect storage", exact: true })).toBeEnabled();
  await expect(notice).toContainText("Downloads and chats still save on this device.");
  await expect(notice.locator("p")).toHaveCount(1);
  await expect(notice).not.toContainText("Select Protect storage");
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await notice.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  }
  await page.screenshot({ path: testInfo.outputPath("storage-protection-notice.png") });
  await notice.getByRole("button", { name: "Protect storage", exact: true }).click();
  await expect(notice).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open Drowse", exact: true })).toBeEnabled();
});

test("chat storage protection stays compact above saved chats on desktop and phones", async ({ page }, testInfo) => {
  await page.route("**/fixtureBrowser.worker.ts*", async (route) => {
    const response = await route.fetch();
    await route.fulfill({ response, body: (await response.text()).replace(/persisted:\s*true/g, "persisted: false") });
  });
  await openFixtureWorkbench(page);
  await page.goto(`${devUrl}/app?fixture=1`);
  const banner = page.getByRole("status", { name: "Keep your chats and models" });
  await expect(page.getByText("Local workbench", { exact: true })).toHaveCount(0);
  await expect(banner).toBeVisible();
  const guide = banner.locator("details");
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const theme of ["Light", "Dark"]) {
    await setAppearance(page, theme);
    await expect(banner).toHaveCSS("background-color", theme === "Light" ? "rgba(17, 22, 36, 0.03)" : "rgba(232, 234, 238, 0.03)");
    await expect(banner).toHaveCSS("background-image", "none");
    await expect(banner.getByRole("button", { name: "Protect storage", exact: true })).toHaveCSS("background-color", "rgb(245, 215, 110)");
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await banner.screenshot({ path: testInfo.outputPath(`storage-neutral-${theme}-${width}.png`) });
    }
  }
  await expect(guide).not.toHaveAttribute("open");
  await banner.getByText("How to keep your data", { exact: true }).click();
  await expect(guide).toHaveAttribute("open", "");
  await expect(guide).toContainText("Your browser may decline the request");
  await expect(guide).toContainText("Download beside a saved chat");
  await expect(guide).toContainText("Your existing chats and models won't transfer to it");
  expect((await guide.innerText()).split(/\s+/).length).toBeLessThan(120);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(() => banner.evaluate(element =>
      element.getBoundingClientRect().bottom < document.querySelector("#saved-chats")!.getBoundingClientRect().top,
    )).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    expect(await guide.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  }
  await banner.screenshot({ path: testInfo.outputPath("storage-protection-guide-phone.png") });
  await banner.getByText("How to keep your data", { exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(guide).not.toHaveAttribute("open");
  await page.evaluate(() => {
    Object.defineProperty(navigator.storage, "persist", { configurable: true, value: async () => false });
    Object.defineProperty(navigator.storage, "persisted", { configurable: true, value: async () => false });
  });
  await banner.getByRole("button", { name: "Protect storage", exact: true }).click();
  await expect(banner.getByRole("button", { name: "Protect storage", exact: true })).toBeEnabled();
  await expect(guide).toHaveAttribute("open", "");
  expect(await banner.evaluate((element) => Boolean(element.compareDocumentPosition(document.querySelector("#saved-chats")!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBe(true);
});

test("a returning user sees their chat home instead of onboarding", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.goto(`${devUrl}/`);

  const open = page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse", exact: true });
  await expect(open).toBeVisible();
  await expect(open).toHaveAttribute("href", "/app");
  await page.goto(`${devUrl}/app?fixture=1`);

  const logo = page.locator(".page-route:not([inert]) .page-brand");
  await expect(logo).toHaveAttribute("href", "/");
  await logo.click();
  await expect(page).toHaveURL(`${devUrl}/`);
  await expect(page.locator("#hero-title")).toBeVisible();
  await page.goto(`${devUrl}/app?fixture=1`);

  await continueFromChatHome(page);
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page).toHaveTitle("Drowse");
  await expect(page.getByRole("heading", { name: "This device is ready" })).toHaveCount(0);
  await expect(logo).toHaveAttribute("href", "/");
  await logo.click();
  await expect(page).toHaveURL(`${devUrl}/`);
  await expect(page.locator("#hero-title")).toBeVisible();
});

test("a stale workbench module reloads once and preserves the installed model", async ({ page }) => {
  let failedImports = 0;
  await page.route("**/src/App.svelte*", async (route) => {
    if (failedImports === 0) {
      failedImports += 1;
      await route.abort("failed");
      return;
    }
    await route.continue();
  });

  await openFixtureWorkbench(page);

  expect(failedImports).toBe(1);
  await expect(page).toHaveURL(`${devUrl}/app?fixture=1&reopen=1`);
  await expect(page.locator(".shell")).toBeVisible();
  expect(await fixtureInstallIds(page)).toContain("qwen3-1.7b-fixture");
});

test("a stale hosted bootstrap module reloads once instead of leaving a broken page", async ({ page }) => {
  let failedImports = 0;
  await page.route("**/src/hosted/runtime/workerFixtureRuntime.ts*", async (route) => {
    if (failedImports === 0) {
      failedImports += 1;
      await route.abort("failed");
      return;
    }
    await route.continue();
  });

  await page.goto(`${devUrl}/app?fixture=1`);

  expect(failedImports).toBe(1);
  await expect(page).toHaveURL(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
});

test("persistent missing workbench assets stop after one automatic reload and allow manual recovery", async ({ page }) => {
  let failedImports = 0;
  let missing = true;
  await page.route("**/src/App.svelte*", async (route) => {
    if (!missing) return route.continue();
    failedImports += 1;
    await route.abort("failed");
  });

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await page.getByRole("button", { name: "Download and open", exact: true }).click();

  await expect(page.getByText(
    "Some app files could not load. This can happen after an update or a connection problem. Reload Drowse to refresh its app files.",
  )).toBeVisible();
  await expect(page.getByRole("button", { name: "Reload Drowse" })).toBeVisible();
  expect(failedImports).toBe(2);
  await page.waitForTimeout(750);
  expect(failedImports).toBe(2);

  missing = false;
  await page.getByRole("button", { name: "Reload Drowse", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  expect(failedImports).toBe(2);
  expect(await fixtureInstallIds(page)).toContain("qwen3-1.7b-fixture");
  expect(new URL(page.url()).searchParams.has("app-recovery")).toBe(false);
});

test("hosted navigation is cross-origin isolated", async ({ page }) => {
  await page.goto("/");
  expect(await page.evaluate(() => globalThis.crossOriginIsolated)).toBe(true);
});

test("unknown paths render the hosted 404", async ({ page }) => {
  await page.goto("/outside-the-workbench");

  await expect(page).toHaveTitle("Page not found · Drowse");
  await expect(
    page.getByRole("heading", { name: "Page not found", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "Open Drowse", exact: true })).toHaveAttribute(
    "href",
    "/app",
  );
});

test("landing and onboarding fit a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBe(true);

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Download and open", exact: true })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBe(true);
});

test("first-run offline notice does not cover the mobile primary action and restores focus", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) throw new Error("Service workers unavailable");
    await navigator.serviceWorker.ready;
  });

  const notice = page.getByRole("status").filter({ hasText: "The interface is ready offline." });
  await expect(notice).toBeVisible({ timeout: 15_000 });
  const primary = page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" });
  const [noticeBox, primaryBox] = await Promise.all([notice.boundingBox(), primary.boundingBox()]);
  expect(noticeBox).not.toBeNull();
  expect(primaryBox).not.toBeNull();
  expect(noticeBox!.y).toBeGreaterThanOrEqual(0);
  expect(noticeBox!.y + noticeBox!.height).toBeLessThanOrEqual(844);
  expect(
    noticeBox!.x < primaryBox!.x + primaryBox!.width &&
      noticeBox!.x + noticeBox!.width > primaryBox!.x &&
      noticeBox!.y < primaryBox!.y + primaryBox!.height &&
      noticeBox!.y + noticeBox!.height > primaryBox!.y,
  ).toBe(false);

  await primary.focus();
  const dismiss = notice.getByRole("button", { name: "Dismiss" });
  await dismiss.focus();
  await page.keyboard.press("Enter");
  await expect(notice).toHaveCount(0);
  await expect(primary).toBeFocused();
});

test("dismissing the landing notice does not make the hero a selectable focus container", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  const notice = page.getByRole("status").filter({ hasText: "The interface is ready offline." });
  await expect(notice).toBeVisible({ timeout: 15_000 });
  await notice.getByRole("button", { name: "Dismiss" }).click();
  await expect(notice).toHaveCount(0);
  await expect(page.locator(".hero .primary-action")).toBeFocused();
  await expect(page.locator("main")).not.toHaveAttribute("tabindex");
  await expect(page.locator("main")).toHaveCSS("outline-style", "none");
});

test("a controlled offline launch resumes PWA update monitoring after reconnect", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) throw new Error("Service workers unavailable");
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);

  await page.addInitScript(() => {
    const root = globalThis as typeof globalThis & {
      __drowsePwaReconnect?: { online: boolean; registrations: number };
    };
    const state = { online: false, registrations: 0 };
    root.__drowsePwaReconnect = state;
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      get: () => state.online,
    });
    const serviceWorkers = navigator.serviceWorker;
    const register = serviceWorkers.register.bind(serviceWorkers);
    Object.defineProperty(serviceWorkers, "register", {
      configurable: true,
      value: (...args: Parameters<ServiceWorkerContainer["register"]>) => {
        state.registrations += 1;
        return register(...args);
      },
    });
  });
  await page.reload();

  const notice = page.getByRole("status").filter({ hasText: "The interface is ready offline." });
  await expect(notice).toBeVisible();
  expect(await page.evaluate(() => {
    const root = globalThis as typeof globalThis & {
      __drowsePwaReconnect?: { online: boolean; registrations: number };
    };
    return root.__drowsePwaReconnect?.registrations;
  })).toBe(0);

  await page.evaluate(() => {
    const root = globalThis as typeof globalThis & {
      __drowsePwaReconnect?: { online: boolean; registrations: number };
    };
    const state = root.__drowsePwaReconnect;
    if (!state) throw new Error("Missing PWA reconnect fixture");
    state.online = true;
    window.dispatchEvent(new Event("online"));
    window.dispatchEvent(new Event("online"));
  });
  await expect.poll(
    () => page.evaluate(() => {
      const root = globalThis as typeof globalThis & {
        __drowsePwaReconnect?: { online: boolean; registrations: number };
      };
      return root.__drowsePwaReconnect?.registrations;
    }),
  ).toBe(1);
});

test.describe("touchscreen tablet", () => {
test.use({ hasTouch: true });
test("compact workbench controls keep touch targets at tablet widths", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await openFixtureWorkbench(page);

  await showWorkspaceTools(page, "chat");
  const chatTargetHeights = await page.locator(".chat-header button").evaluateAll(
    (elements) => elements.map((element) => element.getBoundingClientRect().height),
  );
  expect(chatTargetHeights.length).toBeGreaterThan(0);
  expect(Math.min(...chatTargetHeights)).toBeGreaterThanOrEqual(44);

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Create a tablet thread");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await openWorkspace(page, "Branches");
  const treeTargetHeights = await page.getByRole("treeitem").evaluateAll(
    (elements) => elements.map((element) => element.getBoundingClientRect().height),
  );
  expect(treeTargetHeights.length).toBeGreaterThan(0);
  expect(Math.min(...treeTargetHeights)).toBeGreaterThanOrEqual(44);
});
});

test("fixture installs, opens the shared workbench, stops, and generates", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  const submit = sendButton(page);
  const stop = page.getByRole("button", { name: /^Stop$/i });
  await openWorkspace(page, "Controls");
  await page.getByRole("button", { name: "Sampling settings" }).click();
  const samplingDrawer = page.getByRole("dialog", { name: "Sampling settings" });
  await expect(samplingDrawer.getByText(
    "Set how the model samples tokens and what it saves.",
    { exact: true },
  )).toHaveCount(0);
  await expect(samplingDrawer.getByText(
    "Limit the token choices and change how repetition is scored.",
    { exact: true },
  )).toHaveCount(0);
  const alternatives = page.getByRole("spinbutton", {
    name: "Return top K",
  });
  await expect(alternatives).toBeEnabled();
  await expect(alternatives).toHaveAttribute("max", String(Number.MAX_SAFE_INTEGER));
  await expect(alternatives).toHaveValue("5");
  await page.getByRole("button", { name: "Close drawer" }).click();
  await openWorkspace(page, "Conversation");

  await composer.fill("Stop this response early");
  await submit.click();
  await expect(stop).toBeEnabled();
  await stop.dispatchEvent("click");
  await expect(stop).toBeDisabled();
  await expect(page.locator(".msg .response-body").last()).not.toHaveText(fixtureResponse);

  await composer.fill("Finish this response");
  await submit.click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect(stop).toBeDisabled();
  const storage = await fixtureStorageSnapshot(page);
  expect(storage.install).toMatchObject({
    id: "qwen3-1.7b-fixture",
    kind: "model",
    objectHashes: expect.arrayContaining([...fixtureModelHashes]),
  });
  expect(storage.installs.map((install) => install.id).sort()).toEqual([
    "fixture-jlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);
  expect(storage.sizes).toEqual({
    [fixtureHashes[0]]: 45,
    [fixtureHashes[1]]: 53,
    [fixtureHashes[2]]: 57,
    [fixtureHashes[3]]: 54,
  });
  expect(pageErrors).toEqual([]);
});

test("bottom conversation action clears the active path without deleting its branches", async ({ page }) => {
  await openFixtureWorkbench(page);

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Keep this branch available");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  const clear = page.getByRole("button", { name: "Clear conversation" });
  await expect(clear).toHaveClass(/clear-conversation/);
  const [clearBox, headerBox] = await Promise.all([
    clear.boundingBox(),
    page.locator(".app-header").boundingBox(),
  ]);
  expect(clearBox).not.toBeNull();
  expect(headerBox).not.toBeNull();
  expect(clearBox!.y).toBeGreaterThan(headerBox!.y + headerBox!.height);
  await clear.click();
  const confirm = page.getByRole("button", { name: "Confirm clear conversation" });
  await expect(confirm).toBeVisible();
  await confirm.click();

  await expect(page.getByText("Type a prompt to get started", { exact: true })).toBeVisible();
  await expect(page.locator(".conversation-empty h2, .empty-mark, .starter-prompts")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Clear conversation" })).toHaveCount(0);

  await openWorkspace(page, "Branches");
  await expect(page.getByText("Keep this branch available", { exact: true })).toBeVisible();
});

test("first run includes J-lens while additional model tools remain manageable", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  const toolPicker = page.locator(".tool-picker");
  const featureChoice = toolPicker.getByRole("checkbox", { name: /Feature explorer/ });
  const includedJlens = toolPicker.locator(".required-tool", { hasText: "Word insights" });
  await expect(includedJlens).toContainText("J-lens");
  await expect(includedJlens.locator('input[type="checkbox"]')).toHaveCount(0);
  await expect(featureChoice).toBeChecked();
  await expect(toolPicker.locator(".tool-total")).toContainText("209 B");

  await featureChoice.uncheck();
  await expect(featureChoice).not.toBeChecked();
  await expect(toolPicker.locator(".tool-total")).toContainText("155 B");
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "qwen3-1.7b-fixture",
  ]);

  const drawer = await openModelAndStorage(page);
  const optionalSection = drawer.locator("section.panel", {
    hasText: "Included files and additions",
  });
  const jlensOption = optionalSection.locator('li[data-pack-id="fixture-jlens"]');
  await expect(jlensOption.getByText("Ready", { exact: true })).toBeVisible();
  await expect(jlensOption.getByRole("button")).toHaveCount(0);

  const saeOption = optionalSection.locator('li[data-pack-id="fixture-sae"]');
  const downloadSae = saeOption.getByRole("button", { name: "Download" });
  await expect(downloadSae).toBeEnabled();
  await downloadSae.click();
  await expect(saeOption.getByText("Ready", { exact: true })).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);

  const rLensOption = optionalSection.locator('li[data-pack-id="fixture-rlens"]');
  const downloadRLens = rLensOption.getByRole("button", { name: "Download" });
  await expect(downloadRLens).toBeEnabled();
  await downloadRLens.click();
  await expect(rLensOption.getByText("Ready", { exact: true })).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);

  const installedSae = downloadedPackRow(drawer, "fixture-sae");
  await attemptPackDeletion(installedSae);
  await page.waitForURL(`${devUrl}/app?fixture=1&reopen=1`);
  await expect(page.locator(".shell")).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "qwen3-1.7b-fixture",
  ]);

  const reopenedDrawer = await openModelAndStorage(page);
  const reopenedTools = reopenedDrawer.locator("section.panel", {
    hasText: "Included files and additions",
  });
  const reinstallSae = reopenedTools.locator('li[data-pack-id="fixture-sae"]');
  await reinstallSae.getByRole("button", { name: "Download" }).click();
  await expect(reinstallSae.getByText("Ready", { exact: true })).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);
});

test("installed J-lens and SAE packs are usable without preparation errors", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  await openWorkspace(page, "Controls");
  const instrumentTabs = page.getByRole("group", { name: "Response guidance type" });
  await instrumentTabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  await expect(lens.getByRole("button", { name: "Compatible data pack" })).toContainText(
    "fixture-jlens",
  );
  await expect(lens.getByRole("button", { name: "In use", exact: true })).toBeDisabled();
  await lens.getByRole("button", { name: "Compatible data pack" }).click();
  await expect(lens.getByRole("option", { name: "fixture-jlens", exact: true })).toBeVisible();
  await expect(lens.getByRole("option", { name: /workspace|neuronpedia/i })).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(lens.getByText("Create on this device", { exact: true })).toHaveCount(0);
  await expect(lens.getByLabel("J-lens corpus prompts")).toHaveCount(0);
  await expect(lens.getByRole("button", { name: /^(confirm )?fit$/i })).toHaveCount(0);
  await expect(lens.locator('[aria-description="turn live readout off"]')).toBeVisible();
  await lens.locator('[aria-description="turn live readout off"]').click();
  await expect(lens.locator('[aria-description="turn live readout on"]')).toBeVisible();
  await lens.locator('[aria-description="turn live readout on"]').click();
  await expect(lens.locator('[aria-description="turn live readout off"]')).toBeVisible();
  await lens.getByRole("textbox", { name: "Add a word prediction direction" }).fill("fixture");
  await lens.getByRole("button", { name: "Add word", exact: true }).click();
  await expect(lens.locator(".steer-cards")).toContainText("fixture");
  const lensOperation = lens.getByRole("group", {
    name: "steering operation for jlens/fixture",
  });
  await lensOperation.getByRole("button", { name: "ablate", exact: true }).click();
  await expect(lensOperation.getByRole("button", { name: "ablate", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await lens.getByRole("textbox", { name: "Watch a prediction word" }).fill("fixture");
  await lens.getByRole("button", { name: "Watch word", exact: true }).click();
  await expect(lens.getByRole("list", { name: "J-lens probe tokens" })).toBeVisible();
  await expect(lens.getByText("Not measured", { exact: true })).toBeVisible();

  await instrumentTabs.getByRole("button", { name: "SAE", exact: true }).click();
  const sae = page.getByLabel("Model feature controls");
  await expect(sae.getByRole("button", { name: "Compatible data pack" })).toContainText(
    "fixture-sae",
  );
  await expect(sae.getByRole("button", { name: "In use", exact: true })).toBeDisabled();
  await sae.getByRole("button", { name: "Compatible data pack" }).click();
  await expect(sae.getByRole("option", { name: "fixture-sae", exact: true })).toBeVisible();
  await expect(sae.getByRole("option", { name: /saelens|release/i })).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(sae.getByText("Create on this device", { exact: true })).toHaveCount(0);
  await expect(sae.getByLabel("SAE training tokens")).toHaveCount(0);
  await expect(sae.getByRole("button", { name: /^(confirm )?train$/i })).toHaveCount(0);
  await expect(sae.locator('[aria-description="turn live readout off"]')).toBeVisible();
  await sae.locator('[aria-description="turn live readout off"]').click();
  await expect(sae.locator('[aria-description="turn live readout on"]')).toBeVisible();
  await sae.locator('[aria-description="turn live readout on"]').click();
  await expect(sae.locator('[aria-description="turn live readout off"]')).toBeVisible();
  await sae.getByRole("textbox", { name: "Add a model feature direction" }).fill("7");
  await sae.getByRole("button", { name: "Add feature", exact: true }).click();
  await expect(sae.locator(".steer-cards")).toContainText("7");
  const saeOperation = sae.getByRole("group", {
    name: "steering operation for sae/7",
  });
  await saeOperation.getByRole("button", { name: "ablate", exact: true }).click();
  await expect(saeOperation.getByRole("button", { name: "ablate", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await sae.getByRole("textbox", { name: "Watch a model feature" }).fill("7");
  await sae.getByRole("button", { name: "Watch feature", exact: true }).click();
  await expect(sae.getByRole("list", { name: "SAE feature probes" })).toBeVisible();
  await expect(sae.getByText("Not measured", { exact: true })).toBeVisible();

  const steeringExpression = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.currentSteeringExpression();
  }, toastModuleUrl);
  expect(steeringExpression).toContain("0.3 !jlens/fixture");
  expect(steeringExpression).toContain("0.3 !sae/7");

  await openWorkspace(page, "Conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Measure installed instruments");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  await openWorkspace(page, "Controls");
  await instrumentTabs.getByRole("button", { name: "SAE", exact: true }).click();
  await expect(sae.getByLabel("Strength 0.67").first()).toBeVisible();

  await instrumentTabs.getByRole("button", { name: "J-lens", exact: true }).click();
  await expect(lens.getByLabel("Strength 0.755", { exact: true }).first()).toBeVisible();

  await openWorkspace(page, "Conversation");
  await page.getByRole("button", { name: "Reroll assistant message" }).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  await expect.poll(async () => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === "Measure installed instruments",
    );
    if (!user) return [];
    const childIds = stores.loomTree.children_of.get(user.id) ?? [];
    return childIds.map((id: string) => {
      const node = stores.loomTree.nodes.get(id);
      return {
        role: node?.role,
        hasLens: node?.tokens?.some((token: { measurements?: { instruments?: object } }) =>
          token.measurements?.instruments && "lens" in token.measurements.instruments
        ) ?? false,
        hasSae: node?.tokens?.some((token: { measurements?: { instruments?: object } }) =>
          token.measurements?.instruments && "sae" in token.measurements.instruments
        ) ?? false,
      };
    });
  }, toastModuleUrl)).toEqual([
    { role: "assistant", hasLens: true, hasSae: true },
    { role: "assistant", hasLens: true, hasSae: true },
  ]);

  await expect(page.getByText(/Fake runtime method .* is not implemented/)).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test("models without an SAE say that no SAE is available", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1&fixtureSae=0`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: /Feature explorer/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();

  await openWorkspace(page, "Controls");
  await page.getByRole("group", { name: "Response guidance type" })
    .getByRole("button", { name: "SAE", exact: true }).click();
  await expect(page.getByLabel("Model feature controls").getByText(
    "No SAE is available for this model",
    { exact: true },
  )).toBeVisible();

  await openWorkspace(page, "Conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Inspect without an SAE");
  await sendButton(page).click();
  await page.getByRole("button", { name: "Inspect tokens in assistant message" }).click();
  const drawer = await openTokenDetails(page);
  await drawer.getByRole("button", { name: /^sae\b/i }).click();
  await expect(drawer.getByText(
    "No SAE is available for this model",
    { exact: true },
  )).toBeVisible();
});

test("changing models closes the runtime and opens installed model choices", async ({ page }) => {
  await openFixtureWorkbench(page);
  await clickWorkspaceAction(page, "All tools");
  const search = page.getByRole("combobox", { name: "Filter commands" });
  await search.fill("model settings");
  await page.keyboard.press("Enter");
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  await modelControls.getByRole("button", { name: "Change model", exact: true }).click();
  await expect.poll(() => page.url()).toBe(`${devUrl}/app?fixture=1&choose=1`);
  await expect(page.getByRole("heading", { name: "Models" }))
    .toBeVisible();
  await expect(page.getByText("Installed and verified", { exact: true })).toBeVisible();
});

test("installed standard and R-lens packs swap live without fitting", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  const toolPicker = page.locator(".tool-picker");
  const rLensChoice = toolPicker.getByRole("checkbox", { name: /R-lens readouts/ });
  await expect(rLensChoice).not.toBeChecked();
  await rLensChoice.check();
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect.poll(() => fixtureInstallIds(page)).toEqual([
    "fixture-jlens",
    "fixture-rlens",
    "fixture-sae",
    "qwen3-1.7b-fixture",
  ]);

  await openWorkspace(page, "Controls");
  const instrumentTabs = page.getByRole("group", { name: "Response guidance type" });
  await instrumentTabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  const source = lens.getByRole("button", { name: "Compatible data pack" });
  await source.click();
  await lens.getByRole("option", { name: "fixture-rlens", exact: true }).click();
  await lens.getByRole("button", { name: "Use pack", exact: true }).click();
  await expect(source).toContainText("fixture-rlens");
  await expect(lens.getByRole("button", { name: "In use", exact: true })).toBeDisabled();

  await expect(lens.locator('[aria-description="turn live readout off"]')).toBeVisible();
  await lens.getByRole("textbox", { name: "Add a word prediction direction" }).fill("fixture");
  await lens.getByRole("button", { name: "Add word", exact: true }).click();
  await lens.getByRole("textbox", { name: "Watch a prediction word" }).fill("fixture");
  await lens.getByRole("button", { name: "Watch word", exact: true }).click();

  await openWorkspace(page, "Conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Use the active R-lens");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await openWorkspace(page, "Controls");
  await instrumentTabs.getByRole("button", { name: "J-lens", exact: true }).click();
  await expect(lens.getByLabel("Strength 0.755", { exact: true }).first()).toBeVisible();
});

test("hover replay routes the exact token request and exposes a recoverable failure", async ({ page }) => {
  await installFixtureRpcProbe(page);
  await openFixtureWorkbench(page);

  await openWorkspace(page, "Controls");
  const instrumentTabs = page.getByRole("group", { name: "Response guidance type" });
  await instrumentTabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  await lens.locator('[aria-description="turn live readout off"]').click();
  await openWorkspace(page, "Conversation");

  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create a replayable token");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  await openWorkspace(page, "Controls");
  await instrumentTabs.getByRole("button", { name: "J-lens", exact: true }).click();

  await page.evaluate(() => {
    const state = (globalThis as typeof globalThis & {
      __drowseFixtureRpc: {
        requests: FixtureTokenReadoutRequest[];
        failNextFamily: "lens" | "sae" | null;
      };
    }).__drowseFixtureRpc;
    state.requests = [];
    state.failNextFamily = "lens";
  });
  const expected = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const turn = [...stores.chatLog.turns].reverse().find(
      (candidate: { role: string; nodeId?: string | null; tokens?: unknown[] }) =>
        candidate.role === "assistant" && candidate.nodeId && (candidate.tokens?.length ?? 0) > 0,
    );
    const token = turn?.tokens?.[0] as { rawIndex?: number | null } | undefined;
    if (!turn?.nodeId || token?.rawIndex == null) {
      throw new Error("Fixture assistant token is missing its replay identity");
    }
    stores.beginTokenHover(token, turn.nodeId);
    return { nodeId: turn.nodeId, rawIndex: token.rawIndex };
  }, toastModuleUrl);

  await expect.poll(() => page.evaluate(() =>
    (globalThis as typeof globalThis & {
      __drowseFixtureRpc: { requests: FixtureTokenReadoutRequest[] };
    }).__drowseFixtureRpc.requests.length
  )).toBe(1);
  const requests = await page.evaluate(() =>
    (globalThis as typeof globalThis & {
      __drowseFixtureRpc: { requests: FixtureTokenReadoutRequest[] };
    }).__drowseFixtureRpc.requests
  );
  expect(requests[0]).toEqual({
    service: "instruments",
    method: "tokenReadout",
    args: [
      "lens",
      expected.nodeId,
      expected.rawIndex,
      { topK: 5, steered: true, raw: false, layers: "all" },
    ],
  });
  await expect(lens.getByRole("alert")).toHaveText(
    "This token's word-likelihood reading could not be rebuilt. Try again, or close and reopen the model.",
  );
  await expect.poll(async () => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.tokenHoverState.saeReadout?.length ?? 0;
  }, toastModuleUrl)).toBeGreaterThan(0);
});

test("optional SAE packs cannot be deleted while direct steering or readings use them", async ({ page }) => {
  await openFixtureWorkbench(page);
  await openWorkspace(page, "Controls");
  const instrumentTabs = page.getByRole("group", { name: "Response guidance type" });

  await instrumentTabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  await lens.getByRole("textbox", { name: "Add a word prediction direction" }).fill("fixture");
  await lens.getByRole("button", { name: "Add word", exact: true }).click();
  await lens.getByRole("textbox", { name: "Watch a prediction word" }).fill("fixture");
  await lens.getByRole("button", { name: "Watch word", exact: true }).click();

  await instrumentTabs.getByRole("button", { name: "SAE", exact: true }).click();
  const sae = page.getByLabel("Model feature controls");
  await sae.getByRole("textbox", { name: "Add a model feature direction" }).fill("7");
  await sae.getByRole("button", { name: "Add feature", exact: true }).click();
  await sae.getByRole("textbox", { name: "Watch a model feature" }).fill("7");
  await sae.getByRole("button", { name: "Watch feature", exact: true }).click();

  const drawer = await openModelAndStorage(page);
  await expect(downloadedPackRow(drawer, "fixture-jlens")).toHaveCount(0);
  await expect(drawer.locator('li[data-pack-id="fixture-jlens"]')).toContainText(
    "Word insights · included",
  );

  const saeRow = downloadedPackRow(drawer, "fixture-sae");
  await attemptPackDeletion(saeRow);
  await expect(drawer.getByRole("alert")).toContainText(
    "SAE steering: sae/7; SAE readings: sae/7",
  );
  await expect(saeRow).toBeVisible();
});

test("optional SAE packs cannot be deleted while a custom condition depends on them", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    stores.applyCustomSteeringExpression(
      "0.5 fixture/calm.focused@when:sae/7>0.1",
    );
  }, toastModuleUrl);
  const drawer = await openModelAndStorage(page);
  await expect(downloadedPackRow(drawer, "fixture-jlens")).toHaveCount(0);
  const saeRow = downloadedPackRow(drawer, "fixture-sae");
  await attemptPackDeletion(saeRow);
  await expect(drawer.getByRole("alert")).toContainText(
    "SAE steering expression or condition",
  );
  await expect(saeRow).toBeVisible();
});

test("geometry measurements populate and completed timing stays frozen", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  await openWorkspace(page, "Controls");
  const instrumentTabs = page.getByRole("group", { name: "Response guidance type" });
  await instrumentTabs.getByRole("button", { name: "Subspace", exact: true }).click();
  await page.getByRole("button", { name: "Add subspace probe", exact: true }).click();
  const drawer = page.getByRole("dialog");
  await drawer.getByText("attach selector", { exact: true }).click();
  await drawer.getByRole("textbox", { name: "Selector" }).fill("fixture/calm.focused");
  await drawer.getByRole("button", { name: "+ attach", exact: true }).click();
  await expect(page.getByText("probe fixture/calm.focused", { exact: true })).toBeVisible();
  await page.keyboard.press("Escape");

  await openWorkspace(page, "Conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Measure geometry");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  await openWorkspace(page, "Controls");
  const probeRack = page.getByLabel("Probe rack");
  await expect(probeRack).toContainText("calm.focused");
  await expect(probeRack.getByLabel("Per-layer readings for fixture/calm.focused"))
    .not.toHaveAttribute("aria-valuetext", /no data yet/);

  const status = page.getByLabel("Generation status", { exact: true });
  const settledStatus = await status.locator(":scope > .text").allInnerTexts();
  await page.waitForTimeout(750);
  expect(await status.locator(":scope > .text").allInnerTexts()).toEqual(settledStatus);
  expect(pageErrors).toEqual([]);
});

test("comparison controls appear only when usable and auto-compare completes", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(devUrl);
  await page.evaluate(() => {
    localStorage.setItem("drowse.chat.v4.fixture/drowse-tiny", JSON.stringify({
      version: 4,
      model_id: "fixture/drowse-tiny",
      saved_at: Date.now(),
      highlight: {
        target: "__probability__",
        compareTarget: null,
        compareTwo: false,
      },
    }));
  });
  await openFixtureWorkbench(page);

  await showWorkspaceTools(page, "chat");
  const colorPicker = page.getByRole("button", { name: "Color generated words by" });
  await expect(colorPicker).toContainText("Token surprisal");
  await colorPicker.click();
  await expect(page.getByRole("option", { name: "No color", exact: true })).toBeVisible();
  await expect(page.getByRole("option", { name: "Token surprisal", exact: true })).toBeVisible();
  await expect(page.getByRole("option", { name: "Token probability", exact: true })).toHaveCount(0);
  await expect(page.getByRole("option", { name: "Sampler entropy", exact: true })).toHaveCount(0);
  await page.keyboard.press("Escape");

  await expect(page.getByRole("button", { name: "Compare colors" })).toHaveCount(0);

  await openWorkspace(page, "Controls");
  const instrumentTabs = page.getByRole("group", { name: "Response guidance type" });
  await instrumentTabs.getByRole("button", { name: "Subspace", exact: true }).click();
  await page.getByRole("button", { name: "Add subspace probe", exact: true }).click();
  const drawer = page.getByRole("dialog");
  await drawer.getByText("attach selector", { exact: true }).click();
  await drawer.getByRole("textbox", { name: "Selector" }).fill("fixture/calm.focused");
  await drawer.getByRole("button", { name: "+ attach", exact: true }).click();
  await page.keyboard.press("Escape");

  await openWorkspace(page, "Conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create colors to compare");
  await sendButton(page).click();
  const colorComparison = page.getByRole("button", { name: "Compare colors" });
  await expect(colorComparison).toBeVisible();
  await colorComparison.click();
  await expect(page.getByRole("button", { name: "Second word color" }))
    .toContainText("Sampler entropy");
  await page.getByRole("button", { name: "Second word color" }).click();
  await page.getByRole("option", { name: "calm.focused", exact: true }).click();
  await expect(page.getByRole("button", { name: "Second word color" }))
    .toContainText("calm.focused");
  await expect(page.getByText("No second color", { exact: true })).toHaveCount(0);

  const autoCompare = page.getByRole("checkbox", { name: "Compare replies" });
  await autoCompare.check();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Compare this response");
  await sendButton(page).click();

  const shadow = page.locator(".ab-shadow");
  await expect(shadow.locator(".response-body").last()).toHaveText(fixtureResponse);
  await expect(shadow.getByText("pending…", { exact: true })).toHaveCount(0);
  await expect(shadow.getByText("Generating comparison…", { exact: true })).toHaveCount(0);
  await colorPicker.click();
  await expect(page.getByRole("option", { name: "Sampler entropy", exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return {
      active: stores.genStatus.active,
      processing: stores.abState.processingAb,
      pendingTurn: stores.abState.pendingTurnIdx,
    };
  }, toastModuleUrl)).toEqual({ active: false, processing: false, pendingTurn: null });
  expect(pageErrors).toEqual([]);
});

test("fixture survives repeated generation, stop, and reload cycles", async ({ page }) => {
  test.setTimeout(90_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  const runGeneration = async (prompt: string, stopEarly: boolean) => {
    const composer = page.getByRole("textbox", { name: /^Compose as / });
    const stop = page.getByRole("button", { name: /^Stop$/i });
    const responses = page.locator(".msg:has(.model-avatar) .response-body");
    const previousResponses = await responses.count();
    const response = responses.last();
    await composer.fill(prompt);
    await sendButton(page).click();
    await expect(responses).toHaveCount(previousResponses + 1);
    if (stopEarly) {
      await expect(response).toContainText("This");
      await stop.dispatchEvent("click");
      await expect(response).not.toHaveText(fixtureResponse);
    } else {
      await expect(response).toContainText(fixtureResponse);
    }
    await expect(stop).toBeDisabled();
    await expect.poll(() => page.evaluate(async (moduleUrl) => {
      const stores = await import(moduleUrl);
      return {
        active: stores.genStatus.active,
        queued: stores.pendingActions.queue.length,
      };
    }, toastModuleUrl)).toEqual({ active: false, queued: 0 });
  };

  for (let index = 0; index < 20; index += 1) {
    await runGeneration(`Lifecycle generation ${index}`, index % 4 === 0);
  }
  for (let index = 0; index < 6; index += 1) {
    await openInstalledFixtureWorkbench(page);
    await runGeneration(`Reload generation ${index}`, false);
  }

  const storage = await fixtureStorageSnapshot(page);
  expect(storage.names).toEqual(fixtureHashes.map((hash) => `${hash}.data`).sort());
  expect(pageErrors).toEqual([]);
});

test("reload during generation restores the stable user turn without a partial assistant", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Persist before interrupted decode";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();
  await expect(page.locator(".msg .response-body").last()).toContainText("This");
  await expect(page.locator(".msg .response-body").last()).not.toHaveText(fixtureResponse);

  await openInstalledFixtureWorkbench(page);
  await expect(
    page.getByLabel("Conversation", { exact: true }).getByText(prompt, { exact: true }),
  ).toBeVisible();
  await expect(page.locator('.role-chip[title="assistant"]')).toHaveCount(0);

  await page.getByRole("textbox", { name: /^Compose as / }).fill("Continue after recovery");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
});

test("reroll creates an assistant sibling under the original user turn", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Create a reroll branch";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  await page.getByRole("button", { name: "Reroll assistant message" }).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  const branch = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const nodes = [...stores.loomTree.nodes.values()];
    const user = nodes.find((node: { text: string }) => node.text === "Create a reroll branch");
    if (!user) throw new Error("Reroll fixture user node is missing");
    const children = stores.loomTree.children_of.get(user.id) ?? [];
    return {
      childIds: [...children],
      childParents: children.map((id: string) => stores.loomTree.nodes.get(id)?.parent_id),
      childRoles: children.map((id: string) => stores.loomTree.nodes.get(id)?.role),
    };
  }, toastModuleUrl);
  expect(branch.childIds).toHaveLength(2);
  expect(new Set(branch.childParents).size).toBe(1);
  expect(branch.childRoles).toEqual(["assistant", "assistant"]);

  await openWorkspace(page, "Branches");
  const loom = page.locator(".loom-canvas");
  await expect(loom).toHaveAttribute("data-loom-nodes", "4");
  await expect(loom).toHaveAttribute("data-loom-edges", "3");
  await expect(loom.locator("[data-loom-junction]")).toHaveCount(1);
  const parentCard = page.locator(`[data-node-id="${branch.childParents[0]}"]`);
  const firstReply = page.locator(`[data-node-id="${branch.childIds[0]}"]`);
  const secondReply = page.locator(`[data-node-id="${branch.childIds[1]}"]`);
  const [parentBox, firstBox, secondBox] = await Promise.all([
    parentCard.boundingBox(),
    firstReply.boundingBox(),
    secondReply.boundingBox(),
  ]);
  expect(parentBox).not.toBeNull();
  expect(firstBox).not.toBeNull();
  expect(secondBox).not.toBeNull();
  expect(parentBox!.x + parentBox!.width).toBeLessThan(firstBox!.x);
  expect(firstBox!.x).toBeCloseTo(secondBox!.x, 0);
  expect(Math.abs(firstBox!.y - secondBox!.y)).toBeGreaterThan(firstBox!.height);

  const initialZoom = Number(await loom.getAttribute("data-loom-zoom"));
  await page.getByRole("button", { name: "Zoom in" }).click();
  await expect.poll(async () => Number(await loom.getAttribute("data-loom-zoom")))
    .toBeGreaterThan(initialZoom);
  await page.getByRole("button", { name: "Fit whole loom" }).click();

  await parentCard.getByRole("button", { name: "Hide child paths" }).click();
  await expect(loom).toHaveAttribute("data-loom-nodes", "1");
  await expect(loom).toHaveAttribute("data-loom-edges", "0");
  await parentCard.getByRole("button", { name: "Show child paths" }).click();
  await expect(loom).toHaveAttribute("data-loom-nodes", "4");
  await expect(loom).toHaveAttribute("data-loom-edges", "3");

  await page.evaluate(async ({ moduleUrl, childIds }) => {
    const stores = await import(moduleUrl);
    stores.openDrawer("node_compare", { node_ids: childIds });
  }, { moduleUrl: toastModuleUrl, childIds: branch.childIds });
  const comparison = page.getByRole("dialog", { name: "Compare conversation branches" });
  await expect(comparison).toBeVisible();
  await expect(comparison.getByText(/rank-1 changes/)).toBeVisible();
  await expect(comparison.getByRole("alert")).toHaveCount(0);
});

test("the loom background stays steady on hover and follows zoom while respecting reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Show the loom depth field");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await openWorkspace(page, "Branches");

  const viewport = page.locator(".loom-viewport");
  const field = viewport.locator(".loom-depth-field");
  await expect(field).toHaveAttribute("aria-hidden", "true");
  await expect(viewport.locator(".loom-depth-light, .loom-focus-halo")).toHaveCount(0);
  const viewportBox = await viewport.boundingBox();
  expect(viewportBox).not.toBeNull();

  const initialTransform = await field.evaluate(element => getComputedStyle(element).transform);
  await page.mouse.move(
    viewportBox!.x + viewportBox!.width * 0.82,
    viewportBox!.y + viewportBox!.height * 0.76,
  );
  await expect(field).toHaveCSS("transform", initialTransform);
  const gridOpacity = await field.evaluate((element) =>
    Number.parseFloat(getComputedStyle(element, "::before").opacity)
  );
  expect(gridOpacity).toBeLessThanOrEqual(0.52);

  const initialSpacing = await field.evaluate((element) =>
    Number.parseFloat(getComputedStyle(element).getPropertyValue("--loom-grid-size"))
  );
  await page.getByRole("button", { name: "Zoom in" }).click();
  await expect.poll(() => field.evaluate((element) =>
    Number.parseFloat(getComputedStyle(element).getPropertyValue("--loom-grid-size"))
  )).toBeGreaterThan(initialSpacing);

  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(field).toHaveCSS("transform", "none");
});

test("a saved token alternative creates an exact sibling loom branch", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Fork this response from one token";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();

  const response = page.locator(".msg .response-body").last();
  await expect(response).toContainText(fixtureResponse);
  await response.locator(".tok").first().click();

  const drilldown = await openTokenDetails(page);
  await drilldown.getByRole("button", { name: /logits/i }).click();
  await expect(drilldown.getByLabel("Ranked token alternatives")).toBeVisible();
  const startBranch = drilldown.getByRole("button", { name: "Start branch" }).first();
  await expect(startBranch).toBeEnabled();

  const source = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) throw new Error("Token-fork fixture user node is missing");
    const sourceId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    const sourceNode = stores.loomTree.nodes.get(sourceId);
    return {
      parentId: user.id,
      sourceId,
      sourceFirstToken: sourceNode?.raw_token_ids?.[0],
    };
  }, { moduleUrl: toastModuleUrl, prompt });

  await startBranch.click();
  await expect(drilldown).toHaveCount(0);
  await expect(page.locator(".msg .response-body").last()).toContainText("Alternative-0 ");
  await openWorkspace(page, "Conversation");
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  const fork = await page.evaluate(async ({ moduleUrl, parentId, sourceId }) => {
    const stores = await import(moduleUrl);
    const siblings = stores.loomTree.children_of.get(parentId) ?? [];
    const forkId = siblings.find((id: string) => id !== sourceId);
    const node = forkId ? stores.loomTree.nodes.get(forkId) : null;
    return {
      activeId: stores.loomTree.active_node_id,
      siblingIds: [...siblings],
      forkId: forkId ?? null,
      forkParentId: node?.parent_id ?? null,
      forkFirstToken: node?.raw_token_ids?.[0] ?? null,
      forkText: node?.text ?? null,
    };
  }, { moduleUrl: toastModuleUrl, parentId: source.parentId, sourceId: source.sourceId });

  expect(fork.siblingIds).toHaveLength(2);
  expect(fork.forkParentId).toBe(source.parentId);
  expect(fork.activeId).toBe(fork.forkId);
  expect(fork.forkFirstToken).not.toBe(source.sourceFirstToken);
  expect(fork.forkText).toMatch(/^Alternative-0 /);

  await openWorkspace(page, "Branches");
  await expect(page.getByRole("treeitem")).toHaveCount(3);
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-edges", "2");
  await expect(page.locator("[data-loom-junction]")).toHaveCount(1);
  await expect(page.getByText("alternate reply", { exact: true })).toHaveCount(2);
});

test("token replacement streams its new continuation visibly on the Loom", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openFixtureWorkbench(page, "&fixtureSlow=1");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Watch this branch grow.");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled({ timeout: 30_000 });
  const original = await page.evaluate(async url => {
    const stores = await import(url);
    const node = stores.loomTree.nodes.get(stores.loomTree.active_node_id);
    return { id: node.id, saved: JSON.stringify(node) };
  }, toastModuleUrl);
  await page.locator(".msg .response-body").last().locator(".tok").nth(3).click();
  const drawer = await openTokenDetails(page);
  await drawer.getByRole("button", { name: "Replace token…" }).click();
  await drawer.getByRole("textbox", { name: "Replacement text" }).fill("Authored ");
  await drawer.getByRole("region", { name: "Selected token branch point" }).getByRole("button", { name: "Start branch", exact: true }).click();
  await expect(page.locator(".loom-canvas")).toBeVisible();
  const current = page.locator('.tree-node-wrap.current-node .node');
  await expect(current).toHaveClass(/streaming/);
  await expect(current.getByText("End of reply", { exact: true })).toHaveCount(0);
  await expect(current.locator('.token-node').first()).toHaveText("Authored", { timeout: 20_000 });
  await expect.poll(() => current.locator('.token-node').count()).toBeGreaterThan(1);
  await expect(current).toHaveClass(/streaming/);
  await current.screenshot({ path: testInfo.outputPath('replacement-streaming.png') });
  await expect(current).not.toHaveClass(/streaming/, { timeout: 30_000 });
  const final = await page.evaluate(async ({ url, id }) => {
    const stores = await import(url);
    const node = stores.loomTree.nodes.get(stores.loomTree.active_node_id);
    return { original: JSON.stringify(stores.loomTree.nodes.get(id)), id: node.id, text: node.text };
  }, { url: toastModuleUrl, id: original.id });
  expect(final.original).toBe(original.saved);
  expect(final.id).not.toBe(original.id);
  expect(final.text).toContain("Authored ");
  await expect(page.locator('.toast.error')).toHaveCount(0);
});

test("typed text replaces a token in a new sibling loom branch", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Replace this response token with authored text";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();

  const response = page.locator(".msg .response-body").last();
  await expect(response).toContainText(fixtureResponse);
  await response.locator(".tok").first().click();

  const drilldown = await openTokenDetails(page);
  await drilldown.getByRole("button", { name: "Replace token…" }).click();
  const replacement = drilldown.getByRole("textbox", { name: "Replacement text" });
  await expect(replacement).toBeFocused();
  await replacement.fill("Authored ");
  await drilldown.getByRole("region", { name: "Selected token branch point" }).getByRole("button", { name: "Start branch", exact: true }).click();

  await expect(drilldown).toHaveCount(0);
  await expect(page.locator(".msg .response-body").last())
    .toContainText("Authored is a deterministic local Drowse runtime fixture.");
  const siblings = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    return user ? (stores.loomTree.children_of.get(user.id) ?? []).length : 0;
  }, { moduleUrl: toastModuleUrl, prompt });
  expect(siblings).toBe(2);
});

test("token continuation keeps its prefix visible and counts only new output on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep the beginning and write a new ending");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);
  const source = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const node = stores.loomTree.nodes.get(stores.loomTree.active_node_id);
    const target = node.tokens[4];
    const prefix = node.tokens.filter((row: { raw_index: number }) => row.raw_index <= target.raw_index)
      .map((row: { text: string }) => row.text).join("");
    const snapshots: Array<{ type: string; text: string; count: number }> = [];
    (window as any).__continuationSnapshots = snapshots;
    stores.onWsMessage((event: { type: string }) => {
      if (event.type === "generation_progress" || event.type === "token") {
        snapshots.push({ type: event.type, text: stores.chatLog.turns.at(-1)?.text ?? "",
          count: stores.genStatus.tokensSoFar });
      }
    });
    return { id: node.id, text: node.text, prefix, rawIndex: target.raw_index };
  }, toastModuleUrl);
  await page.locator(".msg .response-body").last().locator(".tok").nth(4).click();
  const drawer = await openTokenDetails(page);
  await drawer.getByRole("button", { name: "Continue from here" }).click();
  await expect(drawer).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => (window as any).__continuationSnapshots.length)).toBeGreaterThan(0);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);
  const result = await page.evaluate(async ({ moduleUrl, sourceId }) => {
    const stores = await import(moduleUrl);
    const node = stores.loomTree.nodes.get(stores.loomTree.active_node_id);
    return { snapshots: (window as any).__continuationSnapshots,
      text: stores.chatLog.turns.at(-1)?.text, tokens: node.raw_token_ids.length,
      generated: stores.genStatus.tokensSoFar, seed: node.recipe.sampling.seed,
      sourceText: stores.loomTree.nodes.get(sourceId).text,
      overflow: document.documentElement.scrollWidth > window.innerWidth };
  }, { moduleUrl: toastModuleUrl, sourceId: source.id });
  expect(result.snapshots[0]).toEqual({ type: "generation_progress", text: source.prefix, count: 0 });
  for (const snapshot of result.snapshots) expect(snapshot.text.startsWith(source.prefix)).toBe(true);
  expect(result.text).toBe(fixtureResponse);
  expect(result.sourceText).toBe(source.text);
  expect(result.generated).toBe(result.tokens - source.rawIndex - 1);
  expect(Number.isInteger(result.seed)).toBe(true);
  expect(result.overflow).toBe(false);
});

test("a sentence in the loom starts an exact sibling continuation", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Continue from this sentence";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);

  const source = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) throw new Error("Sentence-branch fixture user node is missing");
    const sourceId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    const node = stores.loomTree.nodes.get(sourceId);
    return {
      parentId: user.id,
      sourceId,
      rawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, { moduleUrl: toastModuleUrl, prompt });

  await openWorkspace(page, "Branches");
  const sourceCard = page.locator(`[data-loom-node-id="${source.sourceId}"]`);
  const sentence = sourceCard.locator("[data-loom-sentence]").first();
  await expect(sentence).toContainText("This is a deterministic local Drowse runtime fixture.");
  const branchButton = sentence.getByRole("button", { name: /^Branch after sentence 1:/ });
  const branchLayout = await Promise.all([
    sourceCard.boundingBox(),
    branchButton.boundingBox(),
  ]);
  expect(branchLayout[0]).not.toBeNull();
  expect(branchLayout[1]).not.toBeNull();
  expect(branchLayout[1]!.y + branchLayout[1]!.height)
    .toBeLessThanOrEqual(branchLayout[0]!.y + branchLayout[0]!.height + 1);
  await branchButton.click();
  await expect.poll(() => page.evaluate(async ({ moduleUrl, parentId }) => {
    const stores = await import(moduleUrl);
    return (stores.loomTree.children_of.get(parentId) ?? []).length;
  }, { moduleUrl: toastModuleUrl, parentId: source.parentId })).toBe(2);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);

  const fork = await page.evaluate(async ({ moduleUrl, parentId, sourceId }) => {
    const stores = await import(moduleUrl);
    const siblings = stores.loomTree.children_of.get(parentId) ?? [];
    const forkId = siblings.find((id: string) => id !== sourceId);
    const node = forkId ? stores.loomTree.nodes.get(forkId) : null;
    return {
      activeId: stores.loomTree.active_node_id,
      forkId: forkId ?? null,
      parentId: node?.parent_id ?? null,
      rawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, { moduleUrl: toastModuleUrl, parentId: source.parentId, sourceId: source.sourceId });

  expect(fork.forkId).not.toBeNull();
  expect(fork.activeId).toBe(fork.forkId);
  expect(fork.parentId).toBe(source.parentId);
  expect(fork.rawTokenIds).toEqual(source.rawTokenIds);
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-edges", "3");
});

test("loom projections stay synchronized across path, options, and starred points", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Show the synchronized loom views";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await page.getByRole("button", { name: "Reroll assistant message" }).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  await openWorkspace(page, "Branches");
  await expect(page.getByRole("navigation", { name: "Loom views" })).toBeVisible();

  await page.getByRole("button", { name: /Current path/ }).click();
  const path = page.getByRole("list", { name: "Current conversation path" });
  await expect(path.locator("[data-path-snippet]")).toHaveCount(2);
  await expect(path.getByText(prompt, { exact: true })).toBeVisible();
  await expect(path.getByRole("button", { name: "Branch after sentence" })).toBeEnabled();

  await page.getByRole("button", { name: /Next options/ }).click();
  const options = page.getByRole("list", { name: "Alternative paths" });
  await expect(options.locator("[data-option-node]")).toHaveCount(2);
  const firstOption = options.locator("[data-option-node]").first();
  await firstOption.getByRole("button", { name: "Save for later" }).click();
  await expect(firstOption.getByText("saved", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /^Starred/ }).click();
  const saved = page.getByRole("list", { name: "Starred conversation points" });
  await expect(saved.locator("[data-saved-node]")).toHaveCount(1);
  await expect(saved).toContainText(fixtureResponse);
  await saved.getByRole("button", { name: "Remove" }).click();
  await expect(page.getByText("Nothing saved yet", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /^Map/ }).click();
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-nodes", "4");
});

test("the loom exposes every generated token as an exact branch point", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Show every token in the loom";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  await expect.poll(() => page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) return 0;
    const nodeId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    return stores.loomTree.nodes.get(nodeId)?.tokens?.length ?? 0;
  }, { moduleUrl: toastModuleUrl, prompt })).toBeGreaterThan(0);

  const generated = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const user = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!user) throw new Error("Token-loom fixture user node is missing");
    const nodeId = (stores.loomTree.children_of.get(user.id) ?? [])[0];
    const node = stores.loomTree.nodes.get(nodeId);
    return {
      userId: user.id,
      nodeId,
      tokens: node?.tokens?.map((token: { text: string }) => token.text) ?? [],
      rawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, { moduleUrl: toastModuleUrl, prompt });

  await openWorkspace(page, "Branches");
  await page.getByRole("button", { name: "Fit whole loom" }).click();
  const card = page.locator(`[data-loom-node-id="${generated.nodeId}"]`);
  const tokenNodes = card.locator("[data-loom-token-node]");
  await expect(tokenNodes).toHaveCount(generated.tokens.length);
  await expect(card).toContainText("deterministic");

  const cardLayout = await card.evaluate((wrapper) => {
    const node = wrapper.querySelector<HTMLElement>(".node")!;
    const tokens = wrapper.querySelector<HTMLElement>(".token-field")!;
    const meta = wrapper.querySelector<HTMLElement>(".node-meta");
    const tools = wrapper.querySelector<HTMLElement>(".node-tools")!;
    const tokenRect = tokens.getBoundingClientRect();
    const nextRect = (meta ?? tools).getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    const wrapperRect = wrapper.getBoundingClientRect();
    const toolsRect = tools.getBoundingClientRect();
    return {
      tokenBottom: tokenRect.bottom,
      nextTop: nextRect.top,
      toolsBottom: toolsRect.bottom,
      nodeBottom: nodeRect.bottom,
      nodeLeft: nodeRect.left,
      nodeRight: nodeRect.right,
      wrapperLeft: wrapperRect.left,
      wrapperRight: wrapperRect.right,
      nodeOverflowX: getComputedStyle(node).overflowX,
      tokenOverflow: getComputedStyle(tokens).overflowY,
    };
  });
  expect(cardLayout.tokenBottom).toBeLessThanOrEqual(cardLayout.nextTop + 1);
  expect(cardLayout.toolsBottom).toBeLessThanOrEqual(cardLayout.nodeBottom + 1);
  expect(cardLayout.nodeLeft).toBeGreaterThanOrEqual(cardLayout.wrapperLeft - 1);
  expect(cardLayout.nodeRight).toBeLessThanOrEqual(cardLayout.wrapperRight + 1);
  expect(cardLayout.nodeOverflowX).toBe("hidden");
  expect(cardLayout.tokenOverflow).toBe("visible");

  const userCard = page.locator(`[data-loom-node-id="${generated.userId}"]`);
  const generatedHeight = await card.evaluate((element) => element.getBoundingClientRect().height);
  const userHeight = await userCard.evaluate((element) => element.getBoundingClientRect().height);
  expect(generatedHeight).toBeGreaterThan(userHeight);

  const targetIndex = Math.min(2, generated.tokens.length - 1);
  const siblingCountBefore = await page.evaluate(async ({ moduleUrl, parentId }) => {
    const stores = await import(moduleUrl);
    return (stores.loomTree.children_of.get(parentId) ?? []).length;
  }, { moduleUrl: toastModuleUrl, parentId: generated.userId });
  await tokenNodes.nth(targetIndex).click();
  const tokenTools = await openTokenDetails(page);
  await expect(tokenTools).toBeVisible();
  const branchPoint = tokenTools.getByLabel("Selected token branch point");
  await expect(branchPoint).toBeVisible();
  const drawerLayout = await Promise.all([tokenTools.boundingBox(), branchPoint.boundingBox()]);
  expect(drawerLayout[0]).not.toBeNull();
  expect(drawerLayout[1]).not.toBeNull();
  expect(drawerLayout[1]!.x).toBeGreaterThanOrEqual(drawerLayout[0]!.x - 1);
  expect(drawerLayout[1]!.x + drawerLayout[1]!.width)
    .toBeLessThanOrEqual(drawerLayout[0]!.x + drawerLayout[0]!.width + 1);
  await expect(tokenTools.getByRole("button", { name: "Replace token…" })).toBeVisible();
  await expect(tokenTools.getByRole("button", { name: "Continue from here" })).toBeEnabled();
  await expect(tokenTools.getByRole("button", { name: /geometry/i })).toBeVisible();
  await expect(tokenTools.getByRole("button", { name: /logits/i })).toHaveAttribute("aria-pressed", "true");
  await expect(tokenTools.getByRole("button", { name: /^sae/i })).toBeVisible();
  await expect(tokenTools.getByRole("button", { name: /j-lens/i })).toBeVisible();

  await tokenTools.getByRole("button", { name: "Continue from here" }).click();
  await expect(tokenTools).toHaveCount(0);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.loomTree.active_node_id;
  }, toastModuleUrl)).not.toBe(generated.nodeId);
  await expect.poll(() => page.evaluate(async ({ moduleUrl, parentId }) => {
    const stores = await import(moduleUrl);
    return (stores.loomTree.children_of.get(parentId) ?? []).length;
  }, { moduleUrl: toastModuleUrl, parentId: generated.userId })).toBe(siblingCountBefore + 1);
  const fork = await page.evaluate(async ({ moduleUrl, parentId, sourceId }) => {
    const stores = await import(moduleUrl);
    const siblings = stores.loomTree.children_of.get(parentId) ?? [];
    const forkId = siblings.find((id: string) => id !== sourceId);
    const source = stores.loomTree.nodes.get(sourceId);
    const node = forkId ? stores.loomTree.nodes.get(forkId) : null;
    return {
      sourceRawTokenIds: [...(source?.raw_token_ids ?? [])],
      forkRawTokenIds: [...(node?.raw_token_ids ?? [])],
    };
  }, {
    moduleUrl: toastModuleUrl,
    parentId: generated.userId,
    sourceId: generated.nodeId,
  });
  expect(fork.sourceRawTokenIds).toEqual(generated.rawTokenIds);
  expect(fork.forkRawTokenIds.slice(0, targetIndex + 1))
    .toEqual(generated.rawTokenIds.slice(0, targetIndex + 1));
});

test("arbitrary role labels and structural role swaps survive generation", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();
  await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    stores.samplingState.user_role = "critic";
    stores.samplingState.assistant_role = "guide";
  }, toastModuleUrl);

  await expect(page.getByRole("combobox", { name: "You write as" })).toHaveValue("critic");
  await expect(page.getByRole("combobox", { name: "Model writes as" })).toHaveValue("guide");
  await page.getByRole("button", { name: "Swap writer roles" }).click();
  await expect(page.getByRole("combobox", { name: "You write as" })).toHaveValue("guide");
  await expect(page.getByRole("combobox", { name: "Model writes as" })).toHaveValue("critic");

  const prompt = "Continue this role-swapped scene";
  await page.getByRole("textbox", { name: "Compose as guide" }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  const roles = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const authored = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!authored) throw new Error("Role-swapped authored node is missing");
    const generatedId = (stores.loomTree.children_of.get(authored.id) ?? [])[0];
    const generated = stores.loomTree.nodes.get(generatedId);
    return {
      authored: [authored.role, authored.role_label],
      generated: [generated?.role, generated?.role_label],
    };
  }, { moduleUrl: toastModuleUrl, prompt });
  expect(roles).toEqual({
    authored: ["assistant", "guide"],
    generated: ["user", "critic"],
  });
});

test("custom roles can be entered directly, configured, and used for generation", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();

  const generatedRole = page.getByRole("combobox", { name: "Model writes as" });
  await generatedRole.fill("critic");
  await expect(generatedRole).toHaveValue("critic");

  await page.getByRole("button", { name: "Role settings" }).click();
  const drawer = page.getByRole("dialog", { name: "Role settings" });
  await expect(drawer.getByText(
    "Set reusable behavior for roles already used in this conversation.",
    { exact: true },
  )).toHaveCount(0);
  await expect(drawer.getByText(
    "Choose a role to configure. Change role names directly beside the message box.",
    { exact: true },
  )).toHaveCount(0);
  await drawer.getByRole("button", { name: /critic/i }).click();
  const note = drawer.getByRole("textbox", { name: "Private note" });
  await note.fill("Reviews the response carefully");
  await drawer.getByRole("button", { name: /^assistant\b/i }).click();
  await drawer.getByRole("button", { name: /^critic\b/i }).click();
  await expect(note).toHaveValue("Reviews the response carefully");
  await expect(drawer.getByRole("status")).toHaveText("Saved");
  await page.getByRole("button", { name: "Close drawer" }).click();
  await expect(generatedRole).toHaveValue("critic");

  const prompt = "Review this as a critic";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  const generated = await page.evaluate(async ({ moduleUrl, prompt }) => {
    const stores = await import(moduleUrl);
    const authored = [...stores.loomTree.nodes.values()].find(
      (node: { text: string }) => node.text === prompt,
    );
    if (!authored) throw new Error("Custom-role authored node is missing");
    const generatedId = (stores.loomTree.children_of.get(authored.id) ?? [])[0];
    const node = stores.loomTree.nodes.get(generatedId);
    return [node?.role, node?.role_label];
  }, { moduleUrl: toastModuleUrl, prompt });
  expect(generated).toEqual(["assistant", "critic"]);
});

test("raw completion mode submits and preserves a flat base-model buffer", async ({ page }) => {
  await openFixtureWorkbench(page);
  await openWorkspace(page, "Controls");
  await page.getByRole("button", { name: "Sampling settings" }).click();
  const settings = page.getByRole("dialog", { name: "Sampling settings" });
  await settings.getByRole("button", { name: "Additional parameters" }).click();
  await settings.getByRole("button", { name: "Raw completion", exact: true }).click();
  await settings.getByRole("button", { name: "Close drawer" }).click();
  await openWorkspace(page, "Conversation");
  const buffer = page.getByLabel("Completion buffer", { exact: true });
  await expect(buffer).toBeVisible();
  const editor = buffer.getByRole("textbox", { name: "Editable completion buffer" });
  await editor.fill("Raw prefix: ");
  await buffer.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(editor).toHaveValue(`Raw prefix: ${fixtureResponse}`);
  await expect.poll(() => page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return stores.genStatus.active;
  }, toastModuleUrl)).toBe(false);

  const rawNodes = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    return [...stores.loomTree.nodes.values()].map(
      (node: { text: string; role: string }) => ({ text: node.text, role: node.role }),
    );
  }, toastModuleUrl);
  expect(rawNodes.some((node) => node.text === "Raw prefix: " && node.role === "user")).toBe(true);
  expect(rawNodes.some((node) => node.text === fixtureResponse && node.role === "assistant")).toBe(true);
});

test("online local generation never sends prompts or activations", async ({ page }) => {
  await openFixtureWorkbench(page);
  const promptSentinel = "DROWSE_PRIVATE_PROMPT_7f6fbb8b";
  const activationSentinel = "DROWSE_PRIVATE_ACTIVATION_2384d3d1";
  const requests: Array<{ url: string; body: string }> = [];
  page.on("request", (request) => {
    requests.push({ url: request.url(), body: request.postData() ?? "" });
  });

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill(`${promptSentinel} ${activationSentinel}`);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  for (const request of requests) {
    expect(decodeURIComponent(request.url)).not.toContain(promptSentinel);
    expect(decodeURIComponent(request.url)).not.toContain(activationSentinel);
    expect(request.body).not.toContain(promptSentinel);
    expect(request.body).not.toContain(activationSentinel);
  }
  expect(
    requests.filter((request) => new URL(request.url).origin !== new URL(devUrl).origin),
  ).toEqual([]);
});

test("browser transcript YAML exports and imports through the worker runtime", async ({ page }) => {
  await openFixtureWorkbench(page);
  const prompt = "Round-trip this local transcript";
  await page.getByRole("textbox", { name: /^Compose as / }).fill(prompt);
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  const drawer = await openTranscriptDrawer(page);
  await drawer.getByRole("button", { name: "preview YAML" }).click();
  const rendered = drawer.getByRole("textbox", { name: "Rendered transcript YAML" });
  await expect(rendered).toHaveValue(/drowse_transcript: 2/);
  await expect(rendered).toHaveValue(new RegExp(prompt));
  const yaml = await rendered.inputValue();

  await drawer.getByRole("tab", { name: "import" }).click();
  await drawer.getByRole("textbox", { name: "Transcript YAML to import" }).fill(yaml);
  await drawer.getByRole("button", { name: "import", exact: true }).click();
  await expect(drawer.getByText(/imported · leaf/)).toBeVisible();
  await drawer.getByRole("button", { name: "close", exact: true }).click();
  await expect(
    page.getByLabel("Conversation", { exact: true }).getByText(prompt, { exact: true }),
  ).toBeVisible();
});

test("worker-backed fixture restores its loom and generates offline without redownloading", async ({
  context,
  page,
}) => {
  await openFixtureWorkbench(page);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Persist this conversation across reload");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  const artifactRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes(".fixture.bin")) artifactRequests.push(request.url());
  });
  await openInstalledFixtureWorkbench(page);
  expect(artifactRequests).toEqual([]);
  await expect(
    page.getByLabel("Conversation", { exact: true })
      .getByText("Persist this conversation across reload", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);

  const offlineRequests: string[] = [];
  page.on("request", (request) => offlineRequests.push(request.url()));
  await context.setOffline(true);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Generate entirely offline");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  expect(offlineRequests).toEqual([]);
});

test("deleting the active model unloads it and removes its files while preserving chats and packs", async ({
  page,
}) => {
  await openFixtureWorkbench(page);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await composer.fill("Delete this model-specific conversation");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();

  await clickWorkspaceAction(page, "All tools");
  const search = page.getByRole("combobox", { name: "Filter commands" });
  await search.fill("model settings");
  await page.keyboard.press("Enter");
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  await modelControls.getByRole("button", { name: "Storage and downloads" }).click();
  const installedModels = modelControls.getByRole("region", { name: "Downloaded models" });
  await installedModels.getByRole("button", { name: "Delete", exact: true }).click();
  await installedModels.getByRole("button", { name: "Delete model", exact: true }).click();

  await page.waitForURL(`${devUrl}/app?fixture=1&choose=1`);
  await expect(page.getByRole("heading", { name: "Models", exact: true })).toBeVisible();
  await expect.poll(async () => fixtureDeletionSnapshot(page)).toEqual({
    install: null,
    session: expect.objectContaining({
      modelVariantId: "qwen3-1.7b-fixture",
      tree: expect.objectContaining({
        nodes: expect.arrayContaining([
          expect.objectContaining({ role: "user", text: "Delete this model-specific conversation" }),
        ]),
      }),
    }),
    packIds: ["fixture-jlens", "fixture-sae"],
    remaining: [...fixturePackHashes],
  });
});

test("an idle tab cooperatively transfers the exclusive GPU lock", async ({ context, page }) => {
  await openFixtureWorkbench(page);
  await expect.poll(() => page.evaluate(async () =>
    (await navigator.locks.query()).held?.some((lock) => lock.name === "drowse-webgpu-runtime") ?? false
  )).toBe(true);

  const requester = await context.newPage();
  await requester.goto(`${devUrl}/app?fixture=1`);
  await continueFromChatHome(requester);

  await expect(requester.locator(".shell")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".shell")).toHaveCount(0);
  await expect.poll(() => requester.evaluate(async () =>
    (await navigator.locks.query()).held?.some((lock) => lock.name === "drowse-webgpu-runtime") ?? false
  )).toBe(true);
});

test("exclusive GPU ownership survives repeated cooperative tab transfers", async ({
  context,
  page,
}) => {
  const pageErrors: string[] = [];
  context.on("page", (candidate) => {
    candidate.on("pageerror", (error) => pageErrors.push(error.message));
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await openFixtureWorkbench(page);

  let owner = page;
  for (let index = 0; index < 6; index += 1) {
    const requester = await context.newPage();
    await requester.goto(`${devUrl}/app?fixture=1`);
    await continueFromChatHome(requester);

    await expect(requester.locator(".shell")).toBeVisible({ timeout: 15_000 });
    await expect(owner.locator(".shell")).toHaveCount(0);
    await expect.poll(() => requester.evaluate(async () =>
      (await navigator.locks.query()).held?.filter(
        (lock) => lock.name === "drowse-webgpu-runtime",
      ).length ?? 0
    )).toBe(1);
    owner = requester;
  }

  expect(pageErrors).toEqual([]);
});

test("a busy owner explicitly approves takeover before generation is stopped", async ({
  context,
  page,
}) => {
  await openFixtureWorkbench(page, "&fixtureSlow=1");
  const requester = await context.newPage();
  await requester.goto(`${devUrl}/app?fixture=1&choose=1`);
  await expect(
    requester.getByRole("heading", { name: "Models" }),
  ).toBeVisible();
  await expect(requester.getByText("Installed and verified")).toBeVisible();

  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep the runtime busy");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeEnabled();
  await requester.getByRole("button", { name: "Open Drowse", exact: true }).click();

  const dialog = page.getByRole("alertdialog", { name: "Another tab wants to use the GPU" });
  await expect(dialog).toBeVisible();
  const keep = dialog.getByRole("button", { name: "Keep working here" });
  const release = dialog.getByRole("button", { name: "Release runtime" });
  await expect(keep).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(release).toBeFocused();
  await release.click();

  await expect(requester.locator(".shell")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".shell")).toHaveCount(0);
});

test("loom tree, node menu, and modal are fully keyboard scoped", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create loom nodes");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  await openWorkspace(page, "Branches");
  const tree = page.getByRole("tree", { name: /Conversation (loom|threads)/ });
  const items = tree.getByRole("treeitem");
  await expect(items).toHaveCount(2);
  await items.first().focus();
  await page.keyboard.press("End");
  const activeItem = items.last();
  await expect(activeItem).toBeFocused();
  await page.keyboard.press("Shift+F10");

  const menu = page.getByRole("menu", { name: "Node actions" });
  await expect(menu).toBeVisible();
  const menuItems = menu.getByRole("menuitem");
  await expect(menuItems.first()).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(menuItems.nth(1)).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(activeItem).toBeFocused();

  const visibleActions = activeItem.getByRole("button", { name: /^Actions for / });
  await visibleActions.focus();
  await page.keyboard.press("Enter");
  await expect(menu).toBeVisible();
  await expect.poll(async () => {
    const [nodeBox, menuBox] = await Promise.all([
      activeItem.boundingBox(),
      menu.boundingBox(),
    ]);
    if (!nodeBox || !menuBox) return Number.POSITIVE_INFINITY;
    const horizontalGap = Math.max(
      0,
      nodeBox.x - (menuBox.x + menuBox.width),
      menuBox.x - (nodeBox.x + nodeBox.width),
    );
    const verticalGap = Math.max(
      0,
      nodeBox.y - (menuBox.y + menuBox.height),
      menuBox.y - (nodeBox.y + nodeBox.height),
    );
    return Math.hypot(horizontalGap, verticalGap);
  }).toBeLessThanOrEqual(32);
  const menuAttachment = await Promise.all([
    activeItem.boundingBox(),
    menu.boundingBox(),
  ]);
  expect(menuAttachment[0]).not.toBeNull();
  expect(menuAttachment[1]).not.toBeNull();
  const [nodeBox, menuBox] = menuAttachment as [
    NonNullable<(typeof menuAttachment)[0]>,
    NonNullable<(typeof menuAttachment)[1]>,
  ];
  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  expect(menuBox.x).toBeGreaterThanOrEqual(8);
  expect(menuBox.x + menuBox.width).toBeLessThanOrEqual(viewport!.width - 8);
  expect(menuBox.y).toBeGreaterThanOrEqual(8);
  expect(menuBox.y + menuBox.height).toBeLessThanOrEqual(viewport!.height - 8);
  await page.keyboard.press("Escape");
  await expect(visibleActions).toBeFocused();

  await activeItem.focus();
  await page.keyboard.press("Shift+F10");
  await menu.getByRole("menuitem", { name: "Edit this turn…" }).click();
  const modal = page.getByRole("dialog", { name: "Edit turn" });
  await expect(modal).toBeVisible();
  await expect(modal.getByRole("textbox", { name: "Node text" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(activeItem).toBeFocused();
});

test("a reserved branch cannot be deleted and a rejected mutation keeps the loom visible", async ({ page }) => {
  await installFixtureTreeDeleteFailure(page);
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create a branch to protect");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  const result = await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const nodeId = stores.loomTree.active_node_id;
    if (!nodeId) throw new Error("Fixture conversation has no active Loom node");
    const before = stores.loomTree.nodes.size;

    stores.genStatus.active = true;
    stores.loomTree.pendingNodeId = nodeId;
    await stores.loomDelete(nodeId);
    const guardedToast = stores.toasts.entries.at(-1)?.message ?? null;

    stores.genStatus.active = false;
    (globalThis as typeof globalThis & { __drowseFailNextTreeDelete?: boolean })
      .__drowseFailNextTreeDelete = true;
    await stores.loomDelete(nodeId);
    stores.genStatus.active = true;
    stores.loomTree.pendingNodeId = nodeId;
    return {
      nodeId,
      before,
      after: stores.loomTree.nodes.size,
      treeError: stores.loomTree.error,
      guardedToast,
      rejectedToast: stores.toasts.entries.at(-1)?.message ?? null,
    };
  }, toastModuleUrl);
  expect(result.after).toBe(result.before);
  expect(result.treeError).toBeNull();
  expect(result.guardedToast).toBe(
    "Finish or stop the current reply before deleting this branch.",
  );
  expect(result.rejectedToast).toBe(
    "delete: Finish or stop the current reply before changing that part of the conversation.",
  );

  await openWorkspace(page, "Branches");
  await expect(page.getByText("tree unavailable", { exact: true })).toHaveCount(0);
  const pendingNode = page.locator(`[data-node-id="${result.nodeId}"]`);
  await pendingNode.getByRole("button", { name: /^Actions for / }).click();
  await expect(
    page.getByRole("menu", { name: "Node actions" })
      .getByRole("menuitem", { name: "Delete this branch…" }),
  ).toBeDisabled();

  await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    stores.genStatus.active = false;
    stores.loomTree.pendingNodeId = null;
  }, toastModuleUrl);
});

test("mobile workbench targets, bidi fields, and transcript radios remain keyboard accessible", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openFixtureWorkbench(page);

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await expect(composer).toHaveAttribute("dir", "auto");
  await composer.fill("Create mobile targets");
  await sendButton(page).click();
  const response = page.locator(".msg .response-body").last();
  await expect(response).toContainText(fixtureResponse);
  await expect(response).toHaveAttribute("dir", "auto");
  await showWorkspaceTools(page, "chat");
  const chatTargetHeights = await page.locator('.chat-header button').evaluateAll(
    (elements) => elements.map((element) => element.getBoundingClientRect().height),
  );
  expect(chatTargetHeights.length).toBeGreaterThan(0);
  expect(Math.min(...chatTargetHeights)).toBeGreaterThanOrEqual(44);
  await openWorkspace(page, "Branches");
  const treeTargetHeights = await page.getByRole("treeitem").evaluateAll(
    (elements) => elements.map((element) => element.getBoundingClientRect().height),
  );
  expect(treeTargetHeights.length).toBeGreaterThan(0);
  expect(Math.min(...treeTargetHeights)).toBeGreaterThanOrEqual(44);
  await openWorkspace(page, "Conversation");

  const drawer = await openTranscriptDrawer(page);
  await drawer.getByRole("tab", { name: "import" }).click();
  const radios = drawer.getByRole("radio");
  await expect(radios).toHaveCount(3);
  await expect(radios.nth(0)).toHaveAccessibleName("start a new path at the root");
  await expect(radios.nth(0)).toHaveAttribute("tabindex", "0");
  await expect(radios.nth(1)).toHaveAttribute("tabindex", "-1");
  await radios.nth(0).focus();
  await page.keyboard.press("ArrowDown");
  await expect(radios.nth(1)).toBeFocused();
  await expect(radios.nth(1)).toHaveAttribute("aria-checked", "true");
  await expect(radios.nth(1)).toHaveAttribute("tabindex", "0");
});

test("unsupported authoring commands are filtered before runtime requests", async ({ page }) => {
  await openFixtureWorkbench(page);
  await clickWorkspaceAction(page, "All tools");
  const paletteSearch = page.getByRole("combobox", { name: "Filter commands" });
  await paletteSearch.fill("templates");
  await expect(page.getByText(
    "No commands match “templates”. Edit or clear the search.",
    { exact: true },
  )).toBeVisible();
  await expect(page.locator('section[aria-label="Template lab"]')).toHaveCount(0);

  await paletteSearch.fill("manifold builder");
  await expect(page.getByText(
    "No commands match “manifold builder”. Edit or clear the search.",
    { exact: true },
  )).toBeVisible();
  await expect(page.locator('section[aria-label="Manifold builder"]')).toHaveCount(0);
});

test("browser Top K controls support the full vocabulary", async ({ page }) => {
  await openFixtureWorkbench(page);
  const maxTokens = page.getByRole("spinbutton", { name: "Max tokens", exact: true });
  if (!await maxTokens.isVisible()) await page.getByRole("button", { name: /^Controls/ }).click();
  await expect(maxTokens).toHaveAttribute("max", "2048");
  await maxTokens.fill("2048");
  await maxTokens.press("Tab");
  await expect(maxTokens).toHaveValue("2048");
  await page.evaluate(async (url) => (await import(url)).openDrawer("advanced_sampling"), toastModuleUrl);
  const drawer = page.getByRole("dialog", { name: "Sampling settings", exact: true });
  const topK = drawer.getByRole("spinbutton", { name: "Top K", exact: true });
  await expect(topK).toHaveAttribute("max", String(Number.MAX_SAFE_INTEGER));
  await expect(topK).toHaveAttribute("placeholder", "Default (1024)");
  await topK.fill("262144");
  await topK.press("Tab");
  await expect(topK).toHaveValue("262144");
  const alternatives = drawer.getByRole("spinbutton", { name: "Return top K", exact: true });
  await alternatives.fill("262144");
  await alternatives.press("Tab");
  await expect(alternatives).toHaveValue("262144");
  await drawer.getByRole("button", { name: "About Top K", exact: true }).click();
  await expect(drawer.getByRole("tooltip").filter({ hasText: /^Top K / })).toContainText("full model vocabulary");
});

test("captured probe readings are not relabeled as unsteered replay results", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openFixtureWorkbench(page);
  await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    await stores.attachProbe("jlens/fixture", { name: "word monitor" });
    await stores.attachProbe("sae/7", { name: "feature monitor" });
    stores.addJLensToRack("fixture");
  }, toastModuleUrl);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Inspect measurement provenance");
  await sendButton(page).click();
  await expect(page.locator(".msg .response-body").last()).toContainText(fixtureResponse);
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  // The deterministic backend does not simulate steering; stamp its synthetic
  // capture so this test can exercise the captured/counterfactual UI boundary.
  await page.evaluate(async (moduleUrl) => {
    const stores = await import(moduleUrl);
    const turn = stores.chatLog.turns.at(-1);
    for (const token of turn.tokens) {
      for (const family of ["lens", "sae"]) {
        token.measurements.instruments[family].binding.steering = turn.appliedSteering;
      }
    }
  }, toastModuleUrl);
  await page.getByRole("button", { name: "Inspect tokens in assistant message" }).click();
  const drawer = await openTokenDetails(page);
  for (const [tab, name] of [[/^j-lens\b/i, "word monitor"], [/^sae\b/i, "feature monitor"]] as const) {
    await drawer.getByRole("button", { name: tab }).click();
    await expect(drawer.locator(".inst-head .origin")).toHaveText("captured");
    await expect(drawer.getByLabel("Pinned probe readings")).toContainText(name);
    await drawer.getByRole("button", { name: "Steering on", exact: true }).click();
    await expect(drawer.locator(".inst-head .origin")).toHaveText("replayed");
    await expect(drawer.locator('[aria-description="Computed without steering so you can compare it with the steered readout."]')).toBeVisible();
    await expect(drawer.getByLabel("Pinned probe readings")).toHaveCount(0);
    await drawer.getByRole("button", { name: "Steering off", exact: true }).click();
    await expect(drawer.locator(".inst-head .origin")).toHaveText("captured");
    await expect(drawer.getByLabel("Pinned probe readings")).toContainText(name);
  }
  await drawer.getByRole("button", { name: "Close drawer" }).click();
  await openWorkspace(page, "Controls");
  const tabs = page.getByRole("group", { name: "Response guidance type" });
  for (const [tab, name] of [["J-lens", "word monitor"], ["SAE", "feature monitor"]]) {
    await tabs.getByRole("button", { name: tab, exact: true }).click();
    await page.getByRole("button", { name: `Unpin probe ${name}`, exact: true }).click();
    await expect(page.getByRole("button", { name: `Unpin probe ${name}`, exact: true })).toHaveCount(0);
    await expect.poll(() => page.evaluate(async ({ moduleUrl, name }) => {
      const stores = await import(moduleUrl);
      return stores.probeRack.active.includes(name);
    }, { moduleUrl: toastModuleUrl, name })).toBe(false);
  }
  expect(errors).toEqual([]);
});

test("token inspection has one keyboard entry point and restores focus", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create inspectable tokens");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  const inspect = page.getByRole("button", { name: "Inspect tokens in assistant message" });
  await expect(inspect).toHaveCount(1);
  const layout = page.locator(".layout");
  const workspace = page.locator(".workspace-frame");
  const workspaceX = await workspace.evaluate((element) => element.getBoundingClientRect().x);
  await inspect.focus();
  await page.keyboard.press("Enter");
  const drawer = await openTokenDetails(page);
  await expect(drawer).toBeVisible();
  await drawer.getByRole("button", { name: /^j-lens\b/i }).click();
  await expect(drawer.locator(".inst-head .origin")).toHaveText("captured");
  await expect(drawer.getByText("Preparing the token", { exact: true })).toHaveCount(0);
  await expect(drawer.getByText("AGGREGATE WORKSPACE", { exact: true })).toBeVisible();
  await expect.poll(() => layout.evaluate((element) => element.scrollLeft)).toBe(0);
  await expect.poll(() => workspace.evaluate((element) => element.getBoundingClientRect().x))
    .toBe(workspaceX);
  await drawer.getByRole("button", { name: /^sae\b/i }).click();
  await expect(drawer.locator(".inst-head .origin")).toHaveText("captured");
  await expect(drawer.getByRole("list", { name: "Top SAE features" })).toBeVisible();
  await expect(drawer.getByText("Reading model features", { exact: true })).toHaveCount(0);
  await expect(drawer.getByRole("button", { name: "Next token" })).toBeVisible();
  await drawer.getByRole("button", { name: "Next token" }).click();
  await expect.poll(() => layout.evaluate((element) => element.scrollLeft)).toBe(0);
  await expect.poll(() => workspace.evaluate((element) => element.getBoundingClientRect().x))
    .toBe(workspaceX);
  await drawer.getByRole("button", { name: /logits/ }).click();
  await expect(drawer.getByLabel("Ranked token alternatives")).toBeVisible();
  await expect(drawer.getByRole("button", { name: "Start branch" }).first()).toBeEnabled();
  await page.keyboard.press("Escape");
  await expect(inspect).toBeFocused();
  await expect.poll(() => layout.evaluate((element) => element.scrollLeft)).toBe(0);
  await expect.poll(() => workspace.evaluate((element) => element.getBoundingClientRect().x))
    .toBe(workspaceX);
});

test("the J-lens matrix scrolls with the drawer and shows cell readings without hover tips", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create a tall J-lens matrix");
  await sendButton(page).click();
  const response = page.locator(".msg .response-body").last();
  await expect(response).toContainText(fixtureResponse);
  await response.locator(".tok").first().click();

  const drawer = await openTokenDetails(page);
  const grid = drawer.locator(".grid-scroll");
  await drawer.getByRole("button", { name: /^j-lens\b/i }).click();
  await expect(grid.locator(".lens-table")).toBeVisible();
  await expect(drawer.locator("footer.drawer-footer")).toHaveCount(0);
  await expect(drawer.getByText(
    "Combines the fitted layers into one score for each token.",
    { exact: true },
  )).toHaveCount(0);
  await expect(drawer.getByText(
    "Each cell shows a token and its probability at that layer. The generated token has an outline.",
    { exact: true },
  )).toHaveCount(0);
  await expect(drawer.getByText(/Each row shows the tokens a layer was preparing/)).toHaveCount(0);
  const readoutHeader = drawer.locator(".inst-head");
  const aggregateSection = drawer.locator(".section").first();
  const sectionGap = await Promise.all([
    readoutHeader.boundingBox(),
    aggregateSection.boundingBox(),
  ]).then(([header, section]) => {
    if (!header || !section) throw new Error("J-lens sections are not visible");
    return section.y - (header.y + header.height);
  });
  expect(sectionGap).toBeGreaterThanOrEqual(16);
  await grid.evaluate((element) => {
    const body = element.querySelector("tbody");
    const rows = body ? [...body.querySelectorAll("tr")] : [];
    if (!body || rows.length === 0) throw new Error("J-lens fixture rows are missing");
    for (let index = rows.length; index < 24; index += 1) {
      body.append(rows[index % rows.length].cloneNode(true));
    }
  });

  const body = drawer.locator('aside[aria-label="Token drilldown"]');
  const containment = await grid.evaluate((element) => {
    return {
      verticalOverflow: element.scrollHeight - element.clientHeight,
    };
  });
  expect(containment.verticalOverflow).toBeLessThanOrEqual(1);
  await expect.poll(() => body.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true);

  await body.evaluate((element) => {
    element.scrollTop = Math.min(240, element.scrollHeight - element.clientHeight);
  });
  const beforeWheel = await body.evaluate((element) => element.scrollTop);
  const visibleGrid = await grid.boundingBox();
  if (!visibleGrid) throw new Error("J-lens matrix is not visible");
  await page.mouse.move(
    visibleGrid.x + Math.min(visibleGrid.width / 2, 240),
    visibleGrid.y + Math.min(visibleGrid.height / 2, 48),
  );
  await page.mouse.wheel(0, 480);
  await expect.poll(() => body.evaluate((element) => element.scrollTop)).toBeGreaterThan(beforeWheel);
  await expect.poll(() => grid.evaluate((element) => element.scrollTop)).toBe(0);

  await grid.locator(".lens-cell").last().hover();
  await expect(page.locator("#drowse-tooltip")).toHaveCount(0);
  await expect(grid.locator(".lens-cell[title]")).toHaveCount(0);
  const description = await grid.locator(".lens-cell").last().getAttribute("aria-description");
  await expect(page.locator(".focused-readout")).toHaveText(description!);
});

test("error notifications are sticky and timed notifications pause while inspected", async ({
  page,
}) => {
  await openFixtureWorkbench(page);
  const errorTtl = await page.evaluate(async (moduleUrl) => {
    const store = await import(moduleUrl);
    const id = store.pushToast("Persistent fixture error", { kind: "error" });
    return store.toasts.entries.find((toast: { id: number }) => toast.id === id)?.ttlMs;
  }, toastModuleUrl);
  expect(errorTtl).toBeNull();
  const error = page.getByRole("alert").filter({ hasText: "Persistent fixture error" });
  await expect(error).toBeVisible();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(async () => {
      const toast = (await error.boundingBox())!;
      const header = (await page.locator(".app-header").boundingBox())!;
      return toast.y >= header.y + header.height;
    }).toBe(true);
    await openWorkspaceMenu(page);
    await expect(page.getByRole("dialog", { name: "Workspace menu", exact: true })).toBeVisible();
    await page.keyboard.press("Escape");
  }
  await error.getByRole("button", { name: /Dismiss notification/ }).click();
  await expect(error).toHaveCount(0);

  await page.evaluate(async (moduleUrl) => {
    const store = await import(moduleUrl);
    store.pushToast("Pause this notification", { ttlMs: 800 });
  }, toastModuleUrl);
  const notice = page.getByRole("status").filter({ hasText: "Pause this notification" });
  await expect(notice).toBeVisible();
  await notice.hover();
  await page.waitForTimeout(1_000);
  await expect(notice).toBeVisible();
  await page.mouse.move(1, 1);
  await expect(notice).toHaveCount(0);
});

test("portable pack management remains available when browser fitting is disabled", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openFixtureWorkbench(page);

  await clickWorkspaceAction(page, "All tools");
  const paletteSearch = page.getByRole("combobox", { name: "Filter commands" });
  await paletteSearch.fill("packs");
  await page.getByRole("option", { name: /^Manage downloaded controls\b/i }).click();
  await expect(page.getByRole("dialog", { name: "Downloaded response controls" })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBe(true);
});

test("production build excludes the deterministic fixture", async () => {
  const assetsDirectory = resolve("dist-hosted", "assets");
  const scripts = (await readdir(assetsDirectory)).filter((name) => name.endsWith(".js"));
  const sources = await Promise.all(
    scripts.map((name) => readFile(resolve(assetsDirectory, name), "utf8")),
  );
  const productionJavaScript = sources.join("\n");

  expect(productionJavaScript).not.toContain("development-fixture");
  expect(productionJavaScript).not.toContain("DeterministicFakeRuntime");
  expect(productionJavaScript).not.toContain("fixtureHostedRuntime");
  expect(productionJavaScript).not.toContain("fixture.invalid");
  expect((await readdir(assetsDirectory)).filter((name) => name.endsWith(".bin"))).toEqual([]);
});

test("installed PWA shell remains available offline without preloading the model engine", async ({ context, page }) => {
  const webLlmChunks = (await readdir(resolve("dist-hosted", "assets")))
    .filter((name) => /^drowse-web-llm-[A-Za-z0-9_-]+\.js$/u.test(name));
  expect(webLlmChunks).toHaveLength(1);
  const webLlmUrl = `/assets/${webLlmChunks[0]}`;

  await page.goto("/");
  await page.evaluate(async () => {
    if (!("serviceWorker" in navigator)) throw new Error("Service workers unavailable");
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await expect
    .poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller)))
    .toBe(true);

  expect(await page.evaluate(async (url) => Boolean(await caches.match(url)), webLlmUrl)).toBe(false);

  await context.setOffline(true);
  await page.goto("/app", { waitUntil: "domcontentloaded" });

  await expect(page.getByRole("link", { name: "Drowse home" })).toBeVisible();
  await expect(page.locator("#device-check")).toBeVisible();
});

test("an installed PWA update waits for consent and reloads without clearing local data", async ({
  context,
  page,
}, testInfo) => {
  test.setTimeout(90_000);
  await page.clock.install();
  let serviceWorkerVersion = 1;
  await context.route("**/sw.js", async (route) => {
    const script = `
      const version = ${serviceWorkerVersion};
      self.addEventListener("install", (event) => {
        if (version === 1) event.waitUntil(self.skipWaiting());
      });
      self.addEventListener("activate", (event) => {
        event.waitUntil(self.clients.claim());
      });
      self.addEventListener("message", (event) => {
        if (event.data?.type === "SKIP_WAITING") {
          event.waitUntil(self.skipWaiting());
          return;
        }
      });
    `;
    await route.fulfill({
      contentType: "text/javascript; charset=utf-8",
      headers: { "Cache-Control": "no-store" },
      body: script,
    });
  });

  await page.goto("/");
  await page.evaluate(async () => {
    localStorage.setItem("drowse-pwa-upgrade-test", "retained");
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);

  serviceWorkerVersion = 2;
  await page.evaluate(async () => {
    const registration = await navigator.serviceWorker.ready;
    await registration.update();
  });

  const notice = page.getByRole("status").filter({ hasText: "A Drowse update is ready." });
  await expect(notice).toBeVisible();
  await expect(notice.getByRole("button", { name: "Update and reload" })).toBeVisible();
  await expect(notice).toHaveCSS("position", "fixed");
  await expect(notice).toHaveCSS("transform", "none");
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    const bounds = (await notice.boundingBox())!;
    expect(bounds.width).toBeLessThanOrEqual(400);
    expect(bounds.x).toBeGreaterThanOrEqual(15);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(width - 15);
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(885);
    expect(await notice.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    await expect(notice.getByRole("button", { name: "Update later" })).toBeVisible();
    await notice.screenshot({ path: testInfo.outputPath(`update-card-${width}.png`) });
  }
  await notice.getByRole("button", { name: "Update later" }).click();
  await expect(notice).toHaveCount(0);
  expect(await page.evaluate(async () => Boolean((await navigator.serviceWorker.getRegistration())?.waiting))).toBe(true);
  await page.reload();
  await expect(notice).toHaveCount(0);
  await page.clock.fastForward(3_600_000);
  await expect(notice).toBeVisible();
  await Promise.all([
    page.waitForEvent("domcontentloaded"),
    notice.getByRole("button", { name: "Update and reload" }).click(),
  ]);

  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("drowse-pwa-upgrade-test"))).toBe("retained");
  await expect.poll(() => page.evaluate(async () => {
    const registration = await navigator.serviceWorker.getRegistration();
    return {
      controlled: Boolean(navigator.serviceWorker.controller),
      active: registration?.active?.state ?? null,
      waiting: Boolean(registration?.waiting),
      installing: Boolean(registration?.installing),
    };
  })).toEqual({ controlled: true, active: "activated", waiting: false, installing: false });
});
