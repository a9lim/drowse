import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const hostedHomeModule = `/@fs/${resolve("src/hosted/ui/HostedHome.svelte")}`;

test("first use chooses a model and later visits open the chat home", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);

  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "This device is ready" })).toHaveCount(0);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect.poll(() => new URL(page.url()).searchParams.get("reopen")).toBe("1");

  await page.getByRole("link", { name: "Drowse home" }).click();
  await expect(page.getByRole("heading", { name: "Your chats" })).toBeVisible();

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Your chats" })).toBeVisible();
  await expect(page.getByRole("button", { name: "New Instance", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Continue", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "This device is ready" })).toHaveCount(0);

  await page.getByRole("button", { name: "Models", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Models", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Back to chats", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Your chats" })).toBeVisible();
});

test("missing model files preserve saved chats and offer a matching re-download", async ({ page }) => {
  await page.goto(`${devUrl}/outside-the-workbench`);
  await page.evaluate(async () => {
    localStorage.setItem("drowse.entry.v1", JSON.stringify({
      version: 1,
      completedAt: Date.now() - 60_000,
      lastModelVariantId: "qwen3-1.7b-fixture",
    }));
    const samplingState = {
      temperature: 0.7,
      top_p: 0.95,
      top_k: null,
      max_tokens: 256,
      seed: null,
      system_prompt: "",
      stop_sequences: "",
      logit_bias_text: "",
      presence_penalty: 0,
      frequency_penalty: 0,
      thinking: false,
      return_top_k: 8,
      user_role: "user",
      assistant_role: "assistant",
    };
    const nodeFields = {
      role_label: null, thinking_text: null, aggregate_readings: {}, applied_steering: null,
      finish_reason: null, starred: false, notes: "", created_at: Date.now(), edited_at: null,
      edit_count: 0, mean_logprob: null, mean_surprise: null, tokens: null,
      thinking_tokens: null, raw_token_ids: null,
    };
    const snapshot = {
      version: 7,
      savedAt: "2026-09-04T12:00:00.000Z",
      model_id: "qwen3-1.7b-fixture",
      session_id: "default",
      tree: {
        tree_format: 2,
        drowse_version: "fixture",
        model_id: "qwen3-1.7b-fixture",
        session_id: "default",
        name: null,
        root_id: "root",
        active_node_id: "user-1",
        rev: 1,
        nodes: [
          { ...nodeFields, id: "root", parent_id: null, role: "system", text: "", recipe: null },
          { ...nodeFields, id: "user-1", parent_id: "root", role: "user", text: "Saved prompt", recipe: null },
        ],
        children_of: { root: ["user-1"], "user-1": [] },
        cast: {},
      },
      steerRack: [],
      subspaceAlong: 0.5,
      customSteeringExpression: null,
      probeRack: { sortMode: "name", active: [], entries: [] },
      highlightState: { target: null, compareTarget: null, compareTwo: false, smoothBlend: false },
      samplingState,
    };
    const record = {
      schemaVersion: 1,
      id: "saved-chat",
      name: "Still here",
      avatarSeed: "saved-avatar",
      modelId: "qwen3-1.7b-fixture",
      createdAt: Date.now() - 60_000,
      updatedAt: Date.now(),
      snapshot,
    };
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("drowse-saved-conversations", 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains("conversations")) {
          request.result.createObjectStore("conversations", { keyPath: "id" });
        }
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction("conversations", "readwrite");
        transaction.objectStore("conversations").put(record);
        transaction.oncomplete = () => {
          database.close();
          resolve();
        };
        transaction.onerror = () => reject(transaction.error);
      };
    });
  });

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Your chats" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "A model needs to be downloaded again" })).toBeVisible();
  await expect(page.getByText("Your saved chats are safe.")).toBeVisible();
  const card = page.locator('[data-saved-conversation="saved-chat"]');
  await expect(card).toContainText("Still here");
  await expect(card).toContainText("Download required");
  await card.getByRole("button", { name: "Download model", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Models", exact: true })).toHaveCount(0);
  await expect(page.locator(".msg").filter({ hasText: "Saved prompt" })).toBeVisible();
});

test("returning home explains unprotected storage with an actionable control", async ({ page }) => {
  await page.goto(`${devUrl}/outside-the-workbench`);
  await page.evaluate(async ({ hostedHomeModule }) => {
    const [{ default: HostedHome }, { mount }] = await Promise.all([
      import(hostedHomeModule),
      import("/@id/svelte"),
    ]);
    const snapshot = {
      phase: "supported",
      headline: "This device is ready",
      detail: "Ready",
      checks: [],
      models: [],
      download: { available: true, phase: "idle", reason: "Ready" },
      runtime: { available: true, phase: "unloaded", reason: "Ready" },
      storage: { availableBytes: 2_000_000_000, persisted: false },
    };
    const controller = {
      capabilities: () => ({ signals: { appleMobile: false } }),
      retryPersistence: async () => false,
      check: async () => {},
      open: async () => {},
    };
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(HostedHome, {
      target,
      props: {
        controller,
        snapshot,
        onChooseModels() {},
      },
    });
  }, { hostedHomeModule });

  await expect(page.getByRole("heading", { name: "Protect chats and models from browser cleanup" })).toBeVisible();
  await expect(page.getByText(/browser may remove local files when storage is low/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Protect storage" })).toBeVisible();
});

test("returning home remains usable at narrow phone widths", async ({ page }) => {
  await page.goto(`${devUrl}/outside-the-workbench`);
  await page.evaluate(() => {
    localStorage.setItem("drowse.entry.v1", JSON.stringify({
      version: 1,
      completedAt: Date.now(),
      lastModelVariantId: "qwen3-1.7b-fixture",
    }));
  });

  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto(`${devUrl}/app?fixture=1`);
    await expect(page.getByRole("heading", { name: "Your chats" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Download a model", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Manage models", exact: true })).toBeVisible();
    expect(await page.evaluate(() =>
      document.documentElement.scrollWidth <= window.innerWidth + 1
    )).toBe(true);
  }
});
