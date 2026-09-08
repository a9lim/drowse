import { clickWorkspaceAction } from "./workbench-navigation";
import { openWorkspaceMenu } from "./workbench-navigation";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";
import { setAppearance } from "./workbench-navigation";

const devUrl = "http://127.0.0.1:4176";
const drawerStoreUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;

const sendButton = (page: Page) => page.getByRole("button", {
  name: /^(Send|Generate reply|Add message)$/,
});

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

async function expectAccessible(page: Page, include?: string): Promise<void> {
  let builder = new AxeBuilder({ page }).withTags([
    "wcag2a",
    "wcag2aa",
    "wcag21aa",
    "wcag22aa",
  ]);
  if (include) builder = builder.include(include);
  const results = await builder.analyze();
  expect(
    results.violations.map(({ id, impact, nodes }) => ({
      id,
      impact,
      nodes: nodes.map((node) => ({
        target: node.target,
        html: node.html,
        failureSummary: node.failureSummary,
      })),
    })),
  ).toEqual([]);
}

async function openFixtureWorkbench(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
}

test("landing, onboarding, and workbench meet automated WCAG checks", async ({ page }) => {
  await page.goto(devUrl);
  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
  await expectAccessible(page);

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await expectAccessible(page);

  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expectAccessible(page);

  await setAppearance(page, "Dark");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expectAccessible(page);

  await setAppearance(page, "Light");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expectAccessible(page);
  await setAppearance(page, "Dark");

  await page.goto(devUrl);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
  await expectAccessible(page);

  await page.goto(`${devUrl}/app?fixture=1&choose=1`);
  await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expectAccessible(page);
});

test("every hosted command drawer meets automated WCAG checks", async ({ page }) => {
  await openFixtureWorkbench(page);
  await clickWorkspaceAction(page, "All tools");
  const palette = page.getByRole("dialog", { name: "Command palette" });
  await expect(palette).toBeVisible();
  await expectAccessible(page, '[role="dialog"]');

  const commands = await palette.getByRole("option").evaluateAll((options) =>
    options.filter((option) =>
      option.querySelector(".group")?.textContent !== "Controls" &&
      option.querySelector(".label")?.textContent?.trim() !== "Model settings"
    )
      .map((option) => option.querySelector(".label")?.textContent?.trim() ?? ""),
  );
  await page.keyboard.press("Escape");

  for (const command of commands) {
    await clickWorkspaceAction(page, "All tools");
    const search = page.getByRole("combobox", { name: "Filter commands" });
    await search.fill(command);
    await page.getByRole("option", { name: new RegExp(`^${command}\\b`, "i") }).click();
    const dialog = page.getByRole("dialog").last();
    await expect(dialog, command).toBeVisible();
    await expect(dialog.getByText("checking…", { exact: true })).toHaveCount(0);
    await expectAccessible(page, '[role="dialog"]');
    await page.keyboard.press("Escape");
  }
});

test("context-launched workbench drawers meet automated WCAG checks", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create inspectable context");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  const drawers: Array<{ name: string; params?: unknown }> = [
    { name: "save_conversation" },
    { name: "load_conversation" },
    { name: "system_prompt" },
    { name: "advanced_sampling" },
    { name: "subspace" },
    { name: "manifolds" },
    { name: "transcript" },
    { name: "token_drilldown", params: { turnIdx: 1, tokenIdx: 0 } },
    { name: "probe_inspector", params: {} },
    { name: "node_compare", params: { node_ids: [] } },
  ];

  for (const drawer of drawers) {
    await page.evaluate(async ({ moduleUrl, name, params }) => {
      const store = await import(moduleUrl);
      store.openDrawer(name, params);
    }, { moduleUrl: drawerStoreUrl, ...drawer });
    const dialog = page.getByRole("dialog").last();
    await expect(dialog, drawer.name).toBeVisible();
    await expectAccessible(page, '[role="dialog"]');
    await page.keyboard.press("Escape");
  }
});

test("a populated saved-chat library meets automated WCAG checks", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A saved accessibility check");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  await page.getByRole("button", { name: /^(Loom|Branches)$/ }).click();
  await openWorkspaceMenu(page);
  await page.getByRole("button", { name: "Show Loom tools", exact: true }).click();
  const loom = page.locator(".loom-sidebar");
  await loom.getByRole("button", { name: "Save", exact: true }).click();
  const saveDrawer = page.getByRole("dialog", { name: "Save chat" });
  await saveDrawer.getByRole("textbox", { name: "Name" }).fill("Accessible saved chat");
  await expectAccessible(page, '[role="dialog"]');
  await saveDrawer.getByRole("button", { name: /^(Save|Update)$/ }).click();
  await loom.getByRole("button", { name: "Open", exact: true }).click();
  const libraryDrawer = page.getByRole("dialog", { name: "Saved chats" });
  await expect(libraryDrawer.getByText("Accessible saved chat", { exact: true })).toBeVisible();
  await expectAccessible(page, '[role="dialog"]');
  await libraryDrawer.getByRole("button", { name: "Delete", exact: true }).click();
  await expectAccessible(page, '[role="dialog"]');
});
