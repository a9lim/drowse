import { expect, type Page } from "@playwright/test";

export async function openWorkspaceMenu(page: Page): Promise<void> {
  const menu = page.getByRole("button", { name: "Workspace menu", exact: true });
  const collapse = page.getByRole("button", { name: "Hide left sidebar", exact: true });
  await expect.poll(async () => await menu.isVisible() || await collapse.isVisible()).toBe(true);
  if (!await menu.isVisible()) await collapse.click();
  await menu.click();
}

export async function clickWorkspaceAction(page: Page, name: string): Promise<void> {
  const action = page.getByRole("button", { name, exact: true });
  await expect.poll(async () => await action.isVisible()
    || await page.getByRole("button", { name: "Workspace menu", exact: true }).isVisible()).toBe(true);
  if (!await action.isVisible()) await openWorkspaceMenu(page);
  await action.click();
}

export async function selectWorkspaceView(page: Page, name: string | RegExp): Promise<void> {
  const navigation = page.getByRole("navigation", { name: "Workspace", exact: true });
  const expand = page.getByRole("button", { name: "Show left sidebar", exact: true });
  const collapsed = await expand.isVisible();
  if (collapsed) await expand.click();
  await navigation.getByRole("button", { name, exact: typeof name === "string" }).click();
  if (collapsed) await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
}

export async function setAppearance(page: Page, name: string): Promise<void> {
  const sidebarWasOpen = await page.getByRole("button", { name: "Hide left sidebar", exact: true }).isVisible();
  const appearance = page.getByRole("group", { name: "Appearance", exact: true });
  await expect.poll(async () => await appearance.isVisible()
    || await page.getByRole("button", { name: "Workspace menu", exact: true }).isVisible()
    || await page.getByRole("button", { name: "Hide left sidebar", exact: true }).isVisible()).toBe(true);
  const openMenu = !(await appearance.isVisible());
  if (openMenu) {
    await openWorkspaceMenu(page);
  }
  if (await page.locator("html").getAttribute("data-theme") !== name.toLowerCase()) {
    await appearance.getByRole("button", { name: `Switch to ${name.toLowerCase()} theme`, exact: true }).click();
  }
  if (openMenu) await page.keyboard.press("Escape");
  if (sidebarWasOpen && await page.getByRole("button", { name: "Show left sidebar", exact: true }).isVisible()) {
    await page.getByRole("button", { name: "Show left sidebar", exact: true }).click();
  }
  await expect(page.locator("html")).toHaveAttribute("data-theme", name.toLowerCase());
}

export async function returnToChats(page: Page): Promise<void> {
  await clickWorkspaceAction(page, "Back to Chats");
}

export async function showWorkspaceTools(page: Page, name: "chat" | "Loom"): Promise<void> {
  const visibleHeader = page.locator(name === "chat" ? ".chat-header" : ".loom-header");
  if (await visibleHeader.isVisible()) return;
  const show = page.getByRole("button", { name: `Show ${name} tools`, exact: true });
  if (!await show.isVisible()) await openWorkspaceMenu(page);
  await show.click();
}

export async function openTokenDetails(page: Page) {
  const expand = page.getByRole("button", { name: "Full token details", exact: true });
  if (await expand.isVisible()) await expand.click();
  const dialog = page.getByRole("dialog", { name: "Generated word details", exact: true });
  await expect(dialog).toBeVisible();
  return dialog;
}

export async function selectLoomView(page: Page, name: string | RegExp): Promise<void> {
  const button = page.getByRole("navigation", { name: "Loom views", exact: true })
    .getByRole("button", { name });
  if (await button.isVisible()) await button.click();
  else {
    await page.getByRole("button", { name: "Loom view", exact: true }).click();
    await page.getByRole("listbox", { name: "Loom view", exact: true })
      .getByRole("option", { name }).click();
  }
}

export async function chooseFixtureModel(page: Page): Promise<void> {
  await expect(page.locator(".model-grid > button").first()).toBeVisible();
  const available = page.locator('.model-grid > button:not(:disabled):not([aria-disabled="true"])');
  if (await page.locator('.model-grid > button[aria-pressed="true"]').count() === 0
    && await available.count() > 0) {
    await available.first().click();
  }
}
