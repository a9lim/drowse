import { expect, type Page } from "@playwright/test";

export async function setAppearance(page: Page, name: string): Promise<void> {
  const appearance = page.getByRole("group", { name: "Appearance", exact: true });
  await expect.poll(async () => await appearance.isVisible()
    || await page.getByRole("button", { name: "Workspace menu", exact: true }).isVisible()).toBe(true);
  const openMenu = !(await appearance.isVisible());
  if (openMenu) {
    await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  }
  await appearance.getByRole("button", { name, exact: true }).click();
  if (openMenu) await page.keyboard.press("Escape");
  await expect(page.locator("html")).toHaveAttribute("data-theme", name.toLowerCase());
}

export async function returnToChats(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  await page.getByRole("dialog", { name: "Workspace menu", exact: true })
    .getByRole("button", { name: "Your chats", exact: true }).click();
}

export async function showWorkspaceTools(page: Page, name: "chat" | "Loom"): Promise<void> {
  await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  const show = page.getByRole("button", { name: `Show ${name} tools`, exact: true });
  if (await show.isVisible()) await show.click();
  else await page.keyboard.press("Escape");
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
