import { chooseFixtureModel } from "./workbench-navigation";
import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const drawerStoreUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;

async function openFixtureWorkbench(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
}

async function openManifoldBuilder(page: Page) {
  await page.evaluate(async (moduleUrl) => {
    const store = await import(moduleUrl);
    store.drawerState.open = "manifold_builder";
    store.drawerState.params = null;
  }, drawerStoreUrl);
  const dialog = page.getByRole("dialog", { name: "Create a concept or scale" });
  await expect(dialog).toBeVisible();
  return dialog;
}

test("custom manifold validation waits for submit and focuses each blocking field", async ({ page }) => {
  await openFixtureWorkbench(page);
  const dialog = await openManifoldBuilder(page);
  const build = dialog.getByRole("button", { name: "build", exact: true });
  const name = dialog.getByLabel("name *");

  await expect(build).toBeEnabled();
  await expect(dialog.getByText(/Check these details before you build/i)).toHaveCount(0);

  await build.click();
  await expect(name).toBeFocused();
  await expect(dialog.getByText(/Check these details before you build/i)).toBeVisible();

  await name.fill("friendly_scale");
  const high = dialog.getByRole("spinbutton", { name: "x high value" });
  await high.fill("0");
  await build.click();
  await expect(high).toBeFocused();
  await expect(high).toHaveAttribute("aria-invalid", "true");
  await expect(dialog.getByText("The high value must be greater than the low value.")).toBeVisible();

  await high.fill("1");
  await build.click();
  const addNode = dialog.getByRole("button", { name: "+ add node", exact: true });
  await expect(addNode).toBeFocused();
  await expect(dialog.getByText("Add 5 more nodes.")).toBeVisible();

  for (let index = 0; index < 5; index += 1) await addNode.click();
  await build.click();

  const firstStatements = dialog.getByRole("textbox", { name: "Statements for node node_1" });
  await expect(firstStatements).toBeFocused();
  await expect(firstStatements).toHaveAttribute("aria-invalid", "true");
  await expect(dialog.getByText("Add at least one example statement.").first()).toBeVisible();
});

test("hosted production omits unvalidated automatic topology controls", async ({ page }) => {
  await openFixtureWorkbench(page);
  const dialog = await openManifoldBuilder(page);
  await dialog.getByRole("checkbox", { name: "auto-domain" }).check();

  await expect(dialog.getByRole("radiogroup", { name: "Fit method" })).toHaveCount(0);
  await expect(dialog.getByText(
    "linear PCA · automatic curved-shape detection is not included in the hosted release",
  )).toBeVisible();
});
