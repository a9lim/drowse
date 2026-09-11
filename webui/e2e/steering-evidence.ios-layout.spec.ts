import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { resolve } from "node:path";

const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;
const registryUrl = `/@fs${resolve("src/lib/runtime/registry.ts")}`;

test("chat records steering, compares the saved recipe, and clears inherited steering", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeVisible();
  await page.evaluate(async ({ storesUrl, registryUrl }) => {
    const stores = await import(storesUrl);
    const registry = await import(registryUrl);
    const channel = await stores.ensureRuntimeChannel();
    const original = channel.send.bind(channel);
    (window as any).steeringRequests = [];
    channel.send = (payload: unknown) => { (window as any).steeringRequests.push(payload); return original(payload); };
    await registry.getRuntimeClient().tree.castPut("assistant", { steering: "0.25 fixture/calm.focused" });
    await stores.attachProbe("fixture/calm.focused");
    stores.applyCustomSteeringExpression("0.5 fixture/calm.focused");
  }, { storesUrl, registryUrl });
  const next = page.getByLabel("Steering for next reply", { exact: true });
  await expect(next).toContainText("0.5 fixture/calm.focused");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Tell me something interesting");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  const recorded = page.getByLabel("Recorded steering", { exact: true });
  await expect(recorded).toContainText("0.5 fixture/calm.focused");
  const compare = page.getByRole("button", { name: "Compare without steering", exact: true });
  await expect(compare).toBeEnabled();
  await page.evaluate(async url => (await import(url)).applyCustomSteeringExpression("0.8 fixture/calm.focused"), storesUrl);
  await expect(recorded).toContainText("0.5 fixture/calm.focused");
  await compare.click();
  await expect(page.locator(".ab-shadow").getByLabel("Recorded steering")).toHaveText("No steering");
  await expect(compare).toBeEnabled();
  const requests = await page.evaluate(() => (window as any).steeringRequests);
  const readings = page.locator(".ab-shadow .recorded-readings");
  await expect(readings).toBeVisible();
  await readings.locator("summary").click();
  await expect(readings).toContainText("fixture/calm.focused");
  expect(requests.find((request: any) => request.type === "submit").steering).toBe("0.5 fixture/calm.focused");
  const comparison = requests.find((request: any) => request.stateless);
  expect(comparison.recipe_override).toBe("unsteered");
  expect(comparison.steering).toBe("");
  expect(comparison.input).toEqual([{ role: "user", content: "Tell me something interesting", label: null }]);
  expect(comparison.parent_node_id).toBeTruthy();
  expect(await page.evaluate(async url => (await import(url)).autoRegenState.enabled, storesUrl)).toBe(false);

  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect((await new AxeBuilder({ page }).include(".chat").analyze()).violations).toEqual([]);
    await page.locator(".chat").screenshot({ path: testInfo.outputPath(`steering-comparison-${width}.png`) });
  }
  await page.getByRole("button", { name: "Hide comparison", exact: true }).click();
  await expect(page.locator(".ab-grid")).toHaveCount(0);
  await page.evaluate(async url => (await import(url)).applyCustomSteeringExpression(""), storesUrl);
  await expect(next).toContainText("No steering");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Now without steering");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(recorded.last()).toHaveText("No steering");
  expect(await page.evaluate(() => (window as any).steeringRequests.filter((request: any) => request.type === "submit").at(-1).steering)).toBe("");
  await expect(recorded.first()).toContainText("0.5 fixture/calm.focused");
  await page.getByRole("button", { name: "Edit steering", exact: true }).click();
  await expect(page.getByRole("group", { name: "Response guidance type" })).toBeVisible();
  expect(errors).toEqual([]);
});
