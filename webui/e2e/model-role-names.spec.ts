import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

test("changing the model role updates requests, messages, and saved recipes", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const messages: any[] = [];
    (window as any).__roleRequests = messages;
    const postMessage = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message, ...rest: any[]) {
      if (message?.command === "submit") messages.push(structuredClone(message.payload));
      return postMessage.call(this, message, ...rest);
    };
  });
  await page.goto("http://127.0.0.1:4176/app?fixture=1");
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("button", { name: /^Roles / }).click();
  const role = page.getByRole("combobox", { name: "Model writes as", exact: true });
  const storesUrl = `/@fs${resolve("src/lib/stores.svelte.ts")}`;
  const labels: string[] = [];
  for (const name of ["pirate", "forest_guide", "assistant"]) {
    await role.fill(name);
    await role.press("Tab");
    await page.getByRole("textbox", { name: /^Compose as / }).fill("Arrgh");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect.poll(() => page.evaluate(() => (window as any).__roleRequests.length)).toBe(labels.length + 1);
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    expect(await page.evaluate(() => (window as any).__roleRequests.at(-1).sampling.assistant_role ?? null))
      .toBe(name === "assistant" ? null : name);
    labels.push(name);
    await expect(page.locator(".msg:has(.model-avatar) .role-label")).toHaveText(labels);
    expect(await page.evaluate(async url => {
      const stores = await import(url);
      const generated = [...stores.loomTree.nodes.values()].filter((node: any) => node.role === "assistant");
      return generated.map((node: any) => [node.role_label, node.recipe.sampling.assistant_role]);
    }, storesUrl)).toEqual(labels.map(label => label === "assistant" ? [null, null] : [label, label]));
  }
});
