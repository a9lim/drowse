import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { chooseFixtureModel, openWorkspaceMenu } from "./workbench-navigation";

const devUrl = "http://127.0.0.1:4176";

test("stale J-lens is repaired without downloading the model or losing chats", async ({ page }, info) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?fixture=1`);
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  await expect(composer).toBeVisible();
  await composer.fill("Keep this chat during pack recovery");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const navigation = page.getByRole("navigation", { name: "Workspace", exact: true });
  const models = navigation.getByRole("button", { name: "Models", exact: true });
  for (const width of [1440, 760, 680, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    if (width > 760) await expect(models).toBeVisible();
    else {
      await openWorkspaceMenu(page);
      await expect(page.getByRole("button", { name: "Models", exact: true })).toBeVisible();
      await page.keyboard.press("Escape");
    }
    expect(await navigation.evaluate(element => element.scrollWidth <= element.clientWidth), `navigation fits ${width}px`).toBe(true);
    const boxes = await navigation.getByRole("button").evaluateAll(buttons => buttons.map(button => {
      const box = button.getBoundingClientRect();
      return { left: box.left, right: box.right, top: box.top };
    }));
    for (let index = 1; index < boxes.length; index++) {
      if (Math.abs(boxes[index].top - boxes[index - 1].top) < 1) {
        expect(boxes[index].left).toBeGreaterThanOrEqual(boxes[index - 1].right - 1);
      }
    }
    await page.locator("#workspace-sidebar").screenshot({ path: info.outputPath(`models-navigation-${width}.png`) });
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await models.click();
  await expect(page.getByRole("heading", { name: "Models", exact: true })).toBeVisible();
  expect(new URL(page.url()).searchParams.get("model")).toBe("qwen3-1.7b-fixture");

  await page.evaluate(async url => {
    const { BrowserContentStore } = await import(url);
    const store = new BrowserContentStore();
    await store.initialize();
    const installs = await store.listInstalls();
    const lens = installs.find((record: any) => record.id === "fixture-jlens");
    const sae = installs.find((record: any) => record.id === "fixture-sae");
    if (!lens || !sae) throw new Error("Fixture instruments were not installed");
    await store.registerInstall({ ...lens, objectHashes: sae.objectHashes });
  }, `/@fs${resolve("src/hosted/runtime/contentStore.ts")}`);
  const requests: string[] = [];
  page.on("request", request => requests.push(request.url()));
  await page.reload();
  await expect(page.getByRole("button", { name: "Add tools and open", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Drowse", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Add tools and open", exact: true }).click();
  await expect(composer).toBeVisible();
  await expect(page.locator(".msg").filter({ hasText: "Keep this chat during pack recovery" })).toBeVisible();
  expect(requests.some(url => /(?:model|core|sae)\.fixture\.bin/.test(url))).toBe(false);
  const fitted = await page.evaluate(async url => {
    const { getRuntimeClient } = await import(url);
    return (await getRuntimeClient().sessions.get()).jlens_fitted;
  }, `/@fs${resolve("src/lib/runtime/registry.ts")}`);
  expect(fitted).toBe(true);
});

test("missing instrument actions use the guarded Models route", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  await chooseFixtureModel(page);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeVisible();
  await page.evaluate(async ({ componentUrl }) => {
    const [{ default: Recovery }, { mount }] = await Promise.all([
      import(componentUrl), import("/e2e/svelte-runtime.ts"),
    ]);
    const target = document.createElement("div");
    target.id = "pack-recovery-fixture";
    document.body.append(target);
    mount(Recovery, { target, props: { kind: "jlens" } });
  }, { componentUrl: `/@fs${resolve("src/hosted/ui/InstrumentPackRecovery.svelte")}` });
  const recovery = page.locator("#pack-recovery-fixture");
  await expect(recovery).toContainText("download or update the compatible pack");
  await page.evaluate(async url => {
    const { genStatus } = await import(url);
    genStatus.active = true;
  }, `/@fs${resolve("src/lib/stores.svelte.ts")}`);
  page.once("dialog", async dialog => {
    expect(dialog.message()).toContain("Stop the current reply and open Models");
    await dialog.dismiss();
  });
  await recovery.getByRole("button", { name: "Open Models", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async url => {
    const { genStatus } = await import(url);
    genStatus.active = false;
  }, `/@fs${resolve("src/lib/stores.svelte.ts")}`);
  await recovery.getByRole("button", { name: "Open Models", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Models", exact: true })).toBeVisible();
  expect(new URL(page.url()).searchParams.get("model")).toBe("qwen3-1.7b-fixture");
});
