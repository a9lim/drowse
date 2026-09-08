import { expect, type Page, type Worker } from "@playwright/test";
import { test } from "./origin-outage-fixture";
import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import ts from "typescript";

test("first-visit model setup caches the engine for a fresh worker during an origin outage", async ({ page, context, outageOrigin }) => {
  test.setTimeout(180_000);
  const assets = await readdir(resolve("dist-hosted/assets"));
  const workerName = assets.find(name => /^browser\.worker-.*\.js$/u.test(name))!;
  const engineName = assets.find(name => /^drowse-web-llm-.*\.js$/u.test(name))!;
  expect(workerName).toBeTruthy();
  expect(engineName).toBeTruthy();
  const workerUrl = new URL(`/assets/${workerName}`, outageOrigin.url).href;
  const engineUrl = new URL(`/assets/${engineName}`, outageOrigin.url).href;
  await page.goto(outageOrigin.url);
  const firstWorker = await startWorker(page);
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller)), {
    timeout: 120_000,
  }).toBe(true);
  expect(await page.evaluate(async url => Boolean(await caches.match(url)), engineUrl)).toBe(false);
  const { outputText } = ts.transpileModule(
    await readFile(resolve("src/hosted/runtime/offlineRuntimeAssets.ts"), "utf8"),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
  );
  await page.evaluate(async source => {
    const exports: { cacheOfflineRuntimeAssets?: () => Promise<void> } = {};
    new Function("exports", source)(exports);
    await exports.cacheOfflineRuntimeAssets!();
  }, outputText);

  async function startWorker(target: Page) {
    const started = target.waitForEvent("worker", { timeout: 15_000 });
    await target.evaluate(url => {
      Object.assign(window, { offlineRuntimeTestWorker: new Worker(url, { type: "module" }) });
    }, workerUrl);
    return started;
  }

  async function importEngine(worker: Worker) {
    expect(await worker.evaluate(async url => {
      const module = await import(url);
      return Object.values(module).some(value =>
        typeof (value as { MLCEngine?: unknown })?.MLCEngine === "function"
      );
    }, engineUrl)).toBe(true);
  }

  await importEngine(firstWorker);
  await expect.poll(() => page.evaluate(async url => Boolean(await caches.match(url, { ignoreVary: true })), engineUrl)).toBe(true);
  await outageOrigin.disconnect();
  const offlinePage = await context.newPage();
  await offlinePage.goto(`${outageOrigin.url}/?outage=1`, { waitUntil: "domcontentloaded" });
  await importEngine(await startWorker(offlinePage));
});
