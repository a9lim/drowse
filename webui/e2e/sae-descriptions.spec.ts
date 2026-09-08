import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const storesUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;
const registryUrl = `/@fs${resolve("src/lib/runtime/registry.ts")}`;

async function openFeatures(page: Page, size: string, layer: number) {
  const index = JSON.parse(await readFile(resolve(`src/lib/data/sae-descriptions/gemma-3-${size}-it.json`), "utf8"));
  await page.evaluate(async ({ storesUrl, registryUrl, source, layer }) => {
    const { chatLog, openDrawer, saeSourceState } = await import(storesUrl);
    const runtime = (await import(registryUrl)).getRuntimeClient();
    const row = { source: "fixture-sae", layer, active: true, description_source: source };
    const original = runtime.instruments.sources.bind(runtime.instruments);
    runtime.instruments.sources = async (family: string) => family === "sae" ? { sources: [row] } : original(family);
    saeSourceState.sources = [row];
    chatLog.turns.at(-1).tokens[0].measurements.instruments.sae = {
      binding: { source: "fixture-sae", steering: null, layer },
      readout: { features: [0, 2286].map(id => ({ id, activation: 4.25, label: null, max_act: 10 })) },
    };
    openDrawer("token_drilldown", { turnIdx: 1, tokenIdx: 0 });
  }, { storesUrl, registryUrl, source: index.source, layer });
  const sheet = page.locator('aside[aria-label="Token drilldown"]');
  await sheet.getByRole("button", { name: /^sae\b/i }).click();
  return { sheet, index };
}

for (const [size, layer] of [["270m", 12], ["1b", 13], ["4b", 17]] as const) {
  test(`published SAE descriptions are visible for Gemma ${size} without external requests`, async ({ page }, testInfo) => {
    test.setTimeout(60_000);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.setViewportSize({ width: 1280, height: 1000 });
    const external: string[] = [];
    await page.route("https://www.neuronpedia.org/**", async route => {
      external.push(route.request().url());
      await route.abort();
    });
    await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
    await expect(page.locator(".shell")).toBeVisible();
    await page.getByRole("textbox", { name: /^Compose as / }).fill("Inspect the published feature descriptions.");
    await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    const { sheet, index } = await openFeatures(page, size, layer);
    const feature = sheet.getByRole("listitem", { name: `SAE feature 2286, layer ${layer}`, exact: true });
    await expect(feature.locator(".feature-description")).toHaveText(index.explanations["2286"][0]);
    await feature.scrollIntoViewIfNeeded();
    await expect(feature.locator(".feature-description")).toBeVisible();
    await expect(feature.getByRole("link", { name: /Neuronpedia/ })).toHaveAttribute("href", `https://www.neuronpedia.org/${index.source.model}/${index.source.source}/2286`);
    expect(external).toEqual([]);
    await page.context().setOffline(true);
    await sheet.getByRole("searchbox", { name: "Find a feature" }).fill(index.explanations["2286"][0]);
    await expect(sheet.getByRole("listitem", { name: /^SAE feature/ })).toHaveCount(1);
    for (const width of [1280, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await feature.scrollIntoViewIfNeeded();
      await expect(feature.locator(".feature-description")).toBeVisible();
      expect(await feature.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
      await sheet.screenshot({ path: testInfo.outputPath(`sae-${size}-${width}.png`) });
    }
    await page.context().setOffline(false);
  });
}

test("published SAE descriptions remain readable in the live feature panel", async ({ page }, testInfo) => {
  const index = JSON.parse(await readFile(resolve("src/lib/data/sae-descriptions/gemma-3-1b-it.json"), "utf8"));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=instruments");
  await expect(page.locator(".shell")).toBeVisible();
  await page.route("https://www.neuronpedia.org/**", route => route.abort());
  await page.evaluate(async ({ storesUrl, registryUrl, source, descriptionUrl }) => {
    const { saeState, setInspectorTab, recordSaeReadoutFrame, backfillSaeMeta } = await import(storesUrl);
    const runtime = (await import(registryUrl)).getRuntimeClient();
    const { loadSaeDescription } = await import(descriptionUrl);
    runtime.instruments.saeFeaturesMetadata = async (ids: number[]) => ({ features: Object.fromEntries(
      await Promise.all(ids.map(async id => [id, {
        label: (await loadSaeDescription(source, id, new AbortController().signal)).label, max_act: 10,
      }]))),
    });
    saeState.live = true;
    saeState.layer = 13;
    recordSaeReadoutFrame([{ id: 1, activation: 4, label: null, max_act: 10 }]);
    await backfillSaeMeta();
    setInspectorTab("sae");
  }, { storesUrl, registryUrl, source: index.source, descriptionUrl: `/@fs${resolve("src/lib/saeDescriptions.ts")}` });
  await page.getByRole("button", { name: /^Controls/ }).click();
  const panel = page.getByRole("list", { name: "SAE feature probes" });
  const label = panel.locator(".feature-description").filter({ hasText: index.explanations["1"][0] });
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await label.scrollIntoViewIfNeeded();
    await expect(label).toBeVisible();
    expect(await label.evaluate(element => {
      const style = getComputedStyle(element);
      return style.whiteSpace !== "nowrap" && style.textOverflow !== "ellipsis" && element.scrollWidth <= element.clientWidth;
    })).toBe(true);
    await panel.screenshot({ path: testInfo.outputPath(`sae-live-${width}.png`) });
  }
});
