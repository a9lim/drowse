import { expect, test } from "@playwright/test";
import type { CatalogIndex, Glossary, NeuronDetail, Pole } from "../src/hosted/ui/neurons/data";

const dev = "http://127.0.0.1:4176";
const glossary: Glossary = {
  "下雨": ["xià yǔ", "to rain; rainy"],
  "雨水": ["yǔ shuǐ", "rainwater; rainfall"],
  "天气": ["tiān qì", "weather; climate"],
};
const overview: [string, string] = ["rain · weather · 下雨 · 雨水", "weather · 天气 · rain · 下雨"];
const positive: Pole = {
  pole: 1, score: 1, shared: [], inout: [], measuredAgree: [], w_norm: 1,
  null_share: 0, silent: false, legible: {}, detector: null, modes: [0, 0],
  reads: [
    { col: "read_S", trust: "primary", twin: false, pct: 0.9, hubs: 0, tokens: [["rain", 1, 0, 0], ["weather", 1, 1, 0], ["下雨", 1, 2, 0], ["雨水", 1, 3, 0]] },
    { col: "write_S", trust: "primary", twin: false, pct: 0.9, hubs: 0, tokens: [["weather", 1, 0, 0], ["天气", 1, 1, 0], ["rain", 1, 2, 0], ["下雨", 1, 3, 0]] },
  ],
};
const index: CatalogIndex = {
  model: "Qwen3-8B", layers: 36, neuronsPerLayer: 12288, total: 1,
  layerMeta: [{ layer: 8, band: "crossover", missing: false, count: 1 }],
  neurons: [[8, 344, "rain · 下雨", "dry", 1, 0, ...overview]],
};

for (const source of ["catalog overview", "lens fallback"] as const) {
  test(`neuron ${source} translates both summary cards and exposes their glossary`, async ({ page, browserName }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    const neuron: NeuronDetail = {
      n: 344, titles: ["rain · 下雨", "dry"],
      poles: [positive, { ...positive, pole: -1, score: 0, reads: [] }],
      ...(source === "catalog overview" ? { overview } : {}),
    };
    await page.route("**/neurons-data/index.json", route => route.fulfill({ json: index }));
    await page.route("**/neurons-data/L08.json", route => route.fulfill({ json: { layer: 8, neurons: [neuron] } }));
    await page.route("**/neurons-data/L08.zh.json", route => route.fulfill({ json: glossary }));
    await page.goto(`${dev}/neurons?layer=8&neuron=344`);

    const cards = page.locator(".overview");
    await expect(cards.getByText("Triggered by:", { exact: true })).toBeVisible();
    await expect(cards.getByText("Steers outputs:", { exact: true })).toBeVisible();
    const inputs = cards.locator("dd").nth(0);
    const outputs = cards.locator("dd").nth(1);
    const rain = inputs.getByRole("button", { name: "下雨", exact: true });
    const weather = outputs.getByRole("button", { name: "天气", exact: true });
    await rain.hover();
    const tip = page.getByRole("tooltip");
    await expect(tip).toContainText("xià yǔ");
    await expect(tip).toContainText("to rain; rainy");
    await expect(rain).toHaveAttribute("aria-describedby", await tip.getAttribute("id") ?? "");
    await page.keyboard.press("Escape");
    await expect(tip).toBeHidden();

    // Tab reaches each term in the two cards, including the output-side translation.
    await rain.focus();
    await page.keyboard.press(browserName === "webkit" ? "Alt+Tab" : "Tab");
    await expect(inputs.getByRole("button", { name: "雨水", exact: true })).toBeFocused();
    await page.keyboard.press(browserName === "webkit" ? "Alt+Tab" : "Tab");
    await expect(weather).toBeFocused();
    await expect(tip).toContainText("tiān qì");
    await expect(tip).toContainText("weather; climate");

    const translate = page.getByRole("checkbox", { name: "Translate Chinese", exact: true });
    await page.getByText("Translate Chinese", { exact: true }).click();
    await expect(translate).toBeChecked();
    await expect(inputs.locator(".morph-source")).toHaveText(["rain", "weather", "to rain", "rainwater"]);
    await expect(outputs.locator(".morph-source")).toHaveText(["weather", "weather", "rain", "to rain"]);
    const translatedWeather = outputs.getByRole("button", { name: "weather (translated from 天气)", exact: true });
    await translatedWeather.focus();
    await expect(tip).toContainText("tiān qì");
    await expect(tip).toContainText("weather; climate");

    await page.keyboard.press("Escape");
    await translatedWeather.hover();
    await expect(tip).toContainText("tiān qì");
    await expect(tip).toContainText("weather; climate");

    await page.getByText("Translate Chinese", { exact: true }).click();
    await expect(translate).not.toBeChecked();
    await expect(inputs.locator(".morph-source")).toHaveText(overview[0].split(" · "));
    await expect(outputs.locator(".morph-source")).toHaveText(overview[1].split(" · "));
    await expect(outputs.getByRole("button", { name: "天气", exact: true })).toHaveAttribute("lang", "zh");
  });
}
