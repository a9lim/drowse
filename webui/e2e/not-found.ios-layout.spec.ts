import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import dataset from "../src/hosted/data/not-found-messages.json" with { type: "json" };

const url = "http://127.0.0.1:4177/a-page-that-is-not-here";
async function current(page: Page) {
  const id = await page.locator(".message").getAttribute("data-message-id");
  return dataset.messages.find(message => message.id === id)!;
}

test("the 404 page renders real tokens, chooses a fresh unique message, and loads no model", async ({ page }) => {
  const requests: string[] = [];
  const errors: string[] = [];
  page.on("request", request => requests.push(request.url()));
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(url);
  await expect(page.getByRole("heading", { name: "Page not found", exact: true })).toBeVisible();
  const first = await current(page);
  await expect(page.locator(".recorded-tokens")).toHaveText(first.text);
  await expect(page.getByRole("link", { name: "Back to home", exact: true })).toHaveAttribute("href", "/");
  await expect(page.getByRole("link", { name: "Open Drowse", exact: true })).toHaveAttribute("href", "/app");
  await page.reload();
  await expect(page.locator(".message")).not.toHaveAttribute("data-message-id", first.id);
  const second = await current(page);
  await page.getByRole("button", { name: "Another message", exact: true }).click();
  await expect(page.locator(".message")).not.toHaveAttribute("data-message-id", second.id);
  await expect(page.locator(".recorded-tokens")).toHaveText((await current(page)).text);
  const download = await page.request.get("/recordings/404-gemma3-4b.json");
  expect(download.ok()).toBe(true);
  expect((await download.json()).runs).toHaveLength(50);
  expect(requests.filter(request => /huggingface|browser\.worker|drowse-web-llm|\.wasm|ndarray|catalog\/v1/.test(request))).toEqual([]);
  expect(errors).toEqual([]);
});

test("token hover, touch, keyboard, and surprisal preserve the recorded probability", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(url);
  const message = await current(page);
  const piece = message.tokens.find(token => token.logprob < -0.1 && /\S/.test(token.text))!;
  const label = piece.text.replaceAll(" ", "␣").replaceAll("\n", "↵");
  const trigger = page.getByRole("button", { name: `Inspect token ${label}`, exact: true }).first();
  const panel = page.getByRole("dialog");
  if (!testInfo.project.name.includes("webkit")) {
    await trigger.hover();
    await expect(panel).toBeVisible();
    await expect(trigger).not.toBeFocused();
    await panel.hover();
    await expect(panel).toBeVisible();
    await page.mouse.move(0, 0);
    await expect(panel).toBeHidden();
  }
  await trigger.click();
  await expect(panel).toBeVisible();
  await expect(panel.locator("h2")).toHaveText(label);
  const probability = Math.exp(piece.logprob);
  await expect(panel.locator("tr.chosen td:last-child")).toHaveText(probability >= 0.001 ? probability.toFixed(3) : probability.toExponential(2));
  await expect(panel.locator("dl")).toContainText(`${(-piece.logprob / Math.LN2).toFixed(2)} bits`);
  await expect(panel.locator("dl")).toContainText(String(piece.tokenId));
  await expect(panel).toContainText("Recorded sampling probabilities, not raw logits.");
  const bounds = await panel.boundingBox();
  const viewport = page.viewportSize()!;
  expect(bounds!.x).toBeGreaterThanOrEqual(8);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width - 8);
  expect(bounds!.y).toBeGreaterThanOrEqual(8);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height - 8);
  await panel.getByRole("button", { name: "Close token probabilities" }).click();
  await expect(panel).toBeHidden();
  await expect(trigger).toBeFocused();
  await trigger.press("Enter");
  await expect(panel).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(panel).toBeHidden();
  await expect(trigger).toBeFocused();
  await trigger.press("ArrowRight");
  await expect(trigger).not.toBeFocused();
  const highlights = page.getByRole("button", { name: "Surprisal", exact: true });
  await highlights.click();
  await expect(highlights).toHaveAttribute("aria-pressed", "false");
  expect(await trigger.evaluate(element => getComputedStyle(element).backgroundColor)).toBe("rgba(0, 0, 0, 0)");
  await highlights.click();
  await expect(highlights).toHaveAttribute("aria-pressed", "true");
  expect(await trigger.evaluate(element => getComputedStyle(element).backgroundColor)).not.toBe("rgba(0, 0, 0, 0)");
});

for (const width of [320, 1440]) {
  test(`all unique messages fit at ${width}px in both themes`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(url);
    for (const theme of ["dark", "light"]) {
      await page.getByRole("button", { name: theme === "dark" ? "Dark" : "Light", exact: true }).click();
      for (const message of dataset.messages) {
        const previous = await current(page);
        if (previous.id !== message.id) {
          const candidates = dataset.messages.filter(item => item.id !== previous.id);
          const value = (candidates.findIndex(item => item.id === message.id) + 0.5) / candidates.length;
          await page.evaluate(value => { Math.random = () => value; }, value);
          await page.getByRole("button", { name: "Another message", exact: true }).click();
        }
        await expect(page.locator(".recorded-tokens")).toHaveText(message.text);
        expect(await page.locator(".message").evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      }
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({ path: testInfo.outputPath(`not-found-${width}-${theme}.png`), fullPage: true });
      const accessibility = await new AxeBuilder({ page }).include(".page-layer").analyze();
      expect(accessibility.violations).toEqual([]);
    }
  });
}

test("the close-up shader honors reduced motion and increased contrast", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(url);
  const shader = page.locator(".hero-visual");
  await expect(shader).toHaveAttribute("data-shader-status", "fallback");
  await expect(shader).toHaveAttribute("data-travel", "0.380");
  await expect(page.getByRole("button", { name: "Pause background", exact: true })).toBeHidden();
  await page.emulateMedia({ reducedMotion: "reduce", contrast: "more" });
  await expect(shader).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Page not found", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Another message", exact: true }).click();
  await expect(page.locator(".recorded-tokens")).toHaveText((await current(page)).text);
});
