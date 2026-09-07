import { expect, test } from "@playwright/test";

const devUrl = "http://127.0.0.1:4176";

test("base completion colors match inspection, retain recorded readings through edits, and follow scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 1100, height: 898 });
  await page.goto(`${devUrl}/app?layoutFixture=base`);
  await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  await page.getByRole("button", { name: "Show chat tools", exact: true }).click();
  const editor = page.getByRole("textbox", { name: "Editable completion buffer" });
  const mirror = page.locator(".color-mirror");
  const model = page.getByLabel("Active model", { exact: true });
  await expect(model).toContainText("pythia-70m-base");
  await editor.fill("A field note: ");
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(mirror.locator("span[style*='background-color']").first()).toBeAttached();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const text = await editor.inputValue();
  const colors = () => mirror.locator("span[style*='background-']").evaluateAll(elements => elements.map(element => element.getAttribute("style")));
  const editColors = await colors();
  expect(editColors.length).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Inspect tokens", exact: true }).click();
  expect(await page.locator(".inspect .tok").evaluateAll(elements => elements.map(element => element.getAttribute("style")))).toEqual(editColors);
  await page.getByRole("button", { name: "Edit text", exact: true }).click();
  await editor.fill(`${text} Unsaved`);
  await expect(mirror).toHaveCount(1);
  expect(await colors()).toEqual(editColors);
  await expect(mirror.locator(".origin-draft")).toHaveText(" Unsaved");
  await expect(page.locator(".color-notice")).toHaveText("Retained tokens keep their original readings; edited text has no recorded probabilities.");
  await page.getByRole("button", { name: "Discard edit", exact: true }).click();
  await expect(editor).toHaveValue(text);
  expect(await colors()).toEqual(editColors);
  await page.getByRole("button", { name: "Compare colors", exact: true }).click();
  await expect(mirror.locator("span[style*='linear-gradient']").first()).toBeAttached();
  await page.getByRole("button", { name: "Use one color", exact: true }).click();
  await page.getByRole("button", { name: "Color completion tokens by", exact: true }).click();
  await page.getByRole("option", { name: "Sampler entropy", exact: true }).click();
  await expect(mirror.locator("span[style*='background-color']").first()).toBeAttached();
  await page.getByRole("button", { name: "Color completion tokens by", exact: true }).click();
  await page.getByRole("option", { name: "No color", exact: true }).click();
  await expect(mirror).toHaveCount(0);
  await page.getByRole("button", { name: "Color completion tokens by", exact: true }).click();
  await page.getByRole("option", { name: "Token surprisal", exact: true }).click();

  await editor.fill(Array.from({ length: 30 }, (_, index) => `Field note ${index + 1}: the marmots gathered beside the rocks.\n`).join(""));
  await page.getByRole("button", { name: "Continue text", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const expectAligned = async () => {
    await expect.poll(() => page.evaluate(() => {
      const editor = document.querySelector<HTMLTextAreaElement>("#completion-buffer")!;
      const mirror = document.querySelector<HTMLElement>(".color-mirror")!;
      const editorStyle = getComputedStyle(editor);
      const mirrorStyle = getComputedStyle(mirror);
      return {
        text: mirror.textContent === `${editor.value}\u200b`,
        scroll: Math.abs(mirror.scrollTop - editor.scrollTop) < 1,
        width: mirror.clientWidth === editor.clientWidth,
        font: mirrorStyle.font === editorStyle.font,
        padding: mirrorStyle.padding === editorStyle.padding,
      };
    })).toEqual({ text: true, scroll: true, width: true, font: true, padding: true });
  };
  await editor.focus();
  await page.keyboard.press("Control+Home");
  await expectAligned();
  await page.keyboard.press("Control+End");
  await expectAligned();
  await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  await page.getByRole("button", { name: "Hide chat tools", exact: true }).click();
  for (const viewport of [{ width: 320, height: 780 }, { width: 1440, height: 900 }]) {
    await page.setViewportSize(viewport);
    await expectAligned();
    await expect.poll(() => model.evaluate(element => {
      const model = element.getBoundingClientRect();
      const panel = element.closest(".chat-panel")!.getBoundingClientRect();
      return Math.abs(model.bottom - panel.bottom) < 1 && model.right <= innerWidth && model.bottom <= innerHeight;
    })).toBe(true);
    await expect.poll(() => page.locator(".generation-actions").evaluate(element => {
      const actions = element.getBoundingClientRect();
      const chat = element.closest(".chat")!.getBoundingClientRect();
      return actions.bottom <= chat.bottom + 1;
    })).toBe(true);
  }
});
