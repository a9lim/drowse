import { expect, test } from "@playwright/test";
import recording from "../src/hosted/data/neighbor-example.json" with { type: "json" };

test("the home example shares the surrounding card container", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/");
  const example = page.getByRole("region", { name: "Recorded Drowse example" });
  const panel = page.locator(".demo-panel");
  for (const width of [320, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    for (const theme of ["dark", "light"]) {
      await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
      await panel.scrollIntoViewIfNeeded();
      const materials = await page.locator(".capabilities ol > li:first-child, .demo-panel, .model-group:first-child")
        .evaluateAll(elements => elements.map(el => {
          const style = getComputedStyle(el);
          return [style.background, style.border, style.borderRadius, style.padding,
            style.backdropFilter, style.color, style.textShadow];
        }));
      expect(materials).toHaveLength(3);
      expect(materials[1]).toEqual(materials[0]);
      expect(materials[2]).toEqual(materials[0]);
      expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await example.getByRole("button", { name: "Compare", exact: true }).click();
      await expect(example.locator(".branch-reply")).toHaveCount(3);
      await example.getByRole("button", { name: "Inspect toward welcoming reply", exact: true }).click();
      await expect(example.locator(".token-text")).toHaveText(recording.runs[5].text);
      await example.getByRole("button", { name: "Steer", exact: true }).click();
      await page.screenshot({ path: testInfo.outputPath(`example-${width}-${theme}.png`) });
    }
  }
  await page.emulateMedia({ contrast: "more" });
  await expect(panel).toHaveCSS("backdrop-filter", "none");
});

test("saved settings and every inspected step retain the actual recording", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/");
  const demo = page.getByRole("region", { name: "Recorded Drowse example" });
  const slider = demo.getByRole("slider");
  await slider.focus();
  await slider.press("Home");
  for (const [index, run] of recording.runs.entries()) {
    await expect(demo.locator(".result")).toHaveText(run.text);
    if (index < recording.runs.length - 1) await slider.press("ArrowRight");
  }
  await demo.getByRole("button", { name: "Inspect", exact: true }).click();
  const tokens = recording.runs.at(-1)!.tokens.filter(token => token.text.length > 0);
  const buttons = demo.locator(".token");
  await expect(buttons).toHaveCount(tokens.length);
  for (let index = 0; index < tokens.length; index++) {
    await buttons.nth(index).click();
    await expect(buttons.nth(index)).toHaveAttribute("aria-pressed", "true");
    const selected = tokens[index];
    const expected = Math.exp(selected.logprob) * 100;
    const probability = expected > 0 && expected < 0.1 ? "<0.1%" : `${expected.toFixed(1)}%`;
    await expect(demo.locator("tr.chosen .probability-value")).toHaveText(probability);
  }
  await expect(demo.getByRole("button", { name: "Next token", exact: true })).toBeDisabled();
  await buttons.last().focus();
  await buttons.last().press("Home");
  await expect(buttons.first()).toBeFocused();
  await expect(demo.getByRole("button", { name: "Previous token", exact: true })).toBeDisabled();
  await demo.getByText("About this recording", { exact: true }).click();
  await expect(demo.locator(".provenance")).toContainText("No model is loaded by this preset");
});
