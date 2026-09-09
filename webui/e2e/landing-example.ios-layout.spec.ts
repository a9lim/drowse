import { expect, test, type Locator } from "@playwright/test";
import recording from "../src/hosted/data/neighbor-example.json" with { type: "json" };

async function readableText(locator: Locator) {
  return locator.evaluate(element => {
    const copy = element.cloneNode(true) as HTMLElement;
    copy.querySelectorAll('[aria-hidden="true"]').forEach(node => node.remove());
    return copy.textContent;
  });
}

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
      await expect.poll(() => readableText(example.locator(".token-text"))).toBe(recording.runs[5].text);
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
    await expect.poll(() => readableText(demo.locator(".result"))).toBe(run.text);
    if (index < recording.runs.length - 1) await slider.press("ArrowRight");
  }
  await demo.getByRole("button", { name: "Inspect", exact: true }).click();
  const tokens = recording.runs.at(-1)!.tokens.filter(token => token.text.length > 0);
  const buttons = demo.locator(".token");
  await expect(buttons).toHaveCount(tokens.length);
  await expect(demo.locator(".mode-content")).toHaveCount(1);
  await expect.poll(() => demo.evaluate(el => Math.abs(el.getBoundingClientRect().height - el.firstElementChild!.getBoundingClientRect().height))).toBeLessThan(1);
  for (let index = 0; index < tokens.length; index++) {
    await buttons.nth(index).click();
    await expect(buttons.nth(index)).toHaveAttribute("aria-pressed", "true");
    const selected = tokens[index];
    const expected = Math.exp(selected.logprob) * 100;
    const probability = expected > 0 && expected < 0.1 ? "<0.1%" : `${expected.toFixed(1)}%`;
    await expect(demo.locator("tr.chosen .probability-value .morph-source")).toHaveText(probability);
  }
  await expect(demo.getByRole("button", { name: "Next token", exact: true })).toBeDisabled();
  await buttons.last().focus();
  await buttons.last().press("Home");
  await expect(buttons.first()).toBeFocused();
  await expect(demo.getByRole("button", { name: "Previous token", exact: true })).toBeDisabled();
  await expect(demo.getByRole("link", { name: "Recorded preset: a new neighbor" })).toHaveAttribute("download", "drowse-neighbor-recording.json");
  await expect(demo).not.toContainText(/Seven saved settings|Three saved paths|Probabilities use|curated example/);
});


test("reply motion resizes the unchanged panel and honors reduced motion", async ({ page }, testInfo) => {
  await page.goto("http://127.0.0.1:4176/");
  await page.setViewportSize({ width: 600, height: 1000 });
  const demo = page.getByRole("region", { name: "Recorded Drowse example" });
  const panel = page.locator(".demo-panel");
  await panel.scrollIntoViewIfNeeded();
  const background = await panel.evaluate(el => getComputedStyle(el).backgroundColor);
  await expect(demo.locator(".result")).toHaveCSS("font-weight", "600");
  await expect.poll(() => demo.locator(".result [data-morph-active]").count()).toBeGreaterThan(0);
  const slider = demo.getByRole("slider");
  await slider.focus();
  await slider.fill("3");
  await expect.poll(() => readableText(demo.locator(".result"))).toBe(recording.runs[3].text);
  await expect.poll(() => demo.evaluate(el => Math.abs(el.getBoundingClientRect().height - el.firstElementChild!.getBoundingClientRect().height))).toBeLessThan(1);
  const heights = await slider.evaluate(async input => {
    const demo = input.closest<HTMLElement>(".capability-demo")!;
    const heights = [demo.getBoundingClientRect().height];
    demo.style.transitionDuration = "2s";
    const started = new Promise<Animation>(resolve => {
      const capture = (event: TransitionEvent) => {
        if (event.target !== demo || event.propertyName !== "height") return;
        demo.removeEventListener("transitionrun", capture);
        const animation = demo.getAnimations().find(animation =>
          "transitionProperty" in animation && animation.transitionProperty === "height",
        )!;
        animation.pause();
        resolve(animation);
      };
      demo.addEventListener("transitionrun", capture);
    });
    (input as HTMLInputElement).value = "5";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    const animation = await started;
    const duration = Number(animation.effect!.getTiming().duration);
    for (const progress of [0.1, 0.25, 0.5, 0.75, 1]) {
      animation.currentTime = duration * progress;
      heights.push(demo.getBoundingClientRect().height);
    }
    animation.finish();
    demo.style.removeProperty("transition-duration");
    return heights;
  });
  expect(Math.abs(heights.at(-1)! - heights[0])).toBeGreaterThan(10);
  expect(new Set(heights.map(height => Math.round(height))).size).toBeGreaterThan(3);
  await expect.poll(() => readableText(demo.locator(".result"))).toBe(recording.runs[5].text);
  await expect(panel).toHaveCSS("background-color", background);
  await demo.getByRole("button", { name: "Inspect", exact: true }).click();
  await expect(demo.locator(".mode-content")).toHaveCount(1);
  const tones = await demo.locator(".token").evaluateAll(tokens => tokens.map(token => getComputedStyle(token).color));
  expect(new Set(tones).size).toBeGreaterThan(1);
  await expect.poll(() => demo.evaluate(el => Math.abs(el.getBoundingClientRect().height - el.firstElementChild!.getBoundingClientRect().height))).toBeLessThan(1);
  await page.screenshot({ path: testInfo.outputPath("demo-inspect.png"), fullPage: true });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await demo.getByRole("button", { name: "Steer", exact: true }).click();
  await expect.poll(() => demo.evaluate(el => Number.parseFloat(getComputedStyle(el).transitionDuration))).toBeLessThan(0.001);
  await expect(demo.locator("[data-morph-active]")).toHaveCount(0);
  await expect(panel).toHaveCSS("background-color", background);
});
