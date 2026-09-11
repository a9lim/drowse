import { expect, test, type Locator } from "@playwright/test";
import recording from "../src/hosted/data/neighbor-example.json" with { type: "json" };

async function readableText(locator: Locator) {
  return locator.evaluate(element => {
    const copy = element.cloneNode(true) as HTMLElement;
    copy.querySelectorAll('[aria-hidden="true"]').forEach(node => node.remove());
    return copy.textContent;
  });
}

test("home example labels remain clear of the slider and its value bubble", async ({ page }, testInfo) => {
  await page.goto("http://127.0.0.1:4176/");
  const demo = page.getByRole("region", { name: "Recorded Drowse example" });
  const slider = demo.getByRole("slider");
  for (const width of [320, 641, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    for (const reducedMotion of ["no-preference", "reduce"] as const) {
      await page.emulateMedia({ reducedMotion });
      for (const mode of ["Steer", "Inspect"]) {
        await demo.getByRole("button", { name: mode, exact: true }).click();
        await expect(slider).toHaveCount(1);
        await slider.press("Home");
        await slider.press("End");
        await expect(demo.locator(".alpha .morph-source")).toHaveText("Toward welcoming · α 0.15");
        await expect.poll(() => demo.evaluate(element => {
          const label = element.querySelector(".alpha")!;
          const labelBox = label.getBoundingClientRect();
          const labelText = document.createRange();
          labelText.selectNodeContents(label.querySelector(".morph-source")!);
          const bubble = element.querySelector(".slider-bubble")!;
          const slider = element.querySelector(".sk-slider")!;
          const clipped = [...label.querySelectorAll("[torph-item]")].some(item => {
            const rect = item.getBoundingClientRect();
            for (let parent = item.parentElement; parent && label.contains(parent); parent = parent.parentElement) {
              const style = getComputedStyle(parent);
              const box = parent.getBoundingClientRect();
              if (/(clip|hidden)/.test(style.overflowX) && (rect.left < box.left - 0.5 || rect.right > box.right + 0.5)) return true;
              if (/(clip|hidden)/.test(style.overflowY) && (rect.top < box.top - 0.5 || rect.bottom > box.bottom + 0.5)) return true;
            }
            return false;
          });
          return {
            unclipped: !clipped,
            fits: labelBox.left >= 0 && labelBox.right <= innerWidth,
            clearsSlider: slider.getBoundingClientRect().top - labelBox.bottom >= 8,
            clearsBubble: bubble.matches(":popover-open") && bubble.getBoundingClientRect().top >= labelText.getBoundingClientRect().bottom,
          };
        })).toEqual({ unclipped: true, fits: true, clearsSlider: true, clearsBubble: true });
      }
    }
    await page.screenshot({ path: testInfo.outputPath(`example-label-${width}.png`) });
  }
});

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
      await example.getByRole("button", { name: "Inspect", exact: true }).click();
      await expect.poll(() => readableText(example.locator(".recorded-tokens"))).toBe(recording.runs[3].text);
      await expect(example.getByRole("group", { name: "Example capability" }).getByRole("button")).toHaveText(["Steer", "Inspect"]);
      await expect(example).not.toContainText("Recorded preset: a new neighbor");
      for (const coloring of ["Surprisal", "Probability"]) {
        await example.getByRole("button", { name: coloring, exact: true }).click();
        await expect(example.getByRole("button", { name: coloring, exact: true })).toHaveAttribute("aria-pressed", "true");
        expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
        await example.locator(".recorded-token").nth(9).click();
        const popover = example.getByRole("dialog");
        await expect(popover).toBeVisible();
        const bounds = (await popover.boundingBox())!;
        expect(bounds.x).toBeGreaterThanOrEqual(8);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(width - 8);
        expect(bounds.y).toBeGreaterThanOrEqual(8);
        expect(bounds.y + bounds.height).toBeLessThanOrEqual(892);
        await page.screenshot({ path: testInfo.outputPath(`inspect-${width}-${theme}-${coloring.toLowerCase()}.png`), fullPage: true });
        await popover.getByRole("button", { name: "Close token probabilities" }).click();
        await expect(popover).toBeHidden();
      }
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
  await page.emulateMedia({ reducedMotion: "reduce" });
  await slider.focus();
  await slider.press("Home");
  for (const [index, run] of recording.runs.entries()) {
    await expect.poll(() => readableText(demo.locator(".result"))).toBe(run.text);
    if (index < recording.runs.length - 1) await slider.press("ArrowRight");
  }
  await demo.getByRole("button", { name: "Inspect", exact: true }).click();
  const tokens = recording.runs.at(-1)!.tokens.filter(token => token.text.length > 0);
  const buttons = demo.locator(".recorded-token");
  await expect(buttons).toHaveCount(tokens.length);
  await expect(demo.locator(".mode-content")).toHaveCount(1);
  await expect.poll(() => demo.evaluate(el => Math.abs(el.getBoundingClientRect().height - el.firstElementChild!.getBoundingClientRect().height))).toBeLessThan(1);
  for (let index = 0; index < tokens.length; index++) {
    await buttons.nth(index).click();
    await expect(buttons.nth(index)).toHaveAttribute("aria-expanded", "true");
    const selected = tokens[index];
    const probability = Math.exp(selected.logprob);
    const popover = demo.getByRole("dialog");
    await expect(popover.locator("tr.chosen td:last-child")).toHaveText(probability >= 0.001 ? probability.toFixed(3) : probability.toExponential(2));
    await expect(popover.locator("dl")).toContainText(`${(-selected.logprob / Math.LN2).toFixed(2)} bits`);
    await popover.getByRole("button", { name: "Close token probabilities" }).click();
    await expect(popover).toBeHidden();
    await expect(buttons.nth(index)).toBeFocused();
  }
  await buttons.last().press("Home");
  await expect(buttons.first()).toBeFocused();
  await expect(demo.locator(".distribution, .step-controls, .token-reading")).toHaveCount(0);
  await expect(demo).not.toContainText(/Seven saved settings|Three saved paths|Probabilities use|curated example/);
});


test("home token popovers support hover, pinning, keyboard, and setting changes", async ({ page }, testInfo) => {
  await page.goto("http://127.0.0.1:4176/");
  const demo = page.getByRole("region", { name: "Recorded Drowse example" });
  await demo.getByRole("button", { name: "Inspect", exact: true }).click();
  await expect(demo.locator(".mode-content")).toHaveCount(1);
  const trigger = demo.locator(".recorded-token").nth(9);
  const panel = demo.getByRole("dialog");
  if (!testInfo.project.name.includes("webkit")) {
    await trigger.hover();
    await page.waitForTimeout(500);
    await expect(panel).toBeHidden();
    await expect(panel).toBeVisible();
    await expect(trigger).not.toBeFocused();
    await panel.hover();
    await expect(panel).toBeVisible();
    await page.mouse.move(0, 0);
    await expect(panel).toBeHidden();
  }
  await trigger.click();
  await expect(panel).toBeVisible();
  await page.mouse.move(0, 0);
  await expect(panel).toBeVisible();
  await panel.getByRole("button", { name: "Close token probabilities" }).click();
  await expect(trigger).toBeFocused();
  await trigger.press("Enter");
  await expect(panel).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(panel).toBeHidden();
  await expect(trigger).toBeFocused();
  await trigger.press("ArrowRight");
  await expect(demo.locator(".recorded-token").nth(10)).toBeFocused();
  await trigger.click();
  await demo.getByRole("slider").press("End");
  await expect(panel).toBeHidden();
  await expect.poll(() => readableText(demo.locator(".recorded-tokens"))).toBe(recording.runs.at(-1)!.text);
  await demo.locator(".recorded-token").first().click();
  await demo.getByRole("button", { name: "Steer", exact: true }).click();
  await expect(panel).toHaveCount(0);
});

test("Inspect morphs token text when steering changes and honors reduced motion", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/");
  await page.setViewportSize({ width: 600, height: 1000 });
  const demo = page.getByRole("region", { name: "Recorded Drowse example" });
  await demo.getByRole("button", { name: "Inspect", exact: true }).click();
  await expect(demo.locator(".mode-content")).toHaveCount(1);
  const reply = demo.locator(".recorded-tokens");
  await reply.scrollIntoViewIfNeeded();
  await expect.poll(() => reply.locator("[data-morph-active]").count()).toBeGreaterThan(0);
  const first = await reply.locator(".recorded-token").first().elementHandle();
  await demo.getByRole("slider").fill("5");
  await expect.poll(() => readableText(reply)).toBe(recording.runs[5].text);
  expect(await first!.evaluate(element => element.isConnected)).toBe(true);
  await expect.poll(() => reply.locator(".morph-paint").evaluateAll(elements => elements.flatMap(element => element.getAnimations({ subtree: true })).filter(animation => animation.playState === "running").length)).toBeGreaterThan(0);
  await expect.poll(() => reply.locator("[data-morph-active]").count()).toBeGreaterThan(0);
  await reply.locator(".recorded-token").nth(9).click();
  await expect(demo.getByRole("dialog")).toBeVisible();
  await expect(demo.getByRole("dialog").locator("dl")).toContainText(String(recording.runs[5].tokens[9].tokenId));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await demo.getByRole("slider").fill("6");
  await expect(demo.getByRole("dialog")).toBeHidden();
  await expect.poll(() => readableText(reply)).toBe(recording.runs[6].text);
  await expect(reply.locator("[data-morph-active]")).toHaveCount(0);
  expect(await reply.evaluate(element => element.getAnimations({ subtree: true }).filter(animation => animation.playState === "running").length)).toBe(0);
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
  const tones = await demo.locator(".recorded-token").evaluateAll(tokens => tokens.map(token => getComputedStyle(token).backgroundColor));
  expect(new Set(tones).size).toBeGreaterThan(1);
  const inks = await demo.locator(".recorded-token").evaluateAll(tokens => tokens.map(token => getComputedStyle(token).color));
  expect(new Set(inks).size).toBe(1);
  await expect.poll(() => demo.evaluate(el => Math.abs(el.getBoundingClientRect().height - el.firstElementChild!.getBoundingClientRect().height))).toBeLessThan(1);
  await page.screenshot({ path: testInfo.outputPath("demo-inspect.png"), fullPage: true });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await demo.getByRole("button", { name: "Steer", exact: true }).click();
  await expect.poll(() => demo.evaluate(el => Number.parseFloat(getComputedStyle(el).transitionDuration))).toBeLessThan(0.001);
  await expect(demo.locator("[data-morph-active]")).toHaveCount(0);
  await expect(panel).toHaveCSS("background-color", background);
});
