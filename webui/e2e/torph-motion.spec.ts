import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";
const dev = "http://127.0.0.1:4176";

async function harness(page: Page) {
  await page.goto(`${dev}/`);
  await page.evaluate(async ({ harness, svelte }) => {
    const { mount } = await import(svelte);
    const { default: Harness } = await import(harness);
    const target = document.createElement("div");
    document.body.append(target);
    document.querySelector<HTMLElement>("#app")!.hidden = true;
    mount(Harness, { target });
  }, { harness: `/@fs${resolve("e2e/fixtures/MotionHarness.svelte")}`, svelte: "/e2e/svelte-runtime.ts" });
  await expect(page.getByTestId("motion-harness")).toBeVisible();
}

test("morph text preserves current readable and selected text across rapid updates", async ({ page }) => {
  await harness(page);
  const display = page.getByTestId("display");
  await expect(display.locator(".morph-text")).toHaveAttribute("data-morph-active", "");
  for (const text of ["−0.01", "0.00", "+0.01", "9.99E-7", "1.00E-6", "10000", "Ready again"]) {
    await page.getByRole("textbox", { name: "Text to display" }).fill(text);
    await expect(display.locator(".morph-source")).toHaveText(text);
  }
  const selected = await display.evaluate(element => {
    const range = document.createRange(); range.selectNodeContents(element);
    const selection = getSelection()!; selection.removeAllRanges(); selection.addRange(range);
    return selection.toString();
  });
  expect(selected).toBe("Ready again");
  await expect(display.locator(".morph-text")).not.toHaveAttribute("data-morph-active", "");
  await page.evaluate(() => getSelection()!.removeAllRanges());
  await page.getByRole("button", { name: "New identity" }).click();
  await expect(display.locator(".morph-source")).toHaveText("Latest 10000");
  await expect(display.locator(".morph-paint")).toHaveAttribute("aria-hidden", "true");
});

test("reduced motion, offscreen text, wrapping and RTL keep exact final content", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await harness(page);
  await page.getByRole("textbox", { name: "Text to display" }).fill("هذه تجربة عربية طويلة للنص");
  await page.getByRole("button", { name: "RTL", exact: true }).click();
  await page.getByRole("button", { name: "Wrap", exact: true }).click();
  await expect(page.locator("[data-morph-active]")).toHaveCount(0);
  expect(await page.getByTestId("display").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(page.getByTestId("display").locator(".morph-text")).not.toHaveAttribute("data-morph-active", "");
  await expect(page.getByTestId("offscreen").locator(".morph-text")).not.toHaveAttribute("data-morph-active", "");
});

test("slider bubble avoids clipped containers and phone edges; native number commits stay exact", async ({ page }, info) => {
  await harness(page);
  for (const width of [320, 390, 768, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    const slider = page.getByRole("slider", { name: "Harness slider" });
    await slider.focus();
    await slider.press("End");
    await expect(slider).toHaveValue("1");
    const bubble = page.locator(".slider-bubble:popover-open");
    await expect(bubble).toBeVisible();
    expect(await bubble.evaluate(el => { const r=el.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth && r.top >= 0; })).toBe(true);
    await page.screenshot({ path: info.outputPath(`controls-${width}.png`) });
    await slider.press("Home");
    await expect(slider).toHaveValue("0");
  }
  const scrollPanel = page.getByTestId("motion-harness");
  const slider = page.getByRole("slider", { name: "Harness slider" });
  await scrollPanel.evaluate(el => { el.style.height = "320px"; el.style.overflow = "auto"; });
  await slider.evaluate(el => el.scrollIntoView({block:"center"}));
  await slider.focus(); await slider.press("ArrowRight");
  const bubble = page.locator(".slider-bubble:popover-open");
  const bubbleTop = (await bubble.boundingBox())!.y;
  await scrollPanel.evaluate(el => el.scrollTop += 20);
  await expect.poll(async () => (await bubble.boundingBox())!.y).toBeCloseTo(bubbleTop - 20, 0);
  await scrollPanel.evaluate(el => { el.style.height = ""; el.style.overflow = ""; });
  const number = page.getByRole("spinbutton", { name: "Harness number" });
  await number.fill("512"); await number.press("Enter");
  await expect(number).toHaveValue("512");
  await expect(page.getByTestId("commits")).toHaveText("1");
  await number.fill(""); await number.press("Enter");
  await expect(number).toHaveValue("");
  await expect(page.getByTestId("number").locator(".morph-source")).toHaveText("-");
});

test("layer, trace and template scrubbing selects real data", async ({ page }) => {
  await harness(page);
  const layers = page.getByRole("slider", {name:"Harness layers"});
  await layers.focus(); await layers.press("ArrowRight");
  await expect(layers).toHaveAttribute("aria-valuetext", "Layer 5 · 0.8");
  await page.getByText("Inspect trace", {exact:true}).click();
  const trace=page.getByRole("slider", {name:"Trace sample"});
  await trace.focus(); await trace.press("Home");
  await expect(page.locator(".trace-readout .morph-source").first()).toHaveText("Sample 1 / 4");
  await trace.press("ArrowRight"); await trace.press("ArrowRight");
  await expect(page.locator(".trace-readout .morph-source").last()).toHaveText("Unavailable");
  await page.getByRole("button", {name:"Preview candidate"}).click();
  await page.getByRole("option", {name:"Tuesday", exact:true}).click();
  await expect(page.locator(".template-preview mark .morph-source")).toHaveText("Tuesday");
  const nodePicker = page.getByRole("button", {name:"Inspect manifold node"});
  await nodePicker.focus(); await nodePicker.press("Enter");
  await page.getByRole("option", {name:"alert",exact:true}).click();
  await expect(page.locator(".node-readout .morph-source")).toHaveText("alert · x 1 · y 0");
});

test("landing demonstration remains labeled and usable at phone widths", async ({page}) => {
  await page.goto(`${dev}/`);
  await page.setViewportSize({width:320,height:900});
  const demo=page.getByRole("region", {name:"Illustrative Drowse example"});
  await expect(demo).toContainText("illustrative data");
  await demo.getByRole("slider", {name:"Example poetic influence"}).focus();
  await demo.getByRole("slider", {name:"Example poetic influence"}).press("End");
  await expect(demo.locator(".result .morph-source")).toHaveText("The sea wears a silver veil of moonlight.");
  await demo.getByRole("button", {name:"Compare",exact:true}).click();
  await demo.getByRole("button", {name:"B · a crowded station."}).click();
  await expect(demo.locator(".result .morph-source")).toHaveText("a crowded station.");
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1)).toBe(true);
});

test("stream reveals only new tokens once and leaves old prefixes selectable", async ({page}) => {
  await page.goto(`${dev}/`);
  const result = await page.evaluate(async url => {
    const { createTokenArrival } = await import(url);
    const stream = createTokenArrival();
    const prefix = Array.from({length: 10000}, () => ({}));
    stream.track(prefix, false, "response");
    let animations = 0;
    const node = document.createElement("span");
    node.textContent = "old prefix";
    document.body.append(node);
    node.animate = (() => { animations++; return {cancel() {}}; }) as typeof node.animate;
    stream.reveal(node, prefix[0]);
    const historical = animations;
    const fresh = Array.from({length: 24}, () => ({}));
    stream.track([...prefix, ...fresh], true, "response");
    for (const token of fresh) stream.reveal(node, token);
    const appended = animations;
    for (const token of [...prefix.slice(-2), ...fresh]) stream.reveal(node, token);
    const remounted = animations;
    const range = document.createRange(); range.selectNodeContents(node);
    getSelection()!.removeAllRanges(); getSelection()!.addRange(range);
    const selectedToken = {};
    stream.track([selectedToken], true, "thinking");
    stream.reveal(node, selectedToken);
    const selected = getSelection()!.toString();
    node.remove(); getSelection()!.removeAllRanges();
    return {historical, appended, remounted, selected};
  }, `/@fs${resolve("src/lib/tokenArrival.ts")}`);
  expect(result).toEqual({historical:0, appended:24, remounted:24, selected:"old prefix"});
});
