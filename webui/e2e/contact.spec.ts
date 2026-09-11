import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const dev = "http://127.0.0.1:4176";
const endpoint = "https://drowse.ai/api/contact";

async function openFeedback(page: Page) {
  await page.goto(`${dev}/app?layoutFixture=1`);
  await expect(page.locator(".shell")).toBeVisible();
  const opener = page.getByRole("button", { name: "Submit Feedback", exact: true });
  if (!await opener.isVisible()) await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  await opener.click();
  await expect(page.getByRole("dialog", { name: "Submit Feedback", exact: true })).toBeVisible();
}

async function fillMessage(page: Page) {
  await page.getByRole("textbox", { name: "Title", exact: true }).fill("A synthetic feedback test");
  await page.getByRole("textbox", { name: "Message", exact: true }).fill("Synthetic test content. No real email is sent.");
}

test("contact email is available in public footers, feedback, help, and the workspace menu", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  const address = "contact@drowse.ai";
  for (const path of ["/", "/contact", "/credits"]) {
    await page.goto(`${dev}${path}`);
    await expect(page.getByRole("navigation", { name: "Footer", exact: true }).getByRole("link", { name: address, exact: true })).toHaveAttribute("href", `mailto:${address}`);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  }
  await openFeedback(page);
  await expect(page.getByRole("dialog", { name: "Submit Feedback", exact: true }).getByRole("link", { name: address, exact: true })).toHaveAttribute("href", `mailto:${address}`);
  await page.getByRole("button", { name: "Close feedback", exact: true }).click();
  await page.getByRole("button", { name: "Workspace menu", exact: true }).click();
  await expect(page.getByRole("navigation", { name: "About Drowse", exact: true }).getByRole("link", { name: address, exact: true })).toHaveAttribute("href", `mailto:${address}`);
  await page.getByRole("button", { name: "Help", exact: true }).click();
  await expect(page.getByRole("region", { name: "Help reference", exact: true }).getByRole("link", { name: address, exact: true })).toHaveAttribute("href", `mailto:${address}`);
});

test("feedback opens below Help, traps focus, validates, and preserves a closed draft", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openFeedback(page);
  const dialog = page.getByRole("dialog", { name: "Submit Feedback", exact: true });
  expect(await page.locator(".sidebar-links button").last().evaluate(button => button.previousElementSibling?.textContent)).toContain("Help and shortcuts");
  await expect(dialog.getByRole("link", { name: /Contact us/ })).toHaveAttribute("href", "/contact");
  await expect(page.getByRole("button", { name: "Close feedback" })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  expect(await dialog.evaluate(node => node.contains(document.activeElement))).toBe(true);
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toBeFocused();
  await expect(page.getByText("Add a short title.")).toBeVisible();
  await fillMessage(page);
  await page.getByRole("textbox", { name: /^Email/ }).fill("invalid");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("textbox", { name: /^Email/ })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(page.getByRole("button", { name: "Submit Feedback", exact: true })).toBeFocused();
  await page.getByRole("button", { name: "Submit Feedback", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
  await page.getByRole("button", { name: "Topic" }).click();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible();
});

test("offline and failed sends retain text and retry the same reference before success", async ({ page, context }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openFeedback(page);
  await fillMessage(page);
  let requests: Record<string, string>[] = [];
  await page.route(endpoint, async route => {
    const body = route.request().postDataJSON();
    requests.push(body);
    await route.fulfill({ status: requests.length === 1 ? 503 : 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify(requests.length === 1 ? { error: "unavailable" } : { status: "sent", reference: body.requestId }) });
  });
  await context.setOffline(true);
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("offline");
  expect(requests).toHaveLength(0);
  await context.setOffline(false);
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("couldn't confirm");
  await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("Synthetic test content. No real email is sent.");
  await page.getByRole("button", { name: "Check delivery", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  expect(requests).toHaveLength(2);
  expect(requests[0].requestId).toBe(requests[1].requestId);
  expect(requests[0].email).toBe("");
  expect(Object.keys(requests[0]).sort()).toEqual(["body", "email", "reason", "requestId", "source", "title", "topic", "website"]);
  await expect(page.getByText("You didn't leave an email, so we can't reply.")).toBeVisible();
  await page.getByRole("button", { name: "Done", exact: true }).click();
  await expect(page.getByRole("button", { name: "Submit Feedback", exact: true })).toBeFocused();
});

test("sending is single-flight and survives closing the popup", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openFeedback(page);
  await fillMessage(page);
  let release!: () => void;
  let count = 0;
  const pending = new Promise<void>(resolve => { release = resolve; });
  await page.route(endpoint, async route => {
    count++;
    await pending;
    await route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify({ status: "sent", reference: route.request().postDataJSON().requestId }) });
  });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("button", { name: "Sending…", exact: true })).toBeDisabled();
  await expect.poll(() => count).toBe(1);
  await page.getByRole("button", { name: "Close feedback" }).click();
  release();
  await page.getByRole("button", { name: "Submit Feedback", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  expect(count).toBe(1);
});

test("unconfirmed, rate-limited, and invalid success responses never lose the draft", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await fillMessage(page);
  const references: string[] = [];
  await page.route(endpoint, async route => {
    const reference = route.request().postDataJSON().requestId;
    references.push(reference);
    const responses = [
      { status: 200, contentType: "text/html", body: "<!doctype html><title>Pages fallback</title>" },
      { status: 202, contentType: "application/json", body: JSON.stringify({ status: "pending", reference }) },
      { status: 429, contentType: "application/json", body: JSON.stringify({ error: "rate_limited" }) },
      { status: 200, contentType: "application/json", body: JSON.stringify({ status: "sent", reference: "wrong-reference" }) },
      { status: 200, contentType: "application/json", body: JSON.stringify({ status: "sent", reference }) },
    ];
    await route.fulfill({ ...responses[references.length - 1], headers: { "Access-Control-Allow-Origin": dev } });
  });
  for (const message of ["couldn't confirm", "haven't confirmed", "Too many attempts", "couldn't confirm"]) {
    await page.getByRole("button", { name: /^(Send message|Check delivery)$/ }).click();
    await expect(page.getByRole("alert")).toContainText(message);
    await expect(page.getByRole("heading", { name: "Message sent" })).not.toBeVisible();
    await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("Synthetic test content. No real email is sent.");
  }
  await page.getByRole("button", { name: /^(Send message|Check delivery)$/ }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  expect(references).toHaveLength(5);
  expect(new Set(references).size).toBe(1);
});

test("a slow connection times out and leaves a recoverable draft", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await fillMessage(page);
  await page.clock.install();
  let release!: () => void;
  let started = false;
  const pending = new Promise<void>(resolve => { release = resolve; });
  await page.route(endpoint, async route => {
    started = true;
    await pending;
    await route.fulfill({ status: 503, contentType: "application/json", body: "{}", headers: { "Access-Control-Allow-Origin": dev } });
  });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect.poll(() => started).toBe(true);
  await page.clock.fastForward(25_001);
  await expect(page.getByRole("alert")).toContainText("Your draft is still here.");
  release();
  await expect(page.getByRole("heading", { name: "Message sent" })).not.toBeVisible();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
});

test("feedback fits a narrow phone and exposes accessible validation and clear controls", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openFeedback(page);
  const dialog = page.getByRole("dialog", { name: "Submit Feedback", exact: true });
  await fillMessage(page);
  await page.getByRole("textbox", { name: /^Email/ }).fill("invalid");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  expect(await dialog.evaluate(node => node.scrollWidth - node.clientWidth)).toBeLessThanOrEqual(1);
  for (const theme of ["dark", "light"]) {
    await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
    const audit = await new AxeBuilder({ page }).include('[role="dialog"]').analyze();
    expect(audit.violations).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath(`feedback-phone-${theme}.png`) });
  }
  await page.getByRole("button", { name: "Clear draft", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
  await page.getByRole("button", { name: "Confirm clear draft", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Workspace menu", exact: true })).toBeFocused();
});

test("topics follow the reason, preserve shared choices, and submit the matching category", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await fillMessage(page);
  const reason = page.getByRole("button", { name: "Reason", exact: true });
  const topic = page.getByRole("button", { name: "Topic", exact: true });
  async function chooseReason(label: string) {
    await reason.click();
    await page.getByRole("option", { name: label, exact: true }).click();
  }
  async function topicLabels() {
    await topic.click();
    const labels = await page.getByRole("listbox", { name: "Topic", exact: true }).getByRole("option").allTextContents();
    await page.keyboard.press("Escape");
    return labels.map(label => label.trim());
  }
  await topic.click();
  await page.getByRole("option", { name: "Models and downloads", exact: true }).click();
  await chooseReason("Help and support");
  await expect(topic).toContainText("Models and downloads");
  expect(await topicLabels()).toContain("Getting started");
  await chooseReason("Bug report");
  await expect(topic).toContainText("Models and downloads");
  expect(await topicLabels()).not.toContain("Getting started");
  await chooseReason("Feature suggestion");
  await expect(topic).toContainText("Models and downloads");
  await chooseReason("Media and press");
  await expect(topic).toContainText("Interview request");
  expect(await topicLabels()).toEqual(["Interview request", "Coverage and fact checking", "Images and press materials", "Something else"]);
  await chooseReason("Research and collaboration");
  await expect(topic).toContainText("Research collaboration");
  expect(await topicLabels()).toEqual(["Research collaboration", "Using Drowse in a study", "Methods and validation", "Sharing results", "Funding & Sponsorship", "Something else"]);
  await topic.click();
  await page.getByRole("option", { name: "Funding & Sponsorship", exact: true }).click();
  await expect(topic).toContainText("Funding & Sponsorship");
  await topic.click();
  await page.getByRole("option", { name: "Something else", exact: true }).click();
  await chooseReason("Media and press");
  await expect(topic).toContainText("Something else");
  await chooseReason("Other inquiry");
  await expect(topic).toContainText("Not applicable");
  await expect(topic).toBeDisabled();
  await chooseReason("Media and press");
  await expect(topic).toBeEnabled();
  await expect(topic).toContainText("Interview request");
  await topic.click();
  await page.getByRole("option", { name: "Images and press materials", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Title", exact: true })).toHaveValue("A synthetic feedback test");
  await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("Synthetic test content. No real email is sent.");
  await page.route(endpoint, async route => {
    const body = route.request().postDataJSON();
    expect(body.reason).toBe("media");
    expect(body.topic).toBe("press-materials");
    await route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify({ status: "sent", reference: body.requestId }) });
  });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
});

test("contact categories, success, and mobile layout are accessible", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${dev}/contact`);
  await expect(page).toHaveTitle("Contact us · Drowse");
  await page.getByRole("button", { name: "Reason" }).click();
  for (const label of ["Help and support", "Bug report", "Media and press", "Research and collaboration"]) await expect(page.getByRole("option", { name: label, exact: true })).toBeVisible();
  await page.getByRole("option", { name: "Media and press", exact: true }).click();
  await fillMessage(page);
  await page.getByRole("textbox", { name: /^Email/ }).fill("test@example.com");
  await page.route(endpoint, async route => {
    const body = route.request().postDataJSON();
    expect(body.source).toBe("contact");
    expect(body.reason).toBe("media");
    expect(body.topic).toBe("interview");
    await route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": dev }, body: JSON.stringify({ status: "sent", reference: body.requestId }) });
  });
  for (const theme of ["dark", "light"]) {
    await page.getByRole("button", { name: theme === "dark" ? "Dark" : "Light", exact: true }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
      const audit = await new AxeBuilder({ page }).include("main").analyze();
      expect(audit.violations).toEqual([]);
      if (width === 390 || width === 1440) {
        await page.screenshot({ path: testInfo.outputPath(`contact-${width}-${theme}.png`), fullPage: true });
      }
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Message sent" })).toBeVisible();
  await expect(page.getByText("We can reply to test@example.com.")).toBeVisible();
  await page.getByRole("button", { name: "Send another message", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("");
});

for (const viewportWidth of [390, 1280]) {
  test(`dropdown width stays aligned through pointer presses and scrolling at ${viewportWidth}px`, async ({ page }) => {
    await page.setViewportSize({ width: viewportWidth, height: 720 });
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto(`${dev}/contact`);
    const field = page.getByRole("button", { name: "Reason", exact: true });
    await field.click({ delay: 160 });
    const list = page.getByRole("listbox", { name: "Reason", exact: true });
    await expect(list).toHaveCSS("transform", "matrix(1, 0, 0, 1, 0, 0)");
    await expect(field).toHaveCSS("scale", "none");
    const width = (await field.boundingBox())!.width;
    const alignment = () => list.evaluate(menu => {
      const field = menu.parentElement!.querySelector("button")!.getBoundingClientRect();
      const popup = menu.getBoundingClientRect();
      return Math.max(Math.abs(popup.width - field.width), Math.abs(popup.left - field.left));
    });
    expect(await alignment()).toBeLessThanOrEqual(1);
    await page.evaluate(() => scrollBy(0, 120));
    await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(0);
    await expect.poll(alignment).toBeLessThanOrEqual(1);
    expect(Math.abs((await list.boundingBox())!.width - width)).toBeLessThanOrEqual(1);
    await list.getByRole("option", { name: "Research and collaboration", exact: true }).click();
    await expect(field).toContainText("Research and collaboration");
    await expect(field).toBeFocused();
  });
}

test("chat spacing and corners remain concentric at desktop, phone, and landscape sizes", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`${dev}/app?layoutFixture=1`);
  await expect(page.locator(".input-row")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Hello, this is a layout test.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg").first()).toBeVisible();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
  expect(await page.locator(".chat").evaluate(chat => {
    const message = chat.querySelector(".msg")!.getBoundingClientRect();
    const composer = chat.querySelector(".input-row")!.getBoundingClientRect();
    return Math.max(Math.abs(message.left - composer.left), Math.abs(message.right - composer.right));
  })).toBeLessThanOrEqual(1);
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 1024, height: 768 }, { width: 390, height: 844 }, { width: 844, height: 390 }, { width: 320, height: 568 }]) {
    await page.setViewportSize(viewport);
    await expect.poll(() => page.locator(".chat").evaluate(chat => {
      chat.scrollTop = chat.scrollHeight;
      const composer = chat.querySelector<HTMLElement>(".input-row")!;
      const box = chat.getBoundingClientRect();
      const input = composer.getBoundingClientRect();
      const outside = getComputedStyle(chat);
      const inside = getComputedStyle(composer);
      const left = input.left - box.left;
      const right = box.right - input.right;
      const bottom = box.bottom - input.bottom;
      return Math.max(Math.abs(left - right), Math.abs(left - bottom), Math.abs(parseFloat(outside.borderBottomLeftRadius) - parseFloat(inside.borderBottomLeftRadius) - left));
    })).toBeLessThanOrEqual(1);
  }
  await page.screenshot({ path: testInfo.outputPath("chat-concentric-phone.png") });
});
