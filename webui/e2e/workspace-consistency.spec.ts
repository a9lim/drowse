import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const moduleUrl = (path: string) => `/@fs${resolve(`src/${path}`)}`;
async function theme(page: Page, value: string) {
  await page.evaluate(async ({ url, value }) => (await import(url)).setTheme(value), { url: moduleUrl("lib/theme.ts"), value });
  await expect(page.locator("html")).toHaveAttribute("data-theme", value);
}
async function mountChats(page: Page) {
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async urls => {
    const [{ default: Home }, { mount }, { conversationLibrary, registerConversationAutosave }, { sessionState }, { captureConversationSnapshot }] = await Promise.all([
      import(urls.home), import("/@id/svelte"), import(urls.saved), import(urls.stores), import(urls.workspace),
    ]);
    const snapshot = captureConversationSnapshot();
    for (const name of ["Marmot notes", "A much longer conversation name to check the library's shared alignment"]) {
      await conversationLibrary.create({ name, snapshot });
    }
    registerConversationAutosave(async () => {});
    const modelId = sessionState.info.model_id;
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(Home, { target, props: {
      controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {}, retryPersistence: async () => false },
      snapshot: { phase: "supported", models: [{ id: modelId, modelId, name: "Pythia 70M", setupComplete: true, fit: "recommended" }], download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: false } },
      onChooseModels() {},
    } });
  }, { home: moduleUrl("hosted/ui/HostedHome.svelte"), saved: moduleUrl("lib/stores/savedConversations.svelte.ts"), stores: moduleUrl("lib/stores.svelte.ts"), workspace: moduleUrl("lib/conversationWorkspace.ts") });
  await expect(page.locator(".chat-card")).toHaveCount(2);
}

test("chat colors float in one horizontal row without changing card layout", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mountChats(page);
  const cards = page.locator(".chat-card");
  const trigger = cards.first().getByRole("button", { name: /^Color / });
  const popup = page.locator(".accent-popover");
  for (const mode of ["light", "dark"]) {
    await theme(page, mode);
    for (const width of [1440, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await trigger.scrollIntoViewIfNeeded();
      const before = await cards.evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().toJSON()));
      await trigger.click();
      await expect(popup.locator(".accent-surface")).toHaveCSS("opacity", "1");
      await expect(popup.locator(".accent-surface")).toHaveCSS("transition-duration", "0s");
      expect(await popup.evaluate(el => el.matches(":popover-open"))).toBe(true);
      expect(await cards.evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().toJSON()))).toEqual(before);
      const positions = await popup.locator(".color-option").evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().toJSON()));
      expect(positions).toHaveLength(6);
      expect(new Set(positions.map(position => position.y)).size).toBe(1);
      await expect.poll(() => popup.evaluate(el => el.getBoundingClientRect().top - el.parentElement!.querySelector("button")!.getBoundingClientRect().bottom)).toBeCloseTo(8, 0);
      const box = (await popup.boundingBox())!;
      const anchor = (await trigger.boundingBox())!;
      expect(box.y).toBeCloseTo(anchor.y + anchor.height + 8, 0);
      expect(box.x).toBeGreaterThanOrEqual(7.9);
      expect(box.x + box.width).toBeLessThanOrEqual(width - 7.9);
      expect(await popup.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`color-popover-${mode}-${width}.png`), fullPage: true });
      await page.keyboard.press("Escape");
      await expect(popup).toHaveCount(0);
      await expect(trigger).toBeFocused();
    }
  }
  await trigger.click();
  await popup.getByRole("radio", { name: "Rose", exact: true }).check();
  await expect(trigger).toHaveText(/Rose/);
  await expect(popup.getByRole("radio", { name: "Rose", exact: true })).toBeChecked();
  await expect.poll(() => page.evaluate(async url => (await (await import(url)).conversationLibrary.list()).conversations.some((record: any) => record.accent === "rose"), moduleUrl("lib/stores/savedConversations.svelte.ts"))).toBe(true);
  await page.keyboard.press("Escape");
  await trigger.focus();
  await page.keyboard.press("Enter");
  await expect(popup.getByRole("radio", { name: "Rose", exact: true })).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await expect(popup.getByRole("radio", { name: "Peach", exact: true })).toBeChecked();
  await page.keyboard.press("Tab");
  await expect(popup).toHaveCount(0);
  await trigger.click();
  await page.getByRole("heading", { name: "Saved chats", exact: true }).click();
  await expect(popup).toHaveCount(0);
  await trigger.click();
  await cards.nth(1).getByRole("button", { name: /^Color / }).click();
  await expect(page.getByRole("dialog", { name: "Chat accent color" })).toHaveCount(1);
});

test("color popovers fade both ways and can reopen during dismissal", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await mountChats(page);
  const trigger = page.locator(".chat-card").first().getByRole("button", { name: /^Color / });
  const popup = page.locator(".accent-popover");
  const surface = popup.locator(".accent-surface");
  await trigger.click();
  await expect(surface).toHaveClass(/is-open/);
  await expect(surface).toHaveCSS("transition-duration", "0.25s, 0.25s");
  await expect(surface).toHaveCSS("opacity", "1");
  const closing = await trigger.evaluate(el => {
    (el as HTMLButtonElement).click();
    const popup = document.querySelector<HTMLElement>(".accent-popover")!;
    const surface = popup.querySelector<HTMLElement>(".accent-surface")!;
    return { closing: surface.classList.contains("is-closing"), inert: surface.inert, inTopLayer: popup.matches(":popover-open"), duration: getComputedStyle(surface).transitionDuration };
  });
  expect(closing).toEqual({ closing: true, inert: true, inTopLayer: true, duration: "0.15s, 0.15s" });
  await trigger.evaluate(el => (el as HTMLButtonElement).click());
  await expect(surface).toHaveClass(/is-open/);
  await expect(surface).toHaveCSS("opacity", "1");
  expect(await popup.evaluate(el => (el as HTMLElement).inert)).toBe(false);
  await page.keyboard.press("Escape");
  await expect(popup).toHaveCount(0);
  await expect(trigger).toBeFocused();
});

test("chat settings color popover escapes the drawer and Escape only dismisses the picker", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async url => (await import(url)).openDrawer("save_conversation"), moduleUrl("lib/stores/drawers.svelte.ts"));
  const trigger = page.locator(".drawer .accent-picker button");
  await expect(trigger).toBeEnabled();
  await trigger.click();
  const popup = page.locator(".accent-popover");
  await expect(popup).toHaveCSS("opacity", "1");
  expect(await popup.evaluate(el => el.matches(":popover-open"))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(popup).toHaveCount(0);
  await expect(trigger).toBeVisible();
  await expect(trigger).toBeFocused();
});

test("models and saved chats share large headings with roomier library cards and flat materials", async ({ page }, testInfo) => {
  test.setTimeout(60_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const surface of ["models", "chats"]) {
    if (surface === "models") {
      await page.goto("http://127.0.0.1:4176/app?layoutFixture=setup");
      await expect(page.locator(".model-grid > button").first()).toBeVisible();
    } else await mountChats(page);
    const card = page.locator(surface === "models" ? ".model-grid > button:not(.selected)" : ".chat-card").first();
    for (const value of ["dark", "light"]) {
      await theme(page, value);
      for (const width of [1440, 320]) {
        await page.setViewportSize({ width, height: 1000 });
        await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
        await expect(card).toHaveCSS("border-radius", "8px");
        if (surface === "models") {
          await expect(card).toHaveCSS("padding", "24px");
          if (width > 760) await expect(card).toHaveCSS("min-height", "100px");
        } else {
          await expect(card).toHaveCSS("padding", "16px");
          await expect(card).toHaveCSS("box-shadow", "none");
          await expect(card).toHaveCSS("background-image", "none");
          await expect(page.getByRole("button", { name: "New Instance", exact: true })).toBeVisible();
        }
        await expect(card).toHaveCSS("backdrop-filter", "none");
        const fit = await page.evaluate(() => {
          const main = document.querySelector("main")!;
          const heading = main.querySelector("h1")!;
          const header = document.querySelector(".page-header")!;
          const mainStyle = getComputedStyle(main);
          return {
            pageFits: document.documentElement.scrollWidth <= window.innerWidth,
            headerFits: header.scrollWidth <= header.clientWidth,
            inset: parseFloat(mainStyle.paddingInlineStart),
            headingInset: heading.getBoundingClientRect().left - main.getBoundingClientRect().left,
            headingSize: parseFloat(getComputedStyle(heading).fontSize),
          };
        });
        expect(fit.pageFits).toBe(true);
        expect(fit.headerFits).toBe(true);
        const inset = surface === "models" ? 24 : 16;
        expect(fit.inset).toBe(inset);
        expect(fit.headingInset).toBeCloseTo(inset, 0);
        expect(fit.headingSize).toBeGreaterThanOrEqual(24);
        if (surface === "chats") {
          expect(await page.getByRole("heading", { name: "Saved chats", exact: true }).evaluate(el => parseFloat(getComputedStyle(el).fontSize))).toBe(fit.headingSize);
          const notice = page.locator(".storage-notice");
          await expect(notice).toBeVisible();
          expect(await notice.locator("h2").evaluate(el => parseFloat(getComputedStyle(el).fontSize))).toBe(fit.headingSize);
          expect(await notice.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
        }
        expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
        await page.screenshot({ path: testInfo.outputPath(`${surface}-${value}-${width}.png`), fullPage: true });
      }
    }
    if (surface === "models") {
      const index = await card.evaluate(el => Array.from(el.parentElement!.children).indexOf(el));
      const selection = page.locator(".model-grid > button").nth(index);
      await selection.click();
      await expect(selection).toHaveAttribute("aria-pressed", "true");
      await page.keyboard.press("Tab");
      await selection.focus();
      await expect.poll(() => selection.evaluate(el => getComputedStyle(el).outlineStyle)).not.toBe("none");
    }
  }
});

test("chat and text completion share a subtle rounded surface without obscuring the editor", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const mode of ["base", "1"]) {
    await page.goto(`http://127.0.0.1:4176/app?layoutFixture=${mode}`);
    const chat = page.locator(".chat");
    await expect(chat).toBeVisible();
    const editor = page.getByRole("textbox", { name: mode === "base" ? "Editable completion buffer" : /^Compose as / });
    await editor.fill("A quiet place to write, with the background still visible.");
    for (const value of ["dark", "light"]) {
      await theme(page, value);
      for (const width of [1440, 320]) {
        await page.setViewportSize({ width, height: 1000 });
        await expect(chat).toHaveCSS("border-radius", "8px");
        await expect(chat).toHaveCSS("background-color", /(?:0\.03|3%)/);
        await expect(chat).toHaveCSS("box-shadow", "none");
        await expect(editor).toHaveValue("A quiet place to write, with the background still visible.");
        expect(await chat.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
        await page.screenshot({ path: testInfo.outputPath(`writing-${mode}-${value}-${width}.png`) });
      }
    }
    await page.emulateMedia({ forcedColors: "active" });
    if (await page.evaluate(() => matchMedia("(forced-colors: active)").matches)) {
      await expect.poll(() => chat.evaluate(el => getComputedStyle(el).backgroundColor)).not.toContain("0.03");
    }
    await page.emulateMedia({ forcedColors: "none" });
  }
});
