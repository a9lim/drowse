import { selectWorkspaceView } from "./workbench-navigation";
import { openWorkspaceMenu } from "./workbench-navigation";
import { setAppearance, returnToChats, showWorkspaceTools, selectLoomView, openTokenDetails } from "./workbench-navigation";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";
import { buffer as readStreamBuffer, text as readStreamText } from "node:stream/consumers";
import { CHAT_ACCENTS } from "../src/lib/chatAccent";

const devUrl = "http://127.0.0.1:4176";
const storesUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;
const servicesUrl = `/@fs/${resolve("src/lib/runtime/services.ts")}`;
test.use({ hasTouch: true });

test("model settings end with the versioned Drowse footer", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await page.evaluate(async url => (await import(url)).openDrawer("local_runtime"), storesUrl);
  const footer = page.locator(".product-footer");
  await expect(footer.locator(".wordmark")).toHaveText("Drowse");
  await expect(footer.locator(".version")).toHaveText("v 0.1");
  await expect(footer).toContainText("© 2026 Drowse Contributors");
  await expect(footer).toContainText("GNU AGPL v3 or later");
  await expect(footer.locator('a[href="/LICENSE"]')).toHaveCount(0);
  await expect(footer.getByRole("link", { name: "Contribute" })).toHaveAttribute("href", /github.com\/a9lim\/drowse/);
  await expect(footer.getByRole("link", { name: "Credits" })).toHaveAttribute("href", "/credits");
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: `/@fs${resolve("src/lib/theme.ts")}`, theme });
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await footer.scrollIntoViewIfNeeded();
      await expect(footer).toBeInViewport();
      expect(await footer.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await footer.screenshot({ path: testInfo.outputPath(`product-footer-${theme}-${width}.png`) });
      expect((await new AxeBuilder({ page }).include(".product-footer").analyze()).violations).toEqual([]);
    }
  }
});

test("Credits matches the public theme and GitHub links say Contribute", async ({ page, browserName }, testInfo) => {
  test.slow();
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Primary navigation" });
  await expect(nav.getByRole("link", { name: "Contribute", exact: true })).toHaveAttribute("href", /github.com\/a9lim\/drowse/);
  await expect(page.getByRole("link", { name: /^(Source|View source)$/ })).toHaveCount(0);
  await nav.getByRole("link", { name: "Credits", exact: true }).click();
  await expect(page).toHaveURL(/\/credits$/);
  await expect(page).toHaveTitle("Credits · Drowse");
  await expect(page.getByRole("heading", { name: "Credits.", exact: true })).toBeVisible();
  await expect(page.locator(".team-member")).toHaveCount(2);
  await expect(page.getByRole("link", { name: "@_a9lim", exact: true })).toHaveAttribute("href", "https://x.com/_a9lim");
  await expect(page.getByRole("link", { name: "@treetowntree", exact: true })).toHaveAttribute("href", "https://x.com/treetowntree");
  await expect(page.getByRole("link", { name: "Website a9l.im", exact: true })).toHaveAttribute("href", "https://a9l.im");
  await expect(page.getByRole("link", { name: "Website logits.ml", exact: true })).toHaveAttribute("href", "https://logits.ml");
  await expect(page.getByRole("link", { name: "@transkatgirl", exact: true })).toHaveAttribute("href", "https://x.com/transkatgirl");
  await expect(page.getByRole("link", { name: "@motion_so", exact: true })).toHaveAttribute("href", "https://x.com/motion_so");
  await expect(page.locator(".thanks-list li").nth(1).getByRole("link", { name: "@voooooogel", exact: true })).toHaveAttribute("href", "https://x.com/voooooogel");
  await expect(page.locator('.team-member .verification-badge[data-status="blue"]')).toHaveCount(2);
  await expect(page.locator('.thanks-list a[href="https://x.com/transkatgirl"] .verification-badge')).toHaveAttribute("data-status", "blue");
  await expect(page.locator('.thanks-list a[href="https://x.com/motion_so"] .verification-badge')).toHaveAttribute("data-status", "gold");
  await expect.poll(() => page.locator(".portrait, .thanks-portrait").evaluateAll(
    images => images.every(image => image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0),
  )).toBe(true);
  await expect(page.locator(".page-brand")).toHaveAttribute("href", "/");
  await expect(nav.getByRole("link", { name: "Chats", exact: true })).toHaveAttribute("href", "/app");
  if (browserName === "webkit") {
    // WebKit's default keyboard mode skips links.
    await page.getByRole("link", { name: "Skip to content" }).focus();
  } else {
    await page.keyboard.press("Tab");
  }
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#credits-main")).toBeFocused();
  await expect(page.locator(".hero-visual")).toHaveCSS("position", "absolute");
  await expect(page.locator(".hero-visual")).toHaveAttribute("data-shader-status", "fallback");
  for (const theme of ["light", "dark"]) {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await setAppearance(page, theme === "light" ? "Light" : "Dark");
    await expect(page.locator("html")).not.toHaveAttribute("data-theme-transition", /.+/);
    for (const width of [1440, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await expect.poll(() => page.evaluate(() => [...document.querySelectorAll("body *")]
        .filter(element => !element.closest('.credits-art-frame, nav[aria-label="Primary navigation"]'))
        .filter(element => element.getBoundingClientRect().right > innerWidth + 1)
        .map(element => `${element.tagName}.${element.className}: ${element.getBoundingClientRect().right}`))).toEqual([]);
      for (const link of await nav.getByRole("link").all()) {
        await link.scrollIntoViewIfNeeded();
        const box = (await link.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(width + 1);
      }
      await page.screenshot({ path: testInfo.outputPath(`credits-${theme}-${width}.png`), fullPage: true });
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    }
  }
  await page.reload();
  await expect(page.getByRole("heading", { name: "Credits.", exact: true })).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(page.locator(".hero-visual")).toHaveAttribute("data-shader-status", "ready");
  await expect(page.locator(".hero-visual canvas")).toHaveCSS("opacity", "1");
  await page.screenshot({ path: testInfo.outputPath("credits-shader-active.png"), fullPage: true });
  await page.emulateMedia({ contrast: "more", reducedMotion: "reduce" });
  await expect(page.locator(".credits-art")).toBeHidden();
  await page.getByRole("link", { name: "Drowse home", exact: true }).click();
  await expect(page.getByRole("navigation", { name: "Primary navigation" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("chat backups download, import separately, and reopen the entire Loom", async ({ page, browserName }, testInfo) => {
  test.skip(browserName === "webkit", "The worker requires OPFS; WebKit import/download is covered by the chat card test.");
  test.setTimeout(120_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?fixture=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A complete local backup");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("This is a deterministic local Drowse runtime fixture.");
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const libraryUrl = `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`;
  const original = await page.evaluate(async ({ libraryUrl, workspaceUrl }) => {
    const { conversationLibrary, flushConversationAutosave } = await import(libraryUrl);
    await flushConversationAutosave();
    const { captureConversationSnapshot } = await import(workspaceUrl);
    const snapshot = structuredClone(captureConversationSnapshot());
    const answer = snapshot.tree.nodes.find((node: any) => node.role === "assistant");
    if (!answer?.tokens.length) throw new Error("Generation did not capture token data");
    const branch = { ...structuredClone(answer), id: crypto.randomUUID(), notes: "Keep this alternate branch", starred: true };
    snapshot.tree.nodes.push(branch);
    snapshot.tree.children_of[branch.parent_id].push(branch.id);
    snapshot.tree.children_of[branch.id] = [];
    return conversationLibrary.create({ name: "Backup experiment", avatarSeed: "backup-avatar", snapshot });
  }, { libraryUrl, workspaceUrl: `/@fs${resolve("src/lib/conversationWorkspace.ts")}` });
  await page.goto(`${devUrl}/app?fixture=1`);
  const source = page.locator(`[data-saved-conversation="${original.id}"]`);
  await expect(source).toBeVisible();
  const downloading = page.waitForEvent("download");
  await source.getByRole("button", { name: "Download backup of Backup experiment", exact: true }).click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe("Backup-experiment.drowsechat");
  const text = await readStreamText(await download.createReadStream());
  const backup = JSON.parse(text);
  expect(backup.conversation).toEqual(original);
  const records = () => page.evaluate(async url => (await (await import(url)).conversationLibrary.list()).conversations, libraryUrl);
  const before = await records();
  const fileInput = page.getByLabel("Import chat backup file", { exact: true });
  await fileInput.setInputFiles({ name: download.suggestedFilename(), mimeType: "application/json", buffer: Buffer.from(text) });
  await expect(page.getByRole("status").filter({ hasText: "Imported" })).toContainText("Backup experiment");
  const after = await records();
  expect(after).toHaveLength(before.length + 1);
  const imported = after.find((record: any) => record.name === original.name && record.id !== original.id)!;
  expect({ ...imported, id: original.id }).toEqual(original);
  expect(after.find((record: any) => record.id === original.id)).toEqual(original);
  backup.conversation.name = "Damaged copy";
  await fileInput.setInputFiles({ name: "damaged.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(backup)) });
  await expect(page.getByRole("alert")).toContainText("integrity check");
  expect(await records()).toEqual(after);
  await page.reload();
  const card = page.locator(`[data-saved-conversation="${imported.id}"]`);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: `/@fs${resolve("src/lib/theme.ts")}`, theme });
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await card.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await card.screenshot({ path: testInfo.outputPath(`backup-card-${theme}-${width}.png`) });
    }
    const accessibility = await new AxeBuilder({ page }).include("#saved-chats").analyze();
    expect(accessibility.violations).toEqual([]);
  }
  await card.getByRole("button", { name: "Open chat", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  const restored = await page.evaluate(async url => (await import(url)).apiTree.get(), `/@fs${resolve("src/lib/runtime/services.ts")}`);
  expect({ ...restored, session_id: original.snapshot.tree.session_id, rev: original.snapshot.tree.rev }).toEqual(original.snapshot.tree);
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("A complete local backup");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Continue only the imported copy");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect.poll(() => page.evaluate(async url => (await (await import(url)).apiTree.get()).nodes.length, `/@fs${resolve("src/lib/runtime/services.ts")}`)).toBe(original.snapshot.tree.nodes.length + 2);
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.evaluate(async url => (await import(url)).flushConversationAutosave(), libraryUrl);
  const continued = await records();
  expect(continued.find((record: any) => record.id === original.id)).toEqual(original);
  expect(continued.find((record: any) => record.id === imported.id).snapshot.tree.nodes).toHaveLength(original.snapshot.tree.nodes.length + 2);
});

test("dropdown and disclosure motion is reversible and respects reduced motion", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await workbench(page);
  await page.evaluate(async ({ selectUrl, comboUrl, detailsUrl }) => {
    const { mount, unmount } = await import("/e2e/svelte-runtime.ts");
    const { default: Select } = await import(selectUrl);
    const { default: Combobox } = await import(comboUrl);
    const { animatedDetails } = await import(detailsUrl);
    const target = document.createElement("div");
    target.id = "motion-test";
    target.style.cssText = "position:fixed;bottom:16px;left:16px;width:280px;padding:16px;background:var(--bg-alt);z-index:9999;border-radius:18px";
    document.body.append(target);
    const options = [{ value: "alpha", label: "Alpha" }, { value: "beta", label: "Beta" }];
    const components = [
      mount(Select, { target, props: { value: "alpha", options, ariaLabel: "Motion choice" } }),
      mount(Combobox, { target, props: { value: "alpha", options, ariaLabel: "Motion role" } }),
    ];
    const details = document.createElement("details");
    details.innerHTML = "<summary>Animated details</summary><p style='padding:24px'>Details remain available after a quick reversal.</p>";
    target.append(details);
    const action = animatedDetails(details);
    (window as any).__destroyMotionTest = () => {
      action.destroy();
      components.forEach(component => unmount(component));
      target.remove();
    };
  }, {
    selectUrl: `/@fs${resolve("src/lib/Select.svelte")}`,
    comboUrl: `/@fs${resolve("src/lib/Combobox.svelte")}`,
    detailsUrl: `/@fs${resolve("src/lib/animatedDetails.ts")}`,
  });
  for (const width of [320, 1440]) {
    await page.setViewportSize({ width, height: 700 });
    for (const kind of ["select", "combobox"]) {
      const trigger = page.locator(kind === "select" ? "#motion-test .sk-select-trigger" : "#motion-test .caret");
      await trigger.evaluate(el => (el as HTMLButtonElement).click());
      const list = page.locator("#motion-test .t-dropdown");
      await expect(list).toHaveClass(/is-open/);
      await expect(list).toHaveAttribute("data-origin", "bottom-left");
      await expect(list).toHaveCSS("transition-duration", "0.14s");
      await expect(list).toHaveCSS("transition-property", "opacity");
      await expect(list).toHaveCSS("transform", "matrix(1, 0, 0, 1, 0, 0)");
      await expect(list).toHaveCSS("opacity", "1");
      const box = (await list.boundingBox())!;
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(width);
      const closing = await trigger.evaluate(el => {
        (el as HTMLButtonElement).click();
        const popup = document.querySelector<HTMLElement>("#motion-test .t-dropdown")!;
        return { closing: popup.classList.contains("is-closing"), inert: popup.inert, duration: getComputedStyle(popup).transitionDuration };
      });
      expect(closing).toEqual({ closing: true, inert: true, duration: "0.1s" });
      await trigger.evaluate(el => (el as HTMLButtonElement).click());
      await expect(list).toHaveCount(1);
      await expect(list).toHaveClass(/is-open/);
      await expect(list).not.toHaveAttribute("inert");
      await list.getByRole("option", { name: "Beta", exact: true }).click();
      await expect(list).toHaveCount(0);
      if (kind === "select") await expect(trigger).toBeFocused();
      else await expect(page.getByRole("combobox", { name: "Motion role" })).toBeFocused();
    }
  }
  const details = page.locator("#motion-test details");
  const summary = details.locator("summary");
  const opening = await summary.evaluate(el => {
    (el as HTMLElement).click();
    return el.parentElement!.getAnimations().map(animation => animation.effect?.getTiming().duration);
  });
  expect(opening).toEqual([250]);
  await summary.evaluate(el => { (el as HTMLElement).click(); (el as HTMLElement).click(); });
  await expect(details).toHaveAttribute("open", "");
  await expect.poll(() => details.evaluate(el => el.getAnimations().length)).toBe(0);
  await expect(details).toHaveCSS("overflow", "visible");
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(details).not.toHaveAttribute("open");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await summary.click();
  expect(await details.evaluate(el => el.getAnimations().length)).toBe(0);
  await page.locator("#motion-test .sk-select-trigger").click();
  const popup = page.locator("#motion-test .t-dropdown");
  await expect(popup).toHaveCSS("opacity", "1");
  expect(await popup.evaluate(el => el.getAnimations().length)).toBe(0);
  await page.keyboard.press("Escape");
  await expect(popup).toHaveCount(0);
  await page.locator("#motion-test").screenshot({ path: testInfo.outputPath("expanded-details.png") });
  await page.evaluate(() => (window as any).__destroyMotionTest());
  await expect(page.locator("#motion-test")).toHaveCount(0);
});

for (const theme of ["dark", "light"] as const) {
  test(`semantic color system stays unified in ${theme}`, async ({ page }, testInfo) => {
    test.setTimeout(90_000);
    await page.emulateMedia({ reducedMotion: "reduce", colorScheme: theme });
    await workbench(page);
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), {
      url: `/@fs${resolve("src/lib/theme.ts")}`, theme,
    });
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    const palette = await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      return Object.fromEntries(["accent", "focus-ring", "pillar-lens", "pillar-sae"].map(name => [name, style.getPropertyValue(`--${name}`).trim()]));
    });
    expect(palette["focus-ring"]).toBe(palette.accent);
    expect(palette["pillar-lens"]).toBe(palette.accent);
    expect(palette["pillar-sae"]).not.toBe(palette.accent);
    const primary = page.locator('.input-actions button[type="submit"]');
    await expect(primary).toHaveClass(/solid/);
    expect(await primary.evaluate(el => getComputedStyle(el).getPropertyValue("--btn-accent").trim())).toBe(palette.accent);
    await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain language models.");
    await primary.click();
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    await page.evaluate(async url => { (await import(url)).highlightState.target = "__probability__"; }, storesUrl);
    const tokenStyle = await page.locator(".msg .tok").first().getAttribute("style");
    expect(tokenStyle).toContain("--pillar-lens");
    for (const accent of CHAT_ACCENTS) {
      await page.evaluate(accent => {
        const root = document.documentElement;
        root.dataset.chatAccent = accent.id;
        root.style.setProperty("--chat-accent-dark", accent.dark);
        root.style.setProperty("--chat-accent-light", accent.light);
      }, accent);
      const colors = await page.locator(".msg .tok").first().evaluate(el => {
        const style = getComputedStyle(el);
        return {
          accent: style.getPropertyValue("--accent").trim(),
          lens: style.getPropertyValue("--pillar-lens").trim(),
          cell: style.getPropertyValue("--layer-cell-lens").trim(),
          sae: style.getPropertyValue("--pillar-sae").trim(),
        };
      });
      expect(colors.accent).toBe(accent[theme]);
      expect(colors.lens).toBe(colors.accent);
      expect(colors.cell).toBe(colors.accent);
      expect(colors.sae).toBe(palette["pillar-sae"]);
      const expectedBackground = await page.locator(".msg .tok").first().evaluate(el => {
        const sample = document.createElement("span");
        sample.style.backgroundColor = (el as HTMLElement).style.backgroundColor.replace("var(--pillar-lens)", "var(--accent)");
        el.appendChild(sample);
        const color = getComputedStyle(sample).backgroundColor;
        sample.remove();
        return color;
      });
      await expect(page.locator(".msg .tok").first()).toHaveCSS("background-color", expectedBackground);
      if (accent.id === "rose") {
        await page.locator(".msg").last().screenshot({ path: testInfo.outputPath(`rose-token-highlights-${theme}.png`) });
      }
    }
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await page.screenshot({ path: testInfo.outputPath(`conversation-${theme}-${width}.png`) });
    }
    await selectWorkspaceView(page, "Loom");
    await showWorkspaceTools(page, "Loom");
    await selectLoomView(page, /^Map\b/);
    const loomToken = page.locator("[data-loom-token-node]").first();
    await expect(loomToken).toHaveAttribute("style", tokenStyle!);
    await page.screenshot({ path: testInfo.outputPath(`loom-${theme}.png`) });
    await page.evaluate(async url => { (await import(url)).highlightState.target = null; }, storesUrl);
    await expect(loomToken).not.toHaveAttribute("style", /background-color/);
    await openDrawer(page, "token_drilldown", { turnIdx: 1, tokenIdx: 0 });
    const sheet = page.locator('aside[aria-label="Token drilldown"]');
    const branch = sheet.locator(".branch-point");
    expect(await branch.evaluate(el => {
      const style = getComputedStyle(el);
      const well = document.createElement("div");
      well.style.background = "var(--input-well)";
      el.appendChild(well);
      const expected = getComputedStyle(well).backgroundColor;
      well.remove();
      return style.backgroundColor === expected;
    })).toBe(true);
    for (const name of [/^sae$/, /^j-lens$/]) {
      await sheet.getByRole("button", { name }).click();
      const selected = sheet.locator('.sk-tabs .selection-indicator');
      const selectedColor = await selected.evaluate(el => {
        const swatch = document.createElement("div");
        swatch.style.background = "var(--control-selected-bg)";
        el.appendChild(swatch);
        const color = getComputedStyle(swatch).backgroundColor;
        swatch.remove();
        return color;
      });
      await expect(selected).toHaveCSS("background-color", selectedColor);
    }
    await page.screenshot({ path: testInfo.outputPath(`token-details-${theme}.png`) });
    await page.keyboard.press("Escape");
    await selectWorkspaceView(page, "Controls");
    await expect(page.locator(".controls-page")).toHaveAttribute("aria-hidden", "false");
    await page.locator(".controls-page").screenshot({ path: testInfo.outputPath(`controls-${theme}.png`) });
    const result = await new AxeBuilder({ page }).include(".controls-page").withRules(["color-contrast"]).analyze();
    expect(result.violations).toEqual([]);
    for (const [name, url] of [["landing", `${devUrl}/`], ["chats", `${devUrl}/app?layoutFixture=setup`]]) {
      await page.goto(url);
      await expect(page.locator(".page-brand")).toBeVisible();
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      await page.screenshot({ path: testInfo.outputPath(`${name}-${theme}.png`) });
    }
    await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Models", exact: true }).click();
    await expect(page.locator(".app-shell")).toBeVisible();
    await page.locator(".app-shell").screenshot({ path: testInfo.outputPath(`models-${theme}.png`) });
  });
}

async function emptyGeometry(page: Page, authoring = false) {
  await workbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain language models.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.evaluate(async ({ storesUrl, registryUrl, authoring }) => {
    const { chatLog, probeRack } = await import(storesUrl);
    probeRack.active = [];
    probeRack.entries.clear();
    for (const turn of chatLog.turns) {
      for (const token of turn.tokens ?? []) {
        if (token.measurements?.instruments) delete token.measurements.instruments.geometry;
      }
    }
    const registry = await import(registryUrl);
    const caps = registry.getRuntimeCapabilities() ?? {
      operations: Object.fromEntries(["fitting", "manifold_artifacts", "probe_subspace_trails", "jlens_fitting", "sae_training", "session_admin"].map(name => [name, { available: true, reasons: [] }])),
    };
    caps.operations.fitting.available = authoring;
    registry.installRuntimeCapabilities(caps);
  }, { storesUrl, registryUrl: `/@fs${resolve("src/lib/runtime/registry.ts")}`, authoring });
  await openDrawer(page, "token_drilldown", { turnIdx: 1, tokenIdx: 0 });
  const sheet = page.locator('aside[aria-label="Token drilldown"]');
  await sheet.getByRole("button", { name: /^geometry\b/i }).click();
  await expect(sheet.getByText("Add a probe to see concept readings", { exact: true })).toBeVisible();
  return sheet;
}

test("geometry setup offers clear actions and preserves the current token", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  const sheet = await emptyGeometry(page, true);
  await sheet.getByRole("button", { name: "Next token", exact: true }).click();
  for (const theme of ["dark", "light"] as const) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), {
      url: `/@fs${resolve("src/lib/theme.ts")}`, theme,
    });
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await sheet.getByRole("button", { name: "Create a concept", exact: true }).scrollIntoViewIfNeeded();
      expect(await sheet.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
      for (const name of ["Add a probe", "Create a concept"]) {
        await expect(sheet.getByRole("button", { name, exact: true })).toBeInViewport();
      }
      await page.screenshot({ path: testInfo.outputPath(`geometry-setup-${theme}-${width}.png`) });
    }
  }
  await sheet.getByRole("button", { name: "Add a probe", exact: true }).click();
  const rack = page.getByRole("region", { name: "Subspaces", exact: true });
  await expect(rack).toBeVisible();
  await page.setViewportSize({ width: 320, height: 900 });
  expect(await rack.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  await expect(rack.getByRole("button", { name: "Back to token", exact: true })).toBeInViewport();
  await page.screenshot({ path: testInfo.outputPath("geometry-probe-picker-mobile.png") });
  await rack.getByRole("button", { name: "Back to token", exact: true }).click();
  await expect(sheet).toBeVisible();
  await sheet.getByRole("button", { name: /^geometry\b/i }).click();
  await expect.poll(() => page.evaluate(async url => (await import(url)).drawerState.params.tokenIdx, storesUrl)).toBe(1);
  await sheet.getByRole("button", { name: "Create a concept", exact: true }).click();
  const builder = page.getByRole("region", { name: "Build manifold", exact: true });
  await expect(builder).toBeVisible();
  await expect(builder.getByRole("tablist", { name: "Authoring mode" })).toBeVisible();
  expect(await builder.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("geometry-concept-builder.png") });
  await builder.locator(".drawer-close").click();
  await expect(sheet).toBeVisible();
  await sheet.getByRole("button", { name: /^geometry\b/i }).click();
  await expect.poll(() => page.evaluate(async url => (await import(url)).drawerState.params.tokenIdx, storesUrl)).toBe(1);

  // Simulate successful authoring; the layout fixture never trains model weights.
  await page.evaluate(async url => {
    const { apiManifolds } = await import(url);
    (window as any).__conceptCalls = [];
    apiManifolds.generate = async (request: unknown) => { (window as any).__conceptCalls.push(request); };
    apiManifolds.fit = async (namespace: string, name: string) => { (window as any).__conceptCalls.push({ namespace, name }); };
  }, servicesUrl);
  await sheet.getByRole("button", { name: "Create a concept", exact: true }).click();
  await builder.getByRole("textbox", { name: "name *", exact: true }).fill("Test Concept");
  await builder.getByRole("textbox", { name: /^concepts/ }).fill("formal\ncasual");
  await builder.getByRole("button", { name: "generate + fit", exact: true }).click();
  await expect(rack).toBeVisible();
  expect(await page.evaluate(() => (window as any).__conceptCalls)).toEqual([
    expect.objectContaining({ namespace: "local", name: "test_concept", concepts: ["formal", "casual"] }),
    { namespace: "local", name: "test_concept" },
  ]);
  await rack.getByRole("button", { name: "Back to token", exact: true }).click();
  await expect(sheet).toBeVisible();
  await expect.poll(() => page.evaluate(async url => (await import(url)).drawerState.params.tokenIdx, storesUrl)).toBe(1);
});

test("geometry setup attaches a probe and hides unsupported training", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const sheet = await emptyGeometry(page);
  await expect(sheet.getByRole("button", { name: "Create a concept", exact: true })).toHaveCount(0);
  await sheet.getByRole("button", { name: "Add a probe", exact: true }).click();
  const rack = page.getByRole("region", { name: "Subspaces", exact: true });
  await rack.getByText("attach selector", { exact: true }).click();
  await rack.getByRole("textbox", { name: "Selector", exact: true }).fill("local/test-concept");
  await rack.getByRole("button", { name: "+ attach", exact: true }).click();
  await expect(sheet).toBeVisible();
  await expect.poll(() => page.evaluate(async url => [...(await import(url)).probeRack.active], storesUrl)).toContain("local/test-concept");
  await expect(sheet.getByText("Add a probe to see concept readings", { exact: true })).toHaveCount(0);
  await sheet.locator(".drawer-close").click();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Read this new reply with the attached probe.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await openDrawer(page, "token_drilldown", { turnIdx: 3, tokenIdx: 0 });
  await expect(sheet.locator(".geo-list")).toContainText("local/test-concept");
});

test("sampling sliders accept full-height drags and reset to original", async ({ page }, testInfo) => {
  await workbench(page);
  await selectWorkspaceView(page, "Controls");
  const slider = page.getByRole("slider", { name: "Temperature", exact: true });
  const original = await slider.inputValue();
  for (const width of [320, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await slider.scrollIntoViewIfNeeded();
    await expect(slider).toBeInViewport();
    const box = (await slider.boundingBox())!;
    expect(box.height).toBeGreaterThanOrEqual(44);
    await page.mouse.move(box.x + box.width * 0.25, box.y + 4);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.85, box.y + box.height + 30, { steps: 12 });
    await page.mouse.up();
    await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(1.5);
    await slider.focus();
    await page.keyboard.press("Home");
    await expect(slider).toHaveValue("0");
    await page.keyboard.press("ArrowRight");
    await expect(slider).toHaveValue("0.05");
    await page.getByRole("button", { name: "Reset to original", exact: true }).click();
    await expect(slider).toHaveValue(original);
    await page.screenshot({ path: testInfo.outputPath(`sampling-reset-${width}.png`) });
  }
});

test("model reset preserves chats and restores defaults after reopening", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "Persistent worker storage requires OPFS.");
  await page.goto(`${devUrl}/app?fixture=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep this conversation after reset.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("status").filter({ hasText: "Response complete." })).toBeVisible();
  await selectWorkspaceView(page, "Controls");
  const slider = page.getByRole("slider", { name: "Temperature", exact: true });
  const original = await slider.inputValue();
  await slider.focus();
  await page.keyboard.press("End");
  await expect(slider).toHaveValue("2");
  await page.getByRole("button", { name: "System prompt", exact: true }).click();
  await page.locator(".drawer-shell textarea").fill("Use a changed system prompt.");
  await page.locator(".drawer-shell").getByRole("button", { name: "Save system prompt", exact: true }).click();
  await page.reload();
  await expect(page.locator(".shell")).toBeVisible();
  await selectWorkspaceView(page, "Controls");
  await expect(slider).toHaveValue("2");
  const tabs = page.getByRole("group", { name: "Controls section", exact: true });
  await tabs.getByRole("button", { name: "Model", exact: true }).click();
  const reset = page.getByRole("tabpanel", { name: "Model controls" }).locator(".reset-settings");
  await reset.getByRole("button", { name: "Reset Settings", exact: true }).click();
  await reset.getByRole("button", { name: "Cancel", exact: true }).click();
  await reset.getByRole("button", { name: "Reset Settings", exact: true }).click();
  await reset.getByRole("button", { name: "Reset Settings", exact: true }).click();
  await expect(reset.getByRole("button", { name: "Cancel", exact: true })).toHaveCount(0);
  await tabs.getByRole("button", { name: "Response", exact: true }).click();
  await expect(slider).toHaveValue(original);
  await page.getByRole("button", { name: "System prompt", exact: true }).click();
  await expect(page.locator(".drawer-shell textarea")).toHaveValue("");
  await page.keyboard.press("Escape");
  await selectWorkspaceView(page, "Conversation");
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("Keep this conversation after reset.");
});

test("steering summary only appears for configured directions", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await selectWorkspaceView(page, "Controls");
  const recipe = page.locator(".inspector .recipe");
  const setState = (state: "empty" | "custom-empty" | "custom" | "disabled") => page.evaluate(async ({ url, state }) => {
    const { steerRack } = await import(url);
    steerRack.entries.clear();
    steerRack.customExpression = state === "custom" ? "0.3 jlens/orange" : state === "custom-empty" ? "" : null;
    if (state === "disabled") steerRack.entries.set("jlens/orange", {
      mode: "jlens", ablate: false, alpha: 0.3, trigger: "BOTH", enabled: false,
    });
  }, { url: storesUrl, state });
  await setState("empty");
  await expect(recipe).toHaveCount(0);
  await expect(page.locator(".inspector")).not.toContainText("Original behavior");
  for (const width of [320, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    const grid = page.locator(".control-grid");
    await expect.poll(() => grid.evaluate(el => Math.abs(el.getBoundingClientRect().height - el.parentElement!.getBoundingClientRect().height))).toBeLessThan(1);
    await expect(page.getByRole("heading", { name: "Generation settings", exact: true })).toBeInViewport();
    await setState("custom");
    await expect(recipe).toContainText("Steering");
    await expect(recipe).toContainText("0.3 jlens/orange");
    await expect(recipe).not.toHaveAttribute("title");
      const firstPanel = (await page.locator(".inspector").boundingBox())!;
    const recipeBox = (await recipe.boundingBox())!;
    expect(recipeBox.x).toBeGreaterThanOrEqual(firstPanel.x);
    expect(recipeBox.x + recipeBox.width).toBeLessThanOrEqual(firstPanel.x + firstPanel.width);
    expect(await recipe.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    await setState("custom-empty");
    await expect(recipe).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath(`quiet-response-controls-${width}.png`) });
  }
  await setState("disabled");
  await expect(recipe).toContainText("jlens/orange");
  await expect(recipe.getByRole("button", { name: "Copy response recipe" })).toHaveCount(0);
  await recipe.getByRole("button", { name: "Remove jlens/orange from steering recipe" }).click();
  await expect(recipe).toHaveCount(0);
});

test("Chat controls share the three-tab layout on desktop and phones", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await selectWorkspaceView(page, "Controls");
  const tabs = page.getByRole("group", { name: "Controls section", exact: true });
  await expect(tabs.getByRole("button")).toHaveText(["Response", "Model", "Chat"]);
  await tabs.getByRole("button", { name: "Chat", exact: true }).click();
  const panel = page.getByRole("tabpanel", { name: "Chat controls" });
  await expect(panel.getByRole("heading", { name: "Save and name chat" })).toBeVisible();
  await expect(panel.getByRole("textbox", { name: "Name", exact: true })).toBeEnabled();
  await expect(panel.getByRole("button", { name: "Generate another avatar" })).toBeVisible();
  await expect(panel.locator(".drawer-close")).toHaveCount(0);
  for (const colorScheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme });
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 800 });
      expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
      await expect.poll(() => tabs.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
      await panel.getByRole("button", { name: "Download copy" }).scrollIntoViewIfNeeded();
      await expect(panel.getByRole("button", { name: "Download copy" })).toBeInViewport();
      await page.screenshot({ path: testInfo.outputPath(`chat-controls-${colorScheme}-${width}.png`) });
    }
  }
  await tabs.getByRole("button", { name: "Model", exact: true }).click();
  await expect(page.getByRole("tabpanel", { name: "Model controls" })).toBeVisible();
  await tabs.getByRole("button", { name: "Response", exact: true }).click();
  await expect(page.getByRole("tabpanel", { name: "Response controls" })).toBeVisible();
});

test("Chat controls update the saved name and avatar without leaving the tab", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "The persistent worker fixture requires OPFS, unavailable in headless WebKit.");
  await page.goto(`${devUrl}/app?fixture=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Save this chat from its controls.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("status").filter({ hasText: "Response complete." })).toBeVisible();
  await selectWorkspaceView(page, "Controls");
  const tabs = page.getByRole("group", { name: "Controls section", exact: true });
  await tabs.getByRole("button", { name: "Chat", exact: true }).click();
  const panel = page.getByRole("tabpanel", { name: "Chat controls" });
  const name = panel.getByRole("textbox", { name: "Name", exact: true });
  await expect(name).toBeEnabled();
  const records = () => page.evaluate(async (url) => {
    const { conversationLibrary } = await import(url);
    return (await conversationLibrary.list()).conversations;
  }, `/@fs/${resolve("src/lib/stores/savedConversations.svelte.ts")}`);
  const original = (await records())[0];
  expect(original).toBeTruthy();
  await name.fill("A named chat");
  await panel.getByRole("button", { name: "Generate another avatar" }).click();
  await panel.getByRole("button", { name: "Update", exact: true }).click();
  await expect.poll(async () => (await records())[0].name).toBe("A named chat");
  const updated = (await records())[0];
  expect(updated.id).toBe(original.id);
  expect(updated.avatarSeed).not.toBe(original.avatarSeed);
  await expect(tabs.getByRole("button", { name: "Chat", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(name).toBeEnabled();
  const download = page.waitForEvent("download");
  await panel.getByRole("button", { name: "Download copy" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("A-named-chat.drowsechat");
  const backup = JSON.parse(await readStreamText(await file.createReadStream()));
  expect(backup.format).toBe("drowse-chat-backup");
  expect(backup.conversation.name).toBe(updated.name);
  expect(backup.conversation.avatarSeed).toBe(updated.avatarSeed);
  expect(backup.conversation.accent).toBe(updated.accent);
  expect(backup.conversation.snapshot.tree).toEqual(updated.snapshot.tree);
  await name.fill("A second copy");
  await panel.getByRole("button", { name: "Save as new" }).click();
  await expect.poll(async () => (await records()).length).toBe(2);
  await expect(name).toBeEnabled();
  await tabs.getByRole("button", { name: "Response", exact: true }).click();
  await tabs.getByRole("button", { name: "Chat", exact: true }).click();
  await expect(name).toHaveValue("A second copy");
});

test("shared page headers and footers align across public pages and workbench", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const width of [320, 1000, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ colorScheme: width === 1000 ? "light" : "dark" });
    await page.goto(`${devUrl}/`);
    await page.evaluate(() => document.fonts.ready);
    const anchor = await page.locator(".page-brand").evaluate(el => el.getBoundingClientRect().toJSON());
    const anchorHeaderX = await page.locator(".page-header").evaluate(el => el.getBoundingClientRect().x);
    const footerStyle = await page.locator(".page-footer").evaluate(el => {
      const css = getComputedStyle(el);
      const content = el.querySelector(".footer-content")!;
      return { x: el.getBoundingClientRect().x, width: el.getBoundingClientRect().width, padding: css.paddingInlineStart,
        height: content.getBoundingClientRect().height, text: el.textContent?.trim() };
    });
    const check = async (name: string) => {
      if (width === 320 && name !== "workbench") await expect(page.locator(".page-brand")).toBeHidden();
      else await expect(page.locator(".page-brand")).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      const box = await page.locator(".page-brand").evaluate(el => el.getBoundingClientRect().toJSON());
      const compact = await page.locator(".page-header").evaluate(el => el.classList.contains("compact"));
      await expect.poll(() => page.locator(".page-brand").evaluate((el, expected) =>
        Math.abs(el.getBoundingClientRect().width - expected), anchor.width * (compact ? 5 / 6 : 1)),
      { message: `${name} width` }).toBeLessThanOrEqual(1);
      expect(box.height).toBeCloseTo(await page.locator(".page-brand").evaluate(el => parseFloat(getComputedStyle(el).minHeight)), 0);
      expect(await page.locator(".page-header").evaluate(el => el.scrollWidth - el.clientWidth), name).toBeLessThanOrEqual(1);
      if (name === "workbench") {
        if (width <= 760) await expect(page.getByRole("button", { name: "Workspace menu", exact: true })).toBeVisible();
        else await expect(page.getByRole("button", { name: "Workspace menu", exact: true })).toHaveCount(0);
        await expect(page.locator(".page-header .theme-toggle, .page-header nav")).toHaveCount(0);
        const sidebarToggle = (await page.getByRole("button", { name: /^(Hide|Show) left sidebar$/ }).boundingBox())!;
        const leadingGap = await page.locator(".page-leading").evaluate(el => parseFloat(getComputedStyle(el).columnGap));
        expect(box.x).toBeCloseTo(sidebarToggle.x + sidebarToggle.width + leadingGap, 0);
        await page.screenshot({ path: testInfo.outputPath(`${name}-${width}.png`) });
        return;
      }
      if (!compact && width > 320) for (const key of ["x", "y"] as const) {
        await expect.poll(() => page.locator(".page-brand").evaluate((el, key) => {
          const box = el.getBoundingClientRect();
          return box[key] - (key === "x" ? el.closest(".page-header")!.getBoundingClientRect().x : 0);
        }, key), { message: `${name} ${key}` }).toBeCloseTo(anchor[key] - (key === "x" ? anchorHeaderX : 0), 0);
      }
      else expect(box.x).toBeGreaterThanOrEqual(0);
      await expect(page.locator(".page-header .theme-toggle")).toHaveCount(1);
      await expect(page.locator(".page-header nav a")).toHaveText(["Chats", "Models", "Credits", "Contribute"]);
      await page.screenshot({ path: testInfo.outputPath(`${name}-${width}.png`) });
      const footer = page.locator(".page-footer");
      await footer.scrollIntoViewIfNeeded();
      await expect(footer.getByRole("link")).toHaveText(["Credits", "Contact us", "contact@drowse.ai", "Contribute"]);
      const actualFooter = await footer.evaluate(el => {
        const css = getComputedStyle(el);
        const content = el.querySelector(".footer-content")!;
        return { x: el.getBoundingClientRect().x, width: el.getBoundingClientRect().width, padding: css.paddingInlineStart,
          height: content.getBoundingClientRect().height, text: el.textContent?.trim() };
      });
      const footerContainer = await footer.evaluate(el => el.parentElement!.getBoundingClientRect().toJSON());
      expect(actualFooter.width).toBe(Math.min(footerContainer.width, (compact ? 90 : 80) * 14));
      expect(actualFooter.x).toBe(footerContainer.x + (footerContainer.width - actualFooter.width) / 2);
      if (!compact) expect({ padding: actualFooter.padding, height: actualFooter.height, text: actualFooter.text })
        .toEqual({ padding: footerStyle.padding, height: footerStyle.height, text: footerStyle.text });
      else {
        expect(actualFooter.text).toBe(footerStyle.text);
        expect(actualFooter.padding).toBe("16px");
        expect(actualFooter.height).toBeGreaterThanOrEqual(80);
      }
      expect(await footer.evaluate(el => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
      await footer.screenshot({ path: testInfo.outputPath(`${name}-footer-${width}.png`) });
    };
    await check("landing");
    await page.goto(`${devUrl}/credits`);
    await check("credits");
    await page.goto(`${devUrl}/404.html`);
    await check("not-found");
    await page.goto(`${devUrl}/app?layoutFixture=setup`);
    await expect(page.locator(".app-shell")).toBeVisible();
    await check("models");
    await workbench(page);
    await check("workbench");
    await page.evaluate(async ({ home, stores }) => {
      const [{ default: HostedHome }, { mount }, { sessionState }] = await Promise.all([import(home), import("/e2e/svelte-runtime.ts"), import(stores)]);
      const modelId = sessionState.info.model_id;
      document.body.replaceChildren();
      const target = document.createElement("div");
      document.body.append(target);
      mount(HostedHome, { target, props: {
        controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {} },
        snapshot: { phase: "supported", models: [{ id: modelId, modelId, name: "Qwen3 1.7B", setupComplete: true, fit: "recommended" }], download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: true } },
        onChooseModels() {},
      } });
    }, { home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`, stores: storesUrl });
    await check("chats");
  }
});

test("page navigation keeps the wordmark anchored and fades without trapping outgoing pages", async ({ page, browserName }, testInfo) => {
  test.skip(browserName === "webkit", "The worker fixture requires OPFS; WebKit header geometry is covered separately.");
  test.setTimeout(90_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  const settled = async () => {
    await expect(page.locator(".page-route")).toHaveCount(1);
    await expect.poll(() => page.locator(".page-route").evaluate(el => el.getAnimations().length)).toBe(0);
  };
  for (const width of [320, 1000, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: "no-preference", colorScheme: width === 1000 ? "light" : "dark" });
    await page.goto(`${devUrl}/app?fixture=1&choose=1`);
    const openModel = page.getByRole("button", { name: /^(Download and open|Open Drowse)$/ });
    await expect(openModel).toBeEnabled();
    await settled();
    await page.evaluate(() => document.fonts.ready);
    const modelLogo = await page.locator(".page-brand").evaluate(el => el.getBoundingClientRect().toJSON());
    const assertHeader = async () => {
      const logo = await page.locator(".page-brand").evaluate(el => el.getBoundingClientRect().toJSON());
      const inWorkbench = await page.locator(".shell").isVisible();
      const compact = await page.locator(".page-header").evaluate(el => el.classList.contains("compact"));
      expect(Math.abs(logo.width - modelLogo.width * (compact ? 5 / 6 : 1))).toBeLessThanOrEqual(1);
      expect(logo.height).toBeCloseTo(await page.locator(".page-brand").evaluate(el => parseFloat(getComputedStyle(el).minHeight)), 0);
      if (inWorkbench) {
        const sidebarToggle = (await page.getByRole("button", { name: /^(Hide|Show) left sidebar$/ }).boundingBox())!;
        const leadingGap = await page.locator(".page-leading").evaluate(el => parseFloat(getComputedStyle(el).columnGap));
        expect(logo.x).toBeCloseTo(sidebarToggle.x + sidebarToggle.width + leadingGap, 0);
      }
      else if (!compact && width > 320) for (const key of ["x", "y"] as const) expect(logo[key]).toBeCloseTo(modelLogo[key], 0);
      expect(await page.locator(".page-header").evaluate(el => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      await expect(page.getByText("Local workbench", { exact: true })).toHaveCount(0);
    };
    await openModel.click();
    await expect(page.locator(".shell")).toBeVisible();
    await settled();
    await assertHeader();
    await page.screenshot({ path: testInfo.outputPath(`unified-workbench-${width}.png`) });
    await openWorkspaceMenu(page);
    await page.getByRole("dialog", { name: "Workspace menu", exact: true }).getByRole("button", { name: "Models", exact: true }).click();
    await expect(page.locator(".app-shell")).toBeVisible();
    await settled();
    await assertHeader();
    await page.getByRole("button", { name: /^(Download and open|Open Drowse)$/ }).click();
    await expect(page.locator(".shell")).toBeVisible();
    await settled();
    await returnToChats(page);
    await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
    await settled();
    await assertHeader();
    await page.screenshot({ path: testInfo.outputPath(`unified-chats-${width}.png`) });
    // Trigger before Playwright waits for the animation to finish, then inspect the outgoing surface.
    await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Models", exact: true }).evaluate((el: HTMLAnchorElement) => el.click());
    await expect(page.locator('.page-route[data-route="home"]')).toHaveAttribute("inert", "");
    await expect.poll(() => page.locator('.page-route[data-route="models"]').evaluate(el => el.getAnimations({ subtree: true }).some(a => a.effect?.getKeyframes().some(k => String(k.filter).includes("blur"))))).toBe(true);
    await page.locator('.page-route[data-route="models"]').getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Chats", exact: true }).evaluate((el: HTMLAnchorElement) => el.click());
    await settled();
    await expect(page.locator(".page-route")).toHaveAttribute("data-route", "home");
    await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Models", exact: true }).click();
    await settled();
    await assertHeader();
    await page.screenshot({ path: testInfo.outputPath(`unified-models-${width}.png`) });
    await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Chats", exact: true }).click();
    await settled();
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "Models", exact: true }).click();
    await settled();
    await assertHeader();
    await expect(page.locator(".page-route")).toHaveCSS("filter", "none");
  }
  expect(errors).toEqual([]);
});

test.describe("saved chat card interactions", () => {
  test.use({ hasTouch: false, isMobile: false });
  test("chat names edit in place and avatar shuffle reveals without lighting up the card", async ({ page }, testInfo) => {
    const mountHome = () => page.evaluate(async ({ home, saved, stores, workspace }) => {
      const [{ default: HostedHome }, { mount }, { conversationLibrary, registerConversationAutosave }, { sessionState }, { captureConversationSnapshot }] = await Promise.all([
        import(home), import("/e2e/svelte-runtime.ts"), import(saved), import(stores), import(workspace),
      ]);
      if (!(await conversationLibrary.hasAny())) {
        await conversationLibrary.create({ name: "Chat 1", snapshot: captureConversationSnapshot() });
      }
      registerConversationAutosave(async () => {});
      const modelId = sessionState.info.model_id;
      document.body.replaceChildren();
      const target = document.createElement("div");
      document.body.append(target);
      mount(HostedHome, { target, props: {
        controller: { capabilities: () => ({ signals: { appleMobile: false } }), open: async () => {}, check: async () => {} },
        snapshot: { phase: "supported", models: [{ id: modelId, modelId, name: "Qwen3 1.7B", setupComplete: true, fit: "recommended" }], download: { available: true, phase: "idle", reason: "Ready" }, runtime: { available: true }, storage: { persisted: true } },
        onChooseModels() {},
      } });
    }, { home: `/@fs${resolve("src/hosted/ui/HostedHome.svelte")}`, saved: `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`, stores: `/@fs${resolve("src/lib/stores.svelte.ts")}`, workspace: `/@fs${resolve("src/lib/conversationWorkspace.ts")}` });
    await workbench(page);
    await page.getByRole("textbox", { name: /^Compose as / }).fill("Hello, I love marmots!");
    await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
    await expect(page.getByRole("status").filter({ hasText: "Response complete." })).toBeVisible();
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    await mountHome();
    const card = page.locator("[data-saved-conversation]").first();
    await expect(card).toBeVisible();
    await expect(card.locator("time .morph-source")).toHaveText("just now");
    await expect(card.locator(".chat-counts .morph-source")).toHaveText("2 messages · 1 Loom thread");
    await card.locator(".chat-name").click();
    const input = card.getByRole("textbox", { name: "Chat name" });
    await expect(input).toBeFocused();
    await input.fill("Discard this");
    await input.press("Escape");
    await expect(card.locator(".chat-name")).not.toContainText("Discard this");
    await card.locator(".chat-name").focus();
    await page.keyboard.press("Enter");
    await input.fill(" ");
    await expect(card.getByRole("button", { name: "Save name" })).toBeDisabled();
    await input.fill("Marmot notes");
    await input.press("Enter");
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot notes");
    await card.locator(".chat-name").click();
    await input.fill("Discard this too");
    await card.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot notes");
    await card.locator(".chat-name").click();
    await input.fill(" ");
    await card.locator(".chat-counts").click();
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot notes");
    await card.locator(".chat-name").click();
    await input.fill("Marmot research");
    await card.locator(".chat-counts").click();
    await expect(card.locator(".chat-name .morph-source")).toHaveText("Marmot research");

    const avatar = card.locator(".avatar");
    const image = avatar.locator("img");
    const previous = await image.getAttribute("src");
    const surface = await card.evaluate(el => [getComputedStyle(el).backgroundColor, getComputedStyle(el).boxShadow]);
    await avatar.hover();
    await expect(avatar.locator(".avatar-shuffle")).toHaveCSS("opacity", "1");
    await expect(avatar.locator(".avatar-image")).toHaveCSS("filter", "blur(3px) brightness(0.5)");
    expect(await card.evaluate(el => [getComputedStyle(el).backgroundColor, getComputedStyle(el).boxShadow])).toEqual(surface);
    const timestamp = await card.locator("time").getAttribute("datetime");
    const actionsBefore = await card.locator(".chat-actions").boundingBox();
    await page.evaluate(async url => {
      const { conversationLibrary } = await import(url);
      const update = conversationLibrary.update.bind(conversationLibrary);
      conversationLibrary.update = async (...args: any[]) => {
        conversationLibrary.update = update;
        await new Promise<void>(resolve => { (window as any).releaseAvatarUpdate = resolve; });
        return update(...args);
      };
    }, `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`);
    await avatar.click();
    await expect(avatar).toHaveAttribute("aria-busy", "true");
    await expect(card.locator(".chat-name")).toBeEnabled();
    await expect(card.getByRole("button", { name: "Download backup of Marmot research", exact: true })).toBeEnabled();
    await expect(card.getByRole("button", { name: "Download backup of Marmot research", exact: true }).locator(".morph-source")).toHaveText("Download");
    await expect(card.getByRole("button", { name: "More options for Marmot research", exact: true })).toBeEnabled();
    expect(await card.locator(".chat-actions").boundingBox()).toEqual(actionsBefore);
    await page.evaluate(() => (window as any).releaseAvatarUpdate());
    await expect(avatar).toHaveAttribute("aria-busy", "false");
    await expect(image).not.toHaveAttribute("src", previous!);
    await expect(card.locator("time")).toHaveAttribute("datetime", timestamp!);
    await expect(avatar.locator(".avatar-image")).toHaveCSS("filter", "none");
    await expect(avatar.locator(".avatar-shuffle")).toHaveCSS("opacity", "0");
    await card.getByRole("button", { name: "More options for Marmot research", exact: true }).click();
    await expect(card.getByRole("menuitem", { name: "Delete", exact: true })).toHaveClass(/\bdelete-option\b/);
    await page.keyboard.press("Escape");

    for (const theme of ["light", "dark"]) {
      await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
      for (const width of [320, 1000]) {
        await page.setViewportSize({ width, height: 900 });
        expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
        const aligned = await card.evaluate(el => {
          const time = el.querySelector("time")!.getBoundingClientRect();
          const row = el.querySelector(".chat-title-row")!.getBoundingClientRect();
          const menu = el.querySelector(".chat-menu-trigger")!.getBoundingClientRect();
          return time.right <= menu.left && Math.abs(row.right - menu.right) < 2;
        });
        expect(aligned).toBe(true);
        const picker = card.locator(".chat-color .accent-picker > button");
        const placement = await card.evaluate(el => {
          const outer = el.getBoundingClientRect();
          const picker = el.querySelector(".chat-color .accent-picker > button")!.getBoundingClientRect();
          const css = getComputedStyle(el);
          return {
            left: picker.left - outer.left - parseFloat(css.paddingLeft),
            right: outer.right - picker.right - parseFloat(css.paddingRight),
            bottom: outer.bottom - picker.bottom - parseFloat(css.paddingBottom),
          };
        });
        expect(placement.left).toBeGreaterThanOrEqual(-1);
        expect(placement.right).toBeGreaterThanOrEqual(-1);
        expect(placement.bottom).toBeGreaterThanOrEqual(-1);
        await card.screenshot({ path: testInfo.outputPath(`saved-chat-${theme}-${width}.png`) });
        await picker.click();
        await expect(card.getByRole("radio", { name: "Rose", exact: true })).toBeVisible();
        expect(await card.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
        await card.screenshot({ path: testInfo.outputPath(`saved-chat-colors-open-${theme}-${width}.png`) });
        await picker.click();
        await expect(picker).toHaveAttribute("aria-expanded", "false");
      }
    }
    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect.poll(() => avatar.locator(".avatar-image").evaluate(el => parseFloat(getComputedStyle(el).transitionDuration))).toBeLessThanOrEqual(0.001);
    await page.reload();
    await expect(page.locator(".shell")).toBeVisible();
    await mountHome();
    await expect(page.locator("[data-saved-conversation]").first().locator(".chat-name .morph-source")).toHaveText("Marmot research");
    const downloading = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download backup of Marmot research", exact: true }).click();
    const download = await downloading;
    const text = await readStreamText(await download.createReadStream());
    const backup = JSON.parse(text);
    expect(backup.conversation.name).toBe("Marmot research");
    await page.getByLabel("Import chat backup file").setInputFiles({ name: "backup.json", mimeType: "application/json", buffer: Buffer.from(text) });
    await expect(page.getByRole("status").filter({ hasText: "Imported" })).toContainText("Marmot research");
    const imported = await page.evaluate(async ({ url, id }) => {
      const { conversations } = await (await import(url)).conversationLibrary.list();
      return conversations.find((record: any) => record.name === "Marmot research" && record.id !== id);
    }, { url: `/@fs${resolve("src/lib/stores/savedConversations.svelte.ts")}`, id: backup.conversation.id });
    expect({ ...imported, id: backup.conversation.id }).toEqual(backup.conversation);
  });
});

for (const touch of [false, true]) {
test.describe(touch ? "touch Loom sizing" : "pointer Loom sizing", () => {
test.use({ hasTouch: touch, isMobile: touch });
test("Loom cards fit wrapped tokens and keep the final row clear of the footer", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await workbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Hello, I love marmots!");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: /^Loom\b/ }).click();
  await openWorkspaceMenu(page);
  await page.getByRole("button", { name: "Show Loom tools", exact: true }).click();
  await page.getByRole("button", { name: /^Map\b/ }).click();
  let compactHeight = 0;
  for (const long of [false, true, false]) {
    const id = await page.evaluate(async ({ url, long }) => {
      const { loomTree } = await import(url);
      const id = loomTree.active_node_id;
      const node = loomTree.nodes.get(id);
      const words = (long ? "What do you enjoy most about them? ".repeat(16) : "What do you enjoy most about them?").split(" ");
      const tokens = words.map((word, index) => ({ ...node.tokens[0], text: `${word} `, raw_index: index, token_id: index }));
      loomTree.nodes.set(id, { ...node, text: tokens.map(token => token.text).join(""), tokens, mean_logprob: -0.24 });
      return id;
    }, { url: storesUrl, long });
    const node = page.locator(`[data-loom-node-id="${id}"] .node`);
    const field = node.locator(".token-field");
    await expect.poll(() => field.evaluate(el => el.scrollHeight - el.clientHeight)).toBeLessThanOrEqual(1);
    if (long) await expect.poll(() => node.evaluate(el => (el as HTMLElement).offsetHeight)).toBeGreaterThan(compactHeight + 40);
    else if (compactHeight) await expect.poll(() => node.evaluate(el => (el as HTMLElement).offsetHeight)).toBeLessThanOrEqual(compactHeight + 2);
    else compactHeight = await node.evaluate(el => (el as HTMLElement).offsetHeight);
    await expect.poll(() => node.evaluate(el => {
      const card = el.getBoundingClientRect();
      const field = el.querySelector(".token-field")!.getBoundingClientRect();
      const last = el.querySelector(".sentence-node:last-child")!.getBoundingClientRect();
      const tools = el.querySelector(".node-tools")!.getBoundingClientRect();
      const meta = el.querySelector(".node-meta")?.getBoundingClientRect();
      const zoom = card.height / (el as HTMLElement).offsetHeight;
      return last.bottom <= field.bottom + 1 && field.bottom <= (meta?.top ?? tools.top) && card.bottom - tools.bottom >= 7 * zoom;
    })).toBe(true);
    await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
    await node.screenshot({ path: testInfo.outputPath(`loom-footer-${long ? "long" : "wrapped"}.png`) });
  }
  expect(errors).toEqual([]);
});
});
}

test("Thinking is shown only for supported models and follows model changes", async ({ page }) => {
  await workbench(page);
  await selectWorkspaceView(page, "Controls");
  const thinking = page.getByRole("checkbox", { name: "Thinking", exact: true });
  for (const [supported, optional] of [[false, false], [true, true], [true, false], [false, false]]) {
    await page.evaluate(async ({ url, supported, optional }) => {
      const { sessionState } = await import(url);
      sessionState.info = { ...sessionState.info, supports_thinking: supported, thinking_is_optional: optional };
    }, { url: storesUrl, supported, optional });
    if (!supported) {
      await expect(thinking).toHaveCount(0);
      await expect(page.getByRole("button", { name: "About Thinking", exact: true })).toHaveCount(0);
    } else {
      await expect(thinking).toBeVisible();
      if (optional) await expect(thinking).toBeEnabled();
      else {
        await expect(thinking).toBeDisabled();
        await expect(thinking).toBeChecked();
      }
    }
  }
});

test("controls tabs have balanced padding and unclipped keyboard focus", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await selectWorkspaceView(page, "Controls");
  const nav = page.locator(".controls-nav");
  const tabs = page.getByRole("group", { name: "Controls section", exact: true });
  for (const width of [320, 390, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    for (const section of ["Response", "Model"]) {
      const button = tabs.getByRole("button", { name: section, exact: true });
      await button.focus();
      await page.keyboard.press("Enter");
      const spacing = await nav.evaluate(el => {
        const outer = el.getBoundingClientRect();
        const inner = el.querySelector(".sk-tabs")!.getBoundingClientRect();
        const style = getComputedStyle(el);
        return { top: inner.top - outer.top, bottom: outer.bottom - inner.bottom - parseFloat(style.borderBottomWidth), left: inner.left - outer.left, right: outer.right - inner.right, fits: el.scrollWidth <= el.clientWidth };
      });
      expect(spacing.top).toBeGreaterThanOrEqual(12);
      expect(Math.abs(spacing.top - spacing.bottom)).toBeLessThan(1);
      expect(spacing.left).toBeGreaterThanOrEqual(12);
      expect(spacing.right).toBeGreaterThanOrEqual(12);
      expect(spacing.fits).toBe(true);
      await expect(button).toBeFocused();
    }
    await nav.screenshot({ path: testInfo.outputPath(`controls-tabs-${width}.png`) });
  }
});

test("model summary shows its logo, readable name and Drowse version", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await workbench(page);
  const summary = page.getByRole("button", { name: "Open model controls", exact: true });
  for (const [id, name, provider, base] of [
    ["gemma3-1b-instruct", "Gemma 3 1B", "gemma", false],
    ["Qwen/Qwen3-1.7B", "Qwen3 1.7B", "qwen", false],
    ["google/gemma-3-1b-pt", "Gemma 3 1B[BASE]", "gemma", true],
  ] as const) {
    await page.evaluate(async ({ module, id, base }) => {
      const { sessionState } = await import(module);
      sessionState.info = { ...sessionState.info, model_id: id, is_base_model: base };
    }, { module: storesUrl, id, base });
    await expect(summary.locator(".model")).toHaveText(name);
    await expect(summary.locator(`[data-provider="${provider}"]`)).toBeVisible();
    await expect(summary.locator(".version")).toHaveText("Drowse 0.1");
    await expect(summary).not.toContainText(/webgpu|q4f16|float32/i);
    const layout = await summary.evaluate(el => {
      const logo = el.querySelector(".model-provider-logo")!.getBoundingClientRect();
      const name = el.querySelector(".model")!.getBoundingClientRect();
      const row = el.querySelector(".model-identity")!.getBoundingClientRect();
      const version = el.querySelector(".version")!.getBoundingClientRect();
      return { centerDelta: Math.abs(logo.y + logo.height / 2 - name.y - name.height / 2), edgeDelta: Math.abs(row.x - version.x), fits: el.scrollWidth <= el.clientWidth };
    });
    expect(layout.centerDelta).toBeLessThan(1);
    expect(layout.edgeDelta).toBeLessThan(1);
    expect(layout.fits).toBe(true);
  }
  await summary.screenshot({ path: testInfo.outputPath("model-summary.png") });
  await summary.click();
  await expect(page.getByRole("region", { name: "Model controls", exact: true })).toBeVisible();
});

test("fields use one soft focus shadow without stacked outlines", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1280, height: 900 });
  await workbench(page);
  await page.evaluate(async module => {
    const { sessionState } = await import(module);
    sessionState.info = { ...sessionState.info, is_base_model: false, user_role_supported: true, role_substitution_supported: true };
  }, storesUrl);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  const role = page.getByRole("combobox", { name: "You write as", exact: true });
  await page.getByRole("button", { name: /^Roles / }).click();
  for (const theme of ["light", "dark"] as const) {
    await page.evaluate(async ({ module, theme }) => {
      const { setTheme } = await import(module);
      setTheme(theme);
    }, { module: `/@fs/${resolve("src/lib/theme.ts")}`, theme });
    for (const width of [1280, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await role.click();
      await expect(role).toBeFocused();
      await expect(role).toHaveCSS("outline-style", "none");
      const wrapper = role.locator("..");
      await expect(wrapper).toHaveCSS("outline-style", "none");
      expect(await wrapper.evaluate(el => getComputedStyle(el).boxShadow)).toContain("inset");
      await wrapper.screenshot({ path: testInfo.outputPath(`role-focus-${theme}-${width}.png`) });
      await page.keyboard.press("Escape");
      await composer.fill("A softer place to write.");
      await expect(composer).toHaveCSS("outline-style", "none");
      await expect(composer).toHaveCSS("box-shadow", "none");
      expect(await composer.evaluate(el => getComputedStyle(el).boxShadow)).not.toContain("0px 0px 0px 2px");
      await expect(composer).toBeFocused();
      await composer.screenshot({ path: testInfo.outputPath(`composer-focus-${theme}-${width}.png`) });
      expect(await composer.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    }
  }
  await role.focus();
  await page.keyboard.press("ArrowDown");
  await expect(role).toHaveAttribute("aria-expanded", "true");
  await page.keyboard.press("Escape");
  await expect(role).toHaveAttribute("aria-expanded", "false");
  await page.keyboard.press("Tab");
  expect(await page.locator(":focus").count()).toBe(1);
  await page.emulateMedia({ forcedColors: "active" });
  await composer.focus();
  await expect(composer).toHaveCSS("outline-style", "solid");
  await expect(composer).toHaveCSS("outline-width", "2px");
});

test("Geometry browses only recorded linear probe layers without changing the aggregate", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Inspect probe layers.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.evaluate(async url => {
    const { chatLog, probeRack } = await import(url);
    const reading = {
      fraction: 0.45, coords: [0.125, 0.25], residual: 0, nearest: [],
      coords_per_layer: { "3": [-0.75, 0.2], "13": [0.6, -0.4], "21": [0.95, 0.8] },
      fraction_per_layer: { "3": 0.2, "13": 0.55, "21": 0.7 }, residual_per_layer: {},
    };
    chatLog.turns.at(-1).tokens[0].measurements.instruments.geometry = { readings: {
      "local/linear": reading,
      "local/aggregate-only": { ...reading, coords_per_layer: {}, fraction_per_layer: {} },
      "local/curved": { ...reading, residual: 0 },
      "local/unknown": { ...reading, residual: 0 },
    } };
    for (const name of ["local/linear", "local/aggregate-only", "local/curved"]) {
      probeRack.entries.set(name, { info: { name, family: "geometry", is_affine: name !== "local/curved", node_labels: ["a", "b", "c"], node_coords: [[-1, 0], [0, 1], [1, 0]] }, sparkline: [], current: 0, previous: 0, perLayer: {}, reading: null, aggregate: null });
    }
  }, storesUrl);
  await openDrawer(page, "token_drilldown", { turnIdx: 1, tokenIdx: 0 });
  const sheet = page.locator('aside[aria-label="Token drilldown"]');
  await sheet.getByRole("button", { name: /^geometry\b/i }).click();
  const region = sheet.getByRole("region", { name: "local/linear layer readings", exact: true });
  await expect(region).toBeVisible();
  await expect(sheet.getByRole("region", { name: "local/aggregate-only layer readings" })).toHaveCount(0);
  await expect(sheet.getByRole("region", { name: "local/curved layer readings" })).toHaveCount(0);
  await expect(sheet.getByRole("region", { name: "local/unknown layer readings" })).toHaveCount(0);
  await expect(sheet.locator(".geo-list > .card").filter({ has: page.getByText("local/unknown", { exact: true }) }).locator(".statline")).toContainText("geometry");
  await expect(sheet.locator(".geo-list > .card").filter({ has: page.getByText("local/curved", { exact: true }) }).locator(".statline")).toContainText("manifold");
  await expect(region.getByRole("group", { name: "local/linear layer 3 axis 0", exact: true })).toContainText("-0.750");
  const selector = region.getByRole("button", { name: "local/linear layer", exact: true });
  await selector.click();
  await expect(page.getByRole("option")).toHaveText(["Layer 3", "Layer 13", "Layer 21"]);
  await page.getByRole("option", { name: "Layer 13", exact: true }).click();
  await expect(region.getByRole("group", { name: "local/linear layer 13 axis 0", exact: true })).toContainText("0.600");
  await expect(region.getByRole("group", { name: "local/linear layer 13 axis 1", exact: true })).toContainText("-0.400");
  await expect(region).toContainText("0.550");
  await expect(sheet.getByRole("group", { name: "local/linear axis 0", exact: true })).toContainText("0.125");
  await page.mouse.move(0, 0);
  await selector.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("listbox")).toBeFocused();
  await page.keyboard.press("End");
  await expect(page.getByRole("option", { name: "Layer 21", exact: true })).toHaveClass(/is-highlight/);
  await page.keyboard.press("Enter");
  await expect(region.getByRole("group", { name: "local/linear layer 21 axis 0", exact: true })).toContainText("0.950");
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: `/@fs/${resolve("src/lib/theme.ts")}`, theme });
    for (const width of [1280, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await region.scrollIntoViewIfNeeded();
      expect(await sheet.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await region.screenshot({ path: testInfo.outputPath(`geometry-layers-${theme}-${width}.png`) });
    }
  }
  await page.evaluate(async url => {
    const { chatLog } = await import(url);
    const reading = chatLog.turns.at(-1).tokens[0].measurements.instruments.geometry.readings["local/linear"];
    reading.coords_per_layer = { "13": [0.3, -0.2] };
    reading.fraction_per_layer = {};
  }, storesUrl);
  await expect(region.getByRole("group", { name: "local/linear layer 13 axis 0", exact: true })).toContainText("0.300");
  await expect(region).not.toContainText("Subspace fraction");
});

test("SAE stays on its recorded layer and automatically loads source-matched descriptions", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1280, height: 1000 });
  const requests: string[] = [];
  let failDescriptionOnce = true;
  await page.route("**/sae-descriptions/*.json", route => route.fulfill({ status: 503 }));
  await page.route("https://www.neuronpedia.org/api/feature/**", async route => {
    const request = route.request();
    requests.push(request.url());
    expect(request.postData()).toBeNull();
    const id = request.url().split("/").at(-1)!;
    if (id === "16190" && failDescriptionOnce) {
      failDescriptionOnce = false;
      await route.fulfill({ status: 503, body: "Temporarily unavailable" });
      return;
    }
    const labels: Record<string, string> = { "16190": "terms and phrases", "13181": "particular conjunctions", "2286": "Reuters" };
    await route.fulfill({ json: {
      modelId: "gemma-3-1b-it", layer: "13-gemmascope-2-res-16k", index: id,
      source: { hfRepoId: "google/gemma-scope-2-1b-it", hfFolderId: "resid_post/layer_13_width_16k_l0_medium" },
      explanations: labels[id] ? [{ description: labels[id], explanationModelName: "gemini-2.5-flash-lite" }] : [],
    } });
  });
  await workbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Inspect feature activations.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await page.evaluate(async url => {
    const { chatLog } = await import(url);
    chatLog.turns.at(-1).tokens[0].measurements.instruments.sae = {
      binding: { source: "fixture-sae", steering: null, layer: 13 },
      readout: { features: [396, 16190, 13181, 2286].map((id, index) => ({ id, activation: [975.08, 882.16, 881.86, 798.36][index], label: null, max_act: null })) },
    };
  }, storesUrl);
  await openDrawer(page, "token_drilldown", { turnIdx: 1, tokenIdx: 0 });
  const sheet = page.locator('aside[aria-label="Token drilldown"]');
  await sheet.getByRole("button", { name: /^sae\b/i }).click();
  await expect.poll(() => page.evaluate(async url => {
    const { saeSourceState } = await import(url);
    return !saeSourceState.loading && saeSourceState.sources.length > 0;
  }, storesUrl)).toBe(true);
  await page.evaluate(async url => {
    const { saeSourceState } = await import(url);
    saeSourceState.sources = [{ source: "fixture-sae", active: true, layer: 13, model_layers: Array.from({ length: 26 }, (_, i) => i), description_source: {
      model: "gemma-3-1b-it", source: "13-gemmascope-2-res-16k", repository: "google/gemma-scope-2-1b-it", folder: "resid_post/layer_13_width_16k_l0_medium",
    } }];
  }, storesUrl);
  await expect(sheet.getByRole("region", { name: "SAE layer explorer" })).toHaveCount(0);
  await expect(sheet.getByRole("button", { name: /SAE layer/ })).toHaveCount(0);
  await expect(sheet.getByRole("listitem", { name: /^SAE feature .*layer 13$/ })).toHaveCount(4);
  await expect(sheet.getByRole("alert")).toContainText("Some descriptions could not be loaded");
  await expect(sheet.getByText("Reuters", { exact: true })).toBeVisible();
  await sheet.getByRole("button", { name: "Retry descriptions", exact: true }).click();
  await expect(sheet.getByRole("button", { name: "Descriptions checked" })).toBeDisabled();
  expect(requests.filter(url => url.endsWith("/16190"))).toHaveLength(2);
  expect(requests.filter(url => url.endsWith("/2286"))).toHaveLength(1);
  await expect(sheet.getByText("No description published for this feature", { exact: true })).toHaveCount(1);
  await expect(sheet.locator(".sae-value")).toHaveText(["975.08", "882.16", "881.86", "798.36"]);
  await expect(sheet.getByRole("link", { name: /Neuronpedia/ })).toHaveCount(4);
  await sheet.getByRole("searchbox", { name: "Find a feature" }).fill("Reuters");
  await expect(sheet.getByRole("listitem", { name: /^SAE feature/ })).toHaveCount(1);
  await expect(sheet.getByRole("listitem", { name: "SAE feature 2286, layer 13" })).toContainText("#4");
  await sheet.getByRole("searchbox", { name: "Find a feature" }).fill("");
  await sheet.getByRole("button", { name: "Sort by", exact: true }).click();
  await page.getByRole("option", { name: "Feature ID", exact: true }).click();
  await expect(sheet.getByRole("listitem", { name: /^SAE feature/ }).first()).toHaveAttribute("aria-label", "SAE feature 396, layer 13");
  await expect(sheet.getByRole("listitem", { name: /^SAE feature/ })).toHaveCount(4);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => { (await import(url)).setTheme(theme); }, { url: `/@fs/${resolve("src/lib/theme.ts")}`, theme });
    for (const width of [1280, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await expect.poll(() => sheet.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await sheet.getByRole("list", { name: "Top SAE features" }).scrollIntoViewIfNeeded();
      await sheet.screenshot({ path: testInfo.outputPath(`sae-readings-${theme}-${width}.png`) });
    }
  }
});

test.describe("token detail tooltip copy", () => {
  test.use({ isMobile: false, hasTouch: false });

  test("routine controls stay quiet while technical help and keyboard resizing remain available", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.setViewportSize({ width: 1280, height: 900 });
    await workbench(page);
    await showWorkspaceTools(page, "chat");
    await page.getByRole("button", { name: /^Roles / }).click();
    const tooltip = page.locator("#drowse-tooltip");
    const quiet = [
      page.getByRole("slider", { name: "Resize writing area" }),
      page.getByRole("button", { name: "Color generated words by" }),
      page.getByRole("button", { name: "Swap writer roles" }),
      page.getByRole("button", { name: "Role settings", exact: true }),
      page.locator(".workspace-nav").getByRole("button", { name: "Controls", exact: true }),
    ];
    for (const control of quiet) {
      await expect(control).not.toHaveAttribute("title");
      await control.hover();
      await page.waitForTimeout(350);
      await expect(tooltip).toHaveCount(0);
    }
    const composer = page.getByRole("textbox", { name: /^Compose as / });
    const before = (await composer.boundingBox())!.height;
    await quiet[0].focus();
    await page.keyboard.press("ArrowUp");
    await expect.poll(async () => (await composer.boundingBox())!.height).toBeGreaterThan(before);
    await expect(quiet[0]).toHaveAccessibleDescription("Use arrow keys to resize. Double-click to reset.");
    const help = page.getByRole("button", { name: "About word colors", exact: true });
    await help.focus();
    await expect(page.getByRole("tooltip").filter({ visible: true })).toBeVisible();
    await page.keyboard.press("Escape");
    await page.locator(".workspace-nav").getByRole("button", { name: "Loom", exact: true }).click();
    await showWorkspaceTools(page, "Loom");
    await expect(page.locator(".loom-views button[title]")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Cut branch…", exact: true })).toHaveAttribute("aria-description", "Remove the current turn and all branches that follow it");
  });

  test("readout hints explain provenance without repeating source IDs", async ({ page }, testInfo) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.setViewportSize({ width: 1280, height: 900 });
    await workbench(page);
    await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain language models.");
    await page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ }).click();
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    await openDrawer(page, "token_drilldown", { turnIdx: 1, tokenIdx: 0 });
    const sheet = page.locator('aside[aria-label="Token drilldown"]');
    await expect(sheet.locator('.scrub[title]')).toHaveCount(0);
    await expect(sheet.getByRole("button", { name: "Next token", exact: true })).toHaveAttribute("aria-description", "Inspect the next token");
    for (const tab of [/sae/, /j-lens/]) {
      await sheet.getByRole("group", { name: "Token detail view", exact: true }).getByRole("button", { name: tab }).click();
      const source = sheet.locator(".inst-head .source");
      await expect(source).toBeVisible();
      await expect(source).not.toHaveAttribute("title");
      await source.hover();
      await expect(page.locator("#drowse-tooltip")).toHaveCount(0);
      const origin = sheet.locator(".inst-head .origin");
      await expect(origin).toHaveText("captured");
      await origin.hover();
      await expect(page.locator("#drowse-tooltip")).toHaveCount(0);
      await sheet.getByRole("button", { name: "About readout provenance", exact: true }).click();
      await expect(sheet.locator(".info-popover:popover-open")).toHaveText("Recorded when this token was generated. No new model run was needed.");
      await page.keyboard.press("Escape");
      await expect(page.locator("#drowse-tooltip")).toHaveCount(0);
      for (const width of [320, 1280]) {
        await page.setViewportSize({ width, height: 900 });
        expect(await source.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
        await expect.poll(() => sheet.evaluate(el => el.scrollWidth <= Math.ceil(el.getBoundingClientRect().width))).toBe(true);
      }
    }
    await sheet.screenshot({ path: testInfo.outputPath("readout-tooltip-cleanup.png") });
  });
});

test.describe("token details wheel and keyboard scrolling", () => {
  // Playwright cannot dispatch wheel events in a mobile WebKit context.
  test.use({ isMobile: false });

test("token details scroll past branching controls in every analysis tab", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain language models.");
  await page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await openDrawer(page, "token_drilldown", { turnIdx: 1, tokenIdx: 0 });
  const sheet = page.locator('aside[aria-label="Token drilldown"]');
  const branch = sheet.locator(".branch-point");
  await expect(branch).toBeVisible();
  await expect(sheet).toHaveAttribute("tabindex", "-1");
  expect(await sheet.evaluate(element => {
    const focusRoot = element.closest('[role="dialog"]') ?? element;
    const focused = document.activeElement;
    return focused instanceof HTMLElement && focusRoot.contains(focused) && focused.matches("button, input, select, textarea, a[href]");
  })).toBe(true);
  await expect(sheet).toHaveCSS("overflow-y", "auto");
  await expect(sheet.locator(":scope > .body")).toHaveCSS("overflow-y", "visible");
  for (const width of [320, 390, 1280]) {
    await page.setViewportSize({ width, height: 600 });
    for (const tab of [/geometry/, /logits/, /sae/, /j-lens/]) {
      await sheet.getByRole("group", { name: "Token detail view", exact: true }).getByRole("button", { name: tab }).click();
      const canScroll = await sheet.evaluate(element => {
        const branch = element.querySelector(".branch-point")!;
        const offset = branch.getBoundingClientRect().top - element.getBoundingClientRect().top + element.scrollTop;
        element.scrollTop = Math.max(0, offset - element.clientHeight * 0.65);
        return element.scrollHeight > element.clientHeight;
      });
      if (!canScroll) continue;
      const before = await sheet.evaluate(element => element.scrollTop);
      const bounds = (await branch.boundingBox())!;
      const sheetBounds = (await sheet.boundingBox())!;
      await page.mouse.move(bounds.x + bounds.width / 2, Math.min(bounds.y + 20, sheetBounds.y + sheetBounds.height - 20));
      await page.mouse.wheel(0, 600);
      await expect.poll(() => sheet.evaluate(element => element.scrollTop)).toBeGreaterThan(before);
      expect(await sheet.locator(":scope > .body").evaluate(element => element.scrollTop)).toBe(0);
      expect(await sheet.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    }
  }
  await page.setViewportSize({ width: 390, height: 600 });
  await sheet.evaluate(element => { element.scrollTop = 0; });
  await sheet.focus();
  await expect(sheet).toHaveCSS("outline-style", "none");
  await page.keyboard.press("PageDown");
  await expect.poll(() => sheet.evaluate(element => element.scrollTop)).toBeGreaterThan(0);
  await page.screenshot({ path: testInfo.outputPath("token-details-scrolled-phone.png") });
  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);
});
});

test("dialog containers stay unselected while controls retain keyboard focus", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  const trigger = page.getByRole("button", { name: "Hide left sidebar", exact: true });
  await trigger.focus();
  await openDrawer(page, "help");
  const dialog = page.getByRole("dialog", { name: "Help and shortcuts", exact: true });
  await expect(dialog).toBeVisible();
  const close = dialog.getByRole("button", { name: /close/i }).first();
  for (const theme of ["dark", "light"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    await dialog.focus();
    await expect(dialog).toHaveCSS("outline-style", "none");
    await page.keyboard.press("Tab");
    await expect(close).toBeFocused();
    await expect(close).toHaveCSS("outline-style", "solid");
    await expect(close).toHaveCSS("outline-width", "2px");
    await dialog.screenshot({ path: testInfo.outputPath(`dialog-focus-${theme}.png`) });
    await dialog.click({ position: { x: 20, y: 20 } });
    await expect(dialog).toHaveCSS("outline-style", "none");
  }
  const copyable = dialog.locator(".intro p").first();
  expect(await copyable.evaluate(element => {
    const selection = window.getSelection()!;
    const range = document.createRange();
    range.selectNodeContents(element);
    selection.removeAllRanges();
    selection.addRange(range);
    const result = selection.toString() === element.textContent && getComputedStyle(element).userSelect !== "none";
    selection.removeAllRanges();
    return result;
  })).toBe(true);
  await page.emulateMedia({ forcedColors: "active" });
  await dialog.focus();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await expect(close).toHaveCSS("outline-style", "solid");
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
});

test("section headings stand alone without redundant eyebrow labels", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await workbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain how language models work.");
  await page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await selectWorkspaceView(page, "Controls");
  await expect(page.getByRole("heading", { name: "Generation settings", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Instruments", exact: true })).toBeVisible();
  await expect(page.locator(".inspector .section-heading p")).toHaveCount(0);
  await page.getByRole("group", { name: "Controls section" }).getByRole("button", { name: "Model", exact: true }).click();
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls.getByRole("heading", { name: "qwen3-1.7b-fixture", exact: true })).toBeVisible();
  await expect(modelControls.locator(".eyebrow")).toHaveCount(0);
  await openDrawer(page, "help");
  await expect(page.getByRole("heading", { name: "What each area does", exact: true })).toBeVisible();
  await expect(page.getByText("The workbench", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Close drawer", exact: true }).click();
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  for (const [button, heading] of [[/^Current path/, "Current path"], [/^Next options/, "Next options"], [/^Starred/, "Starred points"]] as const) {
    await selectLoomView(page, button);
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
    await expect(page.locator(".projection-kicker")).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await page.screenshot({ path: testInfo.outputPath("loom-heading-phone.png") });
});

test("theme crossfade gently blends the whole page and survives rapid switches", async ({ page }, testInfo) => {
  test.slow();
  await page.emulateMedia({ reducedMotion: "no-preference", colorScheme: "dark" });
  await page.setViewportSize({ width: 1280, height: 850 });
  await page.goto(`${devUrl}/`);
  const root = page.locator("html");
  await expect(root).toHaveAttribute("data-theme", "dark");
  await expect(root).not.toHaveAttribute("data-theme-transition");
  await page.evaluate(() => {
    const start = document.startViewTransition.bind(document);
    let paused = false;
    document.startViewTransition = ((...args: Parameters<typeof start>) => {
      const transition = start(...args);
      void transition.ready.then(() => {
        if (paused) return;
        paused = true;
        const animation = document.getAnimations().find(animation =>
          animation instanceof CSSAnimation && animation.animationName === "theme-fade-out");
        animation?.pause();
      }, () => {});
      return transition;
    }) as typeof document.startViewTransition;
  });
  await setAppearance(page, "Light");
  await expect(root).toHaveAttribute("data-theme", "light");
  await expect(root).toHaveAttribute("data-theme-transition", "snapshot");
  const fade = await page.evaluate(() => {
    const root = document.documentElement;
    const old = getComputedStyle(root, "::view-transition-old(root)");
    const next = getComputedStyle(root, "::view-transition-new(root)");
    const animation = document.getAnimations().find(animation =>
      animation instanceof CSSAnimation && animation.animationName === "theme-fade-out");
    animation?.pause();
    if (animation) animation.currentTime = 110;
    return { duration: old.animationDuration, easing: old.animationTimingFunction,
      blend: old.mixBlendMode, nextAnimation: next.animationName, found: Boolean(animation) };
  });
  expect(fade).toEqual({ duration: "0.22s", easing: "cubic-bezier(0.4, 0, 0.2, 1)", blend: "normal", nextAnimation: "none", found: true });
  await page.screenshot({ path: testInfo.outputPath("theme-crossfade-midpoint.png") });
  await page.evaluate(() => document.getAnimations().forEach(animation => {
    if (animation instanceof CSSAnimation && animation.animationName === "theme-fade-out") animation.play();
  }));
  await expect(root).not.toHaveAttribute("data-theme-transition");
  await setAppearance(page, "Dark");
  await expect(root).toHaveAttribute("data-theme", "dark");
  await expect(root).not.toHaveAttribute("data-theme-transition");
  await page.evaluate(async module => {
    const { setTheme } = await import(module);
    setTheme("light"); setTheme("dark"); setTheme("light");
  }, `/@fs/${resolve("src/lib/theme.ts")}`);
  await expect(root).toHaveAttribute("data-theme", "light");
  await expect(root).not.toHaveAttribute("data-theme-transition");
  expect(await page.evaluate(() => localStorage.getItem("drowse.theme"))).toBe("light");
});

test("theme transitions honor reduced motion and fall back to gradual colors", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce", colorScheme: "dark" });
  await page.goto(`${devUrl}/`);
  const root = page.locator("html");
  await setAppearance(page, "Light");
  await expect(root).toHaveAttribute("data-theme", "light");
  await expect(root).not.toHaveAttribute("data-theme-transition");
  await page.evaluate(() => Object.defineProperty(document, "startViewTransition", { value: undefined }));
  await page.emulateMedia({ reducedMotion: "no-preference" });
  const fallback = await page.getByRole("group", { name: "Appearance" })
    .getByRole("button", { name: "Dark", exact: true }).evaluate(button => {
      button.click();
      return {
        mode: document.documentElement.dataset.themeTransition,
        duration: getComputedStyle(document.body).transitionDuration,
      };
    });
  expect(fallback).toEqual({ mode: "fallback", duration: "0.22s" });
  await expect(root).toHaveAttribute("data-theme", "dark");
  await expect(root).not.toHaveAttribute("data-theme-transition");
  await page.reload();
  await expect(root).toHaveAttribute("data-theme", "dark");
  await expect(root).not.toHaveAttribute("data-theme-transition");
});

test("tab identity has static share metadata and branded assets", async ({ page, request }) => {
  const response = await request.get("/");
  const html = await response.text();
  expect(html).toContain("<title>Drowse</title>");
  expect(html).toContain('property="og:site_name" content="Drowse"');
  expect(html).toContain('name="twitter:card" content="summary_large_image"');
  expect(html).toContain('property="og:image:width" content="1200"');
  expect(html).not.toContain("__DROWSE_");
  const image = await request.get("/social/drowse.png");
  expect(image.ok()).toBe(true);
  expect(image.headers()["content-type"]).toContain("image/png");
  await page.goto("/");
  await expect(page).toHaveTitle("Drowse");
  await expect(page.locator('link[rel="icon"]')).toHaveCount(1);
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-state", "home");
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("href", /\/icons\/tab-home-(light|dark)\.png/);
});

test("tab icons use squircle tiles and follow every chat accent", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(devUrl);
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-state", "home");
  await page.evaluate(async url => {
    (window as any).__testTab = (await import(url)).createTabIcon();
    (window as any).__testTab.update("home");
  }, `/@fs${resolve("src/lib/tabIdentity.ts")}`);
  for (const theme of ["light", "dark"] as const) {
    for (const accent of CHAT_ACCENTS) {
      const result = await page.evaluate(async ({ theme, accent }) => {
        document.documentElement.dataset.theme = theme;
        document.documentElement.dataset.chatAccent = accent.id;
        const link = document.querySelector<HTMLLinkElement>('link[rel="icon"]')!;
        const expected = accent[theme];
        for (let attempt = 0; attempt < 100; attempt++) {
          await new Promise(requestAnimationFrame);
          const image = new Image();
          image.src = link.href;
          await image.decode();
          const canvas = document.createElement("canvas");
          canvas.width = canvas.height = 96;
          const context = canvas.getContext("2d")!;
          context.drawImage(image, 0, 0);
          const pixel = [...context.getImageData(48, 8, 1, 1).data];
          const hex = `#${pixel.slice(0, 3).map(value => value.toString(16).padStart(2, "0")).join("")}`;
          if (hex === expected) return { hex, corner: context.getImageData(0, 0, 1, 1).data[3], ink: [...context.getImageData(48, 48, 1, 1).data] };
        }
        throw new Error(`Favicon did not adopt ${theme}/${accent.id}`);
      }, { theme, accent });
      expect(result.hex).toBe(accent[theme]);
      expect(result.corner).toBe(0);
      expect(result.ink).toEqual(theme === "dark" ? [20, 24, 34, 255] : [255, 255, 255, 255]);
    }
  }
  await page.evaluate(() => {
    (window as any).__testTab.update("working");
    document.documentElement.dataset.chatAccent = "rose";
    document.documentElement.dataset.chatAccent = "mint";
  });
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-accent", "mint");
  await page.evaluate(() => { delete document.documentElement.dataset.chatAccent; });
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("href", "/icons/tab-working-dark-0.png?v=fluent");
  await page.evaluate(() => (window as any).__testTab.dispose());
  await expect(page.locator('link[rel="icon"]')).not.toHaveAttribute("data-accent");
});

test("tab identity updates each state and releases its animation lifecycle", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1100, height: 950 });
  await page.goto(`${devUrl}/`);
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-state", "home");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.evaluate(async module => {
    const { createTabIcon, faviconPath, tabLabels } = await import(module);
    const icon = createTabIcon();
    (window as any).__tabIcon = icon;
    icon.update("loading");
    const gallery = document.createElement("section");
    gallery.style.cssText = "position:fixed;inset:0;z-index:9999;background:white;padding:32px;display:flex;gap:24px;flex-wrap:wrap;align-content:start;color:black";
    gallery.id = "icon-gallery";
    for (const state of Object.keys(tabLabels)) {
      const item = document.createElement("div");
      item.style.cssText = "width:130px;display:flex;gap:10px;align-items:center;flex-wrap:wrap";
      for (const size of [16, 32, 64]) {
        const image = document.createElement("img");
        image.width = size; image.height = size; image.src = faviconPath(state); image.alt = state;
        item.append(image);
      }
      const label = document.createElement("span"); label.textContent = state; item.append(label);
      gallery.append(item);
    }
    document.body.append(gallery);
  }, `/@fs/${resolve("src/lib/tabIdentity.ts")}`);
  const icon = page.locator('link[rel="icon"]');
  await expect(icon).toHaveAttribute("data-state", "loading");
  const first = await icon.getAttribute("href");
  await expect.poll(() => icon.getAttribute("href")).not.toBe(first);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(icon).toHaveAttribute("href", /\/icons\/tab-loading-(light|dark)-0\.png/);
  const still = await icon.getAttribute("href");
  await page.waitForTimeout(1100);
  await expect(icon).toHaveAttribute("href", still!);
  await expect.poll(() => page.locator("#icon-gallery img").evaluateAll(images => images.every(image => (image as HTMLImageElement).complete && (image as HTMLImageElement).naturalWidth === 96))).toBe(true);
  await page.locator("#icon-gallery").screenshot({ path: testInfo.outputPath("favicon-family.png") });
  await page.evaluate(async module => {
    const { faviconPath } = await import(module);
    document.documentElement.dataset.theme = "dark";
    const gallery = document.querySelector<HTMLElement>("#icon-gallery")!;
    gallery.style.background = "#151a23"; gallery.style.color = "#edf1f5";
    gallery.querySelectorAll("img").forEach(img => { img.src = faviconPath(img.alt, 0, "dark"); });
  }, `/@fs${resolve("src/lib/tabIdentity.ts")}`);
  await expect(icon).toHaveAttribute("href", "/icons/tab-loading-dark-0.png?v=fluent");
  await expect.poll(() => page.locator("#icon-gallery img").evaluateAll(images => images.every(image => (image as HTMLImageElement).complete))).toBe(true);
  await page.locator("#icon-gallery").screenshot({ path: testInfo.outputPath("favicon-family-dark.png") });
  await page.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  await expect(icon).toHaveAttribute("href", "/icons/tab-loading-light-0.png?v=fluent");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  const hidden = await icon.getAttribute("href");
  await page.waitForTimeout(750);
  await expect(icon).toHaveAttribute("href", hidden!);
  await page.evaluate(() => (window as any).__tabIcon.update("error"));
  await expect(icon).toHaveAttribute("data-state", "error");
  await page.evaluate(() => {
    delete (document as any).hidden;
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.evaluate(async module => {
    const old = (window as any).__tabIcon;
    const { createTabIcon } = await import(module);
    const next = createTabIcon(); next.update("loom"); old.dispose();
    (window as any).__tabIcon = next;
  }, `/@fs/${resolve("src/lib/tabIdentity.ts")}`);
  await expect(icon).toHaveAttribute("data-state", "loom");
  await expect(page.locator('link[rel="icon"]')).toHaveCount(1);
  await page.evaluate(() => (window as any).__tabIcon.dispose());
  await expect(icon).not.toHaveAttribute("data-state");
  const restored = await icon.getAttribute("href");
  await page.waitForTimeout(1100);
  await expect(icon).toHaveAttribute("href", restored!);
});

test("tab identity follows generation and Loom navigation", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "Headless WebKit lacks the fixture worker storage required for generation.");
  await page.goto(`${devUrl}/app?fixture=1&fixtureSlow=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page).toHaveTitle("Drowse");
  const icon = page.locator('link[rel="icon"]');
  await expect(icon).toHaveAttribute("data-state", "conversation");
  await selectWorkspaceView(page, "Controls");
  await expect(icon).toHaveAttribute("data-state", "controls");
  const sections = page.getByRole("group", { name: "Controls section", exact: true });
  await sections.getByRole("button", { name: "Model", exact: true }).click();
  await expect(icon).toHaveAttribute("data-state", "models");
  await sections.getByRole("button", { name: "Chat", exact: true }).click();
  await expect(icon).toHaveAttribute("data-state", "chat-settings");
  await selectWorkspaceView(page, "Conversation");
  await expect(icon).toHaveAttribute("data-state", "conversation");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Describe a loom.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page).toHaveTitle("Generating reply · Drowse");
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-state", "working");
  await expect(page.locator(".msg .tok").first()).toBeVisible();
  await selectWorkspaceView(page, "Loom");
  await expect(page).toHaveTitle("Loom · Generating reply · Drowse");
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-state", "loom-working");
  await page.getByRole("button", { name: /^Weave Text/ }).click();
  await page.locator(".weave").getByRole("button", { name: "Stop", exact: true }).click();
  await expect(page).toHaveTitle("Loom · Drowse");
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-state", "loom");
    await selectWorkspaceView(page, "Conversation");
  await page.locator(".msg .tok").first().click();
  await openTokenDetails(page);
  await expect(icon).toHaveAttribute("data-state", "tokens");
  await page.keyboard.press("Escape");
  await expect(icon).toHaveAttribute("data-state", "conversation");
  await returnToChats(page);
  await expect(page).toHaveTitle("Your chats · Drowse");
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("data-state", "chats");
});

test("lens highlights follow rounded table corners at every scroll position", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await page.evaluate(async url => {
    const { mount } = await import("/e2e/svelte-runtime.ts");
    const { default: LensTab } = await import(url);
    document.body.replaceChildren();
    const target = document.createElement("section");
    target.style.cssText = "margin:24px;max-width:900px";
    document.body.append(target);
    mount(LensTab, { target, props: {
      readout: { loading: false, error: null, progress: null, origin: "captured", source: null, data: {
        node_id: "test", raw_index: 0, token_id: 5, token_text: "things", steering: null,
        layers: [0, 13, 25].map(layer => ({ layer, tokens: ["some", "interesting", "small", "little", "things"].map((token, i) => ({ id: i + 1, token, logprob: Math.log(0.045) })) })),
      } },
      steered: true, jlensFitted: true, hasReplayContext: true, pinned: null, modelId: null, replayAvailable: false,
    } });
  }, `/@fs/${resolve("src/drawers/token/LensTab.svelte")}`);
  const grid = page.locator(".grid-scroll");
  const corner = grid.locator("tbody tr:last-child td:last-child");
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const width of [320, 1000]) {
      await page.setViewportSize({ width, height: 800 });
      await grid.evaluate(el => { el.scrollLeft = 0; });
      await expect(grid.locator("tbody tr:last-child th")).toHaveCSS("border-bottom-left-radius", "0px");
      await grid.evaluate(el => { el.scrollLeft = el.scrollWidth; });
      await expect(corner).toHaveCSS("border-bottom-right-radius", await grid.evaluate(el => getComputedStyle(el).borderBottomRightRadius));
      await expect(corner).toHaveCSS("outline-style", "none");
      expect(await corner.evaluate(el => getComputedStyle(el).boxShadow)).toContain("inset");
      expect(await grid.evaluate(el => {
        const outer = el.getBoundingClientRect();
        const cell = el.querySelector("tbody tr:last-child td:last-child")!.getBoundingClientRect();
        return Math.abs(outer.right - cell.right) < 1 && cell.bottom <= outer.bottom;
      })).toBe(true);
      await grid.screenshot({ path: testInfo.outputPath(`lens-corners-${theme}-${width}.png`) });
    }
  }
  await page.emulateMedia({ forcedColors: "active" });
  if (await page.evaluate(() => matchMedia("(forced-colors: active)").matches)) {
    await expect(corner).toHaveCSS("outline-style", "solid");
    await expect(corner).toHaveCSS("outline-offset", "-2px");
  }
});

test("loading pulse keeps readout progress legible and honors motion preferences", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async (root) => {
    const { mount } = await import("/e2e/svelte-runtime.ts");
    document.body.replaceChildren();
    for (const name of ["LensTab", "SaeTab", "GeometryTab"]) {
      const { default: Component } = await import(`${root}/${name}.svelte`);
      const target = document.createElement("section");
      target.style.cssText = "margin:16px;max-width:560px";
      document.body.append(target);
      mount(Component, { target, props: {
        readout: { loading: true, data: null, error: null, progress: {
          progress: 0.42, phase: "readout", message: "Reading the selected token…",
        } },
        steered: true, jlensFitted: true, saeLoaded: true, hasGeometryProbes: true,
        hasReplayContext: true, pinned: null, modelId: null, replayAvailable: true,
      } });
    }
  }, `/@fs/${resolve("src/drawers/token")}`);
  const pulses = page.locator(".readout-progress.loading-pulse");
  await expect(pulses).toHaveCount(3);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect.poll(() => pulses.first().evaluate(el => getComputedStyle(el, "::before").animationName)).toBe("loading-breathe");
  const opacity = () => pulses.first().evaluate(el => getComputedStyle(el, "::before").opacity);
  const initialOpacity = await opacity();
  await expect.poll(opacity).not.toBe(initialOpacity);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme, theme);
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await expect(page.getByRole("progressbar").first()).toHaveAttribute("aria-valuenow", "42");
      const decoration = await pulses.first().evaluate(el => {
        const css = getComputedStyle(el, "::before");
        return { pointerEvents: css.pointerEvents, radius: css.borderRadius, shadow: css.boxShadow };
      });
      expect(decoration.pointerEvents).toBe("none");
      expect(decoration.radius).not.toBe("0px");
      expect(decoration.shadow).toContain("inset");
      await page.screenshot({ path: testInfo.outputPath(`loading-readouts-${theme}-${width}.png`) });
    }
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect.poll(() => pulses.first().evaluate(el => getComputedStyle(el, "::before").animationName)).toBe("none");
  expect(errors).toEqual([]);
});

test("generation uses a stationary card glow and follows the active reply until stop", async ({ page, browserName }, testInfo) => {
  test.skip(browserName === "webkit", "Headless WebKit rejects OPFS; this test needs fixture worker storage.");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto(`${devUrl}/app?fixture=1&fixtureSlow=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explain how a loom works.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg.generation-active")).toHaveCount(1);
  await expect(page.locator(".msg.generation-active")).toContainText(/Preparing reply|Thinking|Writing/);
  await expect(page.locator(".msg.loading-pulse")).toHaveCount(0);
  await expect(page.locator(".generation-wisp")).toHaveCount(0);
  const card = page.locator(".msg.generation-active");
  const label = card.locator(".generation-label");
  const animation = await card.evaluate(element => {
    const before = getComputedStyle(element, "::before");
    return { name: before.animationName, easing: before.animationTimingFunction, transform: before.transform, pointerEvents: before.pointerEvents, shadow: before.boxShadow };
  });
  expect(animation.name).toBe("generation-breathe");
  expect(animation.easing).toBe("ease-in-out");
  expect(animation.transform).toBe("none");
  expect(animation.pointerEvents).toBe("none");
  expect(animation.shadow).toContain("inset");
  await page.emulateMedia({ reducedMotion: "reduce" });
  expect(await card.evaluate(element => getComputedStyle(element, "::before").animationName)).toBe("none");
  expect(await card.evaluate(element => getComputedStyle(element, "::before").opacity)).toBe("0.7");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await label.scrollIntoViewIfNeeded();
  await card.screenshot({ path: testInfo.outputPath("generating-card-light.png") });
  await setAppearance(page, "Dark");
  await expect(page.locator("html")).not.toHaveAttribute("data-theme-transition", /.+/);
  await label.scrollIntoViewIfNeeded();
  await page.locator(".msg.generation-active").screenshot({ path: testInfo.outputPath("generating-reply.png") });
  for (const width of [320, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const reply = page.locator(".msg.generation-active");
    expect(await reply.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    await reply.screenshot({ path: testInfo.outputPath(`loading-reply-${width}.png`) });
  }
  await page.setViewportSize({ width: 1280, height: 900 });
  await selectWorkspaceView(page, "Loom");
  await page.getByRole("button", { name: /^Map/ }).click();
  await expect(page.locator(".node.generation-active .generation-label")).toHaveCount(1);
  expect(await page.locator(".node.generation-active").evaluate(element => getComputedStyle(element, "::before").animationName)).toBe("generation-breathe");
  await expect(page.locator(".node.loading-pulse, .choice.loading-pulse")).toHaveCount(0);
  await page.locator(".node.generation-active").screenshot({ path: testInfo.outputPath("generating-loom.png") });
  await page.getByRole("button", { name: /^Weave/ }).click();
  await expect(page.locator(".choice.generation-active .generation-label")).toHaveCount(1);
  expect(await page.locator(".choice.generation-active").evaluate(element => getComputedStyle(element, "::before").transform)).toBe("none");
  await page.locator(".choice.generation-active").screenshot({ path: testInfo.outputPath("generating-weave.png") });
  await page.locator(".weave").getByRole("button", { name: "Stop", exact: true }).click();
  await expect(page.locator(".choice.generation-active, .node.generation-active, .msg.generation-active, .status-footer .generation-wisp")).toHaveCount(0);
  await expect(page.locator(".toast.error")).toHaveCount(0);
});

test("base model disclosure is accessible and unclipped on compact screens", async ({ page }, testInfo) => {
  await page.goto(`${devUrl}/app?layoutFixture=setup`);
  const disclosure = page.locator("details.base-models");
  await expect(disclosure).toBeVisible();
  await expect(disclosure).not.toHaveAttribute("open", "");
  await disclosure.locator("summary").focus();
  await page.keyboard.press("Enter");
  await expect(disclosure).toContainText("No verified base-model downloads");
  await expect(disclosure.getByRole("button")).toHaveCount(0);
  for (const theme of ["Light", "Dark"]) {
    const toggle = page.getByRole("button", { name: theme, exact: true });
    if (await toggle.isVisible()) await toggle.click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme.toLowerCase());
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await disclosure.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await disclosure.screenshot({ path: testInfo.outputPath(`base-models-${theme}-${width}.png`) });
    }
  }
  const audit = await new AxeBuilder({ page }).include(".base-models").analyze();
  expect(audit.violations).toEqual([]);
});

test("Loom depth fills the viewport through long pans, zoom, resize, and pointer movement", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await setAppearance(page, "Dark");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explore the depth of the loom.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await selectWorkspaceView(page, "Loom");
  await page.getByRole("button", { name: /^Map/ }).click();
  const viewport = page.locator(".loom-viewport");
  const field = viewport.locator(".loom-depth-field");
  const card = page.locator(".tree-node-wrap").last();
  const layers = () => field.evaluate(element => ["::before", "::after"].map(pseudo => {
    const css = getComputedStyle(element, pseudo);
    return { transform: css.transform, size: parseFloat(css.backgroundSize), position: css.backgroundPosition };
  }));
  await expect(field).toHaveAttribute("aria-hidden", "true");
  await expect(field).toHaveCSS("pointer-events", "none");
  await expect(page.locator(".loom-canvas")).not.toHaveClass(/camera-animating/);
  const bounds = (await viewport.boundingBox())!;
  const depth = await layers();
  expect(depth.every(layer => layer.transform === "none")).toBe(true);
  const before = await card.boundingBox();
  await page.mouse.move(bounds.x + bounds.width * 0.1, bounds.y + bounds.height * 0.8);
  await expect.poll(layers).toEqual(depth);
  await page.mouse.move(bounds.x + bounds.width * 0.9, bounds.y + bounds.height * 0.8);
  await expect.poll(layers).toEqual(depth);
  expect(await card.boundingBox()).toEqual(before);
  await page.screenshot({ path: testInfo.outputPath("loom-depth-desktop.png") });
  await page.mouse.move(0, 0);
  await expect.poll(layers).toEqual(depth);

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const delta of [50000, -100000]) {
      await viewport.dispatchEvent("wheel", { deltaX: delta, deltaY: delta, bubbles: true, cancelable: true });
      await viewport.dispatchEvent("wheel", {
        deltaY: Math.sign(delta) * 10000, ctrlKey: true,
        clientX: bounds.x + 100, clientY: bounds.y + 100,
        bubbles: true, cancelable: true,
      });
      await expect.poll(async () => (await layers())[0].position).not.toBe(depth[0].position);
      const cover = await field.evaluate(element => {
        const rect = element.getBoundingClientRect();
        const parent = element.parentElement!.getBoundingClientRect();
        return { width: rect.width, height: rect.height, parentWidth: parent.width, parentHeight: parent.height };
      });
      expect(cover.width).toBeCloseTo(cover.parentWidth, 0);
      expect(cover.height).toBeCloseTo(cover.parentHeight, 0);
      for (const layer of await layers()) {
        expect(layer.transform).toBe("none");
        expect(layer.position.split(" ").every(value => Math.abs(parseFloat(value)) <= layer.size)).toBe(true);
      }
      const screenshot = await viewport.screenshot({ scale: "css", path: testInfo.outputPath(`loom-grid-${width}-${delta}.png`) });
      const cornerRanges = await page.evaluate(async encoded => {
        const image = new Image();
        image.src = `data:image/png;base64,${encoded}`;
        await image.decode();
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(image, 0, 0);
        return [[32, 32], [image.width - 128, 32], [32, image.height - 128], [image.width - 128, image.height - 128]].map(([x, y]) => {
          const pixels = context.getImageData(x, y, 96, 96).data;
          let min = 255;
          let max = 0;
          for (let i = 0; i < pixels.length; i += 4) {
            min = Math.min(min, pixels[i]);
            max = Math.max(max, pixels[i]);
          }
          return max - min;
        });
      }, screenshot.toString("base64"));
      expect(cornerRanges.every(range => range >= 5)).toBe(true);
    }
    await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
    await expect(page.locator(".loom-canvas")).not.toHaveClass(/camera-animating/);
    const initial = await layers();
    await page.getByRole("button", { name: "Zoom in", exact: true }).click();
    await expect.poll(async () => (await layers())[0].size).toBeGreaterThan(initial[0].size);
    expect((await layers())[1].size).toBeGreaterThan(initial[1].size);
  }
  await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
  await expect(page.locator(".loom-canvas")).not.toHaveClass(/camera-animating/);
  const fitted = (await card.boundingBox())!;
  const phoneBounds = (await viewport.boundingBox())!;
  expect(fitted.x).toBeGreaterThanOrEqual(phoneBounds.x);
  expect(fitted.y).toBeGreaterThanOrEqual(phoneBounds.y);
  expect(fitted.x + fitted.width).toBeLessThanOrEqual(phoneBounds.x + phoneBounds.width);
  expect(fitted.y + fitted.height).toBeLessThanOrEqual(phoneBounds.y + phoneBounds.height);
  await page.screenshot({ path: testInfo.outputPath("loom-depth-phone.png") });
  await setAppearance(page, "Light");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.screenshot({ path: testInfo.outputPath("loom-depth-phone-light.png") });
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const pseudo of ["::before", "::after"]) {
    await expect.poll(() => field.evaluate((element, pseudo) => getComputedStyle(element, pseudo).transform, pseudo)).toBe("none");
  }
  await page.emulateMedia({ contrast: "more" });
  await expect(field).toBeHidden();
  await expect(page.locator(".toast.error")).toHaveCount(0);
});

test("Loom shows a shared token prefix once and keeps continuation token actions exact", async ({ page }, testInfo) => {
  test.slow();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Explore two endings.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".msg .response-body").last()).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const original = await page.evaluate(async url => {
    const stores = await import(url);
    const node = stores.loomTree.nodes.get(stores.loomTree.active_node_id);
    const saved = JSON.stringify(node);
    await stores.sendTextFork(node.id, node.tokens[3].raw_index, "different ");
    return { id: node.id, parent: node.parent_id, saved };
  }, storesUrl);
  await expect.poll(() => page.evaluate(async ({ url, parent }) => {
    const stores = await import(url);
    return !stores.genStatus.active && (stores.loomTree.children_of.get(parent)?.length ?? 0) === 2;
  }, { url: storesUrl, parent: original.parent })).toBe(true);
  await selectWorkspaceView(page, "Loom");
  await page.getByRole("button", { name: /^Map/ }).click();
  const shared = page.locator('[data-loom-shared="2"]');
  await expect(shared).toHaveCount(1);
  await expect(shared.locator(".token-node")).toHaveCount(3);
  await expect(shared.locator('.token-field')).toHaveCSS('flex-grow', '0');
  await expect.poll(() => shared.locator('.node').evaluate(element => {
    const field = element.querySelector<HTMLElement>('.token-field')!;
    const last = field.querySelector<HTMLElement>('.sentence-node:last-child')!;
    return field.clientHeight - (last.offsetTop - field.offsetTop + last.offsetHeight);
  })).toBeLessThanOrEqual(8);
  await expect(page.locator(".loom-canvas")).toHaveAttribute("data-loom-nodes", "4");
  const source = page.locator(`[data-loom-node-id="${original.id}"]`);
  await expect(source.locator(".token-node").first()).toHaveAttribute("data-token-index", "3");
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
    expect(await shared.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`shared-prefix-${width}.png`) });
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
  await source.locator(".token-node").first().click();
  await expect(page.getByRole("dialog", { name: "Generated word details" })).toBeVisible();
  const state = await page.evaluate(async ({ url, id }) => {
    const stores = await import(url);
    return { saved: JSON.stringify(stores.loomTree.nodes.get(id)), token: stores.drawerState.params.tokenIdx };
  }, { url: storesUrl, id: original.id });
  expect(state.saved).toBe(original.saved);
  expect(state.token).toBe(3);
  await page.keyboard.press("Escape");
  for (const count of [40, 3, 18, 3]) {
    await page.evaluate(async ({ url, parent, count }) => {
      const stores = await import(url);
      const ids = stores.loomTree.children_of.get(parent);
      ids.forEach((id: string, branch: number) => {
        const node = stores.loomTree.nodes.get(id);
        const tokens = Array.from({ length: count + 1 }, (_, index) => ({
          ...node.tokens[0], raw_index: index, token_id: index === count ? 1000 + branch : index,
          text: index === count ? ` ending ${branch}.` : index % 7 === 6 ? " marmots. " : " wonderful",
        }));
        stores.loomTree.nodes.set(id, { ...node, tokens, text: tokens.map(token => token.text).join(""), raw_token_ids: tokens.map(token => token.token_id) });
      });
    }, { url: storesUrl, parent: original.parent, count });
    await expect(shared.locator(".token-node")).toHaveCount(count);
    if (count === 3) {
      await expect.poll(() => shared.locator('.node').evaluate(element => {
        const field = element.querySelector<HTMLElement>('.token-field')!;
        const last = field.querySelector<HTMLElement>('.sentence-node:last-child')!;
        return field.clientHeight - (last.offsetTop - field.offsetTop + last.offsetHeight);
      })).toBeLessThanOrEqual(8);
    }
  }
  await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
  await shared.screenshot({ path: testInfo.outputPath('shared-prefix-content-height.png') });
  await page.evaluate(async ({ url, parent }) => {
    const stores = await import(url);
    stores.loomTree.children_of.get(parent).forEach((id: string, branch: number) => {
      const node = stores.loomTree.nodes.get(id);
      const tokens = [...node.tokens.slice(0, 3), ...Array.from({ length: 80 }, (_, index) => ({
        ...node.tokens[0], raw_index: index + 3, token_id: 2000 + branch * 100 + index,
        text: index === 79 ? " finished." : branch ? " burrowing" : " marmots",
      }))];
      stores.loomTree.nodes.set(id, { ...node, tokens, text: tokens.map(token => token.text).join(""), raw_token_ids: tokens.map(token => token.token_id) });
    });
  }, { url: storesUrl, parent: original.parent });
  const continuations = page.locator('[data-loom-node-id] .node').filter({ has: page.locator('.fork', { hasText: 'continuation' }) });
  await expect(continuations).toHaveCount(2);
  for (const width of [390, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const theme of ["light", "dark"]) {
      await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
      await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
      for (let zoom = 0; zoom < 2; zoom++) {
        await expect.poll(() => continuations.evaluateAll(cards => cards.every(card => {
          const field = card.querySelector<HTMLElement>('.token-field')!;
          const last = field.querySelector<HTMLElement>('.token-node:last-child')!;
          const meta = card.querySelector<HTMLElement>('.node-meta')!;
          const tools = card.querySelector<HTMLElement>('.node-tools')!;
          return field.scrollHeight <= field.clientHeight + 1
            && last.getBoundingClientRect().bottom <= field.getBoundingClientRect().bottom
            && field.getBoundingClientRect().bottom <= meta.getBoundingClientRect().top
            && tools.getBoundingClientRect().bottom < card.getBoundingClientRect().bottom;
        }))).toBe(true);
        await page.getByRole("button", { name: "Zoom in", exact: true }).click();
      }
      await page.getByRole("button", { name: "Fit whole loom", exact: true }).click();
      await page.screenshot({ path: testInfo.outputPath(`continuation-no-clipping-${width}-${theme}.png`) });
    }
  }
  await expect(page.locator(".toast.error")).toHaveCount(0);
});

test("error notifications use the shared solid popup surface without a side accent", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async url => {
    const { pushToast } = await import(url);
    pushToast("Generation: The model could not finish that reply. Try again. If it keeps happening, reopen the model.", {
      kind: "error", detail: "Your conversation is still available.", ttlMs: null,
    });
  }, storesUrl);
  const toast = page.locator(".toast.error");
  for (const theme of ["Light", "Dark"]) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await setAppearance(page, theme);
    await expect(toast).toHaveCSS("color", theme === "Dark" ? "rgb(255, 199, 196)" : "rgb(160, 34, 48)");
    for (const width of [320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await expect(toast).toBeVisible();
      const style = await toast.evaluate(element => {
        const css = getComputedStyle(element);
        return {
          start: css.borderInlineStartWidth, end: css.borderInlineEndWidth,
          startColor: css.borderInlineStartColor, endColor: css.borderInlineEndColor,
          background: css.backgroundImage,
          color: css.color,
          fits: element.scrollWidth <= element.clientWidth,
        };
      });
      expect(style.start).toBe("1px");
      expect(style.start).toBe(style.end);
      expect(style.startColor).toBe(style.endColor);
      expect(style.background).toMatch(/^none(?:, none)*$/);
      expect(style.color).toBe(theme === "Dark" ? "rgb(255, 199, 196)" : "rgb(160, 34, 48)");
      expect(style.fits).toBe(true);
      await expect(toast.locator(".detail")).toHaveCSS("color", style.color);
      await toast.screenshot({ path: testInfo.outputPath(`error-${theme}-${width}.png`) });
    }
  }
  await page.emulateMedia({ contrast: "more" });
  await expect(toast).toHaveCSS("background-image", "none");
  await toast.getByRole("button", { name: /^Dismiss notification/ }).click();
  await expect(toast).toHaveCount(0);
});

test("Loom stays accessible while a reply streams", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "Headless WebKit rejects OPFS; this lifecycle test requires persistent worker storage.");
  test.setTimeout(60_000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const reads: string[] = [];
    (window as any).__loomReads = reads;
    const postMessage = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(message: any, ...args: any[]) {
      if (message.command === "request" && message.payload?.service === "tree" &&
        ["get", "edgeLabel"].includes(message.payload.method)) reads.push(message.payload.method);
      return (postMessage as any).call(this, message, ...args);
    };
  });
  await page.goto(`${devUrl}/app?fixture=1&fixtureSlow=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Keep writing while I explore the branches.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".conversation-page").getByRole("button", { name: "Stop", exact: true })).toBeEnabled();
  await page.evaluate(() => { (window as any).__loomReads.length = 0; });
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await selectLoomView(page, /^Weave/);
  await expect(page.locator(".weave")).toBeVisible();
  await expect(page.locator(".weave").getByRole("button", { name: "Stop", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: /^Map/ }).click();
  await expect(page.locator(".loom-canvas")).toBeVisible();
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await page.waitForTimeout(2_500);
  await page.getByRole("button", { name: /^Weave/ }).click();
  await expect(page.locator("[data-weave-choice]")).toHaveCount(1);
  expect(errors).toEqual([]);
  await expect(page.locator(".toast.error")).toHaveCount(0);
  expect(await page.evaluate(() => (window as any).__loomReads)).toEqual([]);
  page.once("dialog", dialog => dialog.dismiss());
  await returnToChats(page);
  await expect(page.locator(".weave").getByRole("button", { name: "Stop", exact: true })).toBeEnabled();
  page.once("dialog", dialog => dialog.accept());
  await returnToChats(page);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(page.locator(".shell")).toHaveCount(0);
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.locator(".chat[aria-label='Chat']")).toContainText("Keep writing while I explore the branches.");
});

test("workspace menu returns to chats and restores the current conversation", async ({ page, browserName }, testInfo) => {
  test.skip(browserName === "webkit", "Headless WebKit rejects OPFS; chat restoration requires persistent worker storage.");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto(`${devUrl}/app?fixture=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  const home = page.getByRole("button", { name: "Workspace menu", exact: true });
  const brand = page.locator(".app-header .page-brand");
  for (const width of [320, 390, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    if (width > 760 && await page.getByRole("button", { name: "Hide left sidebar", exact: true }).isVisible()) {
      await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
    }
    const iconBox = (await home.boundingBox())!;
    const brandBox = (await brand.boundingBox())!;
    expect(iconBox.x).toBeGreaterThanOrEqual(brandBox.x + brandBox.width);
    expect(iconBox.width).toBeGreaterThanOrEqual(width <= 760 ? 44 : 40);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await expect(page.locator(".workspace-nav")).not.toContainText("Saved chats");
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Remember this path when I go home.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const reply = await page.locator(".msg .response-body").last().innerText();
  await page.screenshot({ path: testInfo.outputPath("home-navigation.png") });
  await returnToChats(page);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(page.locator(".shell")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await expect(page.locator('.chat[aria-label="Chat"]')).toContainText("Remember this path when I go home.");
  await expect(page.locator(".msg .response-body").last()).toHaveText(reply);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await returnToChats(page);
  await expect(page.getByRole("heading", { name: "Your chats", exact: true })).toBeVisible();
  await expect(page.locator(".shell")).toHaveCount(0);
});

test("workspace menu has a visible touch target after the wordmark", async ({ page }, testInfo) => {
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  const home = page.getByRole("button", { name: "Workspace menu", exact: true });
  for (const width of [320, 390, 768, 1440]) {
    await page.setViewportSize({ width, height: 844 });
    if (width > 760 && await page.getByRole("button", { name: "Hide left sidebar", exact: true }).isVisible()) {
      await page.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
    }
    await expect(home).toHaveAttribute("aria-haspopup", "dialog");
    await expect(home).toBeVisible();
    const icon = (await home.boundingBox())!;
    const brand = (await page.locator(".app-header .page-brand").boundingBox())!;
    expect(icon.width).toBeGreaterThanOrEqual(width <= 760 ? 44 : 40);
    expect(icon.height).toBeGreaterThanOrEqual(width <= 760 ? 44 : 40);
    expect(icon.x).toBeGreaterThanOrEqual(brand.x + brand.width);
    expect(icon.x + icon.width).toBeLessThanOrEqual(width);
    expect(brand.x + brand.width).toBeLessThanOrEqual(width);
    if (width === 390) await page.screenshot({ path: testInfo.outputPath("home-icon-phone.png") });
  }
});

async function workbench(page: Page) {
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(page.locator(".shell")).toBeVisible();
}

test("header download confirms filename and exact backup size without changing the chat", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Preserve this chat in the backup.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  const menu = page.getByRole("button", { name: "Workspace menu", exact: true });
  const button = page.getByRole("button", { name: "Download chat", exact: true });
  const before = await page.evaluate(async url => (await import(url)).captureConversationSnapshot(), `/@fs/${resolve("src/lib/conversationWorkspace.ts")}`);
  await openWorkspaceMenu(page);
  await button.click();
  const dialog = page.getByRole("dialog", { name: "Download chat", exact: true });
  await expect(dialog.getByLabel("File name", { exact: true })).toBeEnabled();
  await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect(menu).toBeFocused();
  await openWorkspaceMenu(page);
  await button.click();
  const filename = dialog.getByLabel("File name", { exact: true });
  await filename.fill("   ");
  await expect(dialog.getByRole("button", { name: "Download", exact: true })).toBeDisabled();
  await filename.fill("My backup.drowse-chat.json");
  const size = Number(await dialog.getByLabel("Backup size").getAttribute("data-bytes"));
  expect(size).toBeGreaterThan(0);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), { url: `/@fs/${resolve("src/lib/theme.ts")}`, theme });
    for (const width of [320, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
      await dialog.screenshot({ path: testInfo.outputPath(`download-chat-${theme}-${width}.png`) });
    }
  }
  const pendingDownload = page.waitForEvent("download");
  await dialog.getByRole("button", { name: "Download", exact: true }).click();
  const download = await pendingDownload;
  expect(download.suggestedFilename()).toBe("My-backup.drowsechat");
  const bytes = await readStreamBuffer(await download.createReadStream());
  expect(bytes.byteLength).toBe(size);
  const backup = JSON.parse(bytes.toString());
  expect(backup.format).toBe("drowse-chat-backup");
  expect(backup.conversation.snapshot.tree).toEqual(before.tree);
  expect(backup.conversation.snapshot.samplingState).toEqual(before.samplingState);
  expect(backup.conversation.name).not.toBe("My backup.drowse-chat.json");
  await expect(dialog).toHaveCount(0);
  await expect(await menu.isVisible() ? menu : page.getByRole("button", { name: "Hide left sidebar", exact: true })).toBeFocused();
});

test("shared controls have comfortable targets and interruptible contextual feedback", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await workbench(page);
  await page.evaluate(async ({ storesUrl, buttonUrl, pinUrl }) => {
    const { mount, createRawSnippet } = await import("/e2e/svelte-runtime.ts");
    const { default: Button } = await import(buttonUrl);
    const { default: Pin } = await import(pinUrl);
    const { genStatus } = await import(storesUrl);
    genStatus.active = false;
    const target = document.createElement("div");
    target.id = "controls-polish-test";
    target.style.cssText = "position:fixed;top:180px;left:16px;display:flex;gap:12px;padding:16px;background:var(--bg-elev);--card-accent:var(--pillar-sae);z-index:9999;border-radius:18px";
    document.body.append(target);
    const children = createRawSnippet(() => ({ render: () => "<span>Test</span>" }));
    mount(Button, { target, props: { children, ariaLabel: "Animated button" } });
    mount(Button, { target, props: { children, static: true, ariaLabel: "Static button" } });
    mount(Pin, { target, props: {
      shape: "triangle", get pinned() { return genStatus.active; },
      onclick: () => { genStatus.active = !genStatus.active; }, ariaLabel: "Toggle test pin", title: "Pin",
    } });
  }, { storesUrl, buttonUrl: `/@fs/${resolve("src/lib/ui/Button.svelte")}`, pinUrl: `/@fs/${resolve("src/panels/rack/ProbePinButton.svelte")}` });
  const target = page.locator("#controls-polish-test");
  const pin = target.getByRole("button", { name: "Toggle test pin" });
  for (const width of [320, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    for (const button of await target.getByRole("button").all()) {
      const box = (await button.boundingBox())!;
      expect(box.width).toBeGreaterThanOrEqual(width < 1280 ? 44 : 40);
      expect(box.height).toBeGreaterThanOrEqual(width < 1280 ? 44 : 40);
    }
  }
  await pin.focus();
  await expect(pin).toHaveCSS("outline-style", "solid");
  await page.keyboard.press("Space");
  await expect(pin).toHaveAttribute("aria-pressed", "true");
  await expect(pin.locator(".pin-icon")).toHaveCount(2);
  await page.keyboard.press("Space");
  await expect(pin).toHaveAttribute("aria-pressed", "false");
  await expect(pin.locator(".pin-icon.visible")).toHaveCSS("opacity", "1");
  for (const [name, scale] of [["Animated button", "0.96"], ["Static button", "none"]]) {
    const button = target.getByRole("button", { name });
    await button.hover();
    await page.mouse.down();
    await expect(button).toHaveCSS("scale", scale);
    await page.mouse.up();
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  await pin.click();
  await expect(pin.locator(".pin-icon.visible")).toHaveCSS("filter", "blur(0px)");
  expect(await pin.evaluate(el => parseFloat(getComputedStyle(el).transitionDuration))).toBeLessThan(0.001);
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    await target.screenshot({ path: testInfo.outputPath(`shared-controls-${theme}.png`) });
  }
});

test("Rolling numbers interrupt cleanly, preserve precision, and honor reduced motion", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await workbench(page);
  await page.evaluate(async ({ storesUrl, componentUrl }) => {
    const { mount, unmount } = await import("/e2e/svelte-runtime.ts");
    const { default: RollingNumber } = await import(componentUrl);
    const { genStatus } = await import(storesUrl);
    genStatus.tokPerSec = 9.99;
    const target = document.createElement("div");
    target.id = "rolling-test";
    target.style.cssText = "position:fixed;top:180px;left:40px;font:inherit;font-size:48px;color:var(--fg);background:var(--bg);padding:24px;z-index:9999";
    document.body.append(target);
    const component = mount(RollingNumber, { target, props: { get value() { return genStatus.tokPerSec; }, digits: 2 } });
    (window as any).disposeRolling = async () => { await unmount(component); target.remove(); };
    (window as any).rollingAnimations = 0;
    const animate = Element.prototype.animate;
    Element.prototype.animate = function(...args) {
      if (this.closest(".rolling-number")) (window as any).rollingAnimations++;
      return animate.apply(this, args);
    };
  }, { storesUrl, componentUrl: `/@fs/${resolve("src/lib/ui/RollingNumber.svelte")}` });
  const number = page.locator("#rolling-test .rolling-number");
  await expect(number.locator(".morph-source")).toHaveText("9.99");
  await expect(number.locator(".morph-text")).toHaveAttribute("data-morph-active", "");
  for (const value of [10, 1234.56, -0.25, 0]) {
    await page.evaluate(async ({ url, value }) => { (await import(url)).genStatus.tokPerSec = value; }, { url: storesUrl, value });
    await expect(number.locator(".morph-source")).toHaveText(value.toFixed(2));
  }
  await expect.poll(() => page.evaluate(() => (window as any).rollingAnimations)).toBeGreaterThan(0);
  await expect.poll(() => number.evaluate(el => el.getAnimations({ subtree: true }).length)).toBe(0);
  expect(await number.locator(".morph-paint").getAttribute("aria-hidden")).toBe("true");
  expect(await number.locator(".morph-source").count()).toBe(1);
  await page.screenshot({ path: testInfo.outputPath("rolling-number-settled.png") });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.evaluate(async url => { (await import(url)).genStatus.tokPerSec = 98.76; }, storesUrl);
  await expect(number.locator(".morph-source")).toHaveText("98.76");
  await expect.poll(() => number.evaluate(el => el.getAnimations({ subtree: true }).length)).toBe(0);
  await page.evaluate(() => (window as any).disposeRolling());
  await expect(number).toHaveCount(0);
});

test("generation statistics and Loom zoom remain readable in both themes", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await page.evaluate(async url => {
    const { genStatus } = await import(url);
    Object.assign(genStatus, { startedAt: performance.now() - 1200, finishedAt: performance.now(), active: false, tokensSoFar: 128, tokPerSec: 12.5 });
  }, storesUrl);
  const footer = page.locator(".status-footer");
  await expect(footer.locator(".speed")).toHaveText("12.5 tokens/s");
  await expect(footer.locator(".token-count")).toContainText("128 tokens");
  for (const theme of ["Light", "Dark"]) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await setAppearance(page, theme);
    await expect(footer.locator(".speed")).toBeVisible();
    await footer.screenshot({ path: testInfo.outputPath(`rolling-stats-${theme}.png`) });
    await page.setViewportSize({ width: 320, height: 844 });
    expect(await footer.evaluate(el => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
  }
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Show a path in the Loom.");
  await page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ }).click();
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await selectLoomView(page, /^Map/);
  const zoom = page.getByLabel("Loom zoom");
  const previous = await zoom.locator(".morph-source").innerText();
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await expect(zoom.locator(".morph-source")).not.toHaveText(previous);
});

async function openDrawer(page: Page, name: string, params?: unknown) {
  await page.evaluate(async ({ url, name, params }) => {
    const stores = await import(url);
    // The deterministic worker disables fitting; render its authoring forms
    // directly for UI checks without claiming to run a fitting job.
    if (["manifold_builder", "template_lab", "health"].includes(name)) {
      stores.drawerState.open = name;
      stores.drawerState.params = params;
    } else stores.openDrawer(name, params);
  }, { url: storesUrl, name, params });
  await expect(page.getByRole("dialog").last(), name).toBeVisible();
}

test("popups share material tokens while side drawers remain translucent in both themes", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  await page.getByRole("button", { name: /^Controls/ }).click();
  await page.evaluate(async url => {
    const { pushToast } = await import(url);
    pushToast("A consistent popup surface", { ttlMs: null });
  }, storesUrl);
  const surface = (locator: ReturnType<Page["locator"]>) => locator.evaluate(element => {
    const css = getComputedStyle(element);
    return [css.backgroundColor, css.backgroundImage, css.borderTopWidth, css.borderTopColor, css.borderRadius, css.boxShadow];
  });
  for (const theme of ["Light", "Dark"]) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme.toLowerCase()), { url: `/@fs/${resolve("src/lib/theme.ts")}`, theme });
    await expect(page.locator(".toast").first()).toHaveCSS("background-color", theme === "Light" ? "rgb(255, 255, 255)" : "rgb(39, 44, 53)");
    const expected = await surface(page.locator(".toast").first());
    expect(expected[0]).toBe(theme === "Light" ? "rgb(255, 255, 255)" : "rgb(39, 44, 53)");
    expect(expected[1]).toBe("none, none");
    expect(expected[2]).toBe("1px");
    expect(expected[4]).toBe("16px");
    await page.getByRole("button", { name: "Sort probes by", exact: true }).click();
    expect(await surface(page.locator(".sk-select-popover:popover-open"))).toEqual(expected.with(4, "8px"));
    await page.keyboard.press("Escape");
    await openDrawer(page, "advanced_sampling");
    const drawerSurface = await surface(page.locator('.drawer[role="dialog"]'));
    expect(drawerSurface.slice(1)).toEqual(expected.slice(1));
    expect(drawerSurface[0]).toMatch(/\/ 0\.9\)$/);
    await page.getByRole("button", { name: "About Top K", exact: true }).tap();
    const tip = page.locator(".info-popover:popover-open");
    await expect(tip).toBeVisible();
    const tipSurface = await surface(tip);
    expect(tipSurface.slice(0, 2)).toEqual(expected.slice(0, 2));
    expect(tipSurface[2]).toBe("0px");
    expect(tipSurface[4]).toBe(expected[4]);
    expect(tipSurface[5]).toBe(`${expected[3]} 0px 0px 0px 1px, ${expected[5]}`);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: testInfo.outputPath(`popup-family-${theme}.png`) });
    await page.keyboard.press("Escape");
    await page.keyboard.press("Escape");
  }
});

test("contextual help survives touch, scrolling, hovering, and nested Escape", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 320, height: 568 });
  await workbench(page);
  await openDrawer(page, "advanced_sampling");
  const drawer = page.getByRole("dialog", { name: "Sampling settings" });
  const trigger = drawer.getByRole("button", { name: "About Top K", exact: true });
  // A real touch click must not be canceled by the focus event that precedes it.
  await trigger.tap();
  const tip = drawer.getByRole("tooltip").filter({ hasText: "Top K" });
  await expect(tip).toBeVisible();
  const bounds = await tip.boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(320);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(568);
  await expect(tip).toHaveJSProperty("popover", "manual");
  await tip.hover();
  await page.waitForTimeout(200);
  await expect(tip).toBeVisible();
  await trigger.press("Escape");
  await expect(tip).not.toBeVisible();
  await expect(drawer).toBeVisible();
  await trigger.press("Escape");
  await expect(drawer).not.toBeVisible();
});

test("open dropdowns reveal complete labels and expose their controlled listbox", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  const select = page.getByRole("button", { name: "Color generated words by", exact: true });
  await showWorkspaceTools(page, "chat");
  await select.evaluate((element) => { (element.parentElement as HTMLElement).style.width = "100px"; });
  await select.click();
  const id = await select.getAttribute("aria-controls");
  expect(id).toBeTruthy();
  const list = page.locator(`[id="${id}"]`);
  await expect(list).toHaveAttribute("role", "listbox");
  for (const option of await list.getByRole("option").all()) {
    const geometry = await option.evaluate((element) => ({
      width: element.clientWidth,
      contentWidth: element.scrollWidth,
      whiteSpace: getComputedStyle(element).whiteSpace,
      lineHeight: parseFloat(getComputedStyle(element).lineHeight) / parseFloat(getComputedStyle(element).fontSize),
    }));
    expect(geometry.whiteSpace).toBe("normal");
    expect(geometry.contentWidth).toBeLessThanOrEqual(geometry.width + 1);
    expect(geometry.lineHeight).toBeGreaterThanOrEqual(1.4);
  }
  await page.keyboard.press("Escape");
  await expect(select).toBeFocused();
});

for (const theme of ["light", "dark"] as const) {
  test(`six-skill pass covers expanded hosted surfaces in ${theme}`, async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    page.setDefaultTimeout(10_000);
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.emulateMedia({ reducedMotion: "reduce", colorScheme: theme });
    await page.setViewportSize({ width: 390, height: 844 });
    // This audit exercises interface states, independently of download approval and OPFS.
    await workbench(page);
    await expect(page.locator(".shell")).toBeVisible();
    const reports: unknown[] = [];

    async function audit(name: string, selector: string) {
      const root = page.locator(selector).last();
      await expect(root, name).toBeVisible();
      await test.step(`${name}: UI`, async () => {
        const unboundedTransitions = await root.locator("*").evaluateAll((elements) => elements.filter((element) => {
          const style = getComputedStyle(element);
          return style.transitionProperty === "all" && style.transitionDuration.split(",").some((duration) => parseFloat(duration) > 0.001);
        }).map((element) => element.className));
        expect(unboundedTransitions).toEqual([]);
      });
      await test.step(`${name}: colors`, async () => {
        const result = await new AxeBuilder({ page }).include(selector).withRules(["color-contrast"]).analyze();
        expect(result.violations.map((violation) => ({ id: violation.id, nodes: violation.nodes.map((node) => node.failureSummary) })), name).toEqual([]);
      });
      await test.step(`${name}: layout`, async () => {
        for (const width of [320, 1440]) {
          await page.setViewportSize({ width, height: 900 });
          await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth), { message: name }).toBeLessThanOrEqual(1);
          await expect.poll(() => root.evaluate((element) => element.scrollWidth - element.clientWidth), { message: name }).toBeLessThanOrEqual(1);
        }
        await page.setViewportSize({ width: 390, height: 844 });
      });
      await test.step(`${name}: writing`, async () => {
        reports.push({ name, theme, copy: await root.innerText() });
      });
      await test.step(`${name}: typography`, async () => {
        const crampedParagraphs = await root.locator("p").evaluateAll((elements) => elements.flatMap((element) => {
          const style = getComputedStyle(element);
          const size = parseFloat(style.fontSize);
          const leading = parseFloat(style.lineHeight);
          const height = element.getBoundingClientRect().height;
          return height >= leading * 2.9 && leading / size < 1.4 - 0.001
            ? [{ text: element.textContent, leading, size }] : [];
        }));
        expect(crampedParagraphs, name).toEqual([]);
      });
      await test.step(`${name}: accessibility`, async () => {
        const result = await new AxeBuilder({ page }).include(selector)
          .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
        expect(result.violations.map((violation) => ({ id: violation.id, nodes: violation.nodes.map((node) => node.failureSummary) })), name).toEqual([]);
      });
      if (["Loom, Weave", "Authoring, custom", "Template editor"].includes(name)) {
        await page.screenshot({ path: testInfo.outputPath(`${theme}-${name.replaceAll(/[^a-z]+/gi, "-")}.png`) });
      }
    }

    await audit("Conversation, empty", ".conversation-page");
    await page.getByRole("textbox", { name: /^Compose as / }).fill("A clear explanation of how language models work.");
    await page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ }).click();
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    await audit("Conversation, generated", ".conversation-page");
    await page.getByRole("button", { name: /^Loom\b/ }).click();
    await audit("Loom, Weave", ".loom-sidebar");
    for (const name of [/^Map\b/, /^Current path/, /^Next options/, /^Starred/]) {
      await page.getByRole("button", { name: "Loom view", exact: true }).click();
      await page.getByRole("option", { name }).click();
      await audit(`Loom, ${name.source}`, ".loom-sidebar");
    }
    await page.getByRole("button", { name: /^Controls\b/ }).click();
    await audit("Controls, response", ".controls-page");
    const sections = page.getByRole("group", { name: "Response guidance type" });
    for (const button of await sections.getByRole("button").all()) {
      const name = await button.innerText();
      await button.click();
      await audit(`Controls, ${name}`, ".controls-page");
    }
    await page.getByRole("group", { name: "Controls section" }).getByRole("button", { name: "Model", exact: true }).click();
    await audit("Controls, model", ".controls-page");

    for (const name of ["help", "cast", "advanced_sampling", "system_prompt", "save_conversation", "load_conversation", "transcript", "manifold_pack", "manifold_builder", "manifold_merge", "template_lab", "health", "subspace", "manifolds", "compare", "correlation", "token_drilldown"]) {
      await openDrawer(page, name, name === "token_drilldown" ? { turnIdx: 1, tokenIdx: 0 } : undefined);
      const drawer = page.getByRole("dialog").last();
      for (const disclosure of await drawer.locator('.sk-disclosure-trigger').all()) {
        if (await disclosure.getAttribute("aria-expanded") === "false") await disclosure.click();
      }
      await audit(`Drawer, ${name}`, name === "token_drilldown" ? "[role='dialog']:has([data-token-details-scroll])" : ".drawer[role='dialog']");
      if (name === "manifold_builder") {
        for (const mode of ["linear", "template", "custom"]) {
          await drawer.getByRole("tab", { name: mode, exact: true }).click();
          if (mode === "custom") await drawer.getByRole("button", { name: "+ add node", exact: true }).click();
          await audit(`Authoring, ${mode}`, ".drawer[role='dialog']");
        }
      }
      if (name === "template_lab") {
        await drawer.getByRole("group", { name: "Template lab view" }).getByRole("button", { name: "build", exact: true }).click();
        await audit("Template editor", ".drawer[role='dialog']");
      }
      await page.keyboard.press("Escape");
    }
    expect(pageErrors).toEqual([]);
    await testInfo.attach("surface-copy-and-coverage.json", { body: JSON.stringify(reports, null, 2), contentType: "application/json" });
  });
}

test("word details use a darker gray without changing light mode or other drawers", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await workbench(page);
  for (const theme of ["dark", "light"]) {
    await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), {
      url: `/@fs${resolve("src/lib/theme.ts")}`, theme,
    });
    await openDrawer(page, "token_drilldown", { turnIdx: 1, tokenIdx: 0 });
    const dialog = page.getByRole("dialog", { name: "Generated word details" });
    const expectedSurface = async (token: string) => page.getByRole("dialog").evaluate((el, token) => {
      const swatch = document.createElement("span");
      swatch.style.backgroundColor = el.matches(".sheet-host.enabled")
        ? "var(--popup-bg)"
        : `color-mix(in srgb, var(${token}) 90%, transparent)`;
      el.append(swatch);
      const color = getComputedStyle(swatch).backgroundColor;
      swatch.remove();
      return color;
    }, token);
    await expect(dialog).toHaveCSS("background-color", await expectedSurface(theme === "dark" ? "--bg-elev" : "--popup-bg"));
    await page.screenshot({ path: testInfo.outputPath(`word-details-${theme}.png`) });
    await openDrawer(page, "help");
    await expect(page.getByRole("dialog")).toHaveCSS("background-color", await expectedSurface("--popup-bg"));
  }
});

test("landing browser copy is consistent across platforms and action labels stay on one line", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const query = new URL(location.href).searchParams;
    const platform = query.get("platform") ?? "macOS";
    const version = query.get("version");
    Object.defineProperties(navigator, {
      userAgent: { configurable: true, value: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Version/26.0 Safari/605.1.15" },
      platform: { configurable: true, value: "MacIntel" },
      maxTouchPoints: { configurable: true, value: 0 },
      userAgentData: { configurable: true, value: version === null ? undefined : {
        platform, mobile: false, getHighEntropyValues: async () => ({ platformVersion: version }),
      } },
    });
  });
  await page.goto(devUrl);
  const contentStoreUrl = `/@fs/${resolve("src/hosted/runtime/contentStore.ts")}`;
  await page.evaluate(async url => {
    await (await import(url)).hasSelectedInstalledModel();
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-runtime", 5);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(["installs", "settings"], "readwrite");
        transaction.objectStore("installs").put({ id: "landing-layout-fixture", kind: "model", objectHashes: [], installedAt: 1 });
        transaction.objectStore("settings").put({ key: "selected_model_variant_id", value: "landing-layout-fixture" });
        transaction.oncomplete = () => { db.close(); resolve(); };
        transaction.onerror = () => { db.close(); reject(transaction.error); };
      };
    });
  }, contentStoreUrl);
  for (const query of [
    "?platform=macOS&version=26.0.0",
    "?platform=macOS&version=15.6.0",
    "?platform=Windows&version=26.0.0",
    "?platform=Linux&version=26.0.0",
    "",
  ]) {
    await page.goto(`${devUrl}/${query}`);
    const recommendation = page.locator(".hero-action-row > span");
    await expect(recommendation).toHaveText("Chrome or Safari · compatible device required");
    const action = page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse", exact: true });
    await expect(action).toHaveAttribute("href", "/app");
    const returningAction = page.locator(".closing-action");
    await expect(returningAction.getByRole("link")).toHaveText("Open Drowse");
    await expect(returningAction.getByRole("link")).toHaveAttribute("href", "/app");
    await expect(returningAction.locator("p")).toHaveText("Choose your model and tools inside the app.");
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const element of [action, recommendation]) {
        if (element === action) {
          expect(await element.evaluate(element => {
            const range = document.createRange();
            range.selectNodeContents(element);
            return range.getClientRects().length;
          })).toBe(1);
        }
        const box = (await element.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(width);
      }
    }
  }
  await page.setViewportSize({ width: 320, height: 900 });
  await page.locator(".hero-action-row").screenshot({ path: testInfo.outputPath("landing-single-line-action.png") });
});

test("smooth scrolling is limited to the homepage and respects reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("/");
  await expect(page.getByText("Scroll to step inside", { exact: true })).toHaveCount(0);
  await expect(page.locator(".scroll-cue")).toHaveCount(0);
  await expect(page.locator("html")).toHaveCSS("scroll-behavior", "smooth");
  const immediatePosition = await page.evaluate(() => {
    window.scrollTo(0, 600);
    return window.scrollY;
  });
  expect(immediatePosition).toBeLessThan(600);
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(600);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator("html")).toHaveCSS("scroll-behavior", "auto");
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse", exact: true }).click();
  await expect(page).toHaveURL(/\/app$/);
  await expect(page.locator(".landing-shell")).toHaveCount(0);
  await expect(page.locator("html")).toHaveCSS("scroll-behavior", "auto");
  await expect(page.locator(".page-header")).toBeVisible();
  await page.goBack();
  await expect(page.locator("html")).toHaveCSS("scroll-behavior", "smooth");
});

test("landing actions and local-compute copy have room at phone and desktop sizes", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(devUrl);
  await page.evaluate(() => document.fonts.ready);
  const action = page.locator(".hero-action-row .primary-action");
  const note = page.locator(".hero-action-row > span");
  const privacy = page.locator(".trust-line");
  await expect(note).toHaveText("Chrome or Safari · compatible device required");
  await expect(privacy).toHaveText("Open source. Inference and saved work stay on your device.");
  await expect(page.getByRole("button", { name: "Pause background animation" })).toHaveCount(0);
  for (const colorScheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme });
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const element of [action, note, privacy]) {
        const box = (await element.boundingBox())!;
        expect(box.x).toBeGreaterThanOrEqual(16);
        expect(box.x + box.width).toBeLessThanOrEqual(width - 16);
        expect(await element.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
      }
      const actionBox = (await action.boundingBox())!;
      const noteBox = (await note.boundingBox())!;
      const privacyBox = (await privacy.boundingBox())!;
      expect(noteBox.y - actionBox.y - actionBox.height).toBeCloseTo(24, 0);
      expect(privacyBox.y - noteBox.y - noteBox.height).toBeCloseTo(24, 0);
      expect(actionBox.x).toBeCloseTo(noteBox.x, 0);
      expect(noteBox.x).toBeCloseTo(privacyBox.x, 0);
      expect(await action.evaluate(element => {
        const style = getComputedStyle(element);
        return [style.paddingTop, style.paddingBottom, style.paddingLeft, style.paddingRight];
      })).toEqual(["16px", "16px", "24px", "24px"]);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator(".hero-copy").screenshot({ path: testInfo.outputPath("landing-action-spacing.png") });
});

test("landing animation follows a changed motion preference without a pause button", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto(devUrl);
  await expect(page.locator(".hero-visual")).toHaveAttribute("data-shader-status", "ready");
  await expect(page.getByRole("button", { name: "Pause background animation" })).toHaveCount(0);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator(".hero-visual")).toHaveAttribute("data-travel", "0.000");
  const pausedTravel = await page.locator(".hero-visual").getAttribute("data-travel");
  await page.evaluate(() => window.scrollTo(0, 900));
  await expect(page.locator(".hero-visual")).toHaveAttribute("data-travel", pausedTravel!);
  await expect(page.locator(".visual-layer")).toHaveCSS("transform", "none");
});

test("orb entrance waits for a rendered scene and wordmark uses the accent on hover", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: "no-preference", colorScheme: "dark" });
  let releaseArtwork!: () => void;
  const artworkGate = new Promise<void>(resolve => { releaseArtwork = resolve; });
  await page.route("**/images/ethereal-orb.jpg", async route => { await artworkGate; await route.continue(); });
  await page.route("**/video/ethereal-orb*.mp4", route => route.abort());
  await page.goto(devUrl, { waitUntil: "domcontentloaded" });
  const visual = page.locator(".hero-visual");
  await expect(visual).toHaveAttribute("data-shader-status", "loading");
  await expect(visual.locator(".fallback")).toHaveCSS("opacity", "0");
  await expect(visual.locator("canvas")).toHaveCSS("opacity", "0");
  releaseArtwork();
  await expect(visual).toHaveAttribute("data-shader-status", "ready");
  await expect(visual.locator("canvas")).toHaveCSS("opacity", "1");
  await expect(visual.locator(".fallback")).toHaveCSS("opacity", "0");
  const brand = page.getByRole("link", { name: "Drowse home", exact: true });
  for (const colorScheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme });
    await brand.hover();
    await expect.poll(() => brand.evaluate(element => {
      const probe = document.createElement("span");
      probe.style.color = "var(--accent)";
      element.append(probe);
      const expected = getComputedStyle(probe).color;
      probe.remove();
      return getComputedStyle(element).color === expected;
    })).toBe(true);
  }
  await page.screenshot({ path: testInfo.outputPath("orb-settled-wordmark.png") });
});

test("ethereal landing renders, enters on scroll, and releases its renderer on navigation", async ({ page, browserName }, testInfo) => {
  test.slow();
  await page.emulateMedia({ reducedMotion: "no-preference", colorScheme: "dark" });
  await page.setViewportSize({ width: 1440, height: 900 });
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(devUrl);
  const visual = page.locator(".hero-visual");
  await expect(visual).toHaveAttribute("data-shader-status", "ready", { timeout: 20_000 });
  await expect(visual).toHaveAttribute("data-travel", "0.000");
  await page.screenshot({ path: testInfo.outputPath("orb-desktop.png") });
  if (browserName === "webkit") await page.evaluate(() => window.scrollTo(0, 900));
  else await page.mouse.wheel(0, 900);
  await expect.poll(async () => Number(await visual.getAttribute("data-travel"))).toBeGreaterThan(0.5);
  await page.screenshot({ path: testInfo.outputPath("orb-entering.png") });
  await page.evaluate(() => window.scrollTo(0, innerHeight * 1.8));
  await expect(visual).toHaveAttribute("data-travel", "1.000");
  await page.waitForTimeout(300);
  await page.screenshot({ path: testInfo.outputPath("orb-inside.png") });
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(visual).toHaveAttribute("data-travel", "0.000");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(visual).toHaveAttribute("data-travel", "0.000");
  const frozen = await visual.locator("canvas").screenshot();
  await page.waitForTimeout(250);
  expect((await visual.locator("canvas").screenshot()).equals(frozen)).toBe(true);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(visual).toHaveAttribute("data-travel", "0.000");
  await page.screenshot({ path: testInfo.outputPath("orb-phone.png") });
  await page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse", exact: true }).click();
  await expect(visual).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("dark desktop orb has bright highlights and settles behind reading content", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "no-preference", colorScheme: "dark" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(devUrl);
  const visual = page.locator(".hero-visual");
  await expect(visual).toHaveAttribute("data-shader-status", "ready");
  await expect(visual.locator("canvas")).toHaveCSS("opacity", "1");
  await expect(visual).toHaveAttribute("data-orb-boost", "2.200");
  await expect.poll(() => page.locator(".visual-layer").evaluate(el => getComputedStyle(el).filter)).not.toContain("brightness(");
  const table = page.locator('#hero-dark-palette feFuncR[type="table"]');
  const values = (await table.getAttribute("tableValues"))!.split(/\s+/).map(Number);
  for (let i = 0; i <= 12; i++) expect(values[i]).toBe(i / 100);
  expect(values[43]).toBeCloseTo(0.946, 3);
  await page.screenshot({ path: testInfo.outputPath("bright-dark-orb.png") });
  await page.evaluate(() => scrollTo({ top: innerHeight * 0.175, behavior: "instant" }));
  await expect.poll(() => visual.evaluate(el => Number(el.getAttribute("data-orb-boost")))).toBeGreaterThan(2.18);
  await page.evaluate(() => scrollTo({ top: innerHeight, behavior: "instant" }));
  await expect.poll(() => visual.evaluate(el => Number(el.getAttribute("data-orb-boost")))).toBeCloseTo(1.918, 2);
  await page.screenshot({ path: testInfo.outputPath("bright-dark-orb-scrolled.png") });
  await page.evaluate(() => scrollTo({ top: innerHeight * 3, behavior: "instant" }));
  const finalBoost = await page.evaluate(() => {
    const progress = Math.min(1, scrollY / (innerHeight * 2.5));
    return 2.2 - 0.8 * progress * progress * (3 - 2 * progress);
  });
  await expect(visual).toHaveAttribute("data-orb-boost", finalBoost.toFixed(3));
  const dimmedValues = (await table.getAttribute("tableValues"))!.split(/\s+/).map(Number);
  for (let i = 0; i <= 12; i++) expect(dimmedValues[i]).toBe(values[i]);
  expect(dimmedValues[43]).toBeCloseTo(0.43 * finalBoost, 3);
  await page.evaluate(() => scrollTo({ top: 0, behavior: "instant" }));
  await expect(visual).toHaveAttribute("data-orb-boost", "2.200");
  await setAppearance(page, "Light");
  await expect.poll(() => page.locator(".visual-layer").evaluate(el => getComputedStyle(el).filter)).not.toContain("brightness(");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(visual).toHaveAttribute("data-orb-boost", "1.000");
  await setAppearance(page, "Dark");
  await page.evaluate(() => scrollTo({ top: innerHeight * 3, behavior: "instant" }));
  await expect(visual).toHaveAttribute("data-orb-boost", "0.900");
  const mobileValues = (await table.getAttribute("tableValues"))!.split(/\s+/).map(Number);
  for (let i = 0; i <= 12; i++) expect(mobileValues[i]).toBe(values[i]);
  await page.evaluate(() => scrollTo({ top: 0, behavior: "instant" }));
  await expect(visual).toHaveAttribute("data-orb-boost", "1.000");
});

test("ethereal landing keeps rendering without video and reflows in both themes", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.route("**/video/ethereal-orb*.mp4", route => route.abort());
  await page.goto(devUrl);
  const visual = page.locator(".hero-visual");
  await expect(visual).toHaveAttribute("data-shader-status", "ready");
  const canvas = visual.locator("canvas");
  await expect(canvas).toHaveCSS("opacity", "1");
  const firstFrame = await canvas.screenshot();
  await expect.poll(async () => (await canvas.screenshot()).equals(firstFrame)).toBe(false);
  await page.evaluate(() => window.scrollTo(0, innerHeight * 1.8));
  await expect(visual).toHaveAttribute("data-travel", "1.000");
  await page.evaluate(() => window.scrollTo(0, 0));
  for (const colorScheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme });
    for (const width of [320, 390, 760, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse", exact: true })).toBeVisible();
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: testInfo.outputPath("orb-phone-fallback.png") });
});

test("ethereal landing animates when autoplay is denied and recovers its graphics context", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.addInitScript(() => {
    HTMLMediaElement.prototype.play = () => Promise.reject(new DOMException("Autoplay denied", "NotAllowedError"));
  });
  await page.goto(devUrl);
  const visual = page.locator(".hero-visual");
  const canvas = visual.locator("canvas");
  await expect(visual).toHaveAttribute("data-shader-status", "ready");
  await expect(canvas).toHaveCSS("opacity", "1");
  const firstFrame = await canvas.screenshot();
  await expect.poll(async () => (await canvas.screenshot()).equals(firstFrame)).toBe(false);
  await canvas.evaluate(element => {
    const gl = (element as HTMLCanvasElement).getContext("webgl2")!;
    const context = gl.getExtension("WEBGL_lose_context")!;
    element.addEventListener("webglcontextlost", () => setTimeout(() => context.restoreContext(), 100), { once: true });
    context.loseContext();
  });
  await expect(visual).toHaveAttribute("data-shader-status", "fallback");
  await expect(visual).toHaveAttribute("data-shader-status", "ready");
  expect(await canvas.evaluate(element => (element as HTMLCanvasElement).getContext("webgl2")!.getError())).toBe(0);
});

test("ethereal landing starts offline even when animation was disabled on its first visit", async ({ page, context, browserName }) => {
  test.setTimeout(90_000);
  test.skip(browserName === "webkit", "This WebKit runner fails cached script requests with an internal error under offline emulation; blocked-video rendering is tested separately.");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.reload();
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);
  await context.setOffline(true);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  const visual = page.locator(".hero-visual");
  await expect(visual).toHaveAttribute("data-shader-status", "ready");
  await page.evaluate(() => window.scrollTo(0, innerHeight * 1.8));
  await expect(visual).toHaveAttribute("data-travel", "1.000");
  await expect(visual.locator("canvas")).toHaveCSS("opacity", "1");
  const firstFrame = await visual.locator("canvas").screenshot();
  await expect.poll(async () => (await visual.locator("canvas").screenshot()).equals(firstFrame), { timeout: 30_000 }).toBe(false);
});

test("ethereal landing restores its renderer after a fully offline reload", async ({ page, context, browserName }) => {
  test.skip(browserName === "webkit", "This WebKit runner reports an internal navigation error on offline reload; offline renderer startup is tested separately.");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.reload();
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);
  await context.setOffline(true);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.reload({ waitUntil: "domcontentloaded" });
  await expect(page.locator(".hero-visual")).toHaveAttribute("data-shader-status", "ready");
  await expect(page.locator(".hero-visual canvas")).toHaveCSS("opacity", "1");
});

test("landing explains the workbench and separates chat and base model uses", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(devUrl);
  const features = page.locator(".capabilities");
  await expect(features.getByRole("heading", { level: 2 })).toHaveText("Test what shapes a reply.");
  await expect(features.locator("li")).toHaveCount(3);
  await expect(features).toContainText("predictions across layers");
  const models = page.locator(".models");
  await expect(models).not.toContainText("Best quality");
  await expect(models).not.toContainText("Larger");
  await expect(models).not.toContainText("GB");
  await expect(models.getByRole("region", { name: "Chat models", exact: true })).toContainText("Gemma 3 1B");
  await expect(models.getByRole("region", { name: "Chat models", exact: true })).toContainText("Gemma 3 4B");
  const base = models.getByRole("region", { name: "Base models", exact: true });
  await expect(base.locator(".model-row")).toHaveCount(4);
  await expect(base).toContainText("Continue raw text.");
  await expect(models).not.toContainText("M means million parameters");
  await expect(models).not.toContainText("These candidates have existing");
  await expect(models).not.toContainText("validation needed");
  await expect(base.locator("a, button")).toHaveCount(0);
  await expect(models.getByRole("link", { name: "Open Drowse", exact: true })).toHaveAttribute("href", "/app");
  await expect(models.locator(".closing-action p")).toHaveText("Choose your model and tools inside the app.");
  for (const theme of ["Light", "Dark"]) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await setAppearance(page, theme);
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme.toLowerCase());
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await models.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      expect(await models.locator(".model-row").evaluateAll(rows => rows.every(row => row.scrollWidth <= row.clientWidth))).toBe(true);
    }
    await features.screenshot({ path: testInfo.outputPath(`homepage-features-${theme}.png`) });
    await models.screenshot({ path: testInfo.outputPath(`homepage-models-${theme}.png`) });
  }
});

test("landing has no gradient scrims and adapts its ink without panel backgrounds", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(devUrl);
  for (const theme of ["Light", "Dark"]) {
    await setAppearance(page, theme);
    for (const width of [390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await page.locator(".capabilities").scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => {
        return [...document.querySelectorAll(".landing-shell, .landing-shell *")]
          .filter(element => !element.closest('button, a, .theme-toggle, .capability-demo'))
          .flatMap(element =>
          [null, "::before", "::after"].flatMap(pseudo => {
            const style = getComputedStyle(element, pseudo);
            return style.backgroundImage.includes("gradient(") ? [element.className] : [];
          })
        );
      })).toEqual([]);
      for (const selector of [".hero", ".capabilities", ".models"]) {
        await expect(page.locator(selector)).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
        expect(await page.locator(selector).evaluate(element => getComputedStyle(element, "::before").content)).toBe("none");
      }
      await expect(page.locator("#capabilities-title")).toHaveCSS("mix-blend-mode", "normal");
      const foreground = await page.locator("#capabilities-title").evaluate(element => getComputedStyle(element).color);
      const channels = foreground.match(/[\d.]+/g)!.slice(0, 3).map(Number);
      const luminance = (rgb: number[]) => rgb.map(value => {
        const channel = value / 255;
        return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
      }).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
      // Bound the mobile backdrop; desktop orb highlights are independently boosted.
      const filter = await page.locator(".visual-layer .fallback").evaluate(element => getComputedStyle(element).filter);
      let darkestBackdrop: number[];
      if (theme === "Light") {
        expect(filter).toContain("#hero-light-palette");
        expect(filter).not.toContain("invert(");
        const values = (await page.locator("#hero-light-palette feColorMatrix").getAttribute("values"))!.trim().split(/\s+/).map(Number);
        darkestBackdrop = [0, 1, 2].map(row => (values[row * 5 + 4] + values.slice(row * 5, row * 5 + 3).reduce((sum, value) => sum + Math.min(0, value), 0)) * 255);
        expect(darkestBackdrop[2] - darkestBackdrop[1]).toBeGreaterThan(25);
        expect(darkestBackdrop[0] - darkestBackdrop[1]).toBeGreaterThan(12);
      } else {
        expect(filter).toContain("#hero-dark-palette");
        const brightness = Number(await page.locator('#hero-dark-palette feFuncR[type="gamma"]').getAttribute("amplitude"));
        darkestBackdrop = [brightness * 255, brightness * 255, brightness * 255];
      }
      const backdropLuminance = luminance(darkestBackdrop);
      const inkLuminance = luminance(channels);
      if (theme === "Light" || width <= 760) {
        expect((Math.max(inkLuminance, backdropLuminance) + 0.05) / (Math.min(inkLuminance, backdropLuminance) + 0.05)).toBeGreaterThanOrEqual(4.5);
      }
      await expect(page.locator(".primary-action").first()).toHaveCSS("mix-blend-mode", "normal");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`clean-landing-${theme}-${width}.png`) });
    }
  }
  await page.emulateMedia({ contrast: "more" });
  await expect(page.locator(".hero-visual")).toHaveCount(0);
  await expect(page.locator("#capabilities-title")).toHaveCSS("mix-blend-mode", "normal");
  await expect(page.locator(".animation-control")).toHaveCount(0);
  await page.emulateMedia({ contrast: "no-preference" });
  await expect(page.locator(".hero-visual")).toHaveCount(1);
  await expect(page.locator("#capabilities-title")).toHaveCSS("mix-blend-mode", "normal");
});

test("homepage fills the viewport without a scrollbar strip and still scrolls", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(devUrl);
  const root = page.locator("html");
  await expect(root).toHaveCSS("scrollbar-gutter", "auto");
  await expect(root).toHaveCSS("scrollbar-width", "none");
  for (const theme of ["Light", "Dark"]) {
    await setAppearance(page, theme);
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await page.evaluate(() => scrollTo({ top: 0, behavior: "instant" }));
      const geometry = await page.evaluate(() => {
        const root = document.documentElement;
        const background = document.querySelector(".hero-visual")!.getBoundingClientRect();
        const shell = document.querySelector(".landing-shell")!.getBoundingClientRect();
        return {
          gutter: innerWidth - root.clientWidth,
          overflow: root.scrollWidth - root.clientWidth,
          left: background.left,
          right: background.right - innerWidth,
          bottom: background.bottom - innerHeight,
          shellWidth: shell.width - innerWidth,
        };
      });
      expect(geometry).toEqual({ gutter: 0, overflow: 0, left: 0, right: 0, bottom: 0, shellWidth: 0 });
      await page.locator("#hero-title").click();
      await page.keyboard.press("PageDown");
      await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(0);
      await page.evaluate(() => scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" }));
      await expect(page.locator("footer")).toBeInViewport();
      await page.screenshot({ path: testInfo.outputPath(`full-bleed-home-${theme}-${width}.png`) });
    }
  }
  await page.goto(`${devUrl}/app?layoutFixture=1`);
  await expect(root).toHaveCSS("scrollbar-gutter", "stable");
  await expect(root).not.toHaveCSS("scrollbar-width", "none");
});

test("mobile hero overlays the orb without selecting the section", async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto(devUrl);
  await expect(page.locator(".hero-visual")).toHaveAttribute("data-shader-status", "ready");
  await expect(page.locator(".hero-visual canvas")).toHaveCSS("opacity", "1");
  for (const theme of ["Light", "Dark"]) {
    if (await page.locator("html").getAttribute("data-theme") !== theme.toLowerCase()) {
      await setAppearance(page, theme);
    }
    for (const width of [320, 390, 760]) {
      await page.setViewportSize({ width, height: 844 });
      await page.evaluate(() => scrollTo({ top: 0, behavior: "instant" }));
      const heading = page.locator("#hero-title");
      const box = (await heading.boundingBox())!;
      const header = (await page.locator(".page-header").boundingBox())!;
      expect(box.y).toBeGreaterThanOrEqual(header.y + header.height);
      expect(box.y).toBeLessThan(page.viewportSize()!.height * 0.3);
      expect(await heading.evaluate(element => {
        const style = getComputedStyle(element);
        return style.getPropertyValue("user-select") || style.getPropertyValue("-webkit-user-select");
      })).toBe("none");
      await heading.click();
      expect(await page.locator("main").evaluate(element => element === document.activeElement)).toBe(false);
      await expect(page.locator("main")).toHaveCSS("outline-style", "none");
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      const action = page.locator(".hero .primary-action");
      await action.focus();
      await expect(action).toBeFocused();
      await page.keyboard.press("Tab");
      await action.focus();
      await expect(action).toBeFocused();
      await expect(action).toHaveCSS("outline-style", "solid");
      await heading.click();
      await page.evaluate(() => scrollTo({ top: 0, behavior: "instant" }));
      await page.screenshot({ path: testInfo.outputPath(`hero-overlay-${theme}-${width}.png`) });
    }
  }
  await page.locator(".hero .primary-action").focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/app(?:\?|$)/);
});

test("offline notices stay dismissible without moving or overflowing the workbench", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "Headless WebKit rejects OPFS; this reload test requires the persistent worker fixture.");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${devUrl}/app?fixture=1`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "onLine", { configurable: true, get: () => false });
    Object.defineProperty(navigator.serviceWorker, "controller", { configurable: true, get: () => ({ state: "activated" }) });
    navigator.serviceWorker.getRegistration = async () => ({ active: { state: "activated" } }) as ServiceWorkerRegistration;
  });
  await page.reload();
  await expect(page.locator(".shell")).toBeVisible();
  const notice = page.getByRole("status").filter({ hasText: "The interface is ready offline." });
  await expect(notice).toBeVisible();
  const shell = page.locator(".shell");
  const noticeBox = (await notice.boundingBox())!;
  expect(noticeBox.y).toBeGreaterThanOrEqual(0);
  expect(noticeBox.y + noticeBox.height).toBeLessThanOrEqual(844);
  await expect.poll(async () => {
    const box = (await shell.boundingBox())!;
    return box.y + box.height;
  }).toBeLessThanOrEqual(845);
  const before = (await shell.boundingBox())!.height;
  await notice.getByRole("button", { name: "Dismiss" }).click();
  const dismissalOverflow = await page.evaluate(async () => {
    let overflow = 0;
    for (let frame = 0; frame < 20; frame++) {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const bottom = document.querySelector(".shell")!.getBoundingClientRect().bottom;
      overflow = Math.max(overflow, bottom - window.innerHeight);
    }
    return overflow;
  });
  expect(dismissalOverflow).toBeLessThanOrEqual(1);
  await expect(notice).not.toBeVisible();
  await expect.poll(async () => (await shell.boundingBox())!.height).toBe(before);
  await expect(page.getByRole("textbox", { name: /^Compose as / })).toBeVisible();
});

test("template deletion names the saved work and requires confirmation", async ({ page, browserName }) => {
  test.skip(browserName === "webkit", "Template persistence requires the OPFS-backed worker; WebKit's temporary profile rejects OPFS.");
  await page.setViewportSize({ width: 320, height: 720 });
  await page.goto(`${devUrl}/app?${browserName === "webkit" ? "layoutFixture=setup" : "fixture=1"}`);
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.evaluate(async (url) => {
    const { apiTemplates } = await import(url);
    await apiTemplates.create({ namespace: "local", name: "weekday", slot: "[DAY]", values: ["Monday", "Tuesday"], contexts: [{ turns: [{ role: "user", content: "What day is it?" }], assistant: "Today is [DAY]." }] });
  }, servicesUrl);
  await openDrawer(page, "template_lab");
  const drawer = page.getByRole("dialog").last();
  const remove = drawer.getByRole("button", { name: "Delete template local/weekday", exact: true });
  await remove.click();
  await expect(drawer.getByText("Delete local/weekday? This removes the saved template and cannot be undone.")).toBeVisible();
  await drawer.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(remove).toBeVisible();
  await remove.click();
  await expect(drawer.getByRole("button", { name: "Delete template", exact: true })).toBeVisible();
  expect(await drawer.evaluate((element) => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1);
  await drawer.getByRole("button", { name: "Delete template", exact: true }).click();
  await expect(drawer.locator(".cat-row")).toHaveCount(0);
});

test("native API access form retains labels and reflows on phones", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 320, height: 720 });
  // Substitute the native component for the hosted alias only in this test.
  await page.route("**/src/drawers/UnavailableRuntimeDrawer.svelte*", async (route) => {
    const response = await route.fetch({ url: `${devUrl}/@fs/${resolve("src/drawers/SessionAdminDrawer.svelte")}` });
    await route.fulfill({ response });
  });
  await workbench(page);
  await page.evaluate(async (url) => {
    const stores = await import(url);
    stores.drawerState.open = "session_admin";
    stores.drawerState.params = null;
  }, storesUrl);
  const drawer = page.getByRole("dialog").last();
  await expect(drawer.getByRole("heading", { name: "API access", exact: true })).toBeVisible();
  await expect(drawer.getByLabel("Drowse API key")).toBeVisible();
  await expect(drawer.getByText("Changes apply only to this tab until you reload.")).toBeVisible();
  expect(await drawer.evaluate((element) => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1);
  const result = await new AxeBuilder({ page }).include('.drawer[role="dialog"]').withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze();
  expect(result.violations).toEqual([]);
});
