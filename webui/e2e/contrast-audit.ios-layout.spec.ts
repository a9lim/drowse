import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page, type TestInfo } from "@playwright/test";
import { resolve } from "node:path";
import { selectWorkspaceView } from "./workbench-navigation";

const devUrl = "http://127.0.0.1:4176";
const moduleUrl = (path: string) => `/@fs${resolve(path)}`;
const storesUrl = moduleUrl("src/lib/stores.svelte.ts");

test.describe.configure({ mode: "parallel" });

async function setTheme(page: Page, theme: string) {
  await page.evaluate(async ({ url, theme }) => (await import(url)).setTheme(theme), {
    url: moduleUrl("src/lib/theme.ts"), theme,
  });
}

async function measureAction(button: Locator) {
  const foreground = await button.evaluate(element => {
    const style = getComputedStyle(element);
    return { color: style.color, opacity: style.opacity, backdrop: getComputedStyle(element.parentElement!).backgroundColor };
  });
  const screenshot = await button.screenshot({
    scale: "css",
    style: "* { -webkit-text-fill-color: transparent !important; text-shadow: none !important; }",
  });
  await button.page().evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  return button.page().evaluate(async ({ bytes, foreground }) => {
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(bytes)], { type: "image/png" }));
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d")!;
    context.drawImage(bitmap, 0, 0);
    const backgrounds = [0.25, 0.5, 0.75].map(x => Array.from(context.getImageData(
      Math.floor(canvas.width * x), Math.floor(canvas.height / 2), 1, 1,
    ).data).slice(0, 3).map(c => c / 255));
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = foreground.color;
    context.fillRect(0, 0, 1, 1);
    let ink = Array.from(context.getImageData(0, 0, 1, 1).data).slice(0, 3).map(c => c / 255);
    if (Number(foreground.opacity) < 1) {
      context.clearRect(0, 0, 1, 1);
      context.fillStyle = foreground.backdrop;
      context.fillRect(0, 0, 1, 1);
      const backdrop = Array.from(context.getImageData(0, 0, 1, 1).data).slice(0, 3).map(c => c / 255);
      ink = ink.map((c, i) => c * Number(foreground.opacity) + backdrop[i] * (1 - Number(foreground.opacity)));
    }
    bitmap.close();
    const luminance = (rgb: number[]) => rgb.map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
      .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
    const apcaLuminance = (rgb: number[]) => rgb.reduce((sum, c, i) => sum + c ** 2.4 * [0.2126729, 0.7151522, 0.072175][i], 0);
    const clamp = (y: number) => y < 0.022 ? y + (0.022 - y) ** 1.414 : y;
    return backgrounds.map(background => {
      const light = [luminance(ink), luminance(background)].sort((a, b) => b - a);
      const textY = clamp(apcaLuminance(ink));
      const bgY = clamp(apcaLuminance(background));
      const raw = bgY > textY ? (bgY ** 0.56 - textY ** 0.57) * 1.14 : (bgY ** 0.65 - textY ** 0.62) * 1.14;
      return {
        ...foreground, background: background.map(c => Math.round(c * 255)),
        wcag: (light[0] + 0.05) / (light[1] + 0.05),
        apca: Math.abs(Math.abs(raw) < 0.1 ? 0 : (raw > 0 ? raw - 0.027 : raw + 0.027) * 100),
      };
    });
  }, { bytes: [...screenshot], foreground });
}

async function expectActionContrast(button: Locator, label: string) {
  const samples = await measureAction(button);
  expect.soft(Math.min(...samples.map(s => s.wcag)), `${label}: ${JSON.stringify(samples)}`).toBeGreaterThanOrEqual(4.5);
  expect.soft(Math.min(...samples.map(s => s.apca)), `${label}: ${JSON.stringify(samples)}`).toBeGreaterThanOrEqual(60);
  return samples;
}

async function mountStorageHome(page: Page) {
  await page.goto(`${devUrl}/outside-the-workbench`);
  await page.evaluate(async ({ componentUrl, libraryUrl }) => {
    const [{ default: HostedHome }, { mount }, { conversationLibrary }] = await Promise.all([
      import(componentUrl), import("/e2e/svelte-runtime.ts"), import(libraryUrl),
    ]);
    conversationLibrary.listSummaries = async () => ({ conversations: [], issues: [] });
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(HostedHome, { target, props: {
      controller: {
        capabilities: () => ({ signals: { appleMobile: false } }),
        retryPersistence: async () => false, check: async () => {}, open: async () => {},
      },
      snapshot: {
        phase: "supported", headline: "Ready", detail: "Ready", checks: [], models: [],
        download: { available: true, phase: "idle", reason: "Ready" },
        runtime: { available: true, phase: "unloaded", reason: "Ready" },
        storage: { availableBytes: 2_000_000_000, persisted: false },
      },
      onChooseModels() {},
    } });
  }, {
    componentUrl: moduleUrl("src/hosted/ui/HostedHome.svelte"),
    libraryUrl: moduleUrl("src/lib/stores/savedConversations.svelte.ts"),
  });
}

async function auditActionStates(page: Page, selector: string, label: string) {
  const candidates = await page.locator(selector).locator("button:visible, a:visible, summary:visible").evaluateAll(elements => {
    const seen = new Set<string>();
    return elements.flatMap((element, index) => {
      if (element.matches(":disabled, .skip-link") || element.closest("[inert]") || !element.textContent?.trim()
        || element.getAttribute("aria-label")?.startsWith("Inspect token")) return [];
      const style = getComputedStyle(element);
      const key = [element.tagName, element.className, style.color, style.backgroundColor, style.backgroundImage].join("|");
      if (seen.has(key)) return [];
      seen.add(key);
      element.setAttribute("data-contrast-control", String(index));
      return { index, text: element.textContent.trim().slice(0, 80) };
    });
  });
  const checked = [];
  for (const candidate of candidates) {
    const control = page.locator(`${selector} [data-contrast-control="${candidate.index}"]`);
    await control.scrollIntoViewIfNeeded();
    for (const state of ["rest", "hover"]) {
      if (state === "hover") await control.hover();
      else await page.mouse.move(0, 0);
      const pair = await control.evaluate(element => {
        const style = getComputedStyle(element);
        let opacity = 1;
        for (let parent: Element | null = element; parent; parent = parent.parentElement) opacity *= Number(getComputedStyle(parent).opacity);
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d")!;
        const rgba = (color: string) => {
          context.clearRect(0, 0, 1, 1);
          context.fillStyle = color;
          context.fillRect(0, 0, 1, 1);
          return Array.from(context.getImageData(0, 0, 1, 1).data).map(c => c / 255);
        };
        const ink = rgba(style.color), background = rgba(style.backgroundColor);
        if (background[3] < 0.99 || ink[3] < 0.99 || opacity < 0.99) return null;
        const luminance = (rgb: number[]) => rgb.slice(0, 3).map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
          .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
        const light = [luminance(ink), luminance(background)].sort((a, b) => b - a);
        return { color: style.color, background: style.backgroundColor, ratio: (light[0] + 0.05) / (light[1] + 0.05) };
      });
      if (pair) {
        checked.push({ ...candidate, state, ...pair });
        if (pair.ratio < 4.5) await expectActionContrast(control, `${label} ${candidate.text} ${state}`);
      }
    }
    await control.evaluate(element => element.removeAttribute("data-contrast-control"));
  }
  await page.mouse.move(0, 0);
  return checked;
}

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    const storage = navigator.storage ?? {};
    Object.defineProperty(navigator, "storage", { configurable: true, value: storage });
    Object.defineProperty(storage, "persist", { configurable: true, value: async () => false });
    Object.defineProperty(storage, "persisted", { configurable: true, value: async () => false });
  });
});

async function auditSurface(page: Page, testInfo: TestInfo, theme: string, name: string, selector = "body") {
  await page.mouse.move(0, 0);
  const result = await new AxeBuilder({ page }).include(selector).withRules(["color-contrast"]).analyze();
  const summarize = ({ nodes }: any) => nodes.map((node: any) => ({ target: node.target, html: node.html, failure: node.failureSummary }));
  const violations = result.violations.flatMap(summarize);
  const controls = await auditActionStates(page, selector, `${theme} ${name}`);
  await testInfo.attach("contrast-surface-audit", {
    body: JSON.stringify({ name, violations, incomplete: result.incomplete.flatMap(summarize), controls }, null, 2),
    contentType: "application/json",
  });
  expect.soft(violations, name).toEqual([]);
  console.log(`${theme}: ${name} (${violations.length} contrast failures)`);
}

async function populatedWorkbench(page: Page, theme: string) {
  await page.goto(`${devUrl}/app?layoutFixture=instruments`);
  await expect(page.locator(".shell")).toBeVisible();
  await setTheme(page, theme);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("A contrast check with a populated conversation.");
  await page.getByRole("button", { name: /^(Send|Generate reply)$/ }).click();
  await expect(page.locator(".chat").getByRole("button", { name: "Stop", exact: true, includeHidden: true })).toBeDisabled();
}

for (const theme of ["dark", "light"]) {
  test(`disabled button labels remain discernible in ${theme}`, async ({ page }, testInfo) => {
    await page.goto(`${devUrl}/outside-the-workbench`);
    await setTheme(page, theme);
    await page.evaluate(async ({ buttonUrl, numberUrl }) => {
      const [{ mount, createRawSnippet }, { default: Button }, { default: NumberInput }] = await Promise.all([
        import("/e2e/svelte-runtime.ts"), import(buttonUrl), import(numberUrl),
      ]);
      document.body.replaceChildren();
      const target = document.createElement("main");
      target.style.cssText = "display:flex;flex-wrap:wrap;gap:16px;padding:24px;background:var(--bg-alt)";
      document.body.append(target);
      for (const variant of ["solid", "ghost", "flat", "danger"]) {
        mount(Button, { target, props: { variant, disabled: true, children: createRawSnippet(() => ({ render: () => `<span>${variant}</span>` })) } });
      }
      mount(NumberInput, { target, props: { value: 42, disabled: true, ariaLabel: "Disabled numeric value" } });
    }, { buttonUrl: moduleUrl("src/lib/ui/Button.svelte"), numberUrl: moduleUrl("src/lib/NumberInput.svelte") });
    await expect(page.getByRole("spinbutton", { name: "Disabled numeric value" })).toBeDisabled();
    await expect(page.locator(".sk-number")).toHaveCSS("opacity", "0.65");
    await expect(page.locator(".sk-number-input")).toHaveCSS("opacity", "1");
    await expect(page.locator(".number-rest")).toHaveCSS("opacity", "1");
    await expect(page.locator(".number-rest")).toHaveText("42");
    await page.locator("main").screenshot({ path: testInfo.outputPath(`disabled-${theme}.png`) });
    for (const button of await page.locator("button").all()) {
      const samples = await measureAction(button);
      console.log(`disabled ${theme}: ${await button.innerText()} ${JSON.stringify(samples)}`);
      expect.soft(Math.min(...samples.map(sample => sample.apca)), JSON.stringify(samples)).toBeGreaterThanOrEqual(30);
    }
  });

  test(`warning actions preserve readable ink through interaction in ${theme}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    const reports: unknown[] = [];
    for (const surface of ["chats", "saved-chats"]) {
      if (surface === "chats") await mountStorageHome(page);
      else {
        await page.goto(`${devUrl}/app?layoutFixture=instruments`);
        await expect(page.locator(".shell")).toBeVisible();
        await page.evaluate(async ({ storesUrl, workspaceUrl }) => {
          const stores = await import(storesUrl);
          const { captureConversationSnapshot } = await import(workspaceUrl);
          await stores.conversationLibrary.create({ name: "Contrast audit", snapshot: captureConversationSnapshot() });
        }, { storesUrl, workspaceUrl: moduleUrl("src/lib/conversationWorkspace.ts") });
        await page.evaluate(async url => (await import(url)).openDrawer("load_conversation"), storesUrl);
      }
      await setTheme(page, theme);
      const button = page.getByRole("button", { name: "Protect storage", exact: true });
      await expect(button).toBeVisible();
      for (const width of [390, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        await page.mouse.move(0, 0);
        reports.push({ surface, width, state: "rest", samples: await expectActionContrast(button, `${surface} ${theme} ${width} rest`) });
        await button.hover();
        reports.push({ surface, width, state: "hover", samples: await expectActionContrast(button, `${surface} ${theme} ${width} hover`) });
        await page.mouse.down();
        reports.push({ surface, width, state: "pressed", samples: await expectActionContrast(button, `${surface} ${theme} ${width} pressed`) });
        await page.mouse.move(0, 0);
        await page.mouse.up();
        await button.focus();
        reports.push({ surface, width, state: "focus", samples: await expectActionContrast(button, `${surface} ${theme} ${width} focus`) });
        await button.screenshot({ path: testInfo.outputPath(`${surface}-${theme}-${width}.png`) });
      }
    }
    await testInfo.attach("rendered-warning-contrast", { body: JSON.stringify(reports, null, 2), contentType: "application/json" });
  });

  for (const path of ["/", "/credits", "/contact", "/missing-page", "/app?layoutFixture=setup"]) {
    test(`public text retains contrast in ${theme}: ${path}`, async ({ page }, testInfo) => {
      await page.goto(`${devUrl}${path}`);
      await setTheme(page, theme);
      await page.evaluate(() => document.fonts.ready);
      for (const width of [390, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        await auditSurface(page, testInfo, theme, `${path} ${width}`);
      }
    });
  }
  test(`chats and storage notice retain contrast in ${theme}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mountStorageHome(page);
    await setTheme(page, theme);
    await auditSurface(page, testInfo, theme, "Chats and storage notice");
  });
  for (const view of ["Chat", "Loom", "Controls"]) {
    test(`${view} text retains contrast in ${theme}`, async ({ page }, testInfo) => {
      await populatedWorkbench(page, theme);
      for (const width of [390, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        await selectWorkspaceView(page, view === "Chat" ? /^(Chat|Conversation)$/ : view);
        await auditSurface(page, testInfo, theme, `${view} ${width}`);
      }
    });
  }
  for (const name of ["appearance", "local_runtime", "help", "feedback", "cast", "advanced_sampling", "system_prompt", "save_conversation", "load_conversation", "download_chat", "transcript", "manifold_pack", "manifold_builder", "manifold_merge", "template_lab", "health", "subspace", "manifolds", "compare", "correlation", "token_drilldown", "node_compare"]) {
    test(`${name} dialog text retains contrast in ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width: 1440, height: 900 });
      await populatedWorkbench(page, theme);
      await selectWorkspaceView(page, "Controls");
      await page.evaluate(async ({ url, name }) => {
        const stores = await import(url);
        if (["manifold_builder", "template_lab", "health"].includes(name)) stores.drawerState.open = name;
        else stores.openDrawer(name, name === "token_drilldown" ? { turnIdx: 1, tokenIdx: 0 } : undefined);
      }, { url: storesUrl, name });
      await expect(page.locator('.drawer[role="dialog"]')).toBeVisible();
      await auditSurface(page, testInfo, theme, name, '.drawer[role="dialog"]');
      await page.evaluate(async url => (await import(url)).closeDrawer(), storesUrl);
    });
  }
  test(`base completion text retains contrast in ${theme}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(`${devUrl}/app?layoutFixture=base`);
    await expect(page.locator(".shell")).toBeVisible();
    await setTheme(page, theme);
    await auditSurface(page, testInfo, theme, "Base completion");
  });
}
