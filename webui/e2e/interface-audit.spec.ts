import { selectWorkspaceView } from "./workbench-navigation";
import { clickWorkspaceAction } from "./workbench-navigation";
import { openWorkspaceMenu } from "./workbench-navigation";
import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";
import { showWorkspaceTools } from "./workbench-navigation";

const devUrl = "http://127.0.0.1:4176";
const drawerStoreUrl = `/@fs/${resolve("src/lib/stores.svelte.ts")}`;

const sendButton = (page: Page) => page.getByRole("button", {
  name: /^(Send|Generate reply|Add message)$/,
});

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

async function useTouchViewport(page: Page): Promise<void> {
  await page.setViewportSize({ width: 320, height: 720 });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Emulation.setTouchEmulationEnabled", {
    enabled: true,
    maxTouchPoints: 5,
  });
}

async function openFixtureWorkbench(page: Page): Promise<void> {
  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
}

async function openTranscriptDrawer(page: Page) {
  await clickWorkspaceAction(page, "All tools");
  const search = page.getByRole("combobox", { name: "Filter commands" });
  await search.fill("conversation transcript");
  await page.keyboard.press("Enter");
  const drawer = page.getByRole("dialog", { name: "Conversation transcript" });
  await expect(drawer).toBeVisible();
  return drawer;
}

async function expectViewportContainment(page: Page): Promise<void> {
  const result = await page.evaluate(() => {
    const viewportWidth = document.documentElement.clientWidth;
    const overflow = document.documentElement.scrollWidth - viewportWidth;
    const escaped = [...document.querySelectorAll<HTMLElement>(
      "button, a[href], input, select, textarea, [role='button'], [role='tab'], [role='menuitem'], [role='radio']",
    )].filter((element) => {
      const style = getComputedStyle(element);
      const box = element.getBoundingClientRect();
      if (style.display === "none" || style.visibility === "hidden" || box.width === 0 || box.height === 0) {
        return false;
      }
      const navigation = element.closest<HTMLElement>('nav[aria-label="Primary navigation"]');
      if (navigation && navigation.scrollWidth > navigation.clientWidth && getComputedStyle(navigation).overflowX === "auto") {
        return false;
      }
      return box.left < -1 || box.right > viewportWidth + 1;
    }).map((element) => ({
      tag: element.tagName,
      name: element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 80),
      box: element.getBoundingClientRect().toJSON(),
    }));
    return { overflow, escaped };
  });
  expect(result.overflow).toBeLessThanOrEqual(0);
  expect(result.escaped).toEqual([]);
  for (const link of await page.getByRole("navigation", { name: "Primary navigation", exact: true }).getByRole("link").all()) {
    await link.scrollIntoViewIfNeeded();
    const box = (await link.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(await page.evaluate(() => innerWidth));
  }
}

async function expectCoherentHeadingOutline(page: Page): Promise<void> {
  const outline = await page.locator("h1, h2, h3, h4, h5, h6").evaluateAll((headings) =>
    headings.filter((heading) => {
      const box = heading.getBoundingClientRect();
      return getComputedStyle(heading).display !== "none" && box.width > 0 && box.height > 0;
    }).map((heading) => ({
      level: Number(heading.tagName.slice(1)),
      text: heading.textContent?.trim() ?? "",
    }))
  );
  expect(outline.filter((heading) => heading.level === 1)).toHaveLength(1);
  for (let index = 1; index < outline.length; index += 1) {
    expect(
      outline[index].level,
      `heading ${JSON.stringify(outline[index].text)} skips a level`,
    ).toBeLessThanOrEqual(outline[index - 1].level + 1);
  }
}

test("workspace titles remain accessible without repeating visible heading blocks", async ({ page }) => {
  await openFixtureWorkbench(page);
  const outline = await page.locator("h1, h2, h3, h4, h5, h6").evaluateAll((headings) =>
    headings.map((heading) => ({
      level: Number(heading.tagName.slice(1)),
      text: heading.textContent?.trim() ?? "",
    }))
  );
  expect(outline[0]).toEqual({ level: 1, text: "Talk with your model" });
  await expect(page.locator(".workspace-heading, .workspace-eyebrow")).toHaveCount(0);
  await expect(page.locator("#conversation-title.workspace-title-sr")).toHaveText("Talk with your model");

  await selectWorkspaceView(page, "Loom");
  await expect(page.locator("#branches-title.workspace-title-sr")).toHaveText("See every path");

  await selectWorkspaceView(page, "Controls");
  await expect(page.locator("#controls-title.workspace-title-sr")).toHaveText("Response, model, and chat controls");
  await expect(page.locator(".workbench h1, .workbench h2, .workbench h3, .workbench h4, .workbench h5, .workbench h6"))
    .toHaveCount(0);
});

test("workspace controls use the compact radius scale", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openFixtureWorkbench(page);
  await selectWorkspaceView(page, "Loom");
  await showWorkspaceTools(page, "Loom");
  await page.getByRole("button", { name: /^Map All branches$/ }).click();

  const geometry = await page.evaluate(() => {
    const root = getComputedStyle(document.querySelector(".shell")!);
    const radius = (selector: string) => getComputedStyle(document.querySelector(selector)!).borderRadius;
    return {
      tokens: ["--radius-sm", "--radius-inset", "--radius", "--radius-lg"]
        .map((name) => root.getPropertyValue(name).trim()),
      toolbar: [...document.querySelectorAll<HTMLElement>(".loom-header .action-btn")]
        .map((element) => getComputedStyle(element).borderRadius),
      views: [...document.querySelectorAll<HTMLElement>(".loom-views button")]
        .map((element) => getComputedStyle(element).borderRadius),
      search: radius(".filter-input"),
    };
  });

  expect(geometry.tokens).toEqual(["4px", "4px", "4px", "8px"]);
  expect(new Set(geometry.toolbar)).toEqual(new Set(["4px"]));
  expect(new Set(geometry.views)).toEqual(new Set(["4px"]));
  expect(geometry.search).toBe("4px");
});

test("response and model settings share one controls workspace", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openFixtureWorkbench(page);

  const sidebar = page.getByRole("complementary", { name: "Workspace sidebar" });
  const conversation = sidebar.getByRole("button", { name: "Conversation", exact: true });
  const controls = sidebar.getByRole("button", { name: "Controls", exact: true });
  await expect(page.locator(".app-header .workspace-nav")).toHaveCount(0);
  await expect(conversation).toBeVisible();
  await expect(controls).toBeVisible();
  await expect(sidebar.getByRole("button", { name: "Open model settings" })).toHaveCount(0);

  const [conversationBox, controlsBox] = await Promise.all([
    conversation.boundingBox(),
    controls.boundingBox(),
  ]);
  expect(controlsBox!.y).toBeGreaterThanOrEqual(conversationBox!.y + conversationBox!.height);
  expect(controlsBox!.x).toBe(conversationBox!.x);

  await controls.click();
  const sections = page.getByRole("group", { name: "Controls section" });
  await expect(sections.getByRole("button", { name: "Response" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("slider", { name: "Temperature" })).toBeVisible();
  await sections.getByRole("button", { name: "Model" }).click();
  await expect(page.getByRole("region", { name: "Model controls", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "Model controls", exact: true }).locator(".model-copy h3")).toBeVisible();
  await expect(page.getByText("Current model", { exact: true })).toHaveCount(0);
});

test("the compact workbench header keeps every primary control on one line", async ({ page }) => {
  await page.setViewportSize({ width: 1056, height: 720 });
  await openFixtureWorkbench(page);

  const header = page.locator(".app-header");
  await header.getByRole("button", { name: "Hide left sidebar", exact: true }).click();
  const left = header.getByRole("button", { name: "Show left sidebar", exact: true });
  const menu = header.getByRole("button", { name: "Workspace menu", exact: true });
  const right = header.getByRole("button", { name: "Show right sidebar", exact: true });
  const boxes = await Promise.all(
    [left, menu, right].map((control) => control.boundingBox()),
  );
  const centers = boxes.map((box) => box!.y + box!.height / 2);

  for (const center of centers.slice(1)) {
    expect(Math.abs(center - centers[0])).toBeLessThanOrEqual(1);
  }
  expect(boxes[1]!.x).toBeGreaterThan(boxes[0]!.x + boxes[0]!.width);
  expect(boxes[2]!.x).toBeGreaterThanOrEqual(boxes[1]!.x + boxes[1]!.width);

  const workspaceTop = await page.locator(".workspace-frame").evaluate(
    (frame) => frame.getBoundingClientRect().top,
  );
  expect(workspaceTop).toBeLessThanOrEqual(80);
  await expectViewportContainment(page);
});

test("help explains the workbench before showing technical references", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: "Help and shortcuts", exact: true }).click();

  const help = page.getByRole("dialog", { name: "Help and shortcuts" });
  await expect(help.getByRole("heading", { name: "What each area does" })).toBeVisible();
  for (const area of ["Conversation", "Controls", "Loom"]) {
    await expect(help.getByRole("heading", { name: area, exact: true })).toBeVisible();
  }
  await expect(help.getByRole("heading", { name: "Common terms" })).toBeVisible();
  await expect(help.getByText("What would you like to do?", { exact: true })).toHaveCount(0);
  await expect(help.getByRole("button", { name: "Done", exact: true })).toHaveCount(0);

  const shortcuts = help.getByRole("button", { name: "Keyboard shortcuts" });
  await expect(shortcuts).toHaveAttribute("aria-expanded", "false");
  await expect(help.getByText("Stops generation or closes a panel", { exact: true })).toHaveCount(0);
  await shortcuts.click();
  await expect(shortcuts).toHaveAttribute("aria-expanded", "true");
  await expect(help.getByText("Stops generation or closes a panel", { exact: true })).toBeVisible();

  await expect(help.getByRole("button", { name: "Steering expression reference" }))
    .toHaveAttribute("aria-expanded", "false");
  await expectViewportContainment(page);
});

test("Drowse headers use a plain wordmark without decorative marks or local badges", async ({ page }) => {
  await page.goto(devUrl);
  const landingBrand = page.getByRole("link", { name: "Drowse home" });
  await expect(landingBrand).toHaveText("Drowse");
  await expect(landingBrand.locator("svg, img, [aria-hidden='true'], span")).toHaveCount(0);

  await openFixtureWorkbench(page);
  const workbenchBrand = page.locator(".app-header").getByRole("link", { name: "Drowse home" });
  await expect(workbenchBrand).toHaveText("Drowse");
  await expect(workbenchBrand.locator("svg, img, [aria-hidden='true'], span")).toHaveCount(0);
  await expect(page.getByText("LOCAL", { exact: true })).toHaveCount(0);
  await expect(page.getByText("local only", { exact: true })).toHaveCount(0);
});

test("appearance defaults to dark and persists across every hosted surface", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await page.goto(devUrl);

  const appearance = page.getByRole("group", { name: "Appearance" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(appearance.getByRole("button", { name: "Dark", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await appearance.getByRole("button", { name: "Light", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(appearance.getByRole("button", { name: "Light", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('meta[name="theme-color"]')).toHaveAttribute("content", "#f2f4f8");

  const lightContrast = await page.evaluate(() => {
    const root = getComputedStyle(document.documentElement);
    const luminance = (hex: string) => {
      const channels = [1, 3, 5].map((index) => Number.parseInt(hex.slice(index, index + 2), 16) / 255)
        .map((channel) => channel <= 0.04045
          ? channel / 12.92
          : ((channel + 0.055) / 1.055) ** 2.4);
      return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
    };
    const ratio = (foreground: string, background: string) => {
      const values = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
      return (values[0] + 0.05) / (values[1] + 0.05);
    };
    return {
      primary: ratio(root.getPropertyValue("--fg").trim(), root.getPropertyValue("--bg").trim()),
      muted: ratio(root.getPropertyValue("--fg-muted").trim(), root.getPropertyValue("--bg-alt").trim()),
    };
  });
  expect(lightContrast.primary).toBeGreaterThanOrEqual(7);
  expect(lightContrast.muted).toBeGreaterThanOrEqual(4.5);

  await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

  await appearance.getByRole("button", { name: "Dark", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.locator('meta[name="theme-color"]')).toHaveAttribute("content", "#0b0e17");
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const onboardingAppearance = page.getByRole("group", { name: "Appearance" });
  await expect(onboardingAppearance.getByRole("button", { name: "Dark", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await onboardingAppearance.getByRole("button", { name: "Light", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await openWorkspaceMenu(page);
  await expect(page.getByRole("group", { name: "Appearance" })
    .getByRole("button", { name: "Light", exact: true }))
    .toHaveAttribute("aria-pressed", "true");

  const workbenchAppearance = page.getByRole("group", { name: "Appearance" });
  await workbenchAppearance.getByRole("button", { name: "Dark", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.locator(".shell")).toHaveCSS("background-color", "rgb(11, 14, 23)");
  await workbenchAppearance.getByRole("button", { name: "Light", exact: true }).click();
  await expect(page.locator(".shell")).toHaveCSS("background-color", "rgb(242, 244, 248)");

  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});

test("the conversation gives reading and writing space priority over secondary controls", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openFixtureWorkbench(page);

  const surface = await page.locator(".conversation-page .workspace-surface").boundingBox();
  const transcript = await page.getByRole("region", { name: "Conversation" }).boundingBox();
  const composer = await page.getByRole("textbox", { name: /^Compose as / }).boundingBox();
  expect(surface?.width ?? 0).toBeGreaterThan(1200);
  expect(transcript?.height ?? 0).toBeGreaterThan(320);
  expect(composer?.height ?? 0).toBeGreaterThanOrEqual(80);

  await expect(page.getByRole("button", { name: "Transcript", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Chat view", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Workspace menu", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "All tools", exact: true })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Compare replies" })).toHaveCount(0);
});

for (const { platform, shortcut, key } of [
  { platform: "MacIntel", shortcut: "Cmd+K", key: "Meta+K" },
  { platform: "Win32", shortcut: "Ctrl+K", key: "Control+K" },
]) {
  test(`${shortcut} no longer launches tools on ${platform}`, async ({ page }) => {
    await page.addInitScript((value) => {
      Object.defineProperty(navigator, "platform", {
        configurable: true,
        get: () => value,
      });
    }, platform);
    await openFixtureWorkbench(page);

    await expect(page.locator(".kbd-hint, .tool-key")).toHaveCount(0);

    await page.keyboard.press(key);
    await expect(page.getByRole("dialog", { name: "Command palette" })).toHaveCount(0);
    await clickWorkspaceAction(page, "All tools");
    await expect(page.getByRole("dialog", { name: "Command palette" })).toBeVisible();
  });
}

test("landing and onboarding retain hierarchy and reflow at 320 px", async ({ page }) => {
  await useTouchViewport(page);
  await page.goto(devUrl);
  await page.evaluate(() => {
    document.documentElement.style.scrollbarGutter = "stable";
  });
  await expect(page.getByRole("group", { name: "Appearance" }).locator("button:visible"))
    .toHaveCount(2);
  await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
  await expectCoherentHeadingOutline(page);
  await expectViewportContainment(page);

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  await expect(page.getByRole("group", { name: "Appearance" }).locator("button:visible"))
    .toHaveCount(2);
  await expect(page.getByRole("button", { name: "Download and open", exact: true })).toBeVisible();
  await expectCoherentHeadingOutline(page);
  await expectViewportContainment(page);
});

test("the first visit uses a full-bleed local-compute background without redundant illustration UI", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto(devUrl);
  await expect(page.getByText("Computed on-device", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Open source. Inference and saved work stay on your device.", { exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "See the privacy boundary" })).toHaveCount(0);
  await expect(page.locator(".hero-visual")).toHaveAttribute("aria-hidden", "true");
  await expect(page.locator(".hero-visual svg")).toHaveAttribute("width", "0");

  const initial = await page.locator(".hero-visual").evaluate((element) => {
    return {
      position: getComputedStyle(element).position,
      radius: getComputedStyle(element).borderRadius,
      travel: element.getAttribute("data-travel"),
    };
  });
  expect(initial.position).toBe("fixed");
  expect(initial.radius).toBe("0px");

  await page.evaluate(() => window.scrollTo(0, window.innerHeight * 1.5));
  await expect.poll(() => page.locator(".hero-visual").getAttribute("data-travel"))
    .not.toBe(initial.travel);
});

test("mobile keeps model settings inside the controls workspace", async ({ page }) => {
  await useTouchViewport(page);
  await openFixtureWorkbench(page);
  await expect(page.getByRole("button", { name: "Open model settings" })).toHaveCount(0);
  await selectWorkspaceView(page, "Controls");
  await page.getByRole("group", { name: "Controls section" })
    .getByRole("button", { name: "Model" }).click();
  await expect(page.getByRole("region", { name: "Model controls", exact: true })).toBeVisible();
  await expectViewportContainment(page);
});

test("drawers expose one unambiguous close action and retain backdrop dismissal", async ({ page }) => {
  await openFixtureWorkbench(page);
  const helpButton = page.getByRole("button", { name: "Help and shortcuts", exact: true });

  await helpButton.click();
  const dialog = page.getByRole("dialog", { name: "Help and shortcuts" });
  await expect(dialog).toBeVisible();
  await expect(page.getByRole("button", { name: "Close drawer", exact: true })).toHaveCount(1);
  await page.getByRole("button", { name: "Close drawer", exact: true }).click();
  await expect(dialog).toHaveCount(0);

  await helpButton.click();
  await expect(dialog).toBeVisible();
  await page.locator(".drawer-backdrop").click({ position: { x: 4, y: 4 } });
  await expect(dialog).toHaveCount(0);
});

test("interactive motion stays brief and reduced motion removes it", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await openFixtureWorkbench(page);

  const normal = await page.getByRole("button", { name: "Generate reply" }).evaluate((button) => {
    const style = getComputedStyle(button);
    const root = getComputedStyle(document.documentElement);
    return {
      durations: style.transitionDuration.split(",").map((value) => Number.parseFloat(value) * 1000),
      enterEase: root.getPropertyValue("--ease-enter").trim(),
      exitEase: root.getPropertyValue("--ease-exit").trim(),
      pressScale: root.getPropertyValue("--press-scale").trim(),
    };
  });
  expect(normal.durations.every((duration) => duration > 0 && duration <= 300)).toBe(true);
  expect(normal.enterEase).toContain("cubic-bezier");
  expect(normal.exitEase).toContain("cubic-bezier");
  expect(normal.pressScale).toBe("0.96");

  await page.emulateMedia({ reducedMotion: "reduce" });
  const reducedDurations = await page.getByRole("button", { name: "Generate reply" }).evaluate(
    (button) => getComputedStyle(button).transitionDuration
      .split(",")
      .map((value) => Number.parseFloat(value) * 1000),
  );
  expect(reducedDurations.every((duration) => duration <= 0.02)).toBe(true);
});

test("roles, model settings, and reply settings use direct editing with progressive disclosure", async ({ page }) => {
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();

  const modelRole = page.getByRole("combobox", { name: "Model writes as" });
  await modelRole.fill("reviewer");
  await expect(modelRole).toHaveValue("reviewer");

  await page.getByRole("button", { name: "Role settings" }).click();
  const roleSettings = page.getByRole("dialog", { name: "Role settings" });
  await expect(roleSettings.getByText("reviewer", { exact: true })).toBeVisible();
  await expect(roleSettings.getByRole("button", { name: /^Add$/ })).toHaveCount(0);
  await expect(roleSettings.getByRole("textbox", { name: /^Label$/ })).toHaveCount(0);
  await roleSettings.getByRole("button", { name: "Close drawer" }).click();

  await selectWorkspaceView(page, "Controls");
  await page.getByRole("group", { name: "Controls section" })
    .getByRole("button", { name: "Model" }).click();
  const modelInfo = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelInfo.locator(".model-copy h3")).toBeVisible();
  await expect(modelInfo.getByText("Current model", { exact: true })).toHaveCount(0);
  await expect(modelInfo.getByRole("heading", { name: "Included files and additions" })).toBeVisible();
  await expect(modelInfo.getByRole("heading", { name: "Downloaded models" })).toHaveCount(0);
  await modelInfo.getByRole("button", { name: "Storage and downloads" }).click();
  await expect(modelInfo.getByRole("heading", { name: "Downloaded models" })).toBeVisible();
  await page.getByRole("group", { name: "Controls section" })
    .getByRole("button", { name: "Response" }).click();
  await expect(page.getByRole("slider", { name: "Temperature" })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "Top K", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Sampling settings" }).click();
  const replySettings = page.getByRole("dialog", { name: "Sampling settings" });
  await expect(replySettings.getByRole("spinbutton", { name: "Top K", exact: true })).toBeVisible();
  await expect(replySettings.getByRole("textbox", { name: "Stop sequences" })).toHaveCount(0);
  await replySettings.getByRole("button", { name: "Additional parameters" }).click();
  await expect(replySettings.getByRole("textbox", { name: "Stop sequences" })).toBeVisible();
});

test("sampling and instrument controls expose persistent contextual help", async ({ page }) => {
  await useTouchViewport(page);
  await openFixtureWorkbench(page);
  await selectWorkspaceView(page, "Controls");

  const temperature = page.getByRole("slider", { name: "Temperature" });
  const temperatureDescription = await temperature.getAttribute("aria-describedby");
  expect(temperatureDescription).toBeTruthy();
  await expect(page.locator(`#${temperatureDescription}`)).toContainText(
    "Temperature controls randomness",
  );
  await expect(page.getByRole("button", { name: "About Temperature" })).toBeVisible();

  const tabs = page.getByRole("group", { name: "Response guidance type" });
  await tabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  await expect(lens.getByText("Word", { exact: true })).toHaveCount(2);
  await expect(lens.getByRole("button", {
    name: "Turn off live predicted-word readings",
  })).toBeVisible();
  await expect(lens.getByRole("button", {
    name: "About live predicted-word readings",
  })).toBeVisible();

  await tabs.getByRole("button", { name: "SAE", exact: true }).click();
  const sae = page.getByLabel("Model feature controls");
  await expect(sae.getByText("Feature number", { exact: true })).toHaveCount(2);
  await expect(sae.getByRole("button", {
    name: /Turn (?:on|off) live model-feature readings/,
  })).toBeVisible();
  await expect(sae.getByRole("button", {
    name: "About live model-feature readings",
  })).toBeVisible();
  await expectViewportContainment(page);
});

test("the semantic type roles load with deliberate variable-font weights", async ({ page }) => {
  await page.goto(devUrl);
  await expect(page.locator("#hero-title")).toBeVisible();

  const loaded = await page.evaluate(async () => {
    await document.fonts.ready;
    const [structure, reading, data] = await Promise.all([
      document.fonts.load('780 48px "Wix Madefor Display"', "Drowse"),
      document.fonts.load('400 18px "Wix Madefor Text"', "Privacy stays local"),
      document.fonts.load('400 12px "Martian Mono"', "0123456789"),
    ]);
    return [structure.length > 0, reading.length > 0, data.length > 0];
  });
  expect(loaded).toEqual([true, true, true]);

  const landingType = await page.evaluate(() => {
    const inspect = (selector: string) => {
      const element = document.querySelector<HTMLElement>(selector);
      if (!element) throw new Error(`Missing typography fixture: ${selector}`);
      const style = getComputedStyle(element);
      return { family: style.fontFamily, weight: Number(style.fontWeight) };
    };
    return {
      heading: inspect("#hero-title"),
      reading: inspect(".lede"),
      control: inspect(".primary-action"),
      data: inspect(".model-row > span"),
    };
  });

  expect(landingType.heading.family).toContain("Wix Madefor Display");
  expect(landingType.heading.weight).toBe(780);
  expect(landingType.reading.family).toContain("Wix Madefor Text");
  expect(landingType.reading.weight).toBe(400);
  expect(landingType.control.family).toContain("Wix Madefor Display");
  expect(landingType.control.weight).toBe(600);
  expect(landingType.data.family).toContain("Wix Madefor Text");
  expect(landingType.data.weight).toBeGreaterThanOrEqual(400);

  await page.goto(`${devUrl}/app?fixture=1`);
  await expect(
    page.getByRole("heading", { name: "Choose your first model" }),
  ).toBeVisible();
  const onboardingType = await page.evaluate(() => {
    const heading = getComputedStyle(document.querySelector("h1")!);
    const button = getComputedStyle(document.querySelector(".model-grid button")!);
    const data = getComputedStyle(document.querySelector(".model-topline")!);
    return {
      heading: { family: heading.fontFamily, weight: Number(heading.fontWeight) },
      button: { family: button.fontFamily, weight: Number(button.fontWeight) },
      data: { family: data.fontFamily, weight: Number(data.fontWeight) },
    };
  });
  expect(onboardingType.heading.family).toContain("Wix Madefor Display");
  expect(onboardingType.heading.weight).toBe(780);
  expect(onboardingType.button.family).toContain("Wix Madefor Display");
  expect(onboardingType.button.weight).toBe(500);
  expect(onboardingType.data.family).toContain("Martian Mono");

  await page.getByRole("button", { name: "Download and open", exact: true }).click();
  await expect(page.locator(".shell")).toBeVisible();
  await page.getByRole("button", { name: /^Roles / }).click();
  const workbenchType = await page.evaluate(() => {
    const tab = getComputedStyle(document.querySelector<HTMLElement>(".workspace-nav button")!);
    const prose = getComputedStyle(document.querySelector<HTMLElement>(".chat textarea")!);
    const hud = getComputedStyle(document.querySelector<HTMLElement>(".status-footer")!);
    const role = getComputedStyle(document.querySelector<HTMLElement>('[aria-label="You write as"]')!);
    const roleActions = [...document.querySelectorAll<HTMLElement>(".plan-actions button")]
      .map((button) => getComputedStyle(button));
    return {
      tab: { family: tab.fontFamily, size: Number.parseFloat(tab.fontSize), weight: Number(tab.fontWeight) },
      prose: { family: prose.fontFamily, size: Number.parseFloat(prose.fontSize), weight: Number(prose.fontWeight) },
      hud: { family: hud.fontFamily, size: Number.parseFloat(hud.fontSize), weight: Number(hud.fontWeight) },
      role: { family: role.fontFamily, size: Number.parseFloat(role.fontSize), weight: Number(role.fontWeight) },
      roleActions: roleActions.map((style) => ({
        size: Number.parseFloat(style.fontSize),
        weight: Number(style.fontWeight),
      })),
    };
  });
  expect(workbenchType.tab.family).toContain("Wix Madefor Display");
  expect(workbenchType.tab.weight).toBe(500);
  expect(workbenchType.prose.family).toContain("Wix Madefor Text");
  expect(workbenchType.prose.weight).toBe(400);
  expect(workbenchType.hud.family).toContain("Martian Mono");
  expect(workbenchType.tab.size).toBe(13);
  expect(workbenchType.prose.size).toBe(14);
  expect(workbenchType.hud.size).toBe(13);
  expect(workbenchType.role.family).toContain("Martian Mono");
  expect(workbenchType.role.size).toBe(13);
  expect(workbenchType.role.weight).toBe(400);
  expect(new Set(workbenchType.roleActions.map((action) => action.size))).toEqual(new Set([11, 13]));
  expect(new Set(workbenchType.roleActions.map((action) => action.weight))).toEqual(new Set([500]));

  await page.setViewportSize({ width: 640, height: 760 });
  await expect.poll(() => page.locator('[aria-label="You write as"]').evaluate((element) =>
    Number.parseFloat(getComputedStyle(element).fontSize)
  )).toBe(14);

  await page.evaluate(() => {
    document.documentElement.style.fontSize = "17.5px";
  });
  await expect.poll(() => page.locator('[aria-label="You write as"]').evaluate((element) =>
    Number.parseFloat(getComputedStyle(element).fontSize)
  )).toBe(17.5);
});

test("the primary hosted flow mirrors without clipping in RTL", async ({ page }) => {
  await useTouchViewport(page);
  await page.goto(devUrl);
  await page.evaluate(() => document.documentElement.setAttribute("dir", "rtl"));
  await expectViewportContainment(page);

  await openFixtureWorkbench(page);
  await page.evaluate(() => document.documentElement.setAttribute("dir", "rtl"));
  const navPositions = await page.locator(".workspace-nav button").evaluateAll((buttons) =>
    buttons.map((button) => button.getBoundingClientRect().x)
  );
  expect(navPositions[0]).toBeGreaterThan(navPositions[1]);
  expect(navPositions[1]).toBeGreaterThan(navPositions[2]);

  await expect(page.locator(".kbd-hint .tool-key")).toHaveCount(0);

  await page.getByRole("textbox", { name: /^Compose as / }).fill("Mirror this message");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();
  const rtlPadding = await page.locator(".role-chip").first().evaluate((chip) => {
    const roleStyle = getComputedStyle(chip);
    const thinking = chip.cloneNode(false) as HTMLElement;
    thinking.classList.remove("role-chip");
    thinking.classList.add("thinking-body");
    thinking.dir = "rtl";
    thinking.textContent = "تفكير";
    thinking.style.position = "absolute";
    thinking.style.visibility = "hidden";
    chip.closest(".chat")!.append(thinking);
    const thinkingStyle = getComputedStyle(thinking);
    const result = {
      role: {
        inlineStart: Number.parseFloat(roleStyle.paddingInlineStart),
        inlineEnd: Number.parseFloat(roleStyle.paddingInlineEnd),
        left: Number.parseFloat(roleStyle.paddingLeft),
        right: Number.parseFloat(roleStyle.paddingRight),
      },
      thinking: {
        inlineStart: Number.parseFloat(thinkingStyle.paddingInlineStart),
        inlineEnd: Number.parseFloat(thinkingStyle.paddingInlineEnd),
        left: Number.parseFloat(thinkingStyle.paddingLeft),
        right: Number.parseFloat(thinkingStyle.paddingRight),
      },
    };
    thinking.remove();
    return result;
  });
  expect(rtlPadding.role).toEqual({ inlineStart: 0, inlineEnd: 0, left: 0, right: 0 });
  expect(rtlPadding.thinking.inlineStart).toBeGreaterThan(0);
  expect(rtlPadding.thinking.inlineEnd).toBe(0);
  expect(rtlPadding.thinking.right).toBeGreaterThan(rtlPadding.thinking.left);
  await expectViewportContainment(page);

  await clickWorkspaceAction(page, "All tools");
  const search = page.getByRole("combobox", { name: "Filter commands" });
  await search.fill("model settings");
  await page.keyboard.press("Enter");
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  const controlsBox = await modelControls.boundingBox();
  expect(controlsBox?.x).toBeGreaterThanOrEqual(0);
  await expectViewportContainment(page);
});

test("the 320 px composer preserves a usable writing surface", async ({ page }) => {
  await useTouchViewport(page);
  await openFixtureWorkbench(page);
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  const metrics = await composer.evaluate((element) => {
    const box = element.getBoundingClientRect();
    return {
      width: box.width,
      height: box.height,
      fontSize: Number.parseFloat(getComputedStyle(element).fontSize),
    };
  });
  expect(metrics.width).toBeGreaterThanOrEqual(264);
  expect(metrics.height).toBeGreaterThanOrEqual(44);
  expect(metrics.fontSize).toBeGreaterThanOrEqual(16);
  const actionHeights = await page.locator(".input-actions button").evaluateAll(
    (buttons) => buttons.map((button) => button.getBoundingClientRect().height),
  );
  expect(Math.min(...actionHeights)).toBeGreaterThanOrEqual(44);
  await showWorkspaceTools(page, "chat");
  const [chatHeader, readingTools, headerActions] = await Promise.all([
    page.locator(".chat-header").boundingBox(),
    page.locator(".reading-tools").boundingBox(),
    page.locator(".header-actions").boundingBox(),
  ]);
  expect(chatHeader).not.toBeNull();
  for (const group of [readingTools, headerActions]) {
    expect(group).not.toBeNull();
    expect(group!.x).toBeGreaterThanOrEqual(chatHeader!.x);
    expect(group!.x + group!.width).toBeLessThanOrEqual(chatHeader!.x + chatHeader!.width + 1);
  }
  await expectViewportContainment(page);
});

test("conversation actions stay in one row", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create a conversation");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: "Clear conversation" })).toBeVisible();

  const boxes = await page.locator(".input-actions button").evaluateAll((buttons) =>
    buttons.map((button) => {
      const box = button.getBoundingClientRect();
      return { top: box.top, bottom: box.bottom };
    })
  );
  const composerBox = await page.getByRole("textbox", { name: /^Compose as / }).boundingBox();
  expect(composerBox).not.toBeNull();
  expect(boxes).toHaveLength(3);
  expect(Math.max(...boxes.map(({ top }) => top)) - Math.min(...boxes.map(({ top }) => top)))
    .toBeLessThanOrEqual(1);
  expect(Math.min(...boxes.map(({ top }) => top))).toBeGreaterThanOrEqual(
    composerBox!.y + composerBox!.height,
  );
});

test("desktop role controls stay compact so the conversation remains primary", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();

  const [rolePlan, conversation, roleInput] = await Promise.all([
    page.locator(".turn-plan").boundingBox(),
    page.getByRole("region", { name: "Conversation" }).boundingBox(),
    page.getByRole("combobox", { name: "You write as" }).boundingBox(),
  ]);
  expect(rolePlan).not.toBeNull();
  expect(conversation).not.toBeNull();
  expect(roleInput).not.toBeNull();
  expect(rolePlan!.height).toBeLessThanOrEqual(64);
  expect(roleInput!.height).toBeGreaterThanOrEqual(40);
  expect(conversation!.height).toBeGreaterThan(rolePlan!.height * 4);
  await expectViewportContainment(page);
});

test("narrow role controls stay compact and inside their panel", async ({ page }) => {
  await page.setViewportSize({ width: 504, height: 760 });
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();

  const plan = page.locator(".turn-plan");
  const planBox = await plan.boundingBox();
  expect(planBox).not.toBeNull();
  expect(planBox!.height).toBeLessThanOrEqual(144);

  const children = await plan.locator(":scope > *").evaluateAll((elements) =>
    elements.map((element) => element.getBoundingClientRect().toJSON())
  );
  for (const child of children) {
    expect(child.left).toBeGreaterThanOrEqual(planBox!.x - 1);
    expect(child.right).toBeLessThanOrEqual(planBox!.x + planBox!.width + 1);
    expect(child.top).toBeGreaterThanOrEqual(planBox!.y - 1);
    expect(child.bottom).toBeLessThanOrEqual(planBox!.y + planBox!.height + 1);
  }
  await expect(page.getByRole("combobox", { name: "You write as" })).toHaveValue("user");
  await expect(page.getByRole("combobox", { name: "Model writes as" })).toHaveValue("assistant");
  await expectViewportContainment(page);
});

test("role settings scroll instead of shrinking their panels", async ({ page }) => {
  await page.setViewportSize({ width: 504, height: 400 });
  await openFixtureWorkbench(page);
  await page.getByRole("button", { name: /^Roles / }).click();
  await page.getByRole("button", { name: "Role settings" }).click();

  const drawer = page.getByRole("dialog", { name: "Role settings" });
  const body = drawer.locator(".body");
  const overflow = await body.evaluate((element) => ({
    clientHeight: element.clientHeight,
    scrollHeight: element.scrollHeight,
  }));
  expect(overflow.scrollHeight).toBeGreaterThan(overflow.clientHeight);

  const [form, actions] = await Promise.all([
    drawer.locator(".panel.form").boundingBox(),
    drawer.locator(".form-actions").boundingBox(),
  ]);
  expect(form).not.toBeNull();
  expect(actions).not.toBeNull();
  expect(actions!.y + actions!.height).toBeLessThanOrEqual(form!.y + form!.height + 1);
  await drawer.getByRole("textbox", { name: "Private note" }).scrollIntoViewIfNeeded();
  await expect(drawer.getByRole("textbox", { name: "Private note" })).toBeVisible();
  await expectViewportContainment(page);
});

test("the composer can be resized and role controls can be minimized", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.setViewportSize({ width: 1280, height: 760 });
  await openFixtureWorkbench(page);

  const composer = page.getByRole("textbox", { name: /^Compose as / });
  const resizer = page.getByRole("slider", { name: "Resize writing area" });
  const status = page.locator(".status-footer");
  const initialHeight = await composer.evaluate((element) => element.getBoundingClientRect().height);

  const [initialResizerBox, statusBox] = await Promise.all([
    resizer.boundingBox(),
    status.boundingBox(),
  ]);
  expect(initialResizerBox).not.toBeNull();
  expect(statusBox).not.toBeNull();
  expect(initialResizerBox!.y + initialResizerBox!.height).toBeLessThanOrEqual(statusBox!.y + 1);

  await resizer.focus();
  await page.keyboard.press("ArrowUp");
  await expect.poll(async () =>
    composer.evaluate((element) => element.getBoundingClientRect().height)
  ).toBeGreaterThan(initialHeight);

  await page.keyboard.press("Home");
  await expect.poll(async () =>
    composer.evaluate((element) => element.getBoundingClientRect().height)
  ).toBeLessThan(initialHeight);

  const [handleBox, shortHeight] = await Promise.all([
    resizer.boundingBox(),
    composer.evaluate((element) => element.getBoundingClientRect().height),
  ]);
  expect(handleBox).not.toBeNull();
  await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y + handleBox!.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox!.x + handleBox!.width / 2, handleBox!.y - 48, { steps: 4 });
  await page.mouse.up();
  await expect.poll(async () =>
    composer.evaluate((element) => element.getBoundingClientRect().height)
  ).toBeGreaterThan(shortHeight);

  const collapseRoles = page.getByRole("button", { name: "Collapse roles" });
  await page.getByRole("button", { name: /^Roles / }).click();
  const rolePlanShell = page.locator(".turn-plan-shell");
  await rolePlanShell.evaluate((element) => {
    const originalAnimate = element.animate.bind(element);
    element.animate = (keyframes, options) => {
      const animation = originalAnimate(keyframes, options);
      const effect = animation.effect;
      if (
        effect instanceof KeyframeEffect
        && effect.getKeyframes().some((frame) => "height" in frame)
      ) {
        element.dataset.testHeightAnimationCount = String(
          Number(element.dataset.testHeightAnimationCount ?? "0") + 1,
        );
        element.dataset.testHeightAnimationDuration = String(effect.getTiming().duration);
        element.dataset.testHeightAnimationEasing = effect.getTiming().easing;
      }
      return animation;
    };
  });
  await expect(collapseRoles).toHaveAttribute("aria-expanded", "true");
  await expect(collapseRoles).toHaveAttribute("aria-controls", "turn-role-controls");
  await collapseRoles.click();
  await expect(rolePlanShell).toHaveAttribute("data-test-height-animation-count", "1");
  await expect(rolePlanShell).toHaveAttribute("data-test-height-animation-duration", "220");
  await expect(rolePlanShell).toHaveAttribute(
    "data-test-height-animation-easing",
    "cubic-bezier(0.4, 0, 0.2, 1)",
  );
  await expect(page.getByRole("combobox", { name: "You write as" })).toHaveCount(0);

  const roleSummary = page.getByRole("button", { name: /^Roles / });
  await expect(roleSummary).toHaveAttribute("aria-expanded", "false");
  await expect(roleSummary).toHaveAttribute("aria-controls", "turn-role-controls");
  await expect(roleSummary).toBeFocused();
  await roleSummary.click();
  await expect(page.getByRole("combobox", { name: "You write as" })).toBeVisible();
  await expect(collapseRoles).toBeFocused();
  await expect(rolePlanShell).toHaveAttribute("data-test-height-animation-count", "2");

  await page.emulateMedia({ reducedMotion: "reduce" });
  await collapseRoles.click();
  await expect(roleSummary).toBeVisible();
  await expect(rolePlanShell).toHaveAttribute("data-test-height-animation-count", "2");
  await expectViewportContainment(page);
});

test("desktop controls and drawer shells use the shared comfortable spacing contract", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await openFixtureWorkbench(page);

  const controlTarget = await page.evaluate(() => Number.parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--control-target"),
  ));
  expect(controlTarget).toBeGreaterThanOrEqual(40);

  await page.getByRole("button", { name: /^(Loom|Branches)$/, exact: true }).click();
  const treeRowHeights = await page.locator('[role="treeitem"]').evaluateAll((items) =>
    items.map((item) => item.getBoundingClientRect().height)
  );
  expect(Math.min(...treeRowHeights)).toBeGreaterThanOrEqual(32);

  const workspaceBefore = await page.locator(".workspace-frame").boundingBox();
  await selectWorkspaceView(page, "Controls");
  await page.getByRole("group", { name: "Controls section" })
    .getByRole("button", { name: "Model" }).click();
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(modelControls).toBeVisible();
  const spacing = await modelControls.evaluate((root) => {
    const body = getComputedStyle(root.querySelector<HTMLElement>(".body")!);
    const panel = getComputedStyle(root.querySelector<HTMLElement>(".panel")!);
    return {
      bodyInline: Number.parseFloat(body.paddingInlineStart),
      bodyGap: Number.parseFloat(body.rowGap),
      panelPadding: Number.parseFloat(panel.paddingInlineStart),
    };
  });
  expect(spacing.bodyInline).toBe(12);
  expect(spacing.bodyGap).toBeGreaterThanOrEqual(8);
  expect(spacing.panelPadding).toBeGreaterThanOrEqual(12);

  const workspaceAfter = await page.locator(".workspace-frame").boundingBox();
  expect(Math.abs((workspaceAfter?.x ?? 0) - (workspaceBefore?.x ?? 0))).toBeLessThanOrEqual(1);
  expect(Math.abs((workspaceAfter?.width ?? 0) - (workspaceBefore?.width ?? 0))).toBeLessThanOrEqual(1);
});

test("mobile drawers retain touch targets, readable measure, and viewport containment", async ({ page }) => {
  await useTouchViewport(page);
  await openFixtureWorkbench(page);

  const transcript = await openTranscriptDrawer(page);
  await expect(transcript).toBeVisible();
  const tabHeights = await transcript.getByRole("tab").evaluateAll(
    (tabs) => tabs.map((tab) => tab.getBoundingClientRect().height),
  );
  expect(Math.min(...tabHeights)).toBeGreaterThanOrEqual(44);
  await expectViewportContainment(page);
  await page.keyboard.press("Escape");

  await clickWorkspaceAction(page, "All tools");
  const search = page.getByRole("combobox", { name: "Filter commands" });
  await search.fill("model settings");
  await page.keyboard.press("Enter");
  const localDevice = page.getByRole("region", { name: "Model controls", exact: true });
  await expect(localDevice).toBeVisible();
  const buttonHeights = await localDevice.getByRole("button").evaluateAll(
    (buttons) => buttons.filter((button) => !button.hasAttribute("disabled"))
      .map((button) => button.getBoundingClientRect().height),
  );
  expect(Math.min(...buttonHeights)).toBeGreaterThanOrEqual(44);
  const paragraphMeasures = await localDevice.locator(".panel > p").evaluateAll((paragraphs) =>
    paragraphs.map((paragraph) => ({
      width: paragraph.getBoundingClientRect().width,
      characterWidth: Number.parseFloat(getComputedStyle(paragraph).fontSize) * 0.6,
    }))
  );
  for (const measure of paragraphMeasures) {
    expect(measure.width / measure.characterWidth).toBeLessThanOrEqual(78);
  }
  await expectViewportContainment(page);
});

test("portable pack discovery exposes complete mobile keyboard and touch controls", async ({ page }) => {
  await useTouchViewport(page);
  await openFixtureWorkbench(page);
  await clickWorkspaceAction(page, "All tools");
  const paletteSearch = page.getByRole("combobox", { name: "Filter commands" });
  await paletteSearch.fill("packs");
  await page.getByRole("option", { name: /^Manage downloaded controls\b/i }).click();

  const drawer = page.getByRole("dialog", { name: "Downloaded response controls" });
  await expect(drawer).toBeVisible();
  const installed = drawer.getByRole("tab", { name: "Installed" });
  const huggingFace = drawer.getByRole("tab", { name: "Hugging Face" });
  await installed.focus();
  await page.keyboard.press("ArrowRight");
  await expect(huggingFace).toHaveAttribute("aria-selected", "true");
  await expect(huggingFace).toBeFocused();

  const search = drawer.getByRole("searchbox", {
    name: "Search HF for drowse-manifold repos",
  });
  await expect(search).toBeVisible();
  const metrics = await drawer.locator("button, input").evaluateAll((controls) =>
    controls.filter((control) => {
      const box = control.getBoundingClientRect();
      return getComputedStyle(control).display !== "none" && box.width > 0 && box.height > 0;
    }).map((control) => ({
      name: control.getAttribute("aria-label") ?? control.textContent?.trim() ?? "",
      width: control.getBoundingClientRect().width,
      height: control.getBoundingClientRect().height,
    }))
  );
  expect(metrics.filter((control) => control.height < 44)).toEqual([]);
  await expectViewportContainment(page);
});

test("every hosted command surface remains operable at 320 px", async ({ page }) => {
  await useTouchViewport(page);
  await openFixtureWorkbench(page);

  await clickWorkspaceAction(page, "All tools");
  const palette = page.getByRole("dialog", { name: "Command palette" });
  await expect(palette).toBeVisible();
  const paletteMetrics = await palette.locator("input, button").evaluateAll((controls) =>
    controls.map((control) => ({
      name: control.getAttribute("aria-label") ?? control.textContent?.trim() ?? "",
      height: control.getBoundingClientRect().height,
    })),
  );
  expect(
    paletteMetrics.filter((control) => control.height < 44),
    "command-palette controls shorter than the touch target",
  ).toEqual([]);

  const drawerCommands = await palette.getByRole("option").evaluateAll((options) =>
    options.filter((option) =>
      option.querySelector(".group")?.textContent !== "Controls" &&
      option.querySelector(".label")?.textContent?.trim() !== "Model settings"
    )
      .map((option) => option.querySelector(".label")?.textContent?.trim() ?? ""),
  );
  await page.keyboard.press("Escape");

  const issues: Array<{ command: string; kind: string; detail: unknown }> = [];
  for (const command of drawerCommands) {
    await clickWorkspaceAction(page, "All tools");
    const search = page.getByRole("combobox", { name: "Filter commands" });
    await search.fill(command);
    await page.getByRole("option", { name: new RegExp(`^${command}\\b`, "i") }).click();
    const dialog = page.getByRole("dialog").last();
    await expect(dialog).toBeVisible();
    await dialog.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    }));

    const audit = await dialog.evaluate((root) => {
      const viewportWidth = document.documentElement.clientWidth;
      const interactive = [...root.querySelectorAll<HTMLElement>(
        "button, a[href], input, select, textarea, [role='button'], [role='tab'], [role='radio']",
      )].filter((element) => {
        const box = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return style.display !== "none" && style.visibility !== "hidden" && box.width > 0 && box.height > 0;
      });
      return {
        small: interactive.filter((element) =>
          element.getBoundingClientRect().height < 44 &&
          !(element.matches("a[href]") && element.closest("p") && getComputedStyle(element).display === "inline")
        )
          .map((element) => ({
            tag: element.tagName,
            name: element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 60),
            height: element.getBoundingClientRect().height,
          })),
        escaped: interactive.filter((element) => {
          const box = element.getBoundingClientRect();
          if (box.left >= -1 && box.right <= viewportWidth + 1) return false;
          let ancestor = element.parentElement;
          while (ancestor && ancestor !== root) {
            const style = getComputedStyle(ancestor);
            if (
              (style.overflowX === "auto" || style.overflowX === "scroll") &&
              ancestor.scrollWidth > ancestor.clientWidth
            ) return false;
            ancestor = ancestor.parentElement;
          }
          return true;
        }).map((element) => ({
          tag: element.tagName,
          name: element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 60),
        })),
        unlabeledFields: interactive.filter((element) =>
          element.matches("input, select, textarea") &&
          (element as HTMLInputElement).labels?.length === 0 &&
          !element.getAttribute("aria-label") &&
          !element.getAttribute("aria-labelledby")
        ).map((element) => ({ tag: element.tagName, type: element.getAttribute("type") })),
        unnamedActions: interactive.filter((element) =>
          element.matches("button, a[href], [role='button'], [role='tab'], [role='radio']") &&
          !element.getAttribute("aria-label") &&
          !element.getAttribute("aria-labelledby") &&
          !element.getAttribute("title") &&
          !element.textContent?.trim()
        ).map((element) => ({ tag: element.tagName })),
        undersizedFieldText: interactive.filter((element) =>
          element.matches("input, select, textarea") &&
          Number.parseFloat(getComputedStyle(element).fontSize) < 16
        ).map((element) => ({
          tag: element.tagName,
          fontSize: getComputedStyle(element).fontSize,
        })),
      };
    });
    for (const [kind, detail] of Object.entries(audit)) {
      if (Array.isArray(detail) && detail.length > 0) issues.push({ command, kind, detail });
    }
    await page.keyboard.press("Escape");
  }

  expect(issues).toEqual([]);
  await expectViewportContainment(page);
});

test("context-launched workbench drawers remain operable at 320 px", async ({ page }) => {
  await useTouchViewport(page);
  await openFixtureWorkbench(page);
  await page.getByRole("textbox", { name: /^Compose as / }).fill("Create inspectable context");
  await sendButton(page).click();
  await expect(page.getByRole("button", { name: /^Stop$/i })).toBeDisabled();

  const drawers: Array<{ name: string; params?: unknown }> = [
    { name: "save_conversation" },
    { name: "load_conversation" },
    { name: "system_prompt" },
    { name: "advanced_sampling" },
    { name: "subspace" },
    { name: "manifolds" },
    { name: "transcript" },
    { name: "token_drilldown", params: { turnIdx: 1, tokenIdx: 0 } },
    { name: "probe_inspector", params: {} },
    { name: "node_compare", params: { node_ids: [] } },
  ];
  const issues: Array<{ drawer: string; kind: string; detail: unknown }> = [];

  for (const drawer of drawers) {
    await page.evaluate(async ({ moduleUrl, name, params }) => {
      const store = await import(moduleUrl);
      store.openDrawer(name, params);
    }, { moduleUrl: drawerStoreUrl, ...drawer });
    const dialog = page.getByRole("dialog").last();
    await expect(dialog).toBeVisible();
    await dialog.evaluate(() => new Promise<void>((resolveFrame) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolveFrame()));
    }));

    const audit = await dialog.evaluate((root) => {
      const viewportWidth = document.documentElement.clientWidth;
      const controls = [...root.querySelectorAll<HTMLElement>(
        "button, a[href], input, select, textarea, [role='button'], [role='tab'], [role='radio']",
      )].filter((element) => {
        const box = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return style.display !== "none" && style.visibility !== "hidden" && box.width > 0 && box.height > 0;
      });
      return {
        small: controls.filter((element) => element.getBoundingClientRect().height < 44)
          .map((element) => ({
            tag: element.tagName,
            name: element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 60),
            height: element.getBoundingClientRect().height,
          })),
        escaped: controls.filter((element) => {
          const box = element.getBoundingClientRect();
          if (box.left >= -1 && box.right <= viewportWidth + 1) return false;
          let ancestor = element.parentElement;
          while (ancestor && ancestor !== root) {
            const style = getComputedStyle(ancestor);
            if (
              (style.overflowX === "auto" || style.overflowX === "scroll") &&
              ancestor.scrollWidth > ancestor.clientWidth
            ) return false;
            ancestor = ancestor.parentElement;
          }
          return true;
        }).map((element) => ({
          tag: element.tagName,
          name: element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 60),
        })),
        unlabeledFields: controls.filter((element) =>
          element.matches("input, select, textarea") &&
          (element as HTMLInputElement).labels?.length === 0 &&
          !element.getAttribute("aria-label") &&
          !element.getAttribute("aria-labelledby")
        ).map((element) => ({ tag: element.tagName, type: element.getAttribute("type") })),
        unnamedActions: controls.filter((element) =>
          element.matches("button, a[href], [role='button'], [role='tab'], [role='radio']") &&
          !element.getAttribute("aria-label") &&
          !element.getAttribute("aria-labelledby") &&
          !element.getAttribute("title") &&
          !element.textContent?.trim()
        ).map((element) => ({ tag: element.tagName })),
        undersizedFieldText: controls.filter((element) =>
          element.matches("input, select, textarea") &&
          Number.parseFloat(getComputedStyle(element).fontSize) < 16
        ).map((element) => ({
          tag: element.tagName,
          fontSize: getComputedStyle(element).fontSize,
        })),
      };
    });
    for (const [kind, detail] of Object.entries(audit)) {
      if (Array.isArray(detail) && detail.length > 0) issues.push({ drawer: drawer.name, kind, detail });
    }
    await page.keyboard.press("Escape");
  }

  expect(issues).toEqual([]);
  await expectViewportContainment(page);
});
