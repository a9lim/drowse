import { clickWorkspaceAction } from "./workbench-navigation";
import { openWorkspaceMenu } from "./workbench-navigation";
import { expect, test, type Locator, type Page } from "@playwright/test";

const devUrl = "http://127.0.0.1:4176";
const desktopViewports = [
  { width: 1024, height: 768 },
  { width: 1280, height: 800 },
  { width: 1440, height: 900 },
] as const;

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
});

function collectPageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

async function expectNoHorizontalOverflow(page: Page): Promise<void> {
  await expect.poll(async () => page.evaluate(() => {
    const root = document.documentElement;
    return Math.max(root.scrollWidth, document.body.scrollWidth) - root.clientWidth <= 1;
  })).toBe(true);
}

async function expectInsideViewport(locator: Locator): Promise<void> {
  await expect.poll(async () => locator.evaluate((element) => {
    const box = element.getBoundingClientRect();
    return (
      box.left >= -1 &&
      box.right <= document.documentElement.clientWidth + 1 &&
      box.top >= -1 &&
      box.bottom <= window.innerHeight + 1
    );
  })).toBe(true);
}

async function expectStorageAndCoordinationCanaries(
  page: Page,
  exerciseOpfs: boolean,
): Promise<void> {
  expect(await page.evaluate(async (shouldExerciseOpfs) => {
    const suffix = crypto.randomUUID();
    const lockName = `drowse-browser-test-${suffix}`;
    const lockWorked = await navigator.locks.request(lockName, (lock) => lock?.name === lockName);

    const workerUrl = URL.createObjectURL(new Blob(["postMessage('ready')"], {
      type: "text/javascript",
    }));
    let workerWorked = false;
    try {
      workerWorked = await new Promise<boolean>((resolve, reject) => {
        const worker = new Worker(workerUrl, { type: "module" });
        const timeout = window.setTimeout(() => {
          worker.terminate();
          reject(new Error("Dedicated worker canary timed out"));
        }, 2_000);
        worker.onmessage = (event) => {
          window.clearTimeout(timeout);
          worker.terminate();
          resolve(event.data === "ready");
        };
        worker.onerror = (event) => {
          window.clearTimeout(timeout);
          worker.terminate();
          reject(new Error(event.message));
        };
      });
    } finally {
      URL.revokeObjectURL(workerUrl);
    }

    let opfs: boolean | null = null;
    if (shouldExerciseOpfs) {
      const directoryName = `.drowse-browser-test-${suffix}`;
      const root = await navigator.storage.getDirectory();
      let created = false;
      try {
        const directory = await root.getDirectoryHandle(directoryName, { create: true });
        created = true;
        const handle = await directory.getFileHandle("canary.txt", { create: true });
        const initial = await handle.createWritable();
        await initial.write("dro");
        await initial.close();
        const resumed = await handle.createWritable({ keepExistingData: true });
        await resumed.seek(3);
        await resumed.write("wse-extra");
        await resumed.truncate(6);
        await resumed.close();
        const contents = await (await handle.getFile()).text();
        const entries = [];
        for await (const [name] of directory.entries()) entries.push(name);
        opfs = contents === "drowse" && entries.includes("canary.txt");
      } finally {
        if (created) await root.removeEntry(directoryName, { recursive: true });
      }
    }

    return { opfs, webLock: lockWorked, worker: workerWorked };
  }, exerciseOpfs)).toEqual({
    opfs: exerciseOpfs ? true : null,
    webLock: true,
    worker: true,
  });
}

test("desktop landing documents supported browsers and supplies runtime prerequisites", async ({
  page,
  browserName,
}, testInfo) => {
  const pageErrors = collectPageErrors(page);
  const response = await page.goto("/");

  expect(response).not.toBeNull();
  expect(response!.headers()["cross-origin-opener-policy"]).toBe("same-origin");
  expect(response!.headers()["cross-origin-embedder-policy"]).toBe("require-corp");
  if (testInfo.project.use.userAgent) {
    expect(await page.evaluate(() => navigator.userAgent)).toBe(testInfo.project.use.userAgent);
  }

  const { opfs, ...prerequisites } = await page.evaluate(() => ({
    secureContext: globalThis.isSecureContext,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    worker: typeof Worker === "function",
    indexedDb: "indexedDB" in globalThis,
    broadcastChannel: typeof BroadcastChannel === "function",
    webAssembly: typeof WebAssembly === "object",
    webLocks: typeof navigator.locks?.request === "function",
    opfs: typeof navigator.storage?.getDirectory === "function",
  }));
  expect(prerequisites).toEqual({
    secureContext: true,
    crossOriginIsolated: true,
    worker: true,
    indexedDb: true,
    broadcastChannel: true,
    webAssembly: true,
    webLocks: true,
  });
  if (browserName !== "webkit") expect(opfs).toBe(true);
  testInfo.annotations.push({ type: "native-opfs-api", description: String(opfs) });
  // Playwright WebKit either omits OPFS on Linux or rejects getDirectory() in
  // ephemeral contexts; the app's live canary still enforces real Safari storage.
  await expectStorageAndCoordinationCanaries(page, browserName !== "webkit");

  for (const viewport of desktopViewports) {
    await page.setViewportSize(viewport);
    await expect(page.locator(".hero-action-row").getByRole("link", { name: "Open Drowse" })).toBeVisible();
    await expectNoHorizontalOverflow(page);
  }

  const browserSupport = page.locator(".hero-action-row > span");
  await expect(browserSupport).toHaveText("Chrome or Safari · compatible device required");
  expect(pageErrors).toEqual([]);
});

test("fixture workbench stays usable across desktop widths and keyboard navigation", async ({
  page,
}) => {
  const pageErrors = collectPageErrors(page);
  await page.goto(`${devUrl}/app?layoutFixture=1`);

  const shell = page.locator(".shell");
  const composer = page.getByRole("textbox", { name: /^Compose as / });
  const sidebar = page.getByRole("complementary", { name: "Workspace sidebar" });
  const send = page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/ });
  await expect(shell).toBeVisible();
  await expect(composer).toBeVisible();

  for (const viewport of desktopViewports) {
    await page.setViewportSize(viewport);
    await composer.focus();
    await expect(send).toBeVisible();
    const conversation = sidebar.getByRole("button", { name: "Conversation", exact: true });
    const controls = sidebar.getByRole("button", { name: "Controls", exact: true });
    const loom = sidebar.getByRole("button", { name: "Loom", exact: true });
    await expect(conversation).toBeVisible();
    await expect(controls).toBeVisible();
    await expect(loom).toBeVisible();
    await expectInsideViewport(conversation);
    await expectInsideViewport(controls);
    await expectInsideViewport(loom);
    await expectInsideViewport(composer);
    await expectInsideViewport(send);
    await expectNoHorizontalOverflow(page);
  }

  await composer.fill("Cross-browser keyboard check");
  await page.keyboard.press("Meta+K");
  const palette = page.getByRole("dialog", { name: "Command palette" });
  await expect(palette).toHaveCount(0);
  await clickWorkspaceAction(page, "All tools");
  const paletteInput = page.getByRole("combobox", { name: "Filter commands" });
  await expect(palette).toBeVisible();
  await expect(paletteInput).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(palette).toHaveCount(0);
  await expect(composer).toHaveValue("Cross-browser keyboard check");

  const controls = sidebar.getByRole("button", { name: "Controls", exact: true });
  await controls.focus();
  await controls.press("Enter");
  await expect(page.locator("#controls-title")).toHaveText("Response, model, and chat controls");
  await expect(controls).toHaveAttribute("aria-current", "page");
  await expect(page.locator('.workspace-page[data-page-id="3"]')).toHaveAttribute(
    "aria-hidden",
    "false",
  );
  expect(pageErrors).toEqual([]);
});
