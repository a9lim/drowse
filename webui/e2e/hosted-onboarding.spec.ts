import { expect, test, type Locator, type Page } from "@playwright/test";
import { resolve } from "node:path";
import type {
  HostedModelOption,
  HostedShellSnapshot,
} from "../src/hosted/ui/types";

const devUrl = "http://127.0.0.1:4176";
const hostedAppModuleUrl = `/@fs/${resolve("src/hosted/ui/HostedApp.svelte")}`;

function model(
  fit: HostedModelOption["fit"],
  overrides: Partial<HostedModelOption> = {},
): HostedModelOption {
  return {
    id: `model-${fit}`,
    modelId: `model-${fit}`,
    tier: fit === "recommended" ? "fastest" : fit === "eligible" ? "balanced" : "quality",
    name: `${fit} model`,
    sourceUrl: "https://huggingface.co/a9lim/polythetic",
    license: "Apache-2.0",
    size: "1 GB",
    context: "2K context",
    contextTokens: 2048,
    fit,
    installed: false,
    reason: `${fit} device fit`,
    language: "English",
    modelDownloadBytes: 1_000_000_000,
    firstRunBytes: 1_100_000_000,
    remainingDownloadBytes: 1_100_000_000,
    firstRunPacks: [],
    setupComplete: false,
    ...overrides,
  };
}

const models = [
  model("recommended"),
  model("eligible"),
  model("uncertain"),
  model("blocked"),
];

test("base models stay collapsed and explain text completion before selection", async ({ page }, testInfo) => {
  await mountHostedApp(page);
  await updateSnapshot(page, {
    download: { ...initialSnapshot.download, phase: "idle" },
    models: [...models, model("eligible", {
      id: "base-fixture", modelType: "base", name: "Base model fixture",
      firstRunPacks: ["jlens", "sae"].map((kind) => ({
        id: `base-${kind}`, kind: kind as "jlens" | "sae", name: kind,
        bytes: 100, requiredForSetup: true, selected: true, installed: false,
      })),
    })],
  });
  const disclosure = page.locator("details.base-models");
  const summary = disclosure.locator("summary");
  const base = disclosure.getByRole("button", { name: /Base model fixture/ });
  await expect(disclosure).not.toHaveAttribute("open", "");
  await expect(base).not.toBeVisible();
  await expect(page.getByRole("button", { name: /recommended model/ })).toHaveAttribute("aria-pressed", "true");
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(base).toBeVisible();
  await expect(disclosure).toContainText("They don't follow instructions reliably");
  await expect(disclosure).toContainText("Each download includes the core pack for generation and concept steering.");
  await base.click();
  await expect(base).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".required-tool").filter({ hasText: "Feature insights" })).toContainText("(SAE)");
  for (const width of [320, 390, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await disclosure.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const bounds = (await disclosure.boundingBox())!;
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(width);
    await disclosure.screenshot({ path: testInfo.outputPath(`base-models-${width}.png`) });
  }
  await updateSnapshot(page, { models });
  await expect(disclosure).toContainText("No verified base-model downloads");
  await expect(disclosure.getByRole("button")).toHaveCount(0);
});

test("model deletion requires confirmation and allows a new download", async ({ page }) => {
  await mountHostedApp(page);
  await updateSnapshot(page, { models: [model("recommended", { installed: true, setupComplete: true })] });
  const remove = page.getByRole("button", { name: "Delete from device", exact: true });
  await expect(remove).toBeDisabled();
  await updateSnapshot(page, { download: { ...initialSnapshot.download, phase: "idle" } });
  await remove.click();
  await expect(page.locator(".delete-confirmation")).toContainText("recommended model");
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(remove).toBeVisible();
  await remove.click();
  await expect(page.locator(".model-storage")).toContainText("Your chats and analysis packs are kept");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole("button", { name: "Delete model", exact: true }).click();
  await expect(remove).toHaveCount(0);
  await expect(page.getByRole("status").filter({ hasText: "removed from this device" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Download and open", exact: true })).toBeEnabled();
});

const initialSnapshot: HostedShellSnapshot = {
  phase: "supported",
  headline: "This device is ready",
  detail: "Choose a local model.",
  checks: [],
  models,
  selectedModelVariantId: "model-recommended",
  download: {
    available: true,
    phase: "downloading",
    reason: "Downloading verified files.",
  },
  runtime: {
    available: true,
    phase: "unloaded",
    reason: "Ready to load after setup.",
  },
  storage: { availableBytes: 4_000_000_000, persisted: true },
};

async function mountHostedApp(
  page: Page,
  retryPersistenceResults: boolean[] = [true],
): Promise<void> {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${devUrl}/outside-the-workbench`);
  await page.evaluate(async ({ componentUrl, snapshot, retryResults }) => {
    const [{ default: HostedApp }, { mount }] = await Promise.all([
      import(componentUrl),
      import("/e2e/svelte-runtime.ts"),
    ]);
    let current = structuredClone(snapshot);
    const listeners = new Set<(next: typeof current) => void>();
    const downloads: Array<{ id: string; options: unknown }> = [];
    const controller = {
      current: () => current,
      subscribe(listener: (next: typeof current) => void) {
        listeners.add(listener);
        listener(current);
        return () => listeners.delete(listener);
      },
      async check() {},
      capabilities: () => ({ signals: { appleMobile: false } }),
      async download(id: string, options: unknown) {
        downloads.push({ id, options });
        current = { ...current, download: { ...current.download, phase: "requesting_persistence", modelVariantId: id } };
        for (const listener of listeners) listener(current);
        await new Promise(() => {});
      },
      async retryPersistence() {
        const protectedStorage = retryResults.shift() ?? false;
        if (protectedStorage) {
          current = {
            ...current,
            download: { ...current.download, persistenceDenied: false },
            storage: { ...current.storage, persisted: true },
          };
          for (const listener of listeners) listener(current);
        }
        return protectedStorage;
      },
      setOptionalPackSelected() {},
      async cancelDownload() {},
      async deleteModel(id: string) {
        current = {
          ...current,
          models: current.models.map(model => model.id === id ? { ...model, installed: false, setupComplete: false } : model),
          download: { ...current.download, phase: "idle" },
        };
        for (const listener of listeners) listener(current);
      },
      async open() {},
    };
    document.body.replaceChildren();
    const target = document.createElement("div");
    document.body.append(target);
    mount(HostedApp, { target, props: { controller } });
    (globalThis as typeof globalThis & {
      __hostedOnboardingHarness: {
        downloads: typeof downloads;
        update(patch: Partial<typeof current>): void;
      };
    }).__hostedOnboardingHarness = {
      downloads,
      update(patch) {
        current = {
          ...current,
          ...patch,
          download: patch.download
            ? { ...current.download, ...patch.download }
            : current.download,
          runtime: patch.runtime
            ? { ...current.runtime, ...patch.runtime }
            : current.runtime,
        };
        for (const listener of listeners) listener(current);
      },
    };
  }, {
    componentUrl: hostedAppModuleUrl,
    snapshot: initialSnapshot,
    retryResults: retryPersistenceResults,
  });
  await expect(page.getByRole("heading", { name: "Choose your first model" })).toBeVisible();
}

async function updateSnapshot(
  page: Page,
  patch: Partial<HostedShellSnapshot>,
): Promise<void> {
  await page.evaluate((next) => {
    (globalThis as typeof globalThis & {
      __hostedOnboardingHarness: {
        update(patch: Partial<HostedShellSnapshot>): void;
      };
    }).__hostedOnboardingHarness.update(next);
  }, patch);
}

async function expectSafeActionFirst(
  page: Page,
  actions: Locator,
  labels: [string, string],
): Promise<void> {
  await expect(actions).toHaveCount(2);
  await expect(actions.nth(0)).toHaveText(labels[0]);
  await expect(actions.nth(1)).toHaveText(labels[1]);
  const boxes = await actions.evaluateAll((buttons) =>
    buttons.map((button) => button.getBoundingClientRect().toJSON())
  );
  expect(boxes[0].y).toBeLessThan(boxes[1].y);
  await actions.nth(0).focus();
  await page.keyboard.press(page.context().browser()?.browserType().name() === "webkit" ? "Alt+Tab" : "Tab");
  await expect(actions.nth(1)).toBeFocused();
}

test("tab identity reflects device checks and real setup states", async ({ page }) => {
  await mountHostedApp(page);
  const icon = page.locator('link[rel="icon"]');
  const states: Array<[Partial<HostedShellSnapshot>, string]> = [
    [{ phase: "checking" }, "checking"],
    [{ phase: "unsupported" }, "unsupported"],
    [{ phase: "failed" }, "error"],
    [{ phase: "supported", download: { ...initialSnapshot.download, phase: "idle" } }, "models"],
    [{ download: { ...initialSnapshot.download, phase: "downloading" } }, "downloading"],
    [{ download: { ...initialSnapshot.download, phase: "paused" } }, "paused"],
    [{ download: { ...initialSnapshot.download, phase: "installed" }, runtime: { ...initialSnapshot.runtime, phase: "loading" } }, "loading"],
    [{ runtime: { ...initialSnapshot.runtime, phase: "failed" } }, "error"],
  ];
  for (const [snapshot, state] of states) {
    await updateSnapshot(page, snapshot);
    await expect(icon).toHaveAttribute("data-state", state);
    await expect(page).toHaveTitle(/Drowse$/);
  }
});

test("loading pulse follows actual setup work and stops on paused or failed states", async ({ page }, testInfo) => {
  await mountHostedApp(page);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  const gate = page.locator(".download-gate");
  await expect(gate).toHaveClass(/loading-pulse/);
  await expect.poll(() => gate.evaluate(el => getComputedStyle(el, "::before").animationName)).toBe("loading-breathe");
  for (const theme of ["Light", "Dark"]) {
    await page.evaluate(theme => document.documentElement.dataset.theme = theme.toLowerCase(), theme);
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await gate.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await gate.screenshot({ path: testInfo.outputPath(`loading-setup-${theme}-${width}.png`) });
    }
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect.poll(() => gate.evaluate(el => getComputedStyle(el, "::before").animationName)).toBe("none");
  for (const phase of ["paused", "failed", "idle"] as const) {
    await updateSnapshot(page, { download: { ...initialSnapshot.download, phase } });
    await expect(gate).not.toHaveClass(/loading-pulse/);
  }
  const progress = {
    modelVariantId: "model-recommended", files: [], bytesReceived: 50, bytesTotal: 100,
    throughputBytesPerSecond: 0, etaSeconds: null, calculatingEta: true,
    stalled: false, offline: false, resumable: true,
  };
  for (const state of [{ offline: true }, { stalled: true }]) {
    await updateSnapshot(page, { download: {
      ...initialSnapshot.download, progress: { ...progress, ...state },
    } });
    await expect(gate).not.toHaveClass(/loading-pulse/);
  }
  await updateSnapshot(page, { download: { ...initialSnapshot.download, progress } });
  await expect(gate).toHaveClass(/loading-pulse/);
  await page.emulateMedia({ forcedColors: "active" });
  await expect.poll(() => gate.evaluate(el => getComputedStyle(el, "::before").borderStyle)).toBe("dashed");
  await page.emulateMedia({ forcedColors: "none" });
  await updateSnapshot(page, { phase: "checking" });
  await expect(page.locator(".check-panel")).toHaveClass(/loading-pulse/);
  await updateSnapshot(page, {
    phase: "supported",
    download: { ...initialSnapshot.download, phase: "installed" },
    models: models.map(item => ({ ...item, installed: true, setupComplete: true, remainingDownloadBytes: 0 })),
    runtime: { ...initialSnapshot.runtime, phase: "loading" },
  });
  await expect(page.locator(".check-panel")).not.toHaveClass(/loading-pulse/);
  await expect(page.locator(".runtime-gate")).toHaveClass(/loading-pulse/);
  await updateSnapshot(page, { runtime: { ...initialSnapshot.runtime, phase: "failed" } });
  await expect(page.locator(".runtime-gate")).not.toHaveClass(/loading-pulse/);
});

test("hosted onboarding uses truthful download actions and semantic fit colors", async ({ page }) => {
  await mountHostedApp(page);

  await expect(page.getByRole("button", { name: "Pause download", exact: true })).toBeVisible();
  await updateSnapshot(page, {
    download: { ...initialSnapshot.download, phase: "cancelling" },
  });
  await expect(page.getByRole("button", { name: "Pausing…", exact: true })).toBeVisible();
  await expect(page.locator(".status-region")).toContainText("Pausing local setup download.");

  await updateSnapshot(page, {
    download: { ...initialSnapshot.download, phase: "paused" },
  });
  await expect(page.getByRole("button", { name: "Resume download", exact: true })).toBeVisible();

  await updateSnapshot(page, {
    download: { ...initialSnapshot.download, phase: "failed" },
  });
  await expect(page.getByRole("button", { name: "Retry download", exact: true })).toBeVisible();

  const colors = await page.evaluate(() => {
    const resolveColor = (value: string) => {
      const probe = document.createElement("span");
      probe.style.color = value;
      document.body.append(probe);
      const color = getComputedStyle(probe).color;
      probe.remove();
      return color;
    };
    const fitColor = (fit: string) =>
      getComputedStyle(document.querySelector<HTMLElement>(`.model-fit.${fit}`)!).color;
    const cardStyle = (fit: string) => {
      const label = document.querySelector<HTMLElement>(`.model-fit.${fit}`)!;
      return getComputedStyle(label.closest("button")!);
    };
    return {
      recommended: fitColor("recommended"),
      eligible: fitColor("eligible"),
      uncertain: fitColor("uncertain"),
      blocked: fitColor("blocked"),
      success: resolveColor("var(--live)"),
      warning: resolveColor("var(--accent-yellow)"),
      danger: resolveColor("var(--accent-red)"),
      eligibleBackground: cardStyle("eligible").backgroundColor,
      blockedBackground: cardStyle("blocked").backgroundColor,
      eligibleBorder: cardStyle("eligible").borderTopWidth,
      blockedBorder: cardStyle("blocked").borderTopWidth,
    };
  });
  expect(colors.recommended).toBe(colors.success);
  expect(colors.eligible).toBe(colors.success);
  expect(colors.uncertain).toBe(colors.warning);
  expect(colors.blocked).toBe(colors.danger);
  expect(colors.blockedBackground).not.toBe(colors.eligibleBackground);
  expect(colors.blockedBorder).toBe("0px");
  expect(colors.eligibleBorder).toBe("0px");
  await expect(page.locator(".model-grid > button.blocked")).toBeDisabled();
});

test("hardware compatibility keeps the page canvas flat", async ({ page }) => {
  await mountHostedApp(page);

  const shell = page.locator(".app-shell");
  await expect(shell).toHaveClass(/hardware-compatible/);
  await expect(shell).toHaveCSS("background-image", "none");
  expect(await shell.evaluate(element => getComputedStyle(element, "::before").content)).toBe("none");

  await updateSnapshot(page, {
    phase: "checking",
    headline: "Checking this device",
  });
  await expect(shell).not.toHaveClass(/hardware-compatible/);
  await expect(shell).toHaveCSS("background-image", "none");

  await page.emulateMedia({ reducedMotion: "reduce" });
  await updateSnapshot(page, {
    phase: "supported",
    headline: "This device is ready",
  });
  await expect(shell).toHaveClass(/hardware-compatible/);
  await expect(shell).toHaveCSS("background-image", "none");
  expect(await shell.evaluate((element) =>
    Number.parseFloat(getComputedStyle(element, "::before").transitionDuration)
  )).toBeLessThanOrEqual(0.001);
});

test("verified setup keeps page and footer backgrounds flat while scrolling", async ({ page }, testInfo) => {
  await mountHostedApp(page);
  const shell = page.locator(".app-shell");
  const footer = shell.locator("footer");
  const decoration = () => footer.evaluate(element => getComputedStyle(element, "::before").content);
  await expect.poll(decoration).toBe("none");
  const installed = models.map(item => ({ ...item, installed: true, setupComplete: true, remainingDownloadBytes: 0 }));
  await updateSnapshot(page, {
    models: installed,
    download: { ...initialSnapshot.download, phase: "installed" },
  });
  await expect(shell).toHaveClass(/download-complete/);
  await expect.poll(decoration).toBe("none");
  await expect(page.locator(".runtime-gate")).toHaveCSS("border-top-width", "0px");
  await expect(shell).toHaveCSS("background-image", "none");
  await expect(footer).toHaveCSS("background-image", "none");
  expect(await shell.evaluate(element => getComputedStyle(element, "::before").content)).toBe("none");
  expect(await shell.evaluate(element => getComputedStyle(element, "::after").content)).toBe("none");
  for (const theme of ["light", "dark"]) {
    await page.evaluate(theme => { document.documentElement.dataset.theme = theme; }, theme);
    for (const width of [390, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      await page.evaluate(() => window.scrollTo(0, 0));
      const card = page.locator(".model-grid > button").first();
      await expect(card).toHaveCSS("background-image", /^linear-gradient\((?:180deg, )?rgba/);
      await card.screenshot({ path: testInfo.outputPath(`top-lit-model-${theme}-${width}.png`) });
      await page.evaluate(() => window.scrollTo(0, 0));
      const initialBottom = await footer.evaluate(element => element.getBoundingClientRect().bottom);
      await footer.scrollIntoViewIfNeeded();
      const scrolled = await footer.evaluate(element => ({
        bottom: element.getBoundingClientRect().bottom,
        scroll: window.scrollY,
      }));
      expect(scrolled.scroll).toBeGreaterThan(0);
      expect(Math.abs(initialBottom - scrolled.bottom - scrolled.scroll)).toBeLessThan(1);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`flat-model-page-${theme}-${width}.png`) });
    }
  }
  await updateSnapshot(page, { models: installed.map(item => ({ ...item, setupComplete: false })) });
  await expect.poll(decoration).toBe("none");
  for (const phase of ["downloading", "paused", "failed"] as const) {
    await updateSnapshot(page, { models: installed, download: { ...initialSnapshot.download, phase } });
    await expect(shell).not.toHaveClass(/download-complete/);
  }
  await updateSnapshot(page, { phase: "checking", download: { ...initialSnapshot.download, phase: "installed" } });
  await expect(shell).not.toHaveClass(/download-complete/);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await updateSnapshot(page, { phase: "supported" });
  await expect.poll(decoration).toBe("none");
  expect(await footer.evaluate(element => parseFloat(getComputedStyle(element, "::before").transitionDuration))).toBeLessThanOrEqual(0.001);
});

test("model cards show provider logos and catalog-backed SAE availability", async ({ page }) => {
  await mountHostedApp(page);
  const jlensPack: HostedModelOption["firstRunPacks"][number] = {
    id: "gemma-jlens",
    kind: "jlens",
    name: "Gemma J-lens",
    bytes: 40_000_000,
    requiredForSetup: true,
    selected: true,
    installed: false,
  };
  const saePack: HostedModelOption["firstRunPacks"][number] = {
    id: "gemma-scope-sae",
    kind: "sae",
    name: "Gemma Scope SAE",
    bytes: 100_000_000,
    requiredForSetup: false,
    selected: true,
    installed: false,
  };
  const catalogModels = [
    model("recommended", {
      id: "qwen3-1.7b-q4f16",
      name: "Qwen3 1.7B",
    }),
    model("eligible", {
      id: "gemma3-1b-instruct-q4f16_1",
      name: "Gemma 3 1B",
      firstRunPacks: [jlensPack, saePack],
    }),
  ];
  await updateSnapshot(page, {
    models: catalogModels,
    download: { ...initialSnapshot.download, phase: "idle" },
  });

  const qwen = page.locator(".model-grid > button", { hasText: "Qwen3 1.7B" });
  const gemma = page.locator(".model-grid > button", { hasText: "Gemma 3 1B" });
  await expect(qwen.locator('[data-provider="qwen"]')).toHaveAttribute("aria-hidden", "true");
  await expect(gemma.locator('[data-provider="gemma"]')).toHaveAttribute("aria-hidden", "true");
  await expect(qwen.getByText("SAE", { exact: true })).toHaveCount(0);
  await expect(gemma.getByText("SAE", { exact: true })).toHaveAttribute(
    "aria-description",
    "Sparse autoencoder available",
  );

  await gemma.click();
  await expect(page.getByRole("heading", { name: "Choose this download" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: /Gemma Scope features/ })).toBeChecked();
  await expect(page.locator(".setup-disclosure")).toContainText(
    "One download installs the model, response controls, and word insights, plus Gemma Scope features, then opens the workbench.",
  );
  await expect(page.locator(".setup-disclosure")).toContainText(
    "You can add or change feature and R-lens packs later.",
  );

  await updateSnapshot(page, {
    models: catalogModels.map((candidate) => ({ ...candidate, firstRunPacks: [] })),
  });
  await expect(page.locator(".sae-available")).toHaveCount(0);
});

test("hosted onboarding keeps technical file progress hidden and formats ETA once", async ({ page }) => {
  await mountHostedApp(page);
  await updateSnapshot(page, {
    download: {
      ...initialSnapshot.download,
      progress: {
        modelVariantId: "model-recommended",
        files: [{
          path: "weights/params_shard_17.bin",
          bytesReceived: 500,
          bytesTotal: 1_000,
          resumable: true,
          verification: "hashing",
        }],
        bytesReceived: 500,
        bytesTotal: 1_000,
        throughputBytesPerSecond: 50,
        etaSeconds: [1_080, 1_620],
        calculatingEta: false,
        stalled: false,
        offline: false,
        resumable: true,
      },
    },
  });

  await expect(page.getByRole("progressbar", { name: "Local setup download" }))
    .toHaveAttribute("aria-valuenow", "50");
  await expect(page.locator(".progress-copy .rolling-number .morph-source")).toHaveText("50");
  await expect(page.locator(".progress-copy .rolling-number")).toHaveAttribute("data-value", "50");
  await expect(page.locator(".progress-copy")).toContainText("% · 18 - 27 minutes remaining");
  await expect(page.getByText(/params_shard/i)).toHaveCount(0);
  await expect(page.getByRole("list", { name: "Local setup files" })).toHaveCount(0);
});

test("an unverified iPhone model starts one download with storage protection declined", async ({ page }) => {
  await mountHostedApp(page);
  await updateSnapshot(page, {
    models: [model("uncertain", { reason: "This model has not yet completed a successful load on this iPhone or iPad" })],
    download: { ...initialSnapshot.download, phase: "idle" },
    storage: { ...initialSnapshot.storage, persisted: false },
  });
  await page.getByRole("button", { name: "Review warning", exact: true }).click();
  await expect(page.locator(".unsafe-warning")).toContainText("Check this model on your device");
  await expect(page.locator(".unsafe-warning")).not.toContainText("may not have enough memory");
  await page.getByRole("button", { name: "Download anyway", exact: true }).click();
  await expect(page.locator(".unsafe-warning")).toHaveCount(0);
  await expect(page.locator(".download-gate")).toContainText("Preparing download");
  await expect(page.locator(".model-grid > button")).toBeDisabled();
  await expect(page.getByRole("status", { name: "Storage protection", exact: true })).toContainText("Downloads and chats save on this device");
  await updateSnapshot(page, { download: { ...initialSnapshot.download, phase: "downloading" } });
  await expect(page.locator(".download-gate")).toContainText("Downloading model files");
  expect(await page.evaluate(() => (globalThis as any).__hostedOnboardingHarness.downloads)).toEqual([
    { id: "model-uncertain", options: { explicitUnsafeOverride: true } },
  ]);
});

test("hosted onboarding explains and retries storage protection without repeating the download", async ({ page }) => {
  await mountHostedApp(page, [false, true]);
  await updateSnapshot(page, {
    download: {
      ...initialSnapshot.download,
      phase: "installed",
      persistenceDenied: true,
    },
    models: models.map((candidate) =>
      candidate.fit === "recommended"
        ? { ...candidate, installed: true, setupComplete: true, remainingDownloadBytes: 0 }
        : candidate
    ),
    storage: { ...initialSnapshot.storage, persisted: false },
  });

  const notice = page.getByRole("status", { name: "Storage protection", exact: true });
  await expect(notice.getByText("Local storage is available", { exact: true })).toBeVisible();
  await expect(notice).toContainText("Storage protection is optional and helps prevent automatic cleanup.");
  await page.getByRole("button", { name: "Protect storage" }).click();
  await expect(notice).toContainText("Downloads and chats still save on this device.");
  await expect(notice).toContainText("so back up your chats.");
  expect(await notice.innerText()).not.toContain("\u2014");
  await page.getByRole("button", { name: "Protect storage" }).click();
  await expect(page.getByRole("button", { name: "Protect storage" })).toHaveCount(0);
  await expect(notice).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open Drowse", exact: true })).toBeVisible();
});

test("hosted onboarding presents model load failures as retryable failures", async ({ page }) => {
  await mountHostedApp(page);
  await updateSnapshot(page, {
    download: { ...initialSnapshot.download, phase: "installed" },
    models: models.map((candidate) =>
      candidate.fit === "recommended"
        ? { ...candidate, installed: true, setupComplete: true, remainingDownloadBytes: 0 }
        : candidate
    ),
    runtime: {
      ...initialSnapshot.runtime,
      phase: "failed",
      reason: "The verified model configuration is incomplete.",
    },
  });

  await expect(page.getByText("The model could not open", { exact: true })).toBeVisible();
  await expect(page.locator(".runtime-gate p")).toHaveText(
    "The verified model configuration is incomplete.",
  );
  await expect(page.getByRole("button", { name: "Try opening again", exact: true })).toBeEnabled();
  await expect(page.getByText("Ready to open", { exact: true })).toHaveCount(0);
});

test("mobile warning actions keep safe choices first visually and in focus order", async ({ page }) => {
  await mountHostedApp(page);
  await updateSnapshot(page, {
    download: { ...initialSnapshot.download, phase: "idle" },
  });

  await page.locator(".model-grid > button").filter({ hasText: "uncertain model" }).click();
  await page.getByRole("button", { name: "Review warning", exact: true }).click();
  const downloadWarning = page.locator(".unsafe-warning").filter({
    hasText: "Check this model on your device",
  });
  const downloadActions = downloadWarning.locator(".warning-actions button");
  await expectSafeActionFirst(page, downloadActions, [
    "Choose another model",
    "Download anyway",
  ]);
  await downloadActions.nth(0).click();

  await updateSnapshot(page, {
    models: models.map((candidate) =>
      candidate.fit === "uncertain" ? { ...candidate, installed: true } : candidate
    ),
    runtime: {
      ...initialSnapshot.runtime,
      resetSessionAvailable: true,
    },
  });
  const resetWarning = page.locator(".unsafe-warning").filter({
    hasText: "saved conversation cannot open",
  });
  await resetWarning.getByRole("button", {
    name: "Review start-over option",
    exact: true,
  }).click();
  await expectSafeActionFirst(page, resetWarning.locator(".warning-actions button"), [
    "Keep conversation",
    "Remove conversation and start over",
  ]);
});
