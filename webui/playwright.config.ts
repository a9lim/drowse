import { defineConfig, devices } from "@playwright/test";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const host = "127.0.0.1";
const devPort = 4176;
const previewPort = 4177;
const browserChannel = process.env.PLAYWRIGHT_CHANNEL;
const runCacheRoot = resolve(tmpdir(), `drowse-playwright-vite-${process.pid}`);
const desktopFirefox = {
  ...devices["Desktop Firefox"],
  userAgent: undefined,
};

export default defineConfig({
  testDir: "./e2e",
  outputDir: resolve(tmpdir(), "drowse-playwright-results"),
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  timeout: 60_000,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : 1,
  reporter: process.env.CI ? [["github"], ["line"]] : "line",
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://${host}:${previewPort}`,
    serviceWorkers: "allow",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        ...(browserChannel ? { channel: browserChannel } : {}),
      },
    },
    {
      name: "webkit-ios-layout",
      testMatch: /(base-editor-layout|tooltip-policy|torph-motion|ios-layout|hosted-onboarding|ui-skill-pass|sae-descriptions|technical-controls|completion-selection|chat-name-stability|loom-search|workspace-consistency|interface-polish|token-sidebar|mobile-sheet|header-layout|notice-surfaces|model-downloads|message-identity|new-chat-start|storage-persistence|theme-default|public-header|saved-chat-menu|contact)\.spec\.ts/,
      use: {
        ...devices["iPhone 13"],
      },
    },
    {
      name: "webkit-macos",
      testMatch: /(base-editor-layout|tooltip-policy|torph-motion|desktop-browser-compat|storage-persistence|header-layout|theme-default|public-header|saved-chat-menu|contact|mobile-sheet)\.spec\.ts/,
      use: {
        ...devices["Desktop Safari"],
      },
    },
    {
      name: "firefox-macos",
      testMatch: /(base-editor-layout|tooltip-policy|torph-motion|desktop-browser-compat|storage-persistence|header-layout|theme-default|public-header|saved-chat-menu|contact|mobile-sheet)\.spec\.ts/,
      use: {
        ...desktopFirefox,
      },
    },
  ],
  webServer: [
    {
      command: `npm run dev:hosted -- --force --host ${host} --port ${devPort} --strictPort`,
      url: `http://${host}:${devPort}`,
      env: { DROWSE_VITE_CACHE_DIR: resolve(runCacheRoot, "dev") },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `npm run build:hosted && npm run preview:hosted -- --host ${host} --port ${previewPort} --strictPort`,
      url: `http://${host}:${previewPort}`,
      env: { DROWSE_VITE_CACHE_DIR: resolve(runCacheRoot, "preview") },
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
