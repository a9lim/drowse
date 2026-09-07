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
  retries: process.env.CI ? 2 : 0,
  workers: 1,
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
      testMatch: /(ios-layout|ui-skill-pass|technical-controls|completion-selection|chat-name-stability|loom-search|workspace-consistency|interface-polish|token-sidebar|header-layout|notice-surfaces|model-downloads|message-identity|new-chat-start)\.spec\.ts/,
      use: {
        ...devices["iPhone 13"],
      },
    },
    {
      name: "webkit-macos",
      testMatch: /desktop-browser-compat\.spec\.ts/,
      use: {
        ...devices["Desktop Safari"],
      },
    },
    {
      name: "firefox-macos",
      testMatch: /desktop-browser-compat\.spec\.ts/,
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
