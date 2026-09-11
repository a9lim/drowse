import { expect, test as base, type Page } from "@playwright/test";
import { createServer } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import AxeBuilder from "@axe-core/playwright";

const test = base.extend<{ recoveryOrigin: string }>({
  context: async ({ playwright, browserName, launchOptions, contextOptions, baseURL, viewport, userAgent, deviceScaleFactor, isMobile, hasTouch, serviceWorkers }, use) => {
    // WebKit's ephemeral contexts discard CacheStorage entries on navigation.
    const profile = await mkdtemp(join(tmpdir(), "drowse-recovery-profile-"));
    const context = await playwright[browserName].launchPersistentContext(profile, {
      ...launchOptions, ...contextOptions, baseURL, viewport, userAgent,
      deviceScaleFactor, isMobile, hasTouch, serviceWorkers,
    });
    try {
      await use(context);
    } finally {
      await context.close();
      await rm(profile, { recursive: true });
    }
  },
  recoveryOrigin: async ({}, use, testInfo) => {
    const upstream = testInfo.project.use.baseURL!;
    const server = createServer(async (request, response) => {
      if (!request.url?.startsWith("/") || request.url.startsWith("//")) {
        response.writeHead(400).end();
        return;
      }
      if (request.url === "/sw.js") {
        response.writeHead(200, { "Content-Type": "text/javascript", "Cache-Control": "no-store" });
        response.end(`
          const cacheName = "drowse-hosted-precache-v2-" + self.location.origin + "/";
          self.addEventListener("install", event => event.waitUntil((async () => {
            const cache = await caches.open(cacheName);
            await cache.add("/index.html");
            await self.skipWaiting();
          })()));
          self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));
          self.addEventListener("fetch", event => {
            const url = new URL(event.request.url);
            if (url.origin !== self.location.origin) return;
            if (event.request.mode === "navigate") {
              event.respondWith(caches.open(cacheName).then(cache => cache.match("/index.html")));
            } else if (/\\/assets\\/LandingRoot-[^/]+\\.js$/.test(url.pathname)) {
              event.respondWith((async () => {
                const cache = await caches.open(cacheName);
                if (await cache.match("/__broken_shell__")) return Response.error();
                return fetch(event.request);
              })());
            }
          });
        `);
        return;
      }
      const target = new URL(upstream);
      const query = request.url.indexOf("?");
      target.pathname = query < 0 ? request.url : request.url.slice(0, query);
      target.search = query < 0 ? "" : request.url.slice(query);
      const result = await fetch(target, { redirect: "error" });
      const headers = Object.fromEntries(result.headers);
      delete headers["content-encoding"];
      delete headers["content-length"];
      response.writeHead(result.status, headers);
      response.end(Buffer.from(await result.arrayBuffer()));
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("Recovery fixture has no TCP address");
    try {
      await use(`http://127.0.0.1:${address.port}`);
    } finally {
      await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    }
  },
  baseURL: async ({ recoveryOrigin }, use) => use(recoveryOrigin),
});

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status === testInfo.expectedStatus) return;
  await testInfo.attach("recovery-state", { contentType: "application/json", body: JSON.stringify(await page.evaluate(async () => ({
    url: location.href,
    controller: navigator.serviceWorker.controller?.scriptURL,
    registration: (await navigator.serviceWorker.getRegistration("/"))?.active?.scriptURL,
    caches: await Promise.all((await caches.keys()).map(async name => ({ name, requests: (await (await caches.open(name)).keys()).map(request => request.url) }))),
    resources: performance.getEntriesByType("resource").map(entry => ({ name: entry.name, duration: entry.duration })),
  })), null, 2) });
});

async function breakCachedShell(page: Page, suppressAutomaticRecovery = false) {
  await page.goto("/");
  await expect(page.locator("#hero-title")).toBeVisible();
  await page.evaluate(async suppress => {
    await navigator.serviceWorker.ready;
    const cache = await caches.open(`drowse-hosted-precache-v2-${location.origin}/`);
    await cache.put("/__broken_shell__", new Response("stale app"));
    if (suppress) sessionStorage.setItem("drowse:chunk-recovery", JSON.stringify({ stage: "bootstrap", pathname: "/", createdAt: Date.now() }));
  }, suppressAutomaticRecovery);
  await expect.poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);
}

test("automatic recovery escapes a stale service worker instead of reloading its cached shell", async ({ page }) => {
  await breakCachedShell(page);
  let navigations = 0;
  page.on("request", request => { if (request.isNavigationRequest() && request.frame() === page.mainFrame()) navigations += 1; });
  await page.reload();
  await expect(page.locator("#hero-title")).toBeVisible();
  await expect(page.locator(".bootstrap-error")).toHaveCount(0);
  expect(navigations).toBe(2);
  await expect.poll(() => page.evaluate(() => location.search)).toBe("");
});

test("the recovery button repairs a repeat-on-reload failure without removing saved data", async ({ page }) => {
  await page.addInitScript(() => {
    const remove = CacheStorage.prototype.delete;
    CacheStorage.prototype.delete = function (name) {
      localStorage.setItem("deleted-test-caches", JSON.stringify([...JSON.parse(localStorage.getItem("deleted-test-caches") ?? "[]"), name]));
      return remove.call(this, name);
    };
  });
  await breakCachedShell(page, true);
  await page.evaluate(async () => {
    localStorage.setItem("drowse.saved-data-test", "keep chats");
    for (const name of ["webllm/model", "webllm/wasm", "unrelated-cache", "drowse-hosted-on-demand-wasm-v1"]) {
      await (await caches.open(name)).put("/keep-model", new Response("keep model"));
    }
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("drowse-saved-conversations", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("conversations");
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction("conversations", "readwrite");
        tx.objectStore("conversations").put("keep conversation", "sentinel");
        tx.oncomplete = () => { db.close(); resolve(); };
        tx.onerror = () => reject(tx.error);
      };
    });
  });
  const cachedModels = () => page.evaluate(async () => Promise.all(["webllm/model", "webllm/wasm", "unrelated-cache", "drowse-hosted-on-demand-wasm-v1"].map(async name =>
    (await (await caches.open(name)).match("/keep-model"))?.text())));
  expect(await cachedModels()).toEqual(Array(4).fill("keep model"));
  await page.reload();
  await expect(page.getByRole("heading", { name: "Drowse could not open" })).toBeVisible();
  expect(await cachedModels()).toEqual(Array(4).fill("keep model"));
  await page.reload();
  await expect(page.getByRole("heading", { name: "Drowse could not open" })).toBeVisible();
  expect(await cachedModels()).toEqual(Array(4).fill("keep model"));
  await page.getByRole("button", { name: "Reload Drowse", exact: true }).click();
  await expect(page.locator("#hero-title")).toBeVisible();
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("deleted-test-caches") ?? "[]"))).toEqual([expect.stringMatching(/^drowse-hosted-precache-v2-/)]);
  expect(await page.evaluate(async () => ({
    local: localStorage.getItem("drowse.saved-data-test"),
    cached: await Promise.all(["webllm/model", "webllm/wasm", "unrelated-cache", "drowse-hosted-on-demand-wasm-v1"].map(async name =>
      (await (await caches.open(name)).match("/keep-model"))?.text())),
    chat: await new Promise(resolve => {
      const request = indexedDB.open("drowse-saved-conversations", 1);
      request.onsuccess = () => {
        const db = request.result;
        const read = db.transaction("conversations").objectStore("conversations").get("sentinel");
        read.onsuccess = () => { db.close(); resolve(read.result); };
      };
    }),
  }))).toEqual({ local: "keep chats", cached: Array(4).fill("keep model"), chat: "keep conversation" });
});

test("offline recovery keeps the cached app and gives an actionable retry", async ({ context, page }) => {
  await breakCachedShell(page, true);
  await page.reload();
  await expect(page.locator(".bootstrap-error")).toBeVisible();
  await context.setOffline(true);
  await page.getByRole("button", { name: "Reload Drowse", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("You’re offline");
  await expect(page.getByRole("button", { name: "Try reloading again" })).toBeEnabled();
  expect(await page.evaluate(async () => Boolean(await navigator.serviceWorker.getRegistration("/")))).toBe(true);
  expect(await page.evaluate(async () => Boolean(await (await caches.open(`drowse-hosted-precache-v2-${location.origin}/`)).match("/__broken_shell__")))).toBe(true);
  await context.setOffline(false);
  await page.getByRole("button", { name: "Try reloading again" }).click();
  await expect(page.locator("#hero-title")).toBeVisible();
});

test("reload guidance matches the OS and the error actions have room at narrow widths", async ({ page }, testInfo) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "platform", { value: "MacIntel" });
    Object.defineProperty(navigator, "maxTouchPoints", { value: 0 });
    Object.defineProperty(navigator, "userAgent", { value: "Macintosh" });
  });
  await breakCachedShell(page, true);
  await page.reload();
  const error = page.locator(".bootstrap-error");
  await expect(error).toContainText("⌘ + R");
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 900 });
    const layout = await error.evaluate(el => {
      const button = el.querySelector("button")!.getBoundingClientRect();
      const detail = el.querySelector("p")!.getBoundingClientRect();
      const safety = el.querySelector(".bootstrap-error-safety")!.getBoundingClientRect();
      return { gapBefore: button.top - detail.bottom, gapAfter: safety.top - button.bottom, width: el.scrollWidth, viewport: innerWidth, buttonHeight: button.height, inset: button.left };
    });
    expect(layout.gapBefore).toBeGreaterThanOrEqual(24);
    expect(layout.gapAfter).toBeGreaterThanOrEqual(12);
    expect(layout.buttonHeight).toBeGreaterThanOrEqual(48);
    expect(layout.inset).toBeGreaterThanOrEqual(16);
    expect(layout.width).toBeLessThanOrEqual(layout.viewport);
    await page.screenshot({ path: testInfo.outputPath(`recovery-${width}.png`) });
  }
  expect((await new AxeBuilder({ page }).include(".bootstrap-error").analyze()).violations).toEqual([]);
});
