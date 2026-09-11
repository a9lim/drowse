import { test as base } from "@playwright/test";
import { createServer } from "node:http";

export const test = base.extend<{ updateOrigin: string }>({
  updateOrigin: async ({}, use, testInfo) => {
    let version = 1;
    const upstream = testInfo.project.use.baseURL!;
    const server = createServer(async (request, response) => {
      if (!request.url?.startsWith("/") || request.url.startsWith("//")) {
        response.writeHead(400).end();
        return;
      }
      if (request.url === "/__drowse_test__/update" && request.method === "POST") {
        version += 1;
        response.writeHead(204).end();
        return;
      }
      if (request.url === "/sw.js") {
        response.writeHead(200, { "Content-Type": "text/javascript", "Cache-Control": "no-store" });
        response.end(`
          const version = ${version};
          self.addEventListener("install", event => {
            if (version === 1) event.waitUntil(self.skipWaiting());
          });
          self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));
          self.addEventListener("message", event => {
            if (event.data?.type === "SKIP_WAITING") event.waitUntil(self.skipWaiting());
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
    if (!address || typeof address === "string") throw new Error("Update fixture has no TCP address");
    try {
      await use(`http://127.0.0.1:${address.port}`);
    } finally {
      await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    }
  },
  baseURL: async ({ updateOrigin }, use) => use(updateOrigin),
});
