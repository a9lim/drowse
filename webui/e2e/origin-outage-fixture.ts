import { test as base } from "@playwright/test";
import { createServer } from "node:http";

export const test = base.extend<{ outageOrigin: { url: string; disconnect(): Promise<void> } }>({
  outageOrigin: async ({}, use, testInfo) => {
    const upstream = testInfo.project.use.baseURL!;
    const server = createServer(async (request, response) => {
      if (!request.url?.startsWith("/") || request.url.startsWith("//")) {
        response.writeHead(400).end();
        return;
      }
      try {
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
      } catch {
        response.writeHead(502).end();
      }
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("Outage fixture has no TCP address");
    const disconnect = async () => {
      if (!server.listening) return;
      const closed = new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
      server.closeAllConnections();
      await closed;
    };
    try {
      await use({ url: `http://127.0.0.1:${address.port}`, disconnect });
    } finally {
      await disconnect();
    }
  },
});
