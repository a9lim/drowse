export function prefersMarkdown(accept = "") {
  const ranges = accept.toLowerCase().split(",").map(part => {
    const [type, ...parameters] = part.trim().split(";");
    const quality = parameters.map(value => value.trim()).find(value => value.startsWith("q="));
    const q = quality ? Number(quality.slice(2)) : 1;
    return { type: type.trim(), q: Number.isFinite(q) && q >= 0 && q <= 1 ? q : 0 };
  });
  const markdown = ranges.find(range => range.type === "text/markdown")?.q ?? 0;
  const html = ["text/html", "text/*", "*/*"].map(type => ranges.find(range => range.type === type)?.q).find(q => q !== undefined) ?? 0;
  return markdown > 0 && markdown >= html;
}

export function createDiscoveryWorker(pages) {
  return {
    async fetch(request, env) {
      const url = new URL(request.url);
      const page = pages[url.pathname];
      if (!page || !["GET", "HEAD"].includes(request.method)) return env.ASSETS.fetch(request);
      const markdown = prefersMarkdown(request.headers.get("Accept") ?? "");
      const assetUrl = new URL(url);
      if (markdown) assetUrl.pathname = page.markdown;
      const assetRequest = new Request(assetUrl, request);
      const response = await env.ASSETS.fetch(assetRequest);
      const headers = new Headers(response.headers);
      const vary = headers.get("Vary");
      if (!vary?.split(",").some(value => ["accept", "*"].includes(value.trim().toLowerCase()))) {
        headers.set("Vary", vary ? `${vary}, Accept` : "Accept");
      }
      if (response.ok) {
        headers.set("Link", page.links);
        if (markdown) headers.set("Content-Type", "text/markdown; charset=utf-8");
      }
      return new Response(request.method === "HEAD" ? null : response.body, {
        status: response.status, statusText: response.statusText, headers,
      });
    },
  };
}

export function discoveryWorkerSource(pages) {
  return `${prefersMarkdown.toString()}\n${createDiscoveryWorker.toString()}\nexport default createDiscoveryWorker(${JSON.stringify(pages)});\n`;
}
