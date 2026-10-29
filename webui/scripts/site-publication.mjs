import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { guidePages, homeSummary, htmlGuidePaths, sitePages } from "./site-content.mjs";
import { agentCatalogPaths, agentResources } from "./agent-discovery.mjs";
import { createDiscoveryWorker, prefersMarkdown } from "./discovery-worker.mjs";
import { siteTitle, siteDescription, sourceRepository } from "./site-metadata.mjs";

const escape = text => String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
export const markdownPath = path => path === "/" ? "/index.md" : `${path}.md`;
/** Indexable HTML pages that also have a negotiated Markdown twin. */
export const publicPaths = ["/", ...htmlGuidePaths];
/** Every indexable HTML page, in sitemap order. */
export const sitemapPaths = [...publicPaths, ...sitePages.map(page => page.path)];
/** Internal rewrite targets that must not be reachable as pages of their own. */
export const hiddenPaths = ["/app-shell", "/app-shell.html"];
const themeInitUrl = new URL("../public-hosted/theme-init.js", import.meta.url);

/** The pre-paint theme script, inlined into hosted HTML, and its CSP source expression. */
export async function themeInitScript() {
  const source = (await readFile(themeInitUrl, "utf8")).trim();
  return { source, cspSource: `'sha256-${createHash("sha256").update(source, "utf8").digest("base64")}'` };
}
const documentPaths = new Set(guidePages.map(page => page.path));
const documentLink = href => documentPaths.has(href) ? markdownPath(href) : href;

export function discoveryLinks(origin, path) {
  const catalogLinks = origin ? `, <${origin}${agentCatalogPaths[0]}>; rel="ard"; type="application/json", <${origin}${agentCatalogPaths[1]}>; rel="ai-catalog"; type="application/json"` : "";
  return `<${origin}/llms.txt>; rel="describedby", <${origin}/developers.md>; rel="service-doc", <${origin}/sitemap.xml>; rel="sitemap", <${origin}${markdownPath(path)}>; rel="alternate"; type="text/markdown", <${origin}${path}>; rel="canonical"${catalogLinks}`;
}

export function discoveryTags(path = "/", origin = "") {
  const catalogTags = origin ? `\n<link rel="ard" href="${agentCatalogPaths[0]}" type="application/json" />\n<link rel="ai-catalog" href="${agentCatalogPaths[1]}" type="application/json" />` : "";
  const alternate = publicPaths.includes(path) ? `\n<link rel="alternate" href="${markdownPath(path)}" type="text/markdown" />` : "";
  return `<link rel="describedby" href="/llms.txt" type="text/plain" />${alternate}\n<link rel="service-doc" href="/developers.md" />${catalogTags}`;
}

export function guideMarkdown(page, origin) {
  const link = item => `- [${item.label}](${item.href.startsWith("/") ? origin : ""}${documentLink(item.href)})${item.description ? `: ${item.description}` : ""}`;
  return `# ${page.heading}\n\n${page.intro}\n\nResource: ${origin}${markdownPath(page.path)}\n\n${page.sections.map(section => `## ${section.heading}\n\n${(section.paragraphs ?? []).join("\n\n")}${section.bullets ? `\n\n${section.bullets.map(text => `- ${text}`).join("\n")}` : ""}${section.code ? `\n\n\`\`\`\n${section.code}\n\`\`\`` : ""}${section.links ? `\n\n${section.links.map(link).join("\n")}` : ""}`).join("\n\n")}\n\n[Documentation index](${origin}/llms.txt) · [Open Drowse](${origin}/app) · [Source code](${sourceRepository})\n`;
}

export function publicationAssets(origin, { lastmod } = {}) {
  const assets = agentResources(origin);
  for (const page of guidePages) {
    assets.set(markdownPath(page.path).slice(1), guideMarkdown(page, origin));
  }
  const overview = `# ${siteTitle}\n\n${siteDescription}\n\n${homeSummary.intro}\n\n${homeSummary.links.map(link => `- [${link.label}](${origin}${documentLink(link.href)}): ${link.description}`).join("\n")}\n\n[Open Drowse](${origin}/app) · [About and source](${origin}/about.md) · [Privacy and data storage](${origin}/privacy.md)\n`;
  assets.set("index.md", overview);
  assets.set("llms.txt", `# Drowse\n\n> ${homeSummary.intro}\n\nThe browser workbench runs supported models locally with WebGPU. Available instruments depend on the selected model and compatible packs. Python also provides a library, CLI, local HTTP APIs, SAE training and lens fitting.\n\n## Documentation\n\n- [Overview](${origin}/index.md): Product summary and entry points.\n${guidePages.map(page => `- [${page.heading}](${origin}${markdownPath(page.path)}): ${page.description}`).join("\n")}\n\n## Agent operation\n\n- [Drowse browser skill](${origin}/skills/drowse/SKILL.md): Operate the existing WebMCP tools in a supported browser.\n- [Source code](${sourceRepository}): Implementation, installation, issues and license.\n\n## Optional\n\n- [Combined documentation](${origin}/llms-full.txt): All public guides in one file.\n`);
  assets.set("llms-full.txt", `${overview}\n${guidePages.map(page => guideMarkdown(page, origin)).join("\n---\n\n")}`);
  if (origin) assets.set("llms.txt", `${assets.get("llms.txt")}\n- [Agent resource catalog](${origin}${agentCatalogPaths[0]}): Machine-readable discovery for the browser-agent skill; not a remote inference endpoint.\n`);
  if (origin) assets.set("sitemap.xml", `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${sitemapPaths.map(path => `  <url><loc>${escape(origin + path)}</loc>${lastmod ? `<lastmod>${escape(lastmod)}</lastmod>` : ""}</url>`).join("\n")}\n</urlset>\n`);
  return assets;
}

function contentType(path) {
  if (path.endsWith(".md")) return "text/markdown; charset=utf-8";
  if (path.endsWith(".json")) return "application/json; charset=utf-8";
  if (path.endsWith(".xml")) return "application/xml; charset=utf-8";
  if (path.endsWith(".txt")) return "text/plain; charset=utf-8";
  return "text/html; charset=utf-8";
}

function createSiteWorker(pages, hidden) {
  const discovery = createDiscoveryWorker(pages);
  return {
    async fetch(request, env) {
      const url = new URL(request.url);
      if (!hidden.includes(url.pathname)) return discovery.fetch(request, env);
      // Serve the ordinary not-found page with a real 404 instead of the internal app shell.
      const missing = new URL("/__drowse-not-found__", url);
      const response = await env.ASSETS.fetch(new Request(missing, { method: request.method === "HEAD" ? "HEAD" : "GET", headers: request.headers }));
      return new Response(request.method === "HEAD" ? null : response.body, { status: 404, statusText: "Not Found", headers: response.headers });
    },
  };
}

export function siteWorkerSource(pages) {
  return `${prefersMarkdown.toString()}\n${createDiscoveryWorker.toString()}\n${createSiteWorker.toString()}\nexport default createSiteWorker(${JSON.stringify(pages)}, ${JSON.stringify(hiddenPaths)});\n`;
}

export function sitePublication(origin, { lastmod } = {}) {
  const assets = publicationAssets(origin, { lastmod });
  const pages = Object.fromEntries(publicPaths.flatMap(path => {
    const value = { markdown: markdownPath(path), links: discoveryLinks(origin, path) };
    // Trailing-slash variants are 301 redirects in _redirects, so only the canonical path negotiates.
    return [[path, value]];
  }));
  const middleware = server => { server.middlewares.use((request, response, next) => {
    const path = new URL(request.url, "http://localhost").pathname;
    if (!["GET", "HEAD"].includes(request.method)) return next();
    const page = pages[path];
    const wantsMarkdown = page && prefersMarkdown(request.headers.accept);
    const key = wantsMarkdown ? page.markdown.slice(1) : path.replace(/^\//, "").replace(/\/$/, "");
    const assetPath = key;
    if (page) { response.setHeader("Link", page.links); response.setHeader("Vary", "Accept"); }
    if (!assets.has(assetPath)) return next();
    response.setHeader("Content-Type", contentType(assetPath));
    response.setHeader("Access-Control-Allow-Origin", "*");
    response.setHeader("X-Robots-Tag", "noindex, nofollow");
    response.end(request.method === "HEAD" ? undefined : assets.get(assetPath));
  }); };
  return {
    name: "drowse-site-publication",
    configureServer: middleware,
    configurePreviewServer: middleware,
    async generateBundle() {
      for (const [fileName, source] of assets) this.emitFile({ type: "asset", fileName, source });
      this.emitFile({ type: "asset", fileName: "_worker.js", source: siteWorkerSource(pages) });
      this.emitFile({ type: "asset", fileName: "_routes.json", source: JSON.stringify({ version: 1, include: [...Object.keys(pages), ...hiddenPaths], exclude: [] }) });
      let headers = await readFile(new URL("../public-hosted/_headers", import.meta.url), "utf8");
      const { cspSource } = await themeInitScript();
      if (!/script-src 'self'/.test(headers)) throw new Error("hosted CSP is missing script-src 'self'");
      headers = headers.replace(/script-src 'self'/, `script-src 'self' ${cspSource}`);
      for (const path of publicPaths) {
        headers += `\n${path}\n  Link: ${discoveryLinks(origin, path)}\n  Vary: Accept\n`;
      }
      for (const fileName of assets.keys()) {
        if (fileName.endsWith(".html") || fileName.endsWith(".xml")) continue;
        const path = `/${fileName}`;
        const canonicalPath = path === "/index.md" ? "/" : path.replace(/\.md$/, "");
        headers += `\n${path}\n  Content-Type: ${contentType(path)}\n  Access-Control-Allow-Origin: *\n`;
        if (fileName !== "index.md") headers += "  X-Robots-Tag: noindex, follow\n";
        if (publicPaths.includes(canonicalPath)) headers += `  Link: <${origin}${canonicalPath}>; rel="canonical"\n`;
      }
      this.emitFile({ type: "asset", fileName: "_headers", source: headers });
    },
  };
}
