import { readFile } from "node:fs/promises";
import { guidePages, homeSummary } from "./site-content.mjs";
import { agentResources } from "./agent-discovery.mjs";
import { discoveryWorkerSource, prefersMarkdown } from "./discovery-worker.mjs";
import { siteTitle, siteDescription, sourceRepository } from "./site-metadata.mjs";

const escape = text => String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
export const markdownPath = path => path === "/" ? "/index.md" : `${path}.md`;
export const publicPaths = ["/"];
const documentPaths = new Set(guidePages.map(page => page.path));
const documentLink = href => documentPaths.has(href) ? markdownPath(href) : href;

export function discoveryLinks(origin, path) {
  return `<${origin}/llms.txt>; rel="describedby", <${origin}/developers.md>; rel="service-doc", <${origin}/sitemap.xml>; rel="sitemap", <${origin}${markdownPath(path)}>; rel="alternate"; type="text/markdown", <${origin}${path}>; rel="canonical"`;
}

export function discoveryTags(path = "/") {
  return `<link rel="describedby" href="/llms.txt" type="text/plain" />\n<link rel="alternate" href="${markdownPath(path)}" type="text/markdown" />\n<link rel="service-doc" href="/developers.md" />`;
}

export function guideMarkdown(page, origin) {
  const link = item => `- [${item.label}](${item.href.startsWith("/") ? origin : ""}${documentLink(item.href)})${item.description ? `: ${item.description}` : ""}`;
  return `# ${page.heading}\n\n${page.intro}\n\nResource: ${origin}${markdownPath(page.path)}\n\n${page.sections.map(section => `## ${section.heading}\n\n${(section.paragraphs ?? []).join("\n\n")}${section.bullets ? `\n\n${section.bullets.map(text => `- ${text}`).join("\n")}` : ""}${section.code ? `\n\n\`\`\`\n${section.code}\n\`\`\`` : ""}${section.links ? `\n\n${section.links.map(link).join("\n")}` : ""}`).join("\n\n")}\n\n[Documentation index](${origin}/llms.txt) · [Open Drowse](${origin}/app) · [Source code](${sourceRepository})\n`;
}

export function publicationAssets(origin) {
  const assets = agentResources(origin);
  for (const page of guidePages) {
    assets.set(markdownPath(page.path).slice(1), guideMarkdown(page, origin));
  }
  const overview = `# ${siteTitle}\n\n${siteDescription}\n\n${homeSummary.intro}\n\n${homeSummary.links.map(link => `- [${link.label}](${origin}${documentLink(link.href)}): ${link.description}`).join("\n")}\n\n[Open Drowse](${origin}/app) · [About and source](${origin}/about.md) · [Privacy and data storage](${origin}/privacy.md)\n`;
  assets.set("index.md", overview);
  assets.set("llms.txt", `# Drowse\n\n> ${homeSummary.intro}\n\nThe browser workbench runs supported models locally with WebGPU. Available instruments depend on the selected model and compatible packs. Python also provides a library, CLI, local HTTP APIs, SAE training and lens fitting.\n\n## Documentation\n\n- [Overview](${origin}/index.md): Product summary and entry points.\n${guidePages.map(page => `- [${page.heading}](${origin}${markdownPath(page.path)}): ${page.description}`).join("\n")}\n\n## Agent operation\n\n- [Drowse browser skill](${origin}/skills/drowse/SKILL.md): Operate the existing WebMCP tools in a supported browser.\n- [Source code](${sourceRepository}): Implementation, installation, issues and license.\n\n## Optional\n\n- [Combined documentation](${origin}/llms-full.txt): All public guides in one file.\n`);
  assets.set("llms-full.txt", `${overview}\n${guidePages.map(page => guideMarkdown(page, origin)).join("\n---\n\n")}`);
  if (origin) assets.set("sitemap.xml", `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${publicPaths.map(path => `  <url><loc>${escape(origin + path)}</loc></url>`).join("\n")}\n</urlset>\n`);
  return assets;
}

function contentType(path) {
  if (path.endsWith(".md")) return "text/markdown; charset=utf-8";
  if (path.endsWith(".json")) return "application/json; charset=utf-8";
  if (path.endsWith(".xml")) return "application/xml; charset=utf-8";
  if (path.endsWith(".txt")) return "text/plain; charset=utf-8";
  return "text/html; charset=utf-8";
}

export function sitePublication(origin) {
  const assets = publicationAssets(origin);
  const pages = Object.fromEntries(publicPaths.flatMap(path => {
    const value = { markdown: markdownPath(path), links: discoveryLinks(origin, path) };
    return path === "/" ? [[path, value]] : [[path, value], [`${path}/`, value]];
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
    response.setHeader("X-Robots-Tag", "noindex, nofollow");
    response.end(request.method === "HEAD" ? undefined : assets.get(assetPath));
  }); };
  return {
    name: "drowse-site-publication",
    configureServer: middleware,
    configurePreviewServer: middleware,
    async generateBundle() {
      for (const [fileName, source] of assets) this.emitFile({ type: "asset", fileName, source });
      this.emitFile({ type: "asset", fileName: "_worker.js", source: discoveryWorkerSource(pages) });
      this.emitFile({ type: "asset", fileName: "_routes.json", source: JSON.stringify({ version: 1, include: Object.keys(pages), exclude: [] }) });
      let headers = await readFile(new URL("../public-hosted/_headers", import.meta.url), "utf8");
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
