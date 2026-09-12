import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

const webuiRoot = fileURLToPath(new URL("../", import.meta.url));
const landingPath = fileURLToPath(new URL("../src/hosted/ui/Landing.svelte", import.meta.url));
const entryId = "virtual:drowse-landing-prerender";
const imageTypes = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".webp": "image/webp" };

export async function prerenderLanding({ define = {}, assetUrl } = {}) {
  const server = await createServer({
    configFile: false,
    root: webuiRoot,
    publicDir: false,
    logLevel: "error",
    mode: "production",
    define,
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
    plugins: [
      {
        name: "drowse-landing-prerender-assets",
        enforce: "pre",
        async load(id) {
          const mime = imageTypes[extname(id)];
          if (!mime) return;
          const source = await readFile(id);
          const url = assetUrl
            ? await assetUrl(id, source)
            : `data:${mime};base64,${source.toString("base64")}`;
          return `export default ${JSON.stringify(url)};`;
        },
      },
      {
        name: "drowse-landing-prerender-entry",
        resolveId(id) {
          if (id === entryId) return `\0${entryId}`;
        },
        load(id) {
          if (id !== `\0${entryId}`) return;
          return `import { render } from "svelte/server";
import Landing from ${JSON.stringify(landingPath)};
export function prerender() { return render(Landing); }`;
        },
      },
      svelte({
        configFile: fileURLToPath(new URL("../svelte.config.js", import.meta.url)),
        emitCss: false,
        compilerOptions: { dev: false, hmr: false },
      }),
    ],
  });
  try {
    const { prerender } = await server.ssrLoadModule(entryId);
    const { body, head } = prerender();
    const css = [...head.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/g)]
      .map((match) => match[1]).join("\n");
    return { body, css };
  } finally {
    await server.close();
  }
}
