import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

const webuiRoot = fileURLToPath(new URL("../", import.meta.url));
const ui = path => fileURLToPath(new URL(`../src/hosted/ui/${path}`, import.meta.url));
const entryId = "virtual:drowse-landing-prerender";
const pwaRegisterId = "virtual:pwa-register";
const imageTypes = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".webp": "image/webp" };

/**
 * Components prerendered into static HTML. Each key is written to `data-prerendered` so the
 * client entry hydrates only when it is about to mount that same component with the same props.
 */
export const prerenderedComponents = {
  landing: ui("LandingRoot.svelte"),
  credits: ui("Credits.svelte"),
  contact: ui("Contact.svelte"),
  privacy: ui("pages/Privacy.svelte"),
  about: ui("pages/About.svelte"),
  learn: ui("pages/Learn.svelte"),
  "learn-guide": ui("pages/LearnGuide.svelte"),
  contribute: ui("pages/Contribute.svelte"),
  neurons: ui("pages/Neurons.svelte"),
  "third-party-licenses": ui("pages/ThirdPartyLicenses.svelte"),
};

/**
 * Server-renders hosted pages. `pages` maps an output id to `{ component, props }`, where
 * `component` is a key of `prerenderedComponents`.
 */
export async function prerenderPages(pages, { define = {}, assetUrl } = {}) {
  const used = [...new Set(Object.values(pages).map(page => page.component))];
  for (const key of used) if (!prerenderedComponents[key]) throw new Error(`Unknown prerendered component ${key}`);
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
        // Svelte runs onDestroy on the server after rendering. Teardown that touches browser globals
        // would throw there, and a server render has nothing to tear down, so drop those callbacks.
        name: "drowse-prerender-server-teardown",
        enforce: "pre",
        transform(code, id) {
          if (!id.endsWith(".svelte") || !/\bonDestroy\(/.test(code)) return;
          return { code: code.replace(/\bonDestroy\(/g, "(() => {})("), map: null };
        },
      },
      {
        name: "drowse-landing-prerender-entry",
        resolveId(id) {
          if (id === entryId) return `\0${entryId}`;
          // The update prompt registers the service worker only after mount; the server never does.
          if (id === pwaRegisterId) return `\0${pwaRegisterId}`;
        },
        load(id) {
          if (id === `\0${pwaRegisterId}`) return "export function registerSW() { return async () => {}; }";
          if (id !== `\0${entryId}`) return;
          return `import { render } from "svelte/server";
${used.map((key, index) => `import C${index} from ${JSON.stringify(prerenderedComponents[key])};`).join("\n")}
const components = { ${used.map((key, index) => `${JSON.stringify(key)}: C${index}`).join(", ")} };
export function prerender(key, props) { return render(components[key], { props }); }`;
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
    const rendered = {};
    for (const [id, page] of Object.entries(pages)) {
      const { body, head } = prerender(page.component, page.props ?? {});
      const css = [...head.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/g)]
        .map((match) => match[1]).join("\n");
      rendered[id] = { body, css, component: page.component };
    }
    return rendered;
  } finally {
    await server.close();
  }
}

export async function prerenderLanding(options = {}) {
  const { landing } = await prerenderPages({ landing: { component: "landing" } }, options);
  return { body: landing.body, css: landing.css };
}
