import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { fileURLToPath, URL } from "node:url";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import { siteTitle, siteDescription, siteAccent, themeColors, socialImagePath, socialImageType, socialImageAlt, publicOrigin, discoveryMetadata, pageDiscoveryMetadata } from "./scripts/site-metadata.mjs";
import { sitePublication, discoveryTags, themeInitScript } from "./scripts/site-publication.mjs";
import { guidePages, htmlGuidePaths, sitePages } from "./scripts/site-content.mjs";
import { prerenderPages } from "./scripts/prerender-hosted.mjs";
import { thirdPartyLicenseText } from "./scripts/third-party-licenses.mjs";

const fromRoot = (path: string) => fileURLToPath(new URL(path, import.meta.url));
const releaseBuild = process.env.npm_lifecycle_event === "build:hosted:release";
const artifactSourceRevision = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: fromRoot("../"),
  encoding: "utf8",
}).trim();
if (!/^[0-9a-f]{40}$/.test(artifactSourceRevision)) {
  throw new Error("Hosted browser artifacts require an exact Drowse source revision");
}
const releaseRevision = releaseBuild ? artifactSourceRevision : "preview";
// Sitemap lastmod follows the source commit, so rebuilding unchanged content does not churn it.
const sourceDate = execFileSync("git", ["log", "-1", "--format=%cs", artifactSourceRevision], {
  cwd: fromRoot("../"),
  encoding: "utf8",
}).trim();
const drowseSource = await readFile(fromRoot("../drowse/__init__.py"), "utf8");
const drowseVersion = /__version__\s*=\s*"([^"]+)"/.exec(drowseSource)?.[1];
if (!drowseVersion) throw new Error("Hosted build could not resolve the Drowse version");
const runtimeLock = JSON.parse(
  await readFile(fromRoot("../browser-runtime/runtime-lock.json"), "utf8"),
) as { runtimeAbi?: unknown; hookAbi?: unknown };
if (
  typeof runtimeLock.runtimeAbi !== "string" ||
  typeof runtimeLock.hookAbi !== "string"
) {
  throw new Error("Hosted build could not resolve the browser runtime and hook ABIs");
}
const sourceRepositoryUrl = "https://github.com/a9lim/drowse";
const sourceUrl = releaseBuild
  ? `${sourceRepositoryUrl}/tree/${artifactSourceRevision}`
  : sourceRepositoryUrl;
const pageDescription = siteDescription;
const manifestDescription = siteDescription;
const pageTitle = siteTitle;
const siteOrigin = publicOrigin(process.env.DROWSE_PUBLIC_ORIGIN);
const projectLicense = await readFile(fromRoot("../LICENSE"), "utf8");
// The fitting kernel has fixed file names, so its URLs carry the pinned digests: a new kernel is a
// new cache key instead of a stale CacheFirst hit.
const fittingKernel = JSON.parse(
  await readFile(fromRoot("./public-hosted/wasm/fitting-kernel.json"), "utf8"),
) as { files: { name: string; sha256: string }[] };
const fittingFile = (name: string) => {
  const file = fittingKernel.files.find((entry) => entry.name === name);
  if (!file || !/^[0-9a-f]{64}$/.test(file.sha256)) throw new Error(`Hosted build could not resolve the ${name} digest`);
  return `/wasm/${name}?v=${file.sha256.slice(0, 16)}`;
};
const fittingWasm = {
  module: fittingFile("drowse_fitting_wasm.js"),
  wasm: fittingFile("drowse_fitting_wasm_bg.wasm"),
};
const appDefine = {
  __DROWSE_HOSTED_RELEASE__: JSON.stringify(releaseBuild),
  __DROWSE_SOURCE_REVISION__: JSON.stringify(releaseRevision),
  __DROWSE_SOURCE_URL__: JSON.stringify(sourceUrl),
  __DROWSE_VERSION__: JSON.stringify(drowseVersion),
  __DROWSE_ARTIFACT_SOURCE_REVISION__: JSON.stringify(artifactSourceRevision),
  __DROWSE_RUNTIME_ABI__: JSON.stringify(runtimeLock.runtimeAbi),
  __DROWSE_HOOK_ABI__: JSON.stringify(runtimeLock.hookAbi),
  __DROWSE_HOSTED_CHANNEL__: JSON.stringify(releaseBuild ? "release" : "preview"),
  __DROWSE_FITTING_WASM__: JSON.stringify(fittingWasm),
};
const themeScript = await themeInitScript();
// Each prerendered route: output file, the `data-prerendered` key main.ts checks, and its metadata.
const guideByPath = new Map(guidePages.map((page) => [page.path, page]));
const pageMetadata = new Map([...guidePages, ...sitePages].map((page) => [page.path, page]));
const prerenderedRoutes = [
  { path: "/credits", key: "credits", component: "credits" },
  { path: "/contact", key: "contact", component: "contact" },
  { path: "/contribute", key: "contribute", component: "contribute" },
  { path: "/neurons", key: "neurons", component: "neurons" },
  { path: "/third-party-licenses", key: "third-party-licenses", component: "third-party-licenses" },
  ...htmlGuidePaths.map((path) => path.startsWith("/learn/")
    ? { path, key: `learn-guide:${path}`, component: "learn-guide", props: { path } }
    : { path, key: path.slice(1), component: path.slice(1) }),
].map((route) => {
  const metadata = pageMetadata.get(route.path);
  if (!metadata) throw new Error(`Hosted page ${route.path} has no metadata`);
  return { ...route, title: metadata.title, description: metadata.description, heading: guideByPath.get(route.path)?.heading };
});
const escapeAttribute = (text: string) => text.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const escapeText = (text: string) => text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};
const httpsCertificatePath = process.env.DROWSE_HTTPS_CERT;
const httpsPrivateKeyPath = process.env.DROWSE_HTTPS_KEY;
if (Boolean(httpsCertificatePath) !== Boolean(httpsPrivateKeyPath)) {
  throw new Error(
    "Hosted HTTPS requires both DROWSE_HTTPS_CERT and DROWSE_HTTPS_KEY",
  );
}
const https =
  httpsCertificatePath && httpsPrivateKeyPath
    ? {
        cert: await readFile(httpsCertificatePath),
        key: await readFile(httpsPrivateKeyPath),
      }
    : undefined;

let isBuild = false;
let landingBody = "";
let landingStyle = "";
const prerenderedPages = new Map<string, { body: string; style: string }>();
const releaseMetadata: Plugin = {
  name: "drowse-hosted-release-metadata",
  configResolved(config) { isBuild = config.command === "build"; },
  async buildStart() {
    if (!isBuild) return;
    const prerendered = await prerenderPages({
      landing: { component: "landing" },
      ...Object.fromEntries(prerenderedRoutes.map((route) => [route.key, { component: route.component, props: route.props }])),
    }, {
      define: appDefine,
      assetUrl: (filename: string, source: Buffer) => `/${this.getFileName(this.emitFile({ type: "asset", name: basename(filename), source }))}`,
    });
    landingBody = prerendered.landing.body;
    landingStyle = this.getFileName(this.emitFile({ type: "asset", name: "landing-prerender.css", source: prerendered.landing.css }));
    for (const route of prerenderedRoutes) {
      const page = prerendered[route.key];
      const name = `prerender-${route.key.replace(/[^a-z0-9]+/g, "-").replace(/-$/, "")}.css`;
      prerenderedPages.set(route.key, { body: page.body, style: this.getFileName(this.emitFile({ type: "asset", name, source: page.css })) });
    }
  },
  transformIndexHtml(html) {
    const replacements: Record<string, string> = {
      __DROWSE_HOSTED_CHANNEL__: releaseBuild ? "release" : "preview",
      __DROWSE_HOSTED_DESCRIPTION__: pageDescription,
      __DROWSE_HOSTED_TITLE__: pageTitle,
      __DROWSE_SOCIAL_IMAGE__: `${siteOrigin}${socialImagePath}`,
      __DROWSE_SOCIAL_IMAGE_TYPE__: socialImageType,
      __DROWSE_THEME_DARK__: themeColors.dark,
      __DROWSE_THEME_LIGHT__: themeColors.light,
      __DROWSE_SOCIAL_ALT__: socialImageAlt,
      __DROWSE_DISCOVERY_METADATA__: `${discoveryMetadata(siteOrigin, { version: drowseVersion })}\n${discoveryTags("/", siteOrigin)}`,
      __DROWSE_SOURCE_REVISION__: releaseRevision,
      __DROWSE_SOURCE_URL__: sourceUrl,
      __DROWSE_VERSION__: drowseVersion,
      __DROWSE_ARTIFACT_SOURCE_REVISION__: artifactSourceRevision,
      __DROWSE_RUNTIME_ABI__: runtimeLock.runtimeAbi,
      __DROWSE_HOOK_ABI__: runtimeLock.hookAbi,
      __DROWSE_ARTIFACT_VERIFICATION__: releaseBuild ? "verified" : "unverified",
    };
    let transformed = html;
    for (const [placeholder, value] of Object.entries(replacements)) {
      transformed = transformed.replaceAll(placeholder, value);
    }
    if (isBuild) {
      // Inline the pre-paint theme script; its digest is added to the CSP script-src.
      transformed = transformed.replace('<script src="/theme-init.js"></script>', `<script>${themeScript.source}</script>`);
      if (!transformed.includes(themeScript.source)) throw new Error("Hosted index is missing the theme-init script tag");
    }
    if (landingBody) {
      transformed = transformed.replace('<div id="app"></div>', `<div id="app" data-prerendered="landing">${landingBody}</div>`)
        .replace("</head>", `<link rel="stylesheet" href="/${landingStyle}" data-prerender-style />\n</head>`);
    }
    if (transformed.includes("__DROWSE_")) {
      throw new Error("Hosted index contains an unresolved Drowse metadata placeholder");
    }
    return transformed;
  },
};

const projectLicenseAsset: Plugin = {
  name: "drowse-hosted-project-license",
  apply: "build",
  buildStart() {
    this.emitFile({ type: "asset", fileName: "LICENSE", source: projectLicense });
  },
};

const fontPreloads = [/^assets\/WixMadeforText-normal-latin-[A-Za-z0-9_-]{8}\.woff2$/, /^assets\/WixMadeforDisplay-latin-[A-Za-z0-9_-]{8}\.woff2$/];

function replaceOnce(html: string, pattern: RegExp, replacement: string, label: string) {
  if (!pattern.test(html)) throw new Error(`Hosted page template is missing ${label}`);
  return html.replace(pattern, replacement);
}

/** Derives a prerendered route's document from the built homepage. */
function routeDocument(index: string, route: (typeof prerenderedRoutes)[number], page: { body: string; style: string }) {
  const title = escapeAttribute(route.title);
  const description = escapeAttribute(route.description);
  let html = index;
  html = replaceOnce(html, /<title>[^<]*<\/title>/, `<title>${escapeText(route.title)}</title>`, "title");
  html = replaceOnce(html, /(<meta\s+name="description"\s+content=")[^"]*(")/, `$1${description}$2`, "description");
  for (const name of ["og:title", "twitter:title"]) {
    html = replaceOnce(html, new RegExp(`(<meta\\s+(?:name|property)="${name}"\\s+content=")[^"]*(")`), `$1${title}$2`, name);
  }
  for (const name of ["og:description", "twitter:description"]) {
    html = replaceOnce(html, new RegExp(`(<meta\\s+(?:name|property)="${name}"\\s+content=")[^"]*(")`), `$1${description}$2`, name);
  }
  html = html.replace(/<link rel="canonical"[^>]*>\n?/g, "")
    .replace(/<meta property="og:url"[^>]*>\n?/g, "")
    .replace(/<script type="application\/ld\+json">[\s\S]*?<\/script>\n?/g, "")
    .replace(/<link rel="describedby"[^>]*>\n(?:<link rel="alternate"[^>]*>\n)?<link rel="service-doc"[^>]*>/, discoveryTags(route.path))
    .replace("</head>", () => `${pageDiscoveryMetadata(siteOrigin, route)}\n</head>`);
  const landing = `<div id="app" data-prerendered="landing">${landingBody}</div>`;
  if (!html.includes(landing)) throw new Error("Hosted page template is missing the prerendered landing");
  html = html.replace(landing, () => `<div id="app" data-prerendered="${escapeAttribute(route.key)}">${page.body}</div>`);
  html = replaceOnce(html, /<link rel="stylesheet" href="[^"]+" data-prerender-style \/>/, `<link rel="stylesheet" href="/${page.style}" data-prerender-style />`, "prerender style");
  if (html.includes("__DROWSE_")) throw new Error(`Hosted page ${route.path} contains an unresolved placeholder`);
  return html;
}

const sitePagesPlugin: Plugin = {
  name: "drowse-hosted-site-pages",
  enforce: "post",
  apply: "build",
  async generateBundle(_, bundle) {
    const index = bundle["index.html"];
    if (!index || index.type !== "asset") throw new Error("Hosted index is missing");
    // Preload the two faces used above the fold; other subsets load on demand by unicode-range.
    const preloads = fontPreloads.map((pattern) => {
      const fonts = Object.keys(bundle).filter((name) => pattern.test(name));
      if (fonts.length !== 1) throw new Error(`Hosted build expected one font matching ${pattern}, found ${fonts.length}`);
      return `<link rel="preload" href="/${fonts[0]}" as="font" type="font/woff2" crossorigin />`;
    }).join("\n    ");
    const indexSource = String(index.source).replace(/(\s*)<link rel="stylesheet"/, `$1${preloads}$1<link rel="stylesheet"`);
    if (!indexSource.includes(preloads)) throw new Error("Hosted index has no stylesheet to preload fonts before");
    index.source = indexSource;
    for (const route of prerenderedRoutes) {
      const page = prerenderedPages.get(route.key);
      if (!page) throw new Error(`Hosted page ${route.path} was not prerendered`);
      this.emitFile({ type: "asset", fileName: `${route.path.slice(1)}.html`, source: routeDocument(indexSource, route, page) });
    }
    const moduleIds = Object.values(bundle).flatMap((output) => output.type === "chunk" ? output.moduleIds : []);
    this.emitFile({ type: "asset", fileName: "third-party-licenses.txt", source: await thirdPartyLicenseText(moduleIds) });
  },
};

const notFoundPage: Plugin = {
  name: "drowse-hosted-not-found-page",
  enforce: "post",
  generateBundle(_, bundle) {
    const index = bundle["index.html"];
    if (!index || index.type !== "asset") throw new Error("Hosted index is missing");
    const shell = String(index.source)
      .replace(/<body>[\s\S]*<\/body>/, '<body><div id="app"></div></body>')
      .replace(/<link rel="canonical"[^>]*>/g, "")
      .replace(/<meta property="og:url"[^>]*>/g, "")
      .replace(/<script type="application\/ld\+json">[\s\S]*?<\/script>/g, "")
      .replace(/<link[^>]*data-prerender-style[^>]*>/g, "")
      .replace(/<title>[^<]*<\/title>/, "<title>Drowse workbench</title>")
      .replace("<head>", '<head>\n<meta name="robots" content="noindex, follow" />');
    this.emitFile({ type: "asset", fileName: "app-shell.html", source: shell });
    this.emitFile({ type: "asset", fileName: "404.html", source: shell.replace("<title>Drowse workbench</title>", "<title>Page not found · Drowse</title>") });
  },
};

const offlineRuntimeAssets: Plugin = {
  name: "drowse-offline-runtime-assets",
  apply: "build",
  enforce: "post",
  generateBundle(_, bundle) {
    const assets = Object.keys(bundle).filter(name =>
      /^assets\/(?:App|browser\.worker|registry|drowse-web-llm)-[^/]+\.(?:css|js)$/u.test(name)
    ).sort();
    this.emitFile({
      type: "asset",
      fileName: "runtime-assets.json",
      source: JSON.stringify({ assets: assets.map(name => `/${name}`) }),
    });
  },
};

export default defineConfig({
  root: fromRoot("./hosted"),
  publicDir: fromRoot("./public-hosted"),
  cacheDir: process.env.DROWSE_VITE_CACHE_DIR ?? fromRoot("./node_modules/.vite-hosted"),
  define: appDefine,
  plugins: [
    releaseMetadata,
    sitePublication(siteOrigin, { lastmod: sourceDate }),
    projectLicenseAsset,
    sitePagesPlugin,
    notFoundPage,
    offlineRuntimeAssets,
    svelte({ configFile: fromRoot("./svelte.config.js") }),
    VitePWA({
      strategies: "generateSW",
      injectRegister: false,
      registerType: "prompt",
      includeAssets: [
        "LICENSE-Martian-Mono.txt",
        "LICENSE-Wix-Madefor.txt",
      ],
      manifest: {
        id: "/app",
        name: "Drowse",
        short_name: "Drowse",
        description: manifestDescription,
        start_url: "/app",
        scope: "/",
        display: "standalone",
        display_override: ["standalone", "minimal-ui"],
        launch_handler: { client_mode: "focus-existing" },
        shortcuts: [
          { name: "Chats", short_name: "Chats", url: "/app", description: "Open your saved chats" },
          { name: "Models", short_name: "Models", url: "/app?choose=1", description: "Choose a model to run" },
        ],
        background_color: "#0b0e17",
        theme_color: siteAccent,
        categories: ["developer", "productivity", "utilities"],
        icons: [
          {
            src: "/icons/drowse-192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "/icons/drowse-512.png",
            sizes: "512x512",
            type: "image/png",
          },
          {
            src: "/icons/drowse-maskable-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      // The manifest icons already match globPatterns; adding them again duplicates precache entries.
      includeManifestIcons: false,
      workbox: {
        cacheId: "drowse-hosted",
        clientsClaim: true,
        cleanupOutdatedCaches: true,
        ignoreURLParametersMatching: [/^utm_/, /^fbclid$/, /^v$/],
        globPatterns: ["**/*.{css,html,js,json,mp4,png,svg,wasm,woff2}", "images/ethereal-orb.jpg", "LICENSE"],
        globIgnores: [
          "_worker.js", "_routes.json", "404.html", "skills/**", ".well-known/**",
          // Prerendered pages are network-first documents (see the page route below), not precache entries.
          "credits.html", "contact.html", "contribute.html", "neurons.html", "third-party-licenses.html", "privacy.html", "about.html",
          "learn.html", "learn/**", "assets/prerender-*.css",
          "assets/{Privacy,About,Learn,LearnGuide,Contribute,Neurons,ThirdPartyLicenses,SitePage,GuideContent,site-content,NotFound}-*",
          // Browser-agent workspace tools load only when the experimental model context is present.
          "assets/{workspaceTools,serviceTools,generation}-*.js",
          // Inlined into every document at build time.
          "theme-init.js",
          // Hundreds of animated tab-icon frames and multi-megabyte SAE label tables load on demand.
          "icons/tab-*.png",
          "assets/gemma-*.json",
          // Non-Latin font subsets load by unicode-range only when a page needs them.
          "assets/*-cyrillic*.woff2", "assets/*-vietnamese*.woff2", "assets/*-latin-ext-*.woff2",
          "assets/App-*.css",
          "assets/App-*.js",
          "assets/browser.worker-*.js",
          "assets/fitting.worker-*.js",
          "assets/registry-*.js",
          "assets/drowse-web-llm-*.js",
          "assets/typescript-*.js",
          // The neuron explorer's catalog loads one layer at a time, on demand.
          "neurons-data/**",
          "social/**",
          "video/**",
          "wasm/**",
        ],
        maximumFileSizeToCacheInBytes: 1024 * 1024,
        // Hashed assets/ files are keyed by URL (revision null). Only unhashed scripts are pinned to the
        // source revision; stamping hashed chunks made every release refetch all JavaScript.
        manifestTransforms: [async (entries) => ({
          manifest: entries.map((entry) => entry.url.endsWith(".js") && !entry.url.startsWith("assets/")
            ? { ...entry, revision: artifactSourceRevision }
            : entry),
          warnings: [],
        })],
        navigateFallback: "/app-shell.html",
        navigateFallbackAllowlist: [/^\/app\/?(?:\?.*)?$/],
        navigateFallbackDenylist: [/[?&]app-recovery=/],
        runtimeCaching: [
          {
            // Prerendered pages come from the network so visitors get current, crawlable HTML. Offline,
            // a page opened before is reused; credits and contact otherwise boot from the app shell.
            urlPattern: ({ request, url }) => request.mode === "navigate" &&
              /^\/(?:credits|contact|contribute|neurons|third-party-licenses|privacy|about|learn(?:\/[a-z-]+)?)$/.test(url.pathname),
            handler: "NetworkFirst",
            options: {
              cacheName: "drowse-hosted-pages-v1",
              networkTimeoutSeconds: 5,
              expiration: { maxEntries: 16, purgeOnQuotaError: true },
              cacheableResponse: { statuses: [200] },
              precacheFallback: { fallbackURL: "/app-shell.html" },
            },
          },
          {
            urlPattern:
              /\/assets\/(?:App|browser\.worker|fitting\.worker|registry|drowse-web-llm)-[^/]+\.(?:css|js)$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-on-demand-assets-v1",
              // Content-hashed app files do not vary between module and ordinary fetch requests.
              matchOptions: { ignoreVary: true },
              // Bounds superseded releases. Files stored directly by offline setup are not evicted.
              expiration: { maxEntries: 64, purgeOnQuotaError: true },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /\/assets\/(?:heroShaderRuntime|typescript)-[^/]+\.js$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-landing-shader-v1",
              expiration: { maxEntries: 8, purgeOnQuotaError: true },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            // Versioned by the fitting-kernel digest (see fittingWasm), so a hit is always current.
            urlPattern: /\/wasm\/[^/]+\.(?:js|json|wasm)(?:\?v=[0-9a-f]+)?$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-on-demand-wasm-v2",
              expiration: { maxEntries: 6, purgeOnQuotaError: true },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /\/icons\/tab-[^/]+\.png(?:\?[^/]*)?$/,
            handler: "StaleWhileRevalidate",
            options: {
              cacheName: "drowse-hosted-tab-icons-v1",
              expiration: { maxEntries: 320, purgeOnQuotaError: true },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /\/assets\/gemma-[^/]+\.json$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-sae-descriptions-v1",
              expiration: { maxEntries: 6, purgeOnQuotaError: true },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /\/assets\/[^/]+\.woff2$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-fonts-v1",
              expiration: { maxEntries: 24, purgeOnQuotaError: true },
              cacheableResponse: { statuses: [200] },
            },
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      "@hosted": fromRoot("./src/hosted"),
      "@runtime": fromRoot("./src/lib/runtime"),
      "@runtime-jlens-source": fromRoot("./src/hosted/ui/JLensSourceSection.svelte"),
      "@runtime-jlens-missing": fromRoot("./src/hosted/ui/JLensMissingState.svelte"),
      "@runtime-local-drawer": fromRoot("./src/drawers/LocalRuntimeDrawer.svelte"),
      "@runtime-sae-source": fromRoot("./src/hosted/ui/SaeSourceSection.svelte"),
      "@runtime-session-drawer": fromRoot("./src/drawers/UnavailableRuntimeDrawer.svelte"),
    },
  },
  optimizeDeps: {
    // This vendored ESM bundle must not outlive an inference-runtime update in Vite's cache.
    exclude: ["@drowse/web-llm"],
    include: [
      "@noble/hashes/blake2.js",
      "@noble/hashes/sha2.js",
      "@noble/hashes/utils.js",
      "fflate",
      "svelte",
      "svelte/reactivity",
      "svelte/transition",
      "yaml",
    ],
  },
  worker: {
    format: "es",
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("/node_modules/@drowse/web-llm/")) return "drowse-web-llm";
        },
      },
    },
  },
  server: { headers: isolationHeaders, https },
  preview: { headers: isolationHeaders, https },
  build: {
    outDir: fromRoot("./dist-hosted"),
    emptyOutDir: true,
    sourcemap: false,
    // Safari can retain failed modulepreloads across reloads (WebKit 270357). Static
    // <link rel="modulepreload"> tags in the prerendered HTML would hit the same bug, so none are emitted.
    modulePreload: false,
    // Fonts stay separate, cacheable files instead of base64 inside the render-blocking stylesheet.
    assetsInlineLimit: (filePath: string) => (filePath.endsWith(".woff2") ? false : undefined),
  },
});
