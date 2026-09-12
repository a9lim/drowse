import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { fileURLToPath, URL } from "node:url";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import { siteTitle, siteDescription, siteAccent, socialImagePath, socialImageAlt, publicOrigin, discoveryMetadata } from "./scripts/site-metadata.mjs";
import { sitePublication, discoveryTags } from "./scripts/site-publication.mjs";
import { prerenderLanding } from "./scripts/prerender-hosted.mjs";

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
const releaseMetadata: Plugin = {
  name: "drowse-hosted-release-metadata",
  configResolved(config) { isBuild = config.command === "build"; },
  async buildStart() {
    if (!isBuild) return;
    const prerendered = await prerenderLanding({
      define: {
        __DROWSE_HOSTED_RELEASE__: JSON.stringify(releaseBuild),
        __DROWSE_SOURCE_REVISION__: JSON.stringify(releaseRevision),
        __DROWSE_SOURCE_URL__: JSON.stringify(sourceUrl),
      },
      assetUrl: (filename: string, source: Buffer) => `/${this.getFileName(this.emitFile({ type: "asset", name: basename(filename), source }))}`,
    });
    landingBody = prerendered.body;
    landingStyle = this.getFileName(this.emitFile({ type: "asset", name: "landing-prerender.css", source: prerendered.css }));
  },
  transformIndexHtml(html) {
    const replacements: Record<string, string> = {
      __DROWSE_HOSTED_CHANNEL__: releaseBuild ? "release" : "preview",
      __DROWSE_HOSTED_DESCRIPTION__: pageDescription,
      __DROWSE_HOSTED_TITLE__: pageTitle,
      __DROWSE_SOCIAL_IMAGE__: `${siteOrigin}${socialImagePath}`,
      __DROWSE_ACCENT__: siteAccent,
      __DROWSE_SOCIAL_ALT__: socialImageAlt,
      __DROWSE_DISCOVERY_METADATA__: `${discoveryMetadata(siteOrigin)}\n${discoveryTags("/", siteOrigin)}`,
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
    if (landingBody) {
      transformed = transformed.replace('<div id="app"></div>', `<div id="app" data-prerendered>${landingBody}</div>`)
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
      .replace(/<title>[^<]*<\/title>/, "<title>Drowse</title>")
      .replace("<head>", '<head>\n<meta name="robots" content="noindex, follow" />');
    this.emitFile({ type: "asset", fileName: "app-shell.html", source: shell });
    this.emitFile({ type: "asset", fileName: "404.html", source: shell.replace("<title>Drowse</title>", "<title>Page not found · Drowse</title>") });
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
  define: {
    __DROWSE_HOSTED_RELEASE__: JSON.stringify(releaseBuild),
    __DROWSE_SOURCE_REVISION__: JSON.stringify(releaseRevision),
    __DROWSE_SOURCE_URL__: JSON.stringify(sourceUrl),
    __DROWSE_VERSION__: JSON.stringify(drowseVersion),
    __DROWSE_ARTIFACT_SOURCE_REVISION__: JSON.stringify(artifactSourceRevision),
    __DROWSE_RUNTIME_ABI__: JSON.stringify(runtimeLock.runtimeAbi),
    __DROWSE_HOOK_ABI__: JSON.stringify(runtimeLock.hookAbi),
    __DROWSE_HOSTED_CHANNEL__: JSON.stringify(releaseBuild ? "release" : "preview"),
  },
  plugins: [
    releaseMetadata,
    sitePublication(siteOrigin),
    projectLicenseAsset,
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
      workbox: {
        cacheId: "drowse-hosted",
        clientsClaim: true,
        cleanupOutdatedCaches: true,
        ignoreURLParametersMatching: [/^utm_/, /^fbclid$/, /^v$/],
        globPatterns: ["**/*.{css,html,js,json,mp4,png,svg,wasm,woff2}", "images/ethereal-orb.jpg", "LICENSE"],
        globIgnores: [
          "_worker.js", "_routes.json", "404.html", "skills/**", ".well-known/**",
          "assets/App-*.css",
          "assets/App-*.js",
          "assets/browser.worker-*.js",
          "assets/fitting.worker-*.js",
          "assets/registry-*.js",
          "assets/drowse-web-llm-*.js",
          "assets/typescript-*.js",
          "social/**",
          "video/**",
          "wasm/**",
        ],
        maximumFileSizeToCacheInBytes: 1024 * 1024,
        manifestTransforms: [async (entries) => ({
          manifest: entries.map((entry) => entry.url.endsWith(".js")
            ? { ...entry, revision: artifactSourceRevision }
            : entry),
          warnings: [],
        })],
        navigateFallback: "/app-shell.html",
        navigateFallbackAllowlist: [/^\/(?:app(?:\/[^?]*)?|credits\/?|contact\/?)(?:\?.*)?$/],
        navigateFallbackDenylist: [/[?&]app-recovery=/],
        runtimeCaching: [
          {
            urlPattern:
              /\/assets\/(?:App|browser\.worker|fitting\.worker|registry|drowse-web-llm)-[^/]+\.(?:css|js)$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-on-demand-assets-v1",
              // Content-hashed app files do not vary between module and ordinary fetch requests.
              matchOptions: { ignoreVary: true },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /\/assets\/(?:heroShaderRuntime|typescript)-[^/]+\.js$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-landing-shader-v1",
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /\/video\/the-uncertain-loom-loop-no-flash\.mp4$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-landing-shader-v1",
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /\/wasm\/[^/]+\.(?:js|json|wasm)$/,
            handler: "CacheFirst",
            options: {
              cacheName: "drowse-hosted-on-demand-wasm-v1",
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
    // Safari can retain failed modulepreloads across reloads (WebKit 270357).
    modulePreload: false,
  },
});
