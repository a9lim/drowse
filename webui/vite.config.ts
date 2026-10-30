import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { readdir, readFile } from "node:fs/promises";
import { relative, resolve } from "node:path";
import { shipsInDashboard } from "./scripts/dashboard-assets.mjs";

const publicDir = resolve(__dirname, "public");

/** Copy only the public files the dashboard uses (see
 *  scripts/dashboard-assets.mjs); the dev server still serves the whole dir. */
function dashboardPublicAssets(): Plugin {
  return {
    name: "drowse-dashboard-public-assets",
    apply: "build",
    async generateBundle() {
      const walk = async (directory: string): Promise<string[]> => {
        const entries = await readdir(directory, { withFileTypes: true });
        const files = await Promise.all(entries.map(entry => entry.isDirectory()
          ? walk(resolve(directory, entry.name)) : [resolve(directory, entry.name)]));
        return files.flat();
      };
      for (const file of await walk(publicDir)) {
        const fileName = relative(publicDir, file).split("\\").join("/");
        if (shipsInDashboard(fileName)) this.emitFile({ type: "asset", fileName, source: await readFile(file) });
      }
    },
  };
}

/** The browser packs' bundled SAE description indexes stay out of the
 *  dashboard; its lookups go to Neuronpedia (src/lib/saeDescriptionBundle.ts). */
function omitBundledSaeDescriptions(): Plugin {
  const bundle = resolve(__dirname, "src/lib/saeDescriptionBundle.ts");
  const none = resolve(__dirname, "src/lib/saeDescriptionBundle.none.ts");
  return {
    name: "drowse-dashboard-omit-sae-descriptions",
    enforce: "pre",
    async resolveId(source, importer, options) {
      if (!source.endsWith("saeDescriptionBundle")) return null;
      const resolved = await this.resolve(source, importer, { ...options, skipSelf: true });
      return resolved?.id === bundle ? none : null;
    },
  };
}

// Build output goes directly into the drowse Python package so the
// committed dist/ ships in the wheel.  emptyOutDir=true wipes any
// stale assets between builds — the directory's only consumer is
// FastAPI's StaticFiles mount, never user-authored files.
export default defineConfig(({ command }) => ({
  plugins: [svelte(), omitBundledSaeDescriptions(), dashboardPublicAssets()],
  publicDir: command === "build" ? false : publicDir,
  resolve: {
    alias: {
      "@runtime-local-drawer": resolve(
        __dirname,
        "src/drawers/UnavailableRuntimeDrawer.svelte",
      ),
      "@runtime-jlens-source": resolve(
        __dirname,
        "src/panels/JLensSourceSection.svelte",
      ),
      "@runtime-jlens-missing": resolve(
        __dirname,
        "src/drawers/token/JLensMissingState.svelte",
      ),
      "@runtime-sae-source": resolve(
        __dirname,
        "src/panels/SaeSourceSection.svelte",
      ),
      "@runtime-session-drawer": resolve(
        __dirname,
        "src/drawers/SessionAdminDrawer.svelte",
      ),
    },
  },
  build: {
    outDir: resolve(__dirname, "../drowse/web/dist"),
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        // Pin asset names so the SPA fallback in drowse/web/routes.py
        // doesn't have to deal with hash variance — the bundle replaces
        // itself on every build, and StaticFiles serves whatever's at
        // the path the index.html references.
        entryFileNames: "assets/drowse.js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  server: {
    // For `npm run dev`, proxy API + WS to the running drowse serve.
    proxy: {
      "/drowse": {
        target: "http://localhost:8000",
        ws: true,
      },
      "/v1": "http://localhost:8000",
      "/api": "http://localhost:8000",
    },
  },
}));
