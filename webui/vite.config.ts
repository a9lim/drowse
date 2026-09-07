import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "node:path";

// Build output goes directly into the drowse Python package so the
// committed dist/ ships in the wheel.  emptyOutDir=true wipes any
// stale assets between builds — the directory's only consumer is
// FastAPI's StaticFiles mount, never user-authored files.
export default defineConfig({
  plugins: [svelte()],
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
});
