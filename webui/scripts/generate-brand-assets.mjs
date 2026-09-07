import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";
import { icon as tabIcon, states, animatedStates, frameCount } from "./tab-icon-artwork.mjs";

const icon = (state) => tabIcon(state, 0, "dark");

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const rasters = new Map();
  const raster = async (svg, size) => {
    const key = `${size}:${svg}`;
    if (rasters.has(key)) return rasters.get(key);
    const data = await page.evaluate(async ({ svg, size }) => {
      const image = new Image();
      image.src = `data:image/svg+xml,${encodeURIComponent(svg)}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = size;
      canvas.getContext("2d").drawImage(image, 0, 0, size, size);
      return canvas.toDataURL("image/png").split(",")[1];
    }, { svg, size });
    const png = Buffer.from(data, "base64");
    rasters.set(key, png);
    return png;
  };
  for (const directory of ["public-hosted", "public"]) {
    const root = resolve(directory);
    await mkdir(resolve(root, "icons"), { recursive: true });
    for (const state of states) {
      for (const theme of ["light", "dark"]) {
        for (let frame = 0; frame < (animatedStates.has(state) ? frameCount : 1); frame++) {
          const name = `tab-${state}-${theme}${animatedStates.has(state) ? `-${frame}` : ""}`;
          const png = await raster(tabIcon(state, frame, theme), 96);
          await writeFile(resolve(root, "icons", `${name}.png`), png);
          if (theme === "light" && frame < 3) {
            const legacyName = `tab-${state}${animatedStates.has(state) ? `-${frame}` : ""}.png`;
            await writeFile(resolve(root, "icons", legacyName), png);
          }
        }
      }
    }
    await writeFile(resolve(root, "icons/tab-home.png"), await raster(tabIcon("home"), 96));
    await writeFile(resolve(root, "favicon.svg"), tabIcon("home").replace('<g ', '<style>@media(prefers-color-scheme:dark){svg{--tile:#c5b3ff!important;--ink:#141822!important}}</style><g '));
    const sizes = [16, 32, 48];
    const pngs = [];
    for (const size of sizes) pngs.push(await raster(tabIcon("home"), size));
    const iconDirectory = Buffer.alloc(6 + sizes.length * 16);
    iconDirectory.writeUInt16LE(1, 2);
    iconDirectory.writeUInt16LE(sizes.length, 4);
    let offset = iconDirectory.length;
    for (const [i, png] of pngs.entries()) {
      const entry = 6 + i * 16;
      iconDirectory[entry] = sizes[i]; iconDirectory[entry + 1] = sizes[i];
      iconDirectory.writeUInt16LE(1, entry + 4); iconDirectory.writeUInt16LE(32, entry + 6);
      iconDirectory.writeUInt32LE(png.length, entry + 8); iconDirectory.writeUInt32LE(offset, entry + 12);
      offset += png.length;
    }
    await writeFile(resolve(root, "favicon.ico"), Buffer.concat([iconDirectory, ...pngs]));
    for (const [name, size, maskable] of [
      ["drowse-192", 192, false], ["drowse-512", 512, false],
      ["drowse-maskable-512", 512, true], ["apple-touch-icon", 180, false],
    ]) await writeFile(resolve(root, "icons", `${name}.png`), await raster(tabIcon("home", 0, "light", maskable), size));
  }
  if (!process.argv.includes("--icons-only")) {
  const social = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <rect width="1200" height="630" fill="#0b0e17"/>
  <g transform="translate(76 72) scale(1.25)" style="--tile:#c5b3ff;--ink:#141822">${icon("home").replace(/<svg[^>]*>|<\/svg>/g, "")}</g>
  <g font-family="Arial, sans-serif" fill="#f4f1ff">
  <text x="184" y="127" font-size="48" font-weight="700" letter-spacing="-2">Drowse</text>
  <text x="76" y="287" font-size="76" font-weight="700" letter-spacing="-3">See inside</text>
  <text x="76" y="371" font-size="76" font-weight="700" letter-spacing="-3">your model.</text>
  <text x="80" y="459" font-size="27" fill="#c9c2dd">Run, inspect, steer, and branch language models.</text>
  <text x="80" y="504" font-size="27" fill="#c9c2dd">On your device. In your browser.</text></g>
  <g fill="none" stroke="#a48be9" stroke-width="3">
  <path d="M858 308h66c35 0 22-100 59-100h95M924 308h154M924 308c35 0 22 100 59 100h95"/>
  </g><g fill="#211738" stroke="#a48be9" stroke-width="3">
  <rect x="794" y="276" width="84" height="64" rx="18"/>
  <rect x="1044" y="180" width="84" height="56" rx="16"/>
  <rect x="1044" y="280" width="84" height="56" rx="16"/>
  <rect x="1044" y="380" width="84" height="56" rx="16"/>
  </g><g fill="#c1adff"><circle cx="816" cy="308" r="4"/><circle cx="834" cy="308" r="4"/><circle cx="852" cy="308" r="4"/></g></svg>`;
  await page.setViewportSize({ width: 1200, height: 630 });
  await page.setContent(`<style>body{margin:0}</style>${social}`);
  const image = await page.screenshot();
  for (const directory of ["public-hosted", "public"]) {
    await mkdir(resolve(directory, "social"), { recursive: true });
    await writeFile(resolve(directory, "social/drowse.svg"), social);
    await writeFile(resolve(directory, "social/drowse.png"), image);
  }
  }
} finally { await browser.close(); }
