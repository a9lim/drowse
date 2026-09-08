import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";
import { icon as tabIcon, states, animatedStates, frameCount } from "./tab-icon-artwork.mjs";
import { socialImageArtwork } from "./social-image-artwork.mjs";

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
  for (const directory of process.argv.includes("--social-only") ? [] : ["public-hosted", "public"]) {
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
  const social = await socialImageArtwork(page);
  await page.setViewportSize({ width: 1200, height: 630 });
  await page.setContent(`<style>body{margin:0}</style>${social}`);
  await page.evaluate(() => document.fonts.ready);
  const image = await page.screenshot();
  for (const directory of ["public-hosted", "public"]) {
    await mkdir(resolve(directory, "social"), { recursive: true });
    await writeFile(resolve(directory, "social/drowse.svg"), social);
    await writeFile(resolve(directory, "social/drowse.png"), image);
  }
  }
} finally { await browser.close(); }
