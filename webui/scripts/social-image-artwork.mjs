import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

export async function socialImageArtwork(page) {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const server = await createServer({
    root, configFile: false, publicDir: "public-hosted", appType: "custom", logLevel: "error",
    server: { host: "127.0.0.1", port: 0, watch: null },
  });
  server.middlewares.use((request, response, next) => {
    if (request.url !== "/__social-card__") return next();
    response.setHeader("Content-Type", "text/html");
    response.end('<!doctype html><html><head><title>Drowse social artwork</title></head><body style="margin:0"><canvas style="width:1200px;height:630px"></canvas></body></html>');
  });
  try {
    await server.listen();
    await page.setViewportSize({ width: 1200, height: 630 });
    await page.goto(new URL("/__social-card__", server.resolvedUrls.local[0]).href);
    const background = await page.evaluate(async () => {
      const { createHeroShaderSource } = await import("/src/hosted/ui/heroShaderRuntime.ts");
      const poster = new Image();
      poster.src = "/images/ethereal-orb.jpg";
      await poster.decode();
      const video = document.createElement("video");
      video.muted = true;
      video.src = "/video/ethereal-orb.mp4";
      await new Promise((resolve, reject) => {
        video.onloadeddata = resolve;
        video.onerror = () => reject(new Error("Home-screen orb video could not load"));
        video.load();
      });
      await new Promise(resolve => { video.onseeked = resolve; video.currentTime = 6; });
      const canvas = document.querySelector("canvas");
      const renderer = createHeroShaderSource(canvas, video, poster);
      try {
        renderer.resize(1200, 630);
        if (!renderer.update(0.26, 2, false, 2.2)) throw new Error("Home-screen shader did not render");
        return canvas.toDataURL("image/png");
      } finally {
        renderer.dispose();
        video.removeAttribute("src");
        video.load();
      }
    });
    const font = await readFile(new URL("../src/assets/fonts/WixMadeforDisplay-latin.woff2", import.meta.url));
    const bodyFont = await readFile(new URL("../src/assets/fonts/WixMadeforText-normal-latin.woff2", import.meta.url));
    return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <style>@font-face{font-family:Drowse;src:url(data:font/woff2;base64,${font.toString("base64")}) format('woff2');font-weight:400 800}@font-face{font-family:DrowseText;src:url(data:font/woff2;base64,${bodyFont.toString("base64")}) format('woff2');font-weight:400 800}</style>
  <image href="${background}" x="-375" y="-160" width="1860" height="976.5"/>
  <text x="76" y="322" font-family="Drowse, sans-serif" font-size="114" font-weight="750" letter-spacing="-5" fill="#ffffff">Drowse</text>
  <text x="80" y="380" font-family="DrowseText, sans-serif" font-size="30" font-weight="400" fill="#ffffff">Interpretability Workbench</text>
</svg>`;
  } finally {
    await server.close();
  }
}
