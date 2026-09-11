import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.addLocatorHandler(page.locator(".pwa-notice.passive"), async notice => {
    await notice.getByRole("button", { name: "Dismiss", exact: true }).click();
  });
  await page.addInitScript(() => {
    localStorage.setItem("drowse.theme", "light");
    const draw = WebGL2RenderingContext.prototype.drawArrays;
    WebGL2RenderingContext.prototype.drawArrays = function (...args) {
      draw.apply(this, args);
      if (this.getParameter(this.FRAMEBUFFER_BINDING) !== null) return;
      const canvas = this.canvas as HTMLCanvasElement;
      const pixel = new Uint8Array(4);
      this.readPixels(0, 0, 1, 1, this.RGBA, this.UNSIGNED_BYTE, pixel);
      canvas.dataset.pixel = Array.from(pixel).join(",");
      canvas.dataset.drawnSize = `${canvas.width}x${canvas.height}`;
    };
  });
});

test("the live canvas paints its own light and dark palettes without an SVG filter", async ({ page }) => {
  await page.goto("/");
  const hero = page.locator(".hero-visual");
  const canvas = hero.locator("canvas");
  await expect(hero).toHaveAttribute("data-shader-status", "ready");
  await expect(hero.locator(".visual-layer")).toHaveCSS("filter", "none");
  for (const theme of ["light", "dark", "light", "dark"] as const) {
    await page.getByRole("button", { name: theme === "light" ? "Light" : "Dark", exact: true }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await expect.poll(async () => {
      const [red, green, blue, alpha] = (await canvas.getAttribute("data-pixel") ?? "").split(",").map(Number);
      return alpha === 255 && (theme === "light"
        ? red >= 156 && green >= 120 && blue >= 218
        : red <= 111 && green <= 111 && blue <= 111);
    }).toBe(true);
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByRole("button", { name: "Light", exact: true }).click();
  await expect.poll(async () => Number((await canvas.getAttribute("data-pixel"))?.split(",")[2])).toBeGreaterThan(218);
});

test("resizing redraws before paint and scrolling does not resize the drawing buffer", async ({ page }) => {
  test.slow();
  await page.goto("/");
  const hero = page.locator(".hero-visual");
  await expect(hero).toHaveAttribute("data-shader-status", "ready");
  await page.evaluate(() => {
    const canvas = document.querySelector<HTMLCanvasElement>(".hero-visual canvas")!;
    new ResizeObserver(() => {
      if (canvas.dataset.drawnSize !== `${canvas.width}x${canvas.height}`) {
        canvas.dataset.blankResize = "true";
      }
    }).observe(canvas);
  });
  for (const height of [780, 900, 844]) {
    await page.setViewportSize({ width: 390, height });
    await expect.poll(() => hero.evaluate(el => el.clientHeight)).toBe(height);
    await expect.poll(() => hero.evaluate(el => {
      const canvas = el.querySelector("canvas")!;
      return Math.abs(canvas.width * el.clientHeight - canvas.height * el.clientWidth)
        <= Math.max(el.clientWidth, el.clientHeight);
    })).toBe(true);
    await expect(hero.locator("canvas")).not.toHaveAttribute("data-blank-resize");
  }
  const size = await hero.locator("canvas").getAttribute("data-drawn-size");
  for (const scrollY of [400, 900, 1600, 500, 0]) {
    await page.evaluate(y => window.scrollTo({ top: y, behavior: "instant" }), scrollY);
    await expect.poll(() => hero.getAttribute("data-travel")).toBe((Math.min(1, scrollY / (844 * 1.8))).toFixed(3));
    await expect(hero.locator("canvas")).toHaveAttribute("data-drawn-size", size!);
  }
});

test("reduced motion keeps a themed static fallback without starting WebGL", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const hero = page.locator(".hero-visual");
  await expect(hero).toHaveAttribute("data-shader-status", "fallback");
  await expect(hero.locator(".fallback")).toHaveCSS("opacity", "1");
  await expect(hero.locator(".fallback")).toHaveCSS("filter", /hero-light-palette/);
  await page.getByRole("button", { name: "Dark", exact: true }).click();
  await expect(hero.locator(".fallback")).toHaveCSS("filter", /hero-dark-palette/);
  await expect(hero.locator("canvas")).not.toHaveAttribute("data-pixel");
});

test("the credits shader stays contained and receives theme updates", async ({ page }) => {
  await page.goto("/credits");
  const hero = page.locator(".hero-visual.contained");
  await expect(hero).toHaveAttribute("data-shader-status", "ready");
  await expect(hero).toHaveCSS("position", "absolute");
  await expect.poll(() => hero.evaluate(el => el.clientHeight)).toBeLessThan(844);
  await expect.poll(async () => Number((await hero.locator("canvas").getAttribute("data-pixel"))?.split(",")[2])).toBeGreaterThan(218);
  await page.getByRole("button", { name: "Dark", exact: true }).click();
  await expect.poll(async () => Number((await hero.locator("canvas").getAttribute("data-pixel"))?.split(",")[2])).toBeLessThan(112);
});
