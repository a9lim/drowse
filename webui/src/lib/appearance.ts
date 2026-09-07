export type BackgroundEffect = "original" | "pixel" | "dither";

export interface BackgroundSettings {
  effect: BackgroundEffect;
  pixelSize: number;
  visibility: number;
}

export const DEFAULT_BACKGROUND: BackgroundSettings = { effect: "dither", pixelSize: 3, visibility: 0.1 };
export const BACKGROUND_MAX_BYTES = 10 * 1024 * 1024;

export function backgroundSettings(value: Partial<BackgroundSettings>): BackgroundSettings {
  return {
    effect: value.effect === "original" || value.effect === "pixel" ? value.effect : "dither",
    pixelSize: Number.isFinite(value.pixelSize) ? Math.min(12, Math.max(1, Math.round(value.pixelSize!))) : 3,
    visibility: Number.isFinite(value.visibility) ? Math.min(0.14, Math.max(0.04, value.visibility!)) : 0.1,
  };
}

export function validateBackgroundFile(file: { type: string; size: number }): void {
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) throw new Error("Choose a PNG, JPEG, or WebP image.");
  if (!file.size || file.size > BACKGROUND_MAX_BYTES) throw new Error("Choose an image smaller than 10 MB.");
}

export function backgroundDimensions(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, 2048 / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}
