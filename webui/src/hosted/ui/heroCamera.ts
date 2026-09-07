export const HERO_FOCAL_LENGTH = 1.8;
export const HERO_VOLUME_RADIUS = 1.65;

export function heroCameraPosition(progress: number, aspect: number, mobile: boolean, entrance = 1) {
  const t = Math.max(0, Math.min(1, progress));
  const startZ = mobile ? 5 : 4.5;
  const lateral = (1 - t) ** 2;
  return [
    (mobile ? 0 : -0.48 * startZ * aspect / HERO_FOCAL_LENGTH) * lateral,
    -0.1 * startZ / HERO_FOCAL_LENGTH * lateral,
    startZ + (-0.35 - startZ) * t + 0.65 * (1 - Math.max(0, Math.min(1, entrance))) ** 3 * lateral,
  ] as const;
}
