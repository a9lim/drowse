export const ROLE_SLUG_PATTERN = /^[a-z0-9._-]+$/;

export function isRoleSlug(value: unknown): value is string {
  return typeof value === "string" && ROLE_SLUG_PATTERN.test(value);
}
