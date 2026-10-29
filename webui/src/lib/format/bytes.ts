const UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

// Decimal SI sizes, matching how browsers and operating systems report storage.
// A value moves to the next unit once it would round to 1000, so a model never
// reads as "1000 MB". Values under 10 keep one decimal; larger values are whole.
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1000) return `${Math.round(bytes)} B`;
  let unit = 0;
  let value = bytes;
  while (unit < UNITS.length - 1 && roundForDisplay(value) >= 1000) {
    value /= 1000;
    unit += 1;
  }
  return `${formatValue(value)} ${UNITS[unit]}`;
}

function roundForDisplay(value: number): number {
  return value < 9.95 ? Math.round(value * 10) / 10 : Math.round(value);
}

function formatValue(value: number): string {
  return value < 9.95 ? value.toFixed(1) : Math.round(value).toString();
}
