export const MOBILE_SHEET_QUERY = "(max-width: 760px), (pointer: coarse) and (max-width: 1024px) and (max-height: 600px)";
export const SHEET_STOPS = ["full", "half", "peek"] as const;
export type SheetStop = typeof SHEET_STOPS[number];
export type SheetPositions = Record<SheetStop, number>;
export type VelocitySample = { y: number; time: number };

export function sheetPositions(height: number, safeTop = 0): SheetPositions {
  const full = Math.max(12, safeTop);
  const peek = Math.max(full, height - Math.min(220, height * 0.42));
  return { full, half: Math.max(full, Math.min(height * 0.45, peek - 48)), peek };
}

export function nearestSheetStop(y: number, positions: SheetPositions): SheetStop {
  return SHEET_STOPS.reduce((best, stop) => Math.abs(positions[stop] - y) < Math.abs(positions[best] - y) ? stop : best, "full");
}

export function rubberBandSheet(y: number, positions: SheetPositions): number {
  const stretch = (distance: number) => 55 * Math.log1p(distance / 55);
  if (y < positions.full) return positions.full - stretch(positions.full - y);
  if (y > positions.peek) return positions.peek + stretch(y - positions.peek);
  return y;
}

export function sampleSheetVelocity(samples: VelocitySample[], y: number, time: number): number {
  samples.push({ y, time });
  while (samples.length > 1 && samples[0].time < time - 100) samples.shift();
  if (samples.length > 8) samples.shift();
  const first = samples[0];
  return (y - first.y) / Math.max(16, time - first.time) * 1000;
}

export function releaseSheet(start: SheetStop, y: number, delta: number, velocity: number, positions: SheetPositions, height: number): SheetStop | "dismiss" {
  const index = SHEET_STOPS.indexOf(start);
  const threshold = Math.min(80, Math.max(50, height * 0.08));
  // Dismiss only from the compact stop; collapsing a taller sheet stays reversible.
  if (start === "peek" && delta > 20 && y + Math.max(0, velocity) * 0.2 > positions.peek + threshold) return "dismiss";
  const direction = Math.abs(velocity) > 150 ? Math.sign(velocity) : Math.sign(delta);
  if (Math.abs(delta) > 20 && Math.abs(velocity) > 600) return direction > 0 ? "peek" : "full";
  if (Math.abs(delta) > 20 && (Math.abs(velocity) > 150 || Math.abs(delta) > 80)) {
    return SHEET_STOPS[Math.max(0, Math.min(2, index + direction))];
  }
  const nearest = SHEET_STOPS.indexOf(nearestSheetStop(y, positions));
  return SHEET_STOPS[Math.max(index - 1, Math.min(index + 1, nearest))];
}

export function sheetSpring(from: number, target: number, velocity: number, seconds: number): number {
  const displacement = from - target;
  const decay = 30;
  return target + (displacement + (velocity + decay * displacement) * seconds) * Math.exp(-decay * seconds);
}

export function sheetSpringVelocity(from: number, target: number, velocity: number, seconds: number): number {
  const decay = 30;
  return (velocity - decay * (velocity + decay * (from - target)) * seconds) * Math.exp(-decay * seconds);
}

export function canScrollSheetContent(target: Element, boundary: HTMLElement, direction: number): boolean {
  for (let element: Element | null = target; element && boundary.contains(element); element = element.parentElement) {
    const style = getComputedStyle(element);
    if (/(auto|scroll)/.test(style.overflowY) && element.scrollHeight > element.clientHeight + 1) {
      if (direction > 0 && element.scrollTop > 1) return true;
      if (direction < 0 && element.scrollTop + element.clientHeight < element.scrollHeight - 1) return true;
    }
    if (element === boundary) break;
  }
  return false;
}
