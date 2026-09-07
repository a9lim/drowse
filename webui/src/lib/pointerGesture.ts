export interface GesturePoint {
  x: number;
  y: number;
}

export interface PinchMetrics {
  center: GesturePoint;
  distance: number;
}

export function pinchMetrics(first: GesturePoint, second: GesturePoint): PinchMetrics {
  const dx = second.x - first.x;
  const dy = second.y - first.y;
  return {
    center: {
      x: first.x + dx / 2,
      y: first.y + dy / 2,
    },
    distance: Math.hypot(dx, dy),
  };
}

export function scaleFromPinch(
  initialScale: number,
  initialDistance: number,
  currentDistance: number,
  minimum: number,
  maximum: number,
): number {
  if (
    !Number.isFinite(initialScale) ||
    !Number.isFinite(initialDistance) ||
    !Number.isFinite(currentDistance) ||
    initialDistance <= 0 ||
    currentDistance <= 0
  ) {
    return Math.min(maximum, Math.max(minimum, initialScale));
  }
  return Math.min(
    maximum,
    Math.max(minimum, initialScale * currentDistance / initialDistance),
  );
}
