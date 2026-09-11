export interface LoomCamera {
  x: number;
  y: number;
  zoom: number;
}

export interface LoomSize {
  width: number;
  height: number;
}

export interface LoomPoint {
  x: number;
  y: number;
}

export const LOOM_MIN_ZOOM = 0.01;
export const LOOM_MAX_ZOOM = 1.4;

export function clampLoomZoom(value: number): number {
  return Math.min(LOOM_MAX_ZOOM, Math.max(LOOM_MIN_ZOOM, value));
}

export function fitLoomCamera(
  viewport: LoomSize,
  graph: LoomSize,
  padding = Math.min(48, viewport.width / 10, viewport.height / 10),
): LoomCamera {
  if (viewport.width <= 0 || viewport.height <= 0 || graph.width <= 0 || graph.height <= 0) {
    return { x: 0, y: 0, zoom: 1 };
  }
  const availableWidth = Math.max(1, viewport.width - padding * 2);
  const availableHeight = Math.max(1, viewport.height - padding * 2);
  const zoom = clampLoomZoom(Math.min(
    1,
    availableWidth / graph.width,
    availableHeight / graph.height,
  ));
  const scaledWidth = graph.width * zoom;
  const scaledHeight = graph.height * zoom;
  return {
    x: scaledWidth <= availableWidth ? (viewport.width - scaledWidth) / 2 : padding,
    y: scaledHeight <= availableHeight ? (viewport.height - scaledHeight) / 2 : padding,
    zoom,
  };
}

export function centerLoomRect(
  viewport: LoomSize,
  rect: LoomPoint & LoomSize,
  zoom: number,
): LoomCamera {
  const nextZoom = clampLoomZoom(zoom);
  return {
    x: viewport.width / 2 - (rect.x + rect.width / 2) * nextZoom,
    y: viewport.height / 2 - (rect.y + rect.height / 2) * nextZoom,
    zoom: nextZoom,
  };
}

export function resizeLoomCamera(
  camera: LoomCamera,
  previous: LoomSize,
  viewport: LoomSize,
): LoomCamera {
  return {
    x: camera.x + (viewport.width - previous.width) / 2,
    y: camera.y + (viewport.height - previous.height) / 2,
    zoom: camera.zoom,
  };
}

export function zoomLoomAt(
  camera: LoomCamera,
  nextZoom: number,
  anchor: LoomPoint,
): LoomCamera {
  const zoom = clampLoomZoom(nextZoom);
  const worldX = (anchor.x - camera.x) / camera.zoom;
  const worldY = (anchor.y - camera.y) / camera.zoom;
  return {
    x: anchor.x - worldX * zoom,
    y: anchor.y - worldY * zoom,
    zoom,
  };
}
