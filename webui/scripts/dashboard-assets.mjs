import { states, animatedStates } from "./tab-icon-artwork.mjs";

// Which files of webui/public ship in the dashboard bundle that `drowse serve`
// mounts (and so in the wheel).  The public dir also holds the hosted site's
// brand set, which drowse.ai copies into its own public-hosted dir and checks
// against these sources: social preview art, the PWA icons, the legacy
// unthemed tab icons, and every animation frame of the busy tab icons.  The
// dashboard renders animated tab icons on a canvas and needs only each
// state's static (frame 0) light/dark raster as its fallback.
const dashboardIcons = new Set(["icons/tab-home.png", "icons/apple-touch-icon.png"]);
for (const state of states) {
  for (const theme of ["light", "dark"]) {
    dashboardIcons.add(`icons/tab-${state}-${theme}${animatedStates.has(state) ? "-0" : ""}.png`);
  }
}

/** Whether a public-dir file (path relative to the public dir, `/`-separated)
 *  belongs in the dashboard build. */
export function shipsInDashboard(path) {
  if (path.startsWith("social/")) return false;
  if (path.startsWith("icons/")) return dashboardIcons.has(path);
  return true;
}

/** Icons the dashboard references at runtime; the build must contain them. */
export const requiredDashboardIcons = [...dashboardIcons];
