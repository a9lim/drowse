import type { ManifoldInfo } from "../types";

/** The one fit-mode classifier.  Every surface that routes a manifold by
 *  its geometry (the subspace vs manifold drawers, the inspector's
 *  diagnostics gate, the builder's post-create jump, WebMCP rack adds)
 *  goes through here instead of re-deriving it from ``fit_mode``.
 *
 *  - ``flat``: an affine fit (``pca`` / ``baked``) — the subspace family.
 *  - ``curved``: an RBF surface (``spectral`` / ``authored``) — the
 *    manifold family.
 *  - ``unresolved``: an ``auto`` discover folder that is not fitted for
 *    this model yet; its geometry is chosen per-model at fit time
 *    (``resolved_fit_mode`` stays null until then). */
export type FitGeometry = "flat" | "curved" | "unresolved";

type FitModeFields = Pick<ManifoldInfo, "fit_mode" | "resolved_fit_mode">;

export function fitGeometry(m: FitModeFields): FitGeometry {
  const mode = m.resolved_fit_mode ?? m.fit_mode;
  if (mode === "pca" || mode === "baked") return "flat";
  if (mode === "spectral" || mode === "authored") return "curved";
  return "unresolved";
}

/** Discover folders (``pca`` / ``spectral`` / ``auto``) derive their node
 *  coords per model and carry fit diagnostics; authored and baked
 *  folders do not.  The folder-level answer rides ``is_discover``; the
 *  per-tensor fit records only carry ``fit_mode``. */
export function isDiscoverFitMode(mode: string): boolean {
  return mode === "pca" || mode === "spectral" || mode === "auto";
}

export function isDiscoverFit(m: Pick<ManifoldInfo, "is_discover">): boolean {
  return m.is_discover;
}

/** The steering drawer a manifold belongs in.  An unresolved ``auto``
 *  folder is listed in both; this picks the subspace drawer for it. */
export function steeringDrawerFor(m: FitModeFields | undefined): "manifolds" | "subspace" {
  return m && fitGeometry(m) === "curved" ? "manifolds" : "subspace";
}
