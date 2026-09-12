export type ArtifactArea = "manifolds" | "templates";

const listeners = new Set<(area: ArtifactArea) => void | Promise<void>>();

export function onArtifactUpdate(listener: (area: ArtifactArea) => void | Promise<void>): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export async function notifyArtifactUpdate(area: ArtifactArea): Promise<void> {
  await Promise.all([...listeners].map((listener) => listener(area)));
}
