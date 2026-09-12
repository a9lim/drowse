export interface HomeOpenResult {
  state: "opening" | "model_required";
  chat_id: string;
  model_id: string;
  model_variant_id: string | null;
}

interface HomeController {
  busy(): boolean;
  metadataDraft?(): { id: string; name: string; dirty: boolean } | null;
  metadataState?(): unknown;
  open(id: string): Promise<HomeOpenResult>;
}

let active: HomeController | null = null;

export function registerHomeController(controller: HomeController): () => void {
  active = controller;
  return () => { if (active === controller) active = null; };
}

export function getHomeController(): HomeController | null { return active; }
