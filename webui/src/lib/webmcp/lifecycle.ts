export interface PageActions {
  state(): { route: string; recovery: string | null };
  home(): Promise<void> | void;
  models(): Promise<void> | void;
  leave(url: string): Promise<void>;
  retry(): Promise<void>;
  allowTakeover(): Promise<void>;
  denyTakeover(): void;
}

export interface UpdateActions {
  state(): { available: boolean; applying: boolean; error: string | null };
  apply(): Promise<void>;
  remindLater(): void;
}

let page: PageActions | null = null;
let update: UpdateActions | null = null;
let modelPicker: ((id: string) => void) | null = null;

export function selectVisibleModel(id: string): boolean {
  if (!modelPicker) return false;
  modelPicker(id);
  return true;
}
export function registerModelPicker(select: (id: string) => void): () => void {
  modelPicker = select;
  return () => { if (modelPicker === select) modelPicker = null; };
}

export function pageActions(): PageActions | null { return page; }
export function updateActions(): UpdateActions | null { return update; }
export function registerPageActions(actions: PageActions): () => void {
  page = actions;
  return () => { if (page === actions) page = null; };
}
export function registerUpdateActions(actions: UpdateActions): () => void {
  update = actions;
  return () => { if (update === actions) update = null; };
}
