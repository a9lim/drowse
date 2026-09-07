import { DEFAULT_BACKGROUND, backgroundDimensions, backgroundSettings, validateBackgroundFile, type BackgroundSettings } from "../appearance";

interface StoredAppearance extends BackgroundSettings { image: Blob | null; name: string }
export const appearanceState = $state({ ...DEFAULT_BACKGROUND, url: "", name: "", ready: false, busy: false, error: "" });
let stored: StoredAppearance = { ...DEFAULT_BACKGROUND, image: null, name: "" };
let loading: Promise<void> | null = null;

async function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("drowse-appearance", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("settings");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function apply(value: StoredAppearance): void {
  if (stored.image !== value.image || !appearanceState.url) {
    if (appearanceState.url) URL.revokeObjectURL(appearanceState.url);
    appearanceState.url = value.image ? URL.createObjectURL(value.image) : "";
  }
  stored = value;
  Object.assign(appearanceState, backgroundSettings(value), { name: value.name });
}

export function loadAppearance(): Promise<void> {
  return loading ??= (async () => {
    try {
      const db = await database();
      const value = await new Promise<StoredAppearance | undefined>((resolve, reject) => {
        const transaction = db.transaction("settings", "readonly");
        const request = transaction.objectStore("settings").get("background");
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
        transaction.oncomplete = () => db.close();
        transaction.onabort = () => db.close();
      });
      if (value) apply({ ...backgroundSettings(value), image: value.image instanceof Blob ? value.image : null, name: typeof value.name === "string" ? value.name : "" });
    } catch {
      appearanceState.error = "Background storage is unavailable in this browser. Your workspace still works without an image.";
    } finally { appearanceState.ready = true; }
  })();
}

async function persist(value: StoredAppearance): Promise<void> {
  const db = await database();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction("settings", "readwrite");
    transaction.objectStore("settings").put(value, "background");
    transaction.oncomplete = () => { db.close(); resolve(); };
    transaction.onabort = () => { db.close(); reject(transaction.error); };
  });
  apply(value);
}

export async function updateBackground(change: Partial<BackgroundSettings>): Promise<void> {
  if (appearanceState.busy) return;
  appearanceState.busy = true;
  appearanceState.error = "";
  try {
    await loadAppearance();
    await persist({ ...stored, ...backgroundSettings({ ...stored, ...change }) });
  } catch { appearanceState.error = "The background setting could not be saved. Try again or free some browser storage."; }
  finally { appearanceState.busy = false; }
}

export async function uploadBackground(file: File): Promise<void> {
  if (appearanceState.busy) return;
  appearanceState.busy = true;
  appearanceState.error = "";
  try {
    validateBackgroundFile(file);
    await loadAppearance();
    const bitmap = await createImageBitmap(file);
    let image: Blob;
    try {
      const size = backgroundDimensions(bitmap.width, bitmap.height);
      const canvas = document.createElement("canvas");
      canvas.width = size.width;
      canvas.height = size.height;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Image processing is unavailable in this browser.");
      context.drawImage(bitmap, 0, 0, size.width, size.height);
      image = await new Promise<Blob>((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("This image could not be processed.")), "image/webp", 0.85));
    } finally { bitmap.close(); }
    await persist({ ...stored, image, name: file.name.slice(0, 160) });
  } catch (error) {
    appearanceState.error = error instanceof Error && (error.message.startsWith("Choose ") || error.message.startsWith("Image processing") || error.message.startsWith("This image"))
      ? error.message : "The image could not be saved. Try another image or free some browser storage.";
  } finally { appearanceState.busy = false; }
}

export async function removeBackground(): Promise<void> {
  if (appearanceState.busy) return;
  appearanceState.busy = true;
  appearanceState.error = "";
  try {
    await loadAppearance();
    await persist({ ...stored, image: null, name: "" });
  } catch { appearanceState.error = "The background could not be removed. Try again."; }
  finally { appearanceState.busy = false; }
}
