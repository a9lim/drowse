import { migrateLegacyStorageItem, removeLegacyStorageItem } from "./brandMigration";

const RECOVERY_KEY = "drowse:chunk-recovery";
const RECOVERY_PARAM = "app-recovery";
const RECOVERY_WINDOW_MS = 45_000;
const RECOVERY_TIMEOUT_MS = 10_000;

export type ChunkRecoveryStage = "bootstrap" | "workbench";

interface ChunkRecoveryRecord {
  stage: ChunkRecoveryStage;
  pathname: string;
  createdAt: number;
}

let recoveryStarted = false;

const errorText = (error: unknown): string =>
  error instanceof Error ? `${error.name}: ${error.message}` : String(error);

export function isChunkLoadError(error: unknown): boolean {
  const message = errorText(error).toLowerCase();
  return [
    "failed to fetch dynamically imported module",
    "error loading dynamically imported module",
    "importing a module script failed",
    "failed to load module script",
    "unable to preload css",
    "load failed for module",
  ].some((fragment) => message.includes(fragment));
}

function readRecovery(): ChunkRecoveryRecord | null {
  try {
    const value = migrateLegacyStorageItem(sessionStorage, RECOVERY_KEY);
    if (!value) return null;
    const record = JSON.parse(value) as Partial<ChunkRecoveryRecord>;
    if (
      (record.stage !== "bootstrap" && record.stage !== "workbench") ||
      typeof record.pathname !== "string" ||
      typeof record.createdAt !== "number"
    ) return null;
    return record as ChunkRecoveryRecord;
  } catch {
    return null;
  }
}

function writeRecovery(stage: ChunkRecoveryStage): void {
  try {
    sessionStorage.setItem(RECOVERY_KEY, JSON.stringify({
      stage,
      pathname: window.location.pathname,
      createdAt: Date.now(),
    } satisfies ChunkRecoveryRecord));
  } catch {}
}

function clearRecoveryRecord(stage: ChunkRecoveryStage): void {
  try {
    const record = readRecovery();
    if (!record || record.stage === stage) removeLegacyStorageItem(sessionStorage, RECOVERY_KEY);
  } catch {}
}

async function withTimeout<T>(task: Promise<T>, message: string): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      task,
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(new Error(message)), RECOVERY_TIMEOUT_MS);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

async function clearAppShell(): Promise<void> {
  if ("serviceWorker" in navigator) {
    const registration = await navigator.serviceWorker.getRegistration("/");
    if (registration?.scope === new URL("/", location.href).href) {
      const workers = [registration.active, registration.waiting, registration.installing];
      const script = new URL("/sw.js", location.href).href;
      const owned = workers.some(worker => worker?.scriptURL === script);
      if (owned) await registration.unregister();
    }
  }
  if (typeof caches === "undefined") return;
  for (const name of await caches.keys()) {
    // These are app code caches, never the model, chat, or artifact stores.
    if (/^(?:drowse|saklas|polythetic)-hosted-(?:precache-v\d+-|on-demand-assets-v\d+$|landing-shader-v\d+$)/.test(name)) {
      await caches.delete(name);
    }
  }
}

export async function reloadHostedApp(
  stage: ChunkRecoveryStage,
  prepare?: () => Promise<void>,
): Promise<void> {
  if (recoveryStarted) return;
  if (navigator.onLine === false) throw new Error("You’re offline. Reconnect, then reload Drowse.");
  recoveryStarted = true;
  try {
    const checkUrl = new URL("/index.html", location.href);
    checkUrl.searchParams.set(RECOVERY_PARAM, String(Date.now()));
    let response: Response;
    try {
      response = await fetch(checkUrl, { cache: "no-store", signal: AbortSignal.timeout(RECOVERY_TIMEOUT_MS) });
    } catch {
      throw new Error("Drowse could not reach the site. Check your connection, then try again.");
    }
    if (!response.ok || !response.headers.get("content-type")?.includes("text/html") ||
      !(await response.text()).includes('name="drowse-source-revision"')) {
      throw new Error("The current app files are unavailable. Please try again in a moment.");
    }
    if (prepare) await withTimeout(prepare(), "Your work could not be saved in time. Reload was cancelled; please try again.");
    await withTimeout(clearAppShell(), "App refresh did not finish. Close other Drowse tabs, then try again.");
    writeRecovery(stage);
    const url = new URL(window.location.href);
    url.searchParams.set(RECOVERY_PARAM, `${stage}:${Date.now()}`);
    window.location.replace(url.href);
  } catch (error) {
    recoveryStarted = false;
    throw error;
  }
}

export async function recoverFromChunkLoadError(
  error: unknown,
  stage: ChunkRecoveryStage,
  prepare?: () => Promise<void>,
): Promise<boolean> {
  if (!isChunkLoadError(error) || recoveryStarted || navigator.onLine === false) return false;
  if (new URL(location.href).searchParams.get(RECOVERY_PARAM)?.startsWith(`${stage}:`)) return false;
  const previous = readRecovery();
  if (
    previous?.stage === stage &&
    previous.pathname === window.location.pathname &&
    Date.now() - previous.createdAt < RECOVERY_WINDOW_MS
  ) return false;

  try {
    await reloadHostedApp(stage, prepare);
    return true;
  } catch {
    return false;
  }
}

export function completeChunkRecovery(stage: ChunkRecoveryStage): void {
  clearRecoveryRecord(stage);
  const url = new URL(window.location.href);
  const recovery = url.searchParams.get(RECOVERY_PARAM);
  if (!recovery || (recovery.startsWith("workbench:") && stage !== "workbench")) return;
  url.searchParams.delete(RECOVERY_PARAM);
  window.history.replaceState(window.history.state, "", url);
}

export const chunkLoadFailureMessage =
  "Some app files could not load. This can happen after an update or a connection problem. Reload Drowse to refresh its app files.";

export function reloadInstructions(): string {
  const mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    (/Mac/i.test(navigator.platform) && navigator.maxTouchPoints > 1);
  if (mobile) return "If this keeps happening, use Reload in your browser’s menu. You can also close this tab and reopen Drowse.";
  const shortcut = /Mac/i.test(navigator.platform) ? "⌘ + R" : "Ctrl + R";
  return `If this keeps happening, press ${shortcut} to reload the page. If it still cannot open, close other Drowse tabs and try again.`;
}

export const appRefreshSafetyMessage = "Only app files are refreshed. Your saved chats and downloaded models stay on this device.";
