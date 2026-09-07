import { migrateLegacyStorageItem, removeLegacyStorageItem } from "./brandMigration";

const RECOVERY_KEY = "drowse:chunk-recovery";
const RECOVERY_PARAM = "app-recovery";
const RECOVERY_WINDOW_MS = 45_000;
const SERVICE_WORKER_WAIT_MS = 2_500;
const PREPARE_WAIT_MS = 4_000;

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

const wait = (milliseconds: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

async function settleWithTimeout(task: Promise<unknown>, milliseconds: number): Promise<void> {
  await Promise.race([task.catch(() => undefined), wait(milliseconds)]);
}

function waitForWorkerState(worker: ServiceWorker): Promise<void> {
  return new Promise((resolve) => {
    const finish = () => {
      window.clearTimeout(timeout);
      worker.removeEventListener("statechange", onStateChange);
      resolve();
    };
    const onStateChange = () => {
      if (worker.state === "installed" || worker.state === "activated" ||
        worker.state === "redundant") finish();
    };
    const timeout = window.setTimeout(finish, SERVICE_WORKER_WAIT_MS);
    worker.addEventListener("statechange", onStateChange);
  });
}

function waitForControllerChange(previousController: ServiceWorker | null): Promise<void> {
  return new Promise((resolve) => {
    const finish = () => {
      window.clearTimeout(timeout);
      navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
      resolve();
    };
    const onControllerChange = () => {
      if (navigator.serviceWorker.controller !== previousController) finish();
    };
    const timeout = window.setTimeout(finish, SERVICE_WORKER_WAIT_MS);
    navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);
  });
}

async function refreshServiceWorker(): Promise<void> {
  if (!("serviceWorker" in navigator)) return;
  const registration = await navigator.serviceWorker.getRegistration("/").catch(() => undefined);
  if (!registration) return;

  await settleWithTimeout(registration.update(), SERVICE_WORKER_WAIT_MS);
  const installing = registration.installing;
  if (installing && installing.state !== "installed" && installing.state !== "activated") {
    await waitForWorkerState(installing);
  }

  const waiting = registration.waiting;
  if (!waiting) return;
  const previousController = navigator.serviceWorker.controller;
  const controllerChanged = waitForControllerChange(previousController);
  waiting.postMessage({ type: "SKIP_WAITING" });
  await controllerChanged;
}

export async function recoverFromChunkLoadError(
  error: unknown,
  stage: ChunkRecoveryStage,
  prepare?: () => Promise<void>,
): Promise<boolean> {
  if (!isChunkLoadError(error) || recoveryStarted || navigator.onLine === false) return false;
  const previous = readRecovery();
  if (
    previous?.stage === stage &&
    previous.pathname === window.location.pathname &&
    Date.now() - previous.createdAt < RECOVERY_WINDOW_MS
  ) return false;

  recoveryStarted = true;
  writeRecovery(stage);
  if (prepare) await settleWithTimeout(prepare(), PREPARE_WAIT_MS);
  await refreshServiceWorker();

  const url = new URL(window.location.href);
  url.searchParams.set(RECOVERY_PARAM, String(Date.now()));
  window.location.replace(url.href);
  return true;
}

export function completeChunkRecovery(stage: ChunkRecoveryStage): void {
  clearRecoveryRecord(stage);
  const url = new URL(window.location.href);
  if (!url.searchParams.has(RECOVERY_PARAM)) return;
  url.searchParams.delete(RECOVERY_PARAM);
  window.history.replaceState(window.history.state, "", url);
}

export const chunkLoadFailureMessage =
  "Drowse updated while this tab was open. Reload the page to open the current version.";
