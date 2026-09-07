import { migrateLegacyStorageItem, removeLegacyStorageItem } from "./brandMigration";

const OFFLINE_SESSION_KEY = "drowse.hosted.offline";
let offlineSeen = false;

export function hostedNetworkIsOffline(): boolean {
  return navigator.onLine === false || offlineSeen || storedOfflineState();
}

export function startHostedNetworkStateTracking(): () => void {
  const markOffline = () => {
    offlineSeen = true;
    writeOfflineState(true);
  };
  const markOnline = () => {
    offlineSeen = false;
    writeOfflineState(false);
  };
  if (navigator.onLine === false) markOffline();
  // Keep an offline event observed by the previous document across a
  // service-worker navigation. Chromium can report navigator.onLine=true in
  // the newly restored document even while the browser context is still
  // offline. A real online event clears the hint below.
  else if (!storedOfflineState()) markOnline();
  window.addEventListener("offline", markOffline);
  window.addEventListener("online", markOnline);
  return () => {
    window.removeEventListener("offline", markOffline);
    window.removeEventListener("online", markOnline);
  };
}

function storedOfflineState(): boolean {
  try {
    return typeof sessionStorage !== "undefined" &&
      migrateLegacyStorageItem(sessionStorage, OFFLINE_SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

function writeOfflineState(offline: boolean): void {
  try {
    if (typeof sessionStorage === "undefined") return;
    if (offline) sessionStorage.setItem(OFFLINE_SESSION_KEY, "1");
    else removeLegacyStorageItem(sessionStorage, OFFLINE_SESSION_KEY);
  } catch {
    return;
  }
}
