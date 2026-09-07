import { migrateLegacyStorageItem, removeLegacyStorageItem } from "./brandMigration";

export const UPDATE_REMINDER_KEY = "drowse.pwa-update-reminder.v1";

type UpdateReminder = {
  shown: boolean;
  deferrals: number;
  remindAt: number;
};

let fallback: UpdateReminder = { shown: false, deferrals: 0, remindAt: 0 };

export function readUpdateReminder(): UpdateReminder {
  try {
    const raw = migrateLegacyStorageItem(localStorage, UPDATE_REMINDER_KEY);
    if (!raw) return { shown: false, deferrals: 0, remindAt: 0 };
    const value = JSON.parse(raw);
    if (
      typeof value?.shown === "boolean"
      && Number.isSafeInteger(value.deferrals) && value.deferrals >= 0
      && Number.isSafeInteger(value.remindAt) && value.remindAt >= 0
    ) return value;
    return { shown: false, deferrals: 0, remindAt: 0 };
  } catch {
    return fallback;
  }
}

function saveUpdateReminder(value: UpdateReminder): void {
  fallback = value;
  try {
    localStorage.setItem(UPDATE_REMINDER_KEY, JSON.stringify(value));
  } catch {
    // Blocked storage still preserves the reminder for this page session.
  }
}

export function markUpdateShown(): void {
  saveUpdateReminder({ ...readUpdateReminder(), shown: true });
}

export function deferUpdate(now = Date.now()): string {
  const previous = readUpdateReminder();
  const hours = previous.deferrals === 0 ? 1 : previous.deferrals === 1 ? 6 : 24;
  saveUpdateReminder({
    shown: true,
    deferrals: Math.min(previous.deferrals + 1, 3),
    remindAt: now + hours * 60 * 60 * 1000,
  });
  return hours === 24 ? "1 day" : `${hours} ${hours === 1 ? "hour" : "hours"}`;
}

export function clearUpdateReminder(): void {
  fallback = { shown: false, deferrals: 0, remindAt: 0 };
  try {
    removeLegacyStorageItem(localStorage, UPDATE_REMINDER_KEY);
  } catch {
    // Match the in-memory fallback when browser storage is unavailable.
  }
}
