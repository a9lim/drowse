import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({
  configFile: false, appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const originalStorage = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
const values = new Map();
try {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: key => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, value),
      removeItem: key => values.delete(key),
    },
  });
  const reminder = await server.ssrLoadModule("/src/hosted/runtime/updateReminder.ts");
  const fresh = { shown: false, deferrals: 0, remindAt: 0 };
  assert.deepEqual(reminder.readUpdateReminder(), fresh);
  values.set("polythetic.pwa-update-reminder.v1", JSON.stringify({ shown: true, deferrals: 2, remindAt: 1234 }));
  assert.equal(reminder.readUpdateReminder().remindAt, 1234);
  reminder.clearUpdateReminder();
  assert.deepEqual(reminder.readUpdateReminder(), fresh);
  reminder.markUpdateShown();
  assert.equal(reminder.readUpdateReminder().shown, true);
  let now = 1_800_000_000_000;
  for (const [index, hours] of [1, 6, 24, 24, 24].entries()) {
    assert.equal(reminder.deferUpdate(now), hours === 24 ? "1 day" : `${hours} ${hours === 1 ? "hour" : "hours"}`);
    assert.deepEqual(reminder.readUpdateReminder(), {
      shown: true, deferrals: Math.min(index + 1, 3), remindAt: now + hours * 3_600_000,
    });
    now += hours * 3_600_000;
  }
  reminder.clearUpdateReminder();
  assert.deepEqual(reminder.readUpdateReminder(), fresh);
  for (const bad of ["null", "[]", "{}", '{"shown":true,"deferrals":-1,"remindAt":0}', '{"shown":true,"deferrals":1,"remindAt":1e100}']) {
    values.set(reminder.UPDATE_REMINDER_KEY, bad);
    assert.deepEqual(reminder.readUpdateReminder(), fresh);
  }
  values.set(reminder.UPDATE_REMINDER_KEY, "broken-json");
  assert.deepEqual(reminder.readUpdateReminder(), fresh);
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() { throw new DOMException("blocked", "SecurityError"); },
  });
  reminder.markUpdateShown();
  assert.equal(reminder.readUpdateReminder().shown, true);
  assert.equal(reminder.deferUpdate(now), "1 hour");
  assert.equal(reminder.readUpdateReminder().remindAt, now + 3_600_000);
  reminder.clearUpdateReminder();
  assert.deepEqual(reminder.readUpdateReminder(), fresh);
  console.log("update reminders: persistent timing, escalation, reset, corrupt and blocked storage passed");
} finally {
  await server.close();
  if (originalStorage) Object.defineProperty(globalThis, "localStorage", originalStorage);
  else delete globalThis.localStorage;
}
