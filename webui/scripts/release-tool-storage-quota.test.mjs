import assert from "node:assert/strict";
import {
  grantReleaseToolStorageQuota,
  releaseToolQuotaBytes,
} from "./release-tool-storage-quota.mjs";

assert.equal(releaseToolQuotaBytes([{ bytes: 1 }]), 8 * 1024 ** 3);
assert.equal(
  releaseToolQuotaBytes([{ bytes: 4 * 1024 ** 3 }, { bytes: 1 }]),
  10 * 1024 ** 3 + 2,
);
assert.throws(() => releaseToolQuotaBytes([]), /at least one artifact/);
assert.throws(
  () => releaseToolQuotaBytes([{ bytes: 1.5 }]),
  /exact artifact byte sizes/,
);

const calls = [];
const session = {
  async send(method, payload) {
    calls.push([method, payload]);
    return method === "Storage.getUsageAndQuota"
      ? { quota: 8 * 1024 ** 3, usage: 0, overrideActive: true }
      : {};
  },
  async detach() {
    calls.push(["detach"]);
  },
};
const page = {
  context() {
    return {
      async newCDPSession(candidate) {
        assert.equal(candidate, page);
        return session;
      },
    };
  },
};
assert.equal(
  await grantReleaseToolStorageQuota(page, "https://127.0.0.1:1234", [{ bytes: 1 }]),
  8 * 1024 ** 3,
);
assert.deepEqual(calls, [
  [
    "Storage.overrideQuotaForOrigin",
    { origin: "https://127.0.0.1:1234", quotaSize: 8 * 1024 ** 3 },
  ],
  [
    "Storage.getUsageAndQuota",
    { origin: "https://127.0.0.1:1234" },
  ],
  ["detach"],
]);

process.stdout.write("release-tool storage quota tests passed\n");
