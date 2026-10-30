import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  assertCatalogSequenceFloor,
  fetchChecked,
  preflightArtifact,
  retryAfterMilliseconds,
  validatedApprovedUrl,
} from "./preflight-hosted-distribution.mjs";

const tests = [];
const test = (name, run) => tests.push({ name, run });
const timeoutMs = 1_000;

function corsResponse(body, { status = 200, origin = "*", headers = {} } = {}) {
  return new Response(body, {
    status,
    headers: {
      "Access-Control-Allow-Origin": origin,
      ...headers,
    },
  });
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function artifact(bytes = Uint8Array.of(1, 2, 3, 4, 5, 6), path = "weights/model.bin") {
  return {
    path,
    url: "https://source.test/model.bin",
    bytes: bytes.byteLength,
    sha256: sha256(bytes),
  };
}

function validArtifactFetcher(bytes, override = {}) {
  return async (_url, init) => {
    const range = new Headers(init.headers).get("Range");
    if (range !== null) {
      const body = override.rangeBody ?? bytes;
      return corsResponse(body, {
        status: override.rangeStatus ?? 206,
        headers: {
          "Content-Length": String(override.rangeLength ?? body.byteLength),
          "Content-Range": override.contentRange ??
            `bytes 0-${bytes.byteLength - 1}/${bytes.byteLength}`,
          "Access-Control-Expose-Headers": override.rangeExposed ??
            "Content-Length, Content-Range",
          ...(override.rangeHeaders ?? {}),
        },
      });
    }
    const body = override.fullBody ?? bytes;
    const headers = {
      "Content-Length": String(override.fullLength ?? body.byteLength),
      "Access-Control-Expose-Headers": override.fullExposed ?? "Content-Length",
      ...(override.fullHeaders ?? {}),
    };
    if (override.omitFullLength) delete headers["Content-Length"];
    return corsResponse(body, {
      status: override.fullStatus ?? 200,
      headers,
    });
  };
}

test("honors numeric and dated server retry instructions", () => {
  const now = Date.parse("2026-09-12T10:00:00Z");
  assert.equal(retryAfterMilliseconds("20", now), 20_000);
  assert.equal(retryAfterMilliseconds("Sat, 12 Sep 2026 10:00:30 GMT", now), 30_000);
  assert.equal(retryAfterMilliseconds("Sat, 12 Sep 2026 09:59:00 GMT", now), 0);
  for (const value of [null, "", "bad", "-1"]) assert.equal(retryAfterMilliseconds(value, now), null);
});

test("retries rate limiting then requires complete range and hash verification", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6);
  const valid = validArtifactFetcher(bytes);
  const delays = [];
  let calls = 0;
  await preflightArtifact(artifact(bytes), new Set(), 30_000, async (url, init) => {
    calls += 1;
    if (calls === 1) return new Response(null, { status: 429, headers: { "Retry-After": "7" } });
    return valid(url, init);
  }, async delay => { delays.push(delay); });
  assert.equal(calls, 3);
  assert.deepEqual(delays, [7_000]);
});

test("discards interrupted bodies and hashes the full retry from the start", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6);
  const valid = validArtifactFetcher(bytes);
  const delays = [];
  let calls = 0;
  await preflightArtifact(artifact(bytes), new Set(), 30_000, async (url, init) => {
    calls += 1;
    if (calls === 2) return corsResponse(new ReadableStream({
      start(controller) {
        controller.enqueue(bytes.subarray(0, 2));
        controller.error(new TypeError("terminated", { cause: { code: "UND_ERR_SOCKET" } }));
      },
    }), { headers: { "Content-Length": "6", "Access-Control-Expose-Headers": "Content-Length" } });
    return valid(url, init);
  }, async delay => { delays.push(delay); });
  assert.equal(calls, 4);
  assert.deepEqual(delays, [5_000]);
});

test("waits a full rate-limit window when the CDN omits Retry-After", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6);
  const valid = validArtifactFetcher(bytes);
  const delays = [];
  let calls = 0;
  await preflightArtifact(artifact(bytes), new Set(), 600_000, async (url, init) => {
    calls += 1;
    if (calls <= 2) return new Response(null, { status: 429 });
    return valid(url, init);
  }, async delay => { delays.push(delay); });
  assert.equal(calls, 4);
  assert.deepEqual(delays, [300_000, 300_000]);
  await assert.rejects(() => preflightArtifact(artifact(bytes), new Set(), 30_000,
    async () => new Response(null, { status: 429 }),
    async () => { throw new Error("must not exceed the timeout"); },
  ), /HTTP 429/);
});

test("bounds transient retries and never retries integrity or origin failures", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6);
  const delays = [];
  let calls = 0;
  await assert.rejects(() => preflightArtifact(artifact(bytes), new Set(), 30_000, async () => {
    calls += 1;
    return new Response(null, { status: 503 });
  }, async delay => { delays.push(delay); }), /HTTP 503/);
  assert.equal(calls, 3);
  assert.deepEqual(delays, [5_000, 10_000]);
  const noWait = () => { throw new Error("must not retry"); };
  await assert.rejects(() => preflightArtifact(
    { ...artifact(bytes), sha256: "0".repeat(64) }, new Set(), 30_000,
    validArtifactFetcher(bytes), noWait,
  ), /SHA-256 preflight/);
  await assert.rejects(() => preflightArtifact(artifact(bytes), new Set(), 30_000,
    async () => corsResponse(null, { status: 302, headers: { Location: "https://unapproved.test/weights" } }),
    noWait,
  ), /unapproved URL/);
  await assert.rejects(() => preflightArtifact(artifact(bytes), new Set(), 30_000,
    async () => new Response(null, { status: 429, headers: { "Retry-After": "600" } }), noWait,
  ), /HTTP 429/);
});

test("follows only approved HTTPS redirect chains", async () => {
  const calls = [];
  const response = await fetchChecked(
    "https://source.test/file",
    new Set(["https://cdn.test"]),
    timeoutMs,
    {},
    async (url, init) => {
      calls.push({ url, init });
      if (url === "https://source.test/file") {
        return corsResponse(null, {
          status: 302,
          headers: { Location: "https://cdn.test/first" },
        });
      }
      if (url === "https://cdn.test/first") {
        return corsResponse(null, {
          status: 307,
          headers: { Location: "/final" },
        });
      }
      return corsResponse(Uint8Array.of(1));
    },
  );
  assert.equal(response.status, 200);
  assert.deepEqual(calls.map(({ url }) => url), [
    "https://source.test/file",
    "https://cdn.test/first",
    "https://cdn.test/final",
  ]);
  for (const { init } of calls) {
    assert.equal(init.redirect, "manual");
    assert.equal(init.credentials, "omit");
    assert.equal(init.headers.Origin, "https://drowse.invalid");
  }
});

test("rejects unapproved redirect hops before requesting them", async () => {
  let calls = 0;
  await assert.rejects(
    () => fetchChecked(
      "https://source.test/file",
      new Set(["https://cdn.test"]),
      timeoutMs,
      {},
      async () => {
        calls += 1;
        return corsResponse(null, {
          status: 302,
          headers: { Location: "https://unapproved.test/file" },
        });
      },
    ),
    /redirected to unapproved URL/,
  );
  assert.equal(calls, 1);
});

test("allows exactly five approved redirects", async () => {
  let calls = 0;
  const response = await fetchChecked(
    "https://source.test/file",
    new Set(["https://cdn.test"]),
    timeoutMs,
    {},
    async () => {
      calls += 1;
      if (calls === 6) return corsResponse(Uint8Array.of(1));
      return corsResponse(null, {
        status: 302,
        headers: { Location: `https://cdn.test/hop-${calls}` },
      });
    },
  );
  assert.equal(response.status, 200);
  assert.equal(calls, 6);
});

test("rejects a sixth redirect", async () => {
  let redirects = 0;
  await assert.rejects(
    () => fetchChecked(
      "https://source.test/file",
      new Set(["https://cdn.test"]),
      timeoutMs,
      {},
      async () => {
        redirects += 1;
        return corsResponse(null, {
          status: 302,
          headers: { Location: `https://cdn.test/hop-${redirects}` },
        });
      },
    ),
    /exceeded five approved redirects/,
  );
  assert.equal(redirects, 6);
});

test("rejects insecure and credential-bearing URLs", async () => {
  let calls = 0;
  const fetcher = async () => {
    calls += 1;
    return corsResponse(null);
  };
  await assert.rejects(
    () => fetchChecked("http://source.test/file", new Set(), timeoutMs, {}, fetcher),
    /must use HTTPS/,
  );
  await assert.rejects(
    () => fetchChecked(
      "https://user:secret@source.test/file",
      new Set(),
      timeoutMs,
      {},
      fetcher,
    ),
    /redirected to unapproved URL/,
  );
  assert.equal(calls, 0);
  assert.throws(
    () => validatedApprovedUrl(
      "https://user:secret@source.test/file",
      new Set(["https://source.test"]),
      "https://source.test/file",
    ),
    /redirected to unapproved URL/,
  );
});

test("accepts wildcard or an exact echo of the requested origin", async () => {
  const echoed = await fetchChecked(
    "https://source.test/file",
    new Set(),
    timeoutMs,
    {},
    async (_url, init) => corsResponse(null, {
      origin: new Headers(init.headers).get("Origin"),
    }),
  );
  assert.equal(echoed.status, 200);

  await assert.rejects(
    () => fetchChecked(
      "https://source.test/file",
      new Set(),
      timeoutMs,
      {},
      async () => new Response(null, {
        status: 200,
        headers: { "Access-Control-Allow-Origin": "https://app.example.test" },
      }),
    ),
    /did not allow the requested browser origin/,
  );
  let calls = 0;
  await assert.rejects(
    () => fetchChecked(
      "https://source.test/file",
      new Set(["https://cdn.test"]),
      timeoutMs,
      {},
      async () => {
        calls += 1;
        return new Response(null, {
          status: 302,
          headers: { Location: "https://cdn.test/file" },
        });
      },
    ),
    /did not allow the requested browser origin/,
  );
  assert.equal(calls, 1);
});

test("accepts a complete ranged and full artifact verification", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6);
  const requests = [];
  const fetcher = validArtifactFetcher(bytes);
  await preflightArtifact(artifact(bytes), new Set(), timeoutMs, async (url, init) => {
    requests.push({ url, range: new Headers(init.headers).get("Range") });
    return fetcher(url, init);
  });
  assert.deepEqual(requests, [
    { url: "https://source.test/model.bin", range: "bytes=0-5" },
    { url: "https://source.test/model.bin", range: null },
  ]);
});

test("accepts safelisted Content-Length without explicitly exposing it", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  await preflightArtifact(
    artifact(bytes),
    new Set(),
    timeoutMs,
    validArtifactFetcher(bytes, {
      rangeExposed: "Content-Range",
      fullExposed: "",
    }),
  );
});

test("accepts a compressed full response whose decoded length is verified by streaming", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  await preflightArtifact(
    artifact(bytes),
    new Set(),
    timeoutMs,
    validArtifactFetcher(bytes, {
      omitFullLength: true,
      fullHeaders: { "Content-Encoding": "br" },
    }),
  );
});

test("rejects incompatible range metadata", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  await assert.rejects(
    () => preflightArtifact(
      artifact(bytes),
      new Set(),
      timeoutMs,
      validArtifactFetcher(bytes, { contentRange: "bytes 1-3/4" }),
    ),
    /incompatible Content-Range/,
  );
  await assert.rejects(
    () => preflightArtifact(
      artifact(bytes),
      new Set(),
      timeoutMs,
      validArtifactFetcher(bytes, { rangeExposed: "Content-Length" }),
    ),
    /does not expose Content-Range/,
  );
});

test("rejects exact-length and SHA-256 mismatches", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  await assert.rejects(
    () => preflightArtifact(
      artifact(bytes),
      new Set(),
      timeoutMs,
      validArtifactFetcher(bytes, { fullLength: bytes.byteLength + 1 }),
    ),
    /invalid Content-Length/,
  );
  await assert.rejects(
    () => preflightArtifact(
      artifact(bytes),
      new Set(),
      timeoutMs,
      validArtifactFetcher(bytes, {
        fullBody: bytes.slice(0, bytes.byteLength - 1),
        fullLength: bytes.byteLength,
      }),
    ),
    /returned 3 bytes instead of 4/,
  );
  await assert.rejects(
    () => preflightArtifact(
      artifact(bytes),
      new Set(),
      timeoutMs,
      validArtifactFetcher(bytes, { fullBody: Uint8Array.of(4, 3, 2, 1) }),
    ),
    /failed its SHA-256 preflight/,
  );
});

test("accepts catalog advances while rejecting rollback below the embedded floor", () => {
  assert.doesNotThrow(() => assertCatalogSequenceFloor(7, 7));
  assert.doesNotThrow(() => assertCatalogSequenceFloor(8, 7));
  assert.throws(() => assertCatalogSequenceFloor(6, 7), /below the embedded rollback floor/);
});

for (const { name, run } of tests) {
  try {
    await run();
  } catch (error) {
    throw new Error(`Hosted distribution preflight test failed: ${name}`, { cause: error });
  }
}

console.log(`Hosted distribution preflight tools passed (${tests.length} tests)`);
