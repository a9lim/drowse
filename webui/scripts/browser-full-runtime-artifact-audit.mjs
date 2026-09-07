import { createHash } from "node:crypto";

export function artifactUrl(origin, set, path) {
  const url = new URL("/artifact", origin);
  url.searchParams.set("set", set);
  url.searchParams.set("path", path);
  return url.href;
}

export function artifactEntriesDigest(set, files) {
  const tuples = files.map((file) => [set, file.path, file.bytes, file.sha256]);
  return createHash("sha256").update(JSON.stringify(tuples)).digest("hex");
}

export function artifactSetReceipt(set, files) {
  return {
    fileCount: files.length,
    totalBytes: files.reduce((sum, file) => sum + file.bytes, 0),
    entriesSha256: artifactEntriesDigest(set, files),
  };
}

export function validateBrowserArtifactAudit({
  origin,
  expectedSets,
  observedRequests,
  serverLifecycles,
  workerReceipt,
}) {
  const expected = expectedArtifacts(origin, expectedSets);
  validateWorkerReceipt(expectedSets, workerReceipt);
  const requests = recordsByExpectedKey(
    observedRequests,
    expected,
    "browser artifact request",
    (record) => record.url,
  );
  const lifecycles = recordsByExpectedKey(
    serverLifecycles,
    expected,
    "server artifact lifecycle",
    (record) => artifactUrl(origin, record.set, record.path),
  );
  const crossVerifiedArtifactEofAborts = [];
  for (const [url, artifact] of expected) {
    const request = requests.get(url);
    const lifecycle = lifecycles.get(url);
    if (
      request.method !== "GET" || request.resourceType !== "fetch" ||
      request.responseStatus !== 200 || request.responseContentLength !== artifact.bytes ||
      Number(request.finished === true) + Number(request.failed === true) !== 1
    ) {
      throw new Error(`browser artifact request did not close exactly: ${url}`);
    }
    if (request.finished === true) {
      if (request.errorText !== null) {
        throw new Error(`finished browser artifact request carried an error: ${url}`);
      }
    } else {
      if (request.errorText !== "net::ERR_ABORTED") {
        throw new Error(`browser artifact request failed: ${url} (${request.errorText ?? "unknown"})`);
      }
      crossVerifiedArtifactEofAborts.push({
        url,
        set: artifact.set,
        path: artifact.path,
        bytes: artifact.bytes,
        sha256: artifact.sha256,
        errorText: request.errorText,
        serverResponseFinished: true,
        opfsSizeAndSha256Verified: true,
      });
    }
    if (
      lifecycle.method !== "GET" || lifecycle.statusCode !== 200 ||
      lifecycle.expectedBytes !== artifact.bytes || lifecycle.finished !== true ||
      lifecycle.closed !== true || lifecycle.writableFinished !== true
    ) {
      throw new Error(`server artifact response did not finish exactly: ${url}`);
    }
  }
  return {
    files: expected.size,
    bytes: [...expected.values()].reduce((sum, artifact) => sum + artifact.bytes, 0),
    crossVerifiedArtifactEofAborts,
  };
}

function expectedArtifacts(origin, expectedSets) {
  if (!isRecord(expectedSets)) throw new Error("expected artifact sets are invalid");
  const result = new Map();
  for (const [set, files] of Object.entries(expectedSets)) {
    if (!Array.isArray(files) || files.length === 0) {
      throw new Error(`expected artifact set ${set} is empty`);
    }
    for (const file of files) {
      if (
        !isRecord(file) || typeof file.path !== "string" || !file.path ||
        !Number.isSafeInteger(file.bytes) || file.bytes < 1 ||
        typeof file.sha256 !== "string" || !/^[a-f0-9]{64}$/u.test(file.sha256)
      ) throw new Error(`expected artifact entry is invalid in ${set}`);
      const url = artifactUrl(origin, set, file.path);
      if (result.has(url)) throw new Error(`duplicate expected artifact ${set}/${file.path}`);
      result.set(url, { set, path: file.path, bytes: file.bytes, sha256: file.sha256 });
    }
  }
  return result;
}

function recordsByExpectedKey(records, expected, label, urlFor) {
  if (!Array.isArray(records)) throw new Error(`${label} ledger is invalid`);
  const result = new Map();
  for (const record of records) {
    if (!isRecord(record)) throw new Error(`${label} is invalid`);
    let url;
    try {
      url = urlFor(record);
    } catch {
      throw new Error(`${label} URL is invalid`);
    }
    if (typeof url !== "string" || !expected.has(url)) {
      throw new Error(`${label} is unknown or non-canonical: ${String(url)}`);
    }
    if (result.has(url)) throw new Error(`duplicate ${label}: ${url}`);
    result.set(url, record);
  }
  if (result.size !== expected.size) {
    throw new Error(`${label} ledger is incomplete: ${result.size}/${expected.size}`);
  }
  return result;
}

function validateWorkerReceipt(expectedSets, receipt) {
  if (!isRecord(receipt) || receipt.schemaVersion !== 1 || !isRecord(receipt.sets)) {
    throw new Error("worker artifact verification receipt is invalid");
  }
  const expectedNames = Object.keys(expectedSets).sort();
  const actualNames = Object.keys(receipt.sets).sort();
  if (JSON.stringify(actualNames) !== JSON.stringify(expectedNames)) {
    throw new Error("worker artifact verification receipt set names do not match");
  }
  for (const set of expectedNames) {
    const expected = artifactSetReceipt(set, expectedSets[set]);
    const actual = receipt.sets[set];
    if (
      !isRecord(actual) || actual.fileCount !== expected.fileCount ||
      actual.totalBytes !== expected.totalBytes ||
      actual.entriesSha256 !== expected.entriesSha256
    ) throw new Error(`worker artifact verification receipt does not match ${set}`);
  }
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
