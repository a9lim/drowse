import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

const GIB = 1024 ** 3;

export async function launchReleaseToolContext(options) {
  const profileDirectory = await mkdtemp(join(tmpdir(), "drowse-chrome-"));
  try {
    const context = await chromium.launchPersistentContext(profileDirectory, options);
    const page = context.pages()[0] ?? await context.newPage();
    return {
      context,
      page,
      version() {
        const browser = context.browser();
        if (browser === null) throw new Error("release-tool browser is unavailable");
        return browser.version();
      },
      async close() {
        try {
          await context.close();
        } finally {
          await rm(profileDirectory, { recursive: true, force: true });
        }
      },
    };
  } catch (error) {
    await rm(profileDirectory, { recursive: true, force: true });
    throw error;
  }
}

function artifactByteTotal(artifacts) {
  if (!Array.isArray(artifacts) || artifacts.length === 0) {
    throw new Error("release-tool quota requires at least one artifact");
  }
  const artifactBytes = artifacts.reduce((total, artifact) => {
    if (!Number.isSafeInteger(artifact?.bytes) || artifact.bytes < 0) {
      throw new Error("release-tool quota requires exact artifact byte sizes");
    }
    return total + artifact.bytes;
  }, 0);
  if (!Number.isSafeInteger(artifactBytes)) {
    throw new Error("release-tool artifact byte total exceeds the safe integer range");
  }
  return artifactBytes;
}

export function releaseToolQuotaBytes(artifacts) {
  const artifactBytes = artifactByteTotal(artifacts);
  return Math.max(8 * GIB, artifactBytes * 2 + 2 * GIB);
}

export async function grantReleaseToolStorageQuota(page, origin, artifacts) {
  const artifactBytes = artifactByteTotal(artifacts);
  const quotaSize = releaseToolQuotaBytes(artifacts);
  const session = await page.context().newCDPSession(page);
  try {
    await session.send("Storage.overrideQuotaForOrigin", { origin, quotaSize });
    const state = await session.send("Storage.getUsageAndQuota", { origin });
    const requiredAvailableBytes = artifactBytes + 2 * GIB;
    const availableBytes = Number(state.quota) - Number(state.usage);
    if (!Number.isFinite(availableBytes) || availableBytes < requiredAvailableBytes) {
      throw new Error(
        `Chrome reported ${availableBytes} available bytes; the release tool needs ` +
        `${requiredAvailableBytes}`,
      );
    }
  } finally {
    await session.detach();
  }
  return quotaSize;
}
