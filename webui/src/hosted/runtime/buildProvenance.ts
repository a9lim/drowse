export type HostedBuildChannel = "preview" | "release";

export interface BrowserBuildProvenance {
  schemaVersion: 1;
  drowseVersion: string;
  sourceRevision: string;
  runtimeAbi: string;
  hookAbi: string;
  channel: HostedBuildChannel;
  verified: boolean;
}

export interface BrowserArtifactProducer {
  drowseVersion: string;
  producerVersion: string;
}

const COMMIT = /^[0-9a-f]{40}$/;
const VERSION = /^[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)?$/;
const RUNTIME_ABI = /^drowse-web-runtime-v[0-9]+$/;
const HOOK_ABI = /^post-block-residual-v[0-9]+$/;

export function browserBuildProvenance(): Readonly<BrowserBuildProvenance> {
  return validateBrowserBuildProvenance({
    schemaVersion: 1,
    drowseVersion: injected("Drowse version", () => __DROWSE_VERSION__),
    sourceRevision: injected(
      "Drowse source revision",
      () => __DROWSE_ARTIFACT_SOURCE_REVISION__,
    ),
    runtimeAbi: injected("runtime ABI", () => __DROWSE_RUNTIME_ABI__),
    hookAbi: injected("hook ABI", () => __DROWSE_HOOK_ABI__),
    channel: injected("hosted channel", () => __DROWSE_HOSTED_CHANNEL__) as HostedBuildChannel,
    verified: injected("hosted channel", () => __DROWSE_HOSTED_CHANNEL__) === "release",
  });
}

export function browserArtifactProducer(
  provenance: BrowserBuildProvenance = browserBuildProvenance(),
): Readonly<BrowserArtifactProducer> {
  const validated = validateBrowserBuildProvenance(provenance);
  return Object.freeze({
    drowseVersion: validated.drowseVersion,
    producerVersion: JSON.stringify(validated),
  });
}

export function validateBrowserBuildProvenance(
  provenance: BrowserBuildProvenance,
): Readonly<BrowserBuildProvenance> {
  if (
    provenance.schemaVersion !== 1 ||
    !VERSION.test(provenance.drowseVersion) ||
    !COMMIT.test(provenance.sourceRevision) ||
    !RUNTIME_ABI.test(provenance.runtimeAbi) ||
    !HOOK_ABI.test(provenance.hookAbi) ||
    (provenance.channel !== "preview" && provenance.channel !== "release") ||
    provenance.verified !== (provenance.channel === "release")
  ) {
    throw new Error("Hosted browser artifact provenance is invalid or contradictory");
  }
  return Object.freeze({ ...provenance });
}

function injected(label: string, read: () => string): string {
  let value: string;
  try {
    value = read();
  } catch {
    throw new Error(`Hosted build did not inject the exact ${label}`);
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Hosted build did not inject the exact ${label}`);
  }
  return value;
}
