const structuredHookProfiles = new Set(["standard-v1", "standard-v2", "standard-v3"]);

export function structuredHookProfile(value) {
  if (!structuredHookProfiles.has(value)) {
    throw new Error("structuredHookProfile is not allow-listed");
  }
  return value;
}

export function thinkingProfile(value) {
  if (value === null) return null;
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("thinkingProfile must be null or an object");
  }
  const keys = Object.keys(value).sort();
  const expected = ["end", "endTokenIds", "start", "startTokenIds", "startsInThinking"];
  if (keys.join("\0") !== expected.join("\0")) {
    throw new Error("thinkingProfile has unexpected fields");
  }
  const start = delimiter(value.start, "thinkingProfile.start");
  const end = delimiter(value.end, "thinkingProfile.end");
  if (start === end) throw new Error("thinkingProfile delimiters must differ");
  if (typeof value.startsInThinking !== "boolean") {
    throw new Error("thinkingProfile.startsInThinking must be a boolean");
  }
  const startTokenIds = tokenIds(value.startTokenIds, "thinkingProfile.startTokenIds");
  const endTokenIds = tokenIds(value.endTokenIds, "thinkingProfile.endTokenIds");
  if (
    startTokenIds.length === endTokenIds.length &&
    startTokenIds.every((tokenId, index) => tokenId === endTokenIds[index])
  ) {
    throw new Error("thinkingProfile token sequences must differ");
  }
  return {
    start,
    end,
    startsInThinking: value.startsInThinking,
    startTokenIds,
    endTokenIds,
  };
}

export function sameThinkingProfile(left, right) {
  return JSON.stringify(thinkingProfile(left)) === JSON.stringify(thinkingProfile(right));
}

export function lockedModelExecutionProfiles(lock, manifest) {
  const lockedStructuredHookProfile = structuredHookProfile(lock?.structuredHookProfile);
  const artifactStructuredHookProfile = structuredHookProfile(manifest?.structuredHookProfile);
  const lockedThinkingProfile = thinkingProfile(lock?.thinkingProfile);
  const artifactThinkingProfile = thinkingProfile(manifest?.thinkingProfile);
  const quantization = lockedQuantization(lock?.quantization);
  const artifactQuantization = lockedQuantization(manifest?.quantization);
  if (
    artifactStructuredHookProfile !== lockedStructuredHookProfile ||
    JSON.stringify(artifactThinkingProfile) !== JSON.stringify(lockedThinkingProfile) ||
    artifactQuantization !== quantization
  ) {
    throw new Error("hosted model artifacts do not match runtime-lock execution profile");
  }
  return {
    structuredHookProfile: lockedStructuredHookProfile,
    thinkingProfile: lockedThinkingProfile,
    quantization,
    requiredFeatures: quantization === "q4f16_1" ? ["shader-f16"] : [],
  };
}

function lockedQuantization(value) {
  if (value !== "q4f16_1" && value !== "q4f32_1" && value !== "q0f32") {
    throw new Error("hosted quantization is not allow-listed");
  }
  return value;
}

function delimiter(value, label) {
  if (typeof value !== "string" || value.length === 0 || value.length > 256 || value.includes("\0")) {
    throw new Error(`${label} must be 1-256 characters without NULs`);
  }
  return value;
}

function tokenIds(value, label) {
  if (
    !Array.isArray(value) || value.length === 0 || value.length > 32 ||
    value.some((tokenId) => !Number.isSafeInteger(tokenId) || tokenId < 0)
  ) {
    throw new Error(`${label} must contain 1-32 non-negative token IDs`);
  }
  return [...value];
}
