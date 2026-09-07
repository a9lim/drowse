export function selectHostedSaeLayer(layerMap, requested = null) {
  if (
    !Array.isArray(layerMap) || layerMap.length === 0 ||
    layerMap.some((layer, index) =>
      !Number.isSafeInteger(layer) || layer < 0 ||
      (index > 0 && layer <= layerMap[index - 1])
    )
  ) throw new Error("SAE runtime layer map is invalid");

  if (requested !== null) {
    if (!layerMap.includes(requested)) {
      throw new Error(`SAE layer ${requested} is not in the runtime lock`);
    }
    return requested;
  }

  const targetSlot = roundHalfToEven(0.65 * Math.max(layerMap.length - 1, 0));
  return layerMap[targetSlot];
}

function roundHalfToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}
