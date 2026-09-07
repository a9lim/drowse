const FP32_BYTES = 4;
export const JLENS_TARGET_RESIDENT_BYTES = 448 * 1024 * 1024;
export const JLENS_MAX_LAYERS = 8;

export function selectHostedJlensLayers(
  layerMap,
  hiddenSize,
  targetBytes = JLENS_TARGET_RESIDENT_BYTES,
) {
  if (
    !Array.isArray(layerMap) || layerMap.length === 0 ||
    layerMap.some((layer, index) =>
      !Number.isSafeInteger(layer) || layer < 0 ||
      (index > 0 && layer <= layerMap[index - 1])
    ) ||
    !Number.isSafeInteger(hiddenSize) || hiddenSize < 1 ||
    !Number.isSafeInteger(targetBytes) || targetBytes < 1
  ) {
    throw new Error("J-lens layer selection inputs are invalid");
  }
  if (layerMap.length < 2) {
    throw new Error("J-lens fitting requires a source layer before the final residual");
  }
  const sourceLayerMap = layerMap.slice(0, -1);
  const matrixBytes = hiddenSize * hiddenSize * FP32_BYTES;
  const count = Math.max(
    1,
    Math.min(JLENS_MAX_LAYERS, sourceLayerMap.length, Math.floor(targetBytes / matrixBytes)),
  );
  if (count === sourceLayerMap.length) return [...sourceLayerMap];
  if (count === 1) return [sourceLayerMap[sourceLayerMap.length - 1]];
  return Array.from({ length: count }, (_, index) =>
    sourceLayerMap[Math.round(index * (sourceLayerMap.length - 1) / (count - 1))]
  );
}
