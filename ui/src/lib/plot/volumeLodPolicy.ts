export interface VolumeLodLevel {
  level: number;
  stride: readonly [number, number, number];
}

export function selectCameraVolumeLevel(
  levels: readonly VolumeLodLevel[],
  options: {
    nativeSpacingM: number;
    cameraRangeM: number;
    viewportHeightPx: number;
    currentLevel?: number | null;
    interacting?: boolean;
    viewAngleDeg?: number;
    targetCellPixels?: number;
    hysteresisFraction?: number;
  },
): number {
  if (levels.length === 0) return 0;
  const ordered = [...levels].sort((left, right) => left.level - right.level);
  if (!(options.nativeSpacingM > 0) || !(options.cameraRangeM > 0)
      || !(options.viewportHeightPx > 0)) return ordered[0].level;
  const viewAngle = (options.viewAngleDeg ?? 30) * Math.PI / 180;
  const pixelsPerMetre = options.viewportHeightPx
    / (2 * Math.tan(viewAngle * 0.5) * options.cameraRangeM);
  const target = options.targetCellPixels ?? 1.25;
  const projectedPixels = (level: VolumeLodLevel) =>
    options.nativeSpacingM * Math.max(...level.stride) * pixelsPerMetre;
  let candidate = ordered[0].level;
  for (const level of ordered) {
    if (projectedPixels(level) <= target) candidate = level.level;
  }
  const current = options.currentLevel;
  if (options.interacting && current != null
      && ordered.some((level) => level.level === current)) return current;
  if (current == null || current === candidate) return candidate;
  const currentDescriptor = ordered.find((level) => level.level === current);
  if (!currentDescriptor) return candidate;
  const projected = projectedPixels(currentDescriptor);
  const hysteresis = options.hysteresisFraction ?? 0.2;
  if (candidate > current && projected > target * (1 - hysteresis)) return current;
  if (candidate < current && projected < target * (1 + hysteresis)) return current;
  return candidate;
}
