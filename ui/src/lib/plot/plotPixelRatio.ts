export function plotViewerPixelRatio(devicePixelRatio: number): number {
  if (!Number.isFinite(devicePixelRatio) || devicePixelRatio <= 0) return 1;
  return Math.max(1, Math.min(4, devicePixelRatio));
}
