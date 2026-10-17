/** Readable physical tube radius for a prepared streamline lattice.
 *
 * Cell spacing alone is insufficient on a large domain: a 50 m tube is
 * scientifically scaled but becomes sub-pixel when the camera fits 100 km.
 * Preserve at least 0.25% of the horizontal span as radius while retaining a
 * grid-scale floor. The result remains a physical metre value and the user can
 * still override it from the control. */
export function defaultStreamlineTubeRadiusM(
  shape: readonly [number, number, number],
  spacing: readonly [number, number, number],
): number {
  const valid = [...shape, ...spacing].every(Number.isFinite)
    && shape.every((count) => Number.isInteger(count) && count > 0)
    && spacing.every((step) => step > 0);
  if (!valid) return 250;
  const horizontalSpan = Math.max(
    Math.max(1, shape[0] - 1) * spacing[0],
    Math.max(1, shape[1] - 1) * spacing[1],
  );
  const gridRadius = Math.min(...spacing) * 0.75;
  const fittedViewRadius = horizontalSpan / 400;
  const rounded = Math.round(Math.max(gridRadius, fittedViewRadius) / 25) * 25;
  return Math.min(1_000, Math.max(25, rounded));
}
