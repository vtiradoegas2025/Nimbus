/*
 * What to do when a layer's isosurface is too large to retain.
 *
 * A triangle ceiling is a property of the renderer, not of the question the
 * user asked. Refusing to draw makes the ceiling the user's problem; backing
 * the sampling lattice off and saying so keeps the picture and keeps it honest.
 * Pure — no vtk.js, no React.
 */

/** Coarsest lattice the backoff will go to, as a divisor of the layer's voxel budget. */
export const MAX_CONTOUR_LOD_DIVISOR = 64;

/** Never divide below a lattice that can still resolve a storm-scale feature. */
export const MIN_CONTOUR_VOXEL_BUDGET = 250_000;

/** Voxel budget for a layer that has backed off `divisor` steps. */
export function contourVoxelBudget(baseBudget: number, divisor: number): number {
  if (!Number.isFinite(baseBudget) || baseBudget <= 0) return MIN_CONTOUR_VOXEL_BUDGET;
  const safeDivisor = Number.isFinite(divisor) && divisor >= 1 ? divisor : 1;
  return Math.max(MIN_CONTOUR_VOXEL_BUDGET, Math.floor(baseBudget / safeDivisor));
}

/** The next coarser step, or null when already at the coarsest allowed. */
export function nextContourLodDivisor(divisor: number): number | null {
  const safe = Number.isFinite(divisor) && divisor >= 1 ? divisor : 1;
  const next = safe * 2;
  return next <= MAX_CONTOUR_LOD_DIVISOR ? next : null;
}

export interface ContourBackoffOutcome {
  /** Divisor to prepare at next, or null to stay where it is. */
  nextDivisor: number | null;
  /** Message for the user; null when the layer is drawable. */
  error: string | null;
}

/**
 * Decide what a layer should do about its measured contour size.
 *
 * `overBudget` is the renderer's own violation text. Only once the lattice
 * cannot be coarsened further does that become an error the user has to act
 * on — and then it names the control that actually changes the outcome.
 */
export function contourBackoff(
  overBudget: string | null,
  divisor: number,
  thresholdControlLabel: string,
): ContourBackoffOutcome {
  if (!overBudget) return { nextDivisor: null, error: null };
  const next = nextContourLodDivisor(divisor);
  if (next != null) return { nextDivisor: next, error: null };
  return {
    nextDivisor: null,
    error: `This surface stays too large to draw even at the coarsest sampling.`
      + ` Raise the ${thresholdControlLabel} to enclose less of the domain.`,
  };
}
