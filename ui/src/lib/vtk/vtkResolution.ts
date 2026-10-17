/** Pixel-budget policy for vtk.js backing stores. See vtkInteraction.ts for
 *  why the backing store must NOT be resized mid-gesture. */

export interface ResolutionPolicy {
  /** Device pixels per CSS pixel. */
  maxPixelRatio: number;
  /** Total device pixels. */
  maxPixels: number;
}

export interface BackingStoreSize {
  width: number;
  height: number;
  /** Device pixels per CSS pixel actually used, after every ceiling. */
  pixelRatio: number;
}

/** Floor: below 0.5 sub-pixel geometry drops out; a large canvas exceeds its
 *  pixel budget rather than cross this. */
export const MIN_PIXEL_RATIO = 0.5;

/** Full QHD pixels at rest, with a ceiling on larger and Retina canvases. */
export const VOLUME_RESOLUTION_POLICY: ResolutionPolicy = {
  maxPixelRatio: 2,
  maxPixels: 3_700_000,
};

/** Triangle-rasterized scenes: isosurfaces, streamlines, trajectories. */
export const SURFACE_RESOLUTION_POLICY: ResolutionPolicy = {
  maxPixelRatio: 2,
  maxPixels: 8_000_000,
};

/** Resolve backing-store size under the policy; returns null if the container has no layout yet. */
export function backingStoreSize(
  cssWidth: number,
  cssHeight: number,
  devicePixelRatio: number,
  policy: ResolutionPolicy,
): BackingStoreSize | null {
  if (!Number.isFinite(cssWidth) || !Number.isFinite(cssHeight)) return null;
  if (!(cssWidth > 0) || !(cssHeight > 0)) return null;

  const displayRatio = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0
    ? devicePixelRatio
    : 1;
  const ratioCeiling = policy.maxPixelRatio;
  const pixelCeiling = policy.maxPixels;

  // Never supersample past the policy, and never past the display itself:
  // rendering more device pixels than the display can show is pure cost.
  let ratio = Math.min(displayRatio, ratioCeiling);
  if (pixelCeiling > 0) {
    const budgeted = Math.sqrt(pixelCeiling / (cssWidth * cssHeight));
    if (budgeted < ratio) ratio = budgeted;
  }
  if (ratio < MIN_PIXEL_RATIO) ratio = MIN_PIXEL_RATIO;

  return {
    width: Math.max(1, Math.floor(cssWidth * ratio)),
    height: Math.max(1, Math.floor(cssHeight * ratio)),
    pixelRatio: ratio,
  };
}
