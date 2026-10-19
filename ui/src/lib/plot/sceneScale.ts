import type { StepInfo } from "@/lib/run/runOutputs";

/* Display-only scene scaling. Data, integration, and contouring remain in true coordinates. */

/** Default vertical exaggeration: 1 (true aspect). Shown on canvas when overridden. */
export const VERTICAL_EXAG_DEFAULT = 1;
export const VERTICAL_EXAG_RANGE = { min: 1, max: 6, step: 0.5 } as const;

/** Horizontal (x=y) display scale; 1 by default, never auto-seeded. */
export const HORIZONTAL_SCALE_DEFAULT = 1;
export const HORIZONTAL_SCALE_RANGE = { min: 0.5, max: 2, step: 0.1 } as const;

/** Combined horizontal (x = y) + vertical (z) display scaling for a 3D scene. */
export type SceneScale = { xy: number; z: number };

/** Opening vertical exaggeration: always VERTICAL_EXAG_DEFAULT; single home for a future domain-dependent rule. */
export function autoVerticalExag(_grid: StepInfo["grid"] | null | undefined): number {
  return VERTICAL_EXAG_DEFAULT;
}
