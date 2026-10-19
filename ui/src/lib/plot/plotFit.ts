/** How a plane plot spends the pane it is given; pure module with no React/Plotly dependency. */

import { LOCKED_ASPECT_AXIS, lockedAspectPartnerAxis } from "@/lib/science/gridCoords";

/** `true-shape`: 1:1 aspect lock; `fill`: full pane but stretches geometry (opt-in). */
export type PlotFitMode = "true-shape" | "fill";

export const FIT_OPTIONS: ReadonlyArray<{ value: PlotFitMode; label: string }> = [
  { value: "true-shape", label: "True shape" },
  { value: "fill", label: "Fill" },
];

/** Axis lock fragments for Plotly; empty when aspect cannot be locked. Spread unconditionally. */
export function fitAxisFragments(lockAspect: boolean, fit: PlotFitMode) {
  const locked = lockAspect && fit === "true-shape";
  return {
    x: locked ? LOCKED_ASPECT_AXIS : {},
    y: locked ? lockedAspectPartnerAxis() : {},
  };
}
