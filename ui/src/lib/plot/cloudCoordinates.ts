import type { StepInfo } from "@/lib/run/runOutputs";
import { axisFrame, motionMode } from "@/lib/run/frameMotion";
import { isCartesianGrid } from "@/lib/science/cartesianInterp";

export function cloudCoordinateError(step: StepInfo | null | undefined): string | null {
  if (!step?.grid || step.grid.geometryKind === "index" || !isCartesianGrid(step.grid.coordinateSystem)) {
    return "Cloud motion requires physical Cartesian coordinates";
  }
  if (motionMode(step) == null || axisFrame(step).frame !== "earth-fixed") {
    return "Cloud motion requires verified Earth-coordinate registration";
  }
  return null;
}
