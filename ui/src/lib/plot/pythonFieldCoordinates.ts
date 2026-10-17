import type { PythonFieldMetadata } from "../platform/npy";
import type { GridInfo } from "../run/runOutputs";
import { cartesianAxisSpacing, cartesianScalarAxisOrigin } from "../science/cartesianCoordinates";
import { hasPhysicalAxisCoordinates } from "../science/gridCoords";
import type { SliceAxis } from "../platform/npy";

export function pythonFieldCoordinates(
  grid: GridInfo | null | undefined,
  sourceShape: readonly number[],
  outputShape: readonly number[],
  starts: readonly number[],
  steps: readonly number[],
): PythonFieldMetadata["coordinates"] {
  const cylindrical = grid?.coordinateSystem?.toLowerCase().startsWith("cyl") === true;
  const names = cylindrical ? ["r", "theta", "z"] as const : ["x", "y", "z"] as const;
  return outputShape.map((count, dimension) => {
    const axis = dimension as SliceAxis;
    const indices = Array.from({ length: count }, (_, index) => starts[axis] + index * steps[axis]);
    if (axis === 2 && grid?.terrainFollowing) {
      return { name: names[axis], values: indices, unit: "model_level" };
    }
    if (!grid || grid.geometryKind === "index") {
      return { name: names[axis], values: indices, unit: "index" };
    }
    const exact = grid.scalarCoordinatesM?.[axis];
    const exactValid = exact?.length === sourceShape[axis]
      && exact.every((value, index) => Number.isFinite(value) && (index === 0 || value > exact[index - 1]));
    const spacing = cylindrical
      ? [grid.drM, grid.dthetaRad, grid.dzM][axis]
      : cartesianAxisSpacing(grid, axis);
    const physical = cylindrical
      ? exactValid || spacing != null && Number.isFinite(spacing) && spacing > 0
      : hasPhysicalAxisCoordinates(axis, sourceShape[axis], grid)
        && (exactValid || Number.isFinite(spacing) && spacing! > 0);
    if (!physical) return { name: names[axis], values: indices, unit: "index" };
    const horizontalOrigin = axis === 0 ? grid.horizontalScalarOriginM?.[0]
      : axis === 1 ? grid.horizontalScalarOriginM?.[1] : undefined;
    const origin = cylindrical
      ? grid.scalarOriginM?.[axis] ?? horizontalOrigin ?? 0
      : cartesianScalarAxisOrigin(grid, axis, sourceShape[axis], spacing!);
    return {
      name: names[axis],
      values: indices.map(index => exactValid ? exact![index] : origin + index * spacing!),
      unit: cylindrical && axis === 1 ? "rad" : "m",
    };
  });
}
