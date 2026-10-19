export interface ContourTopologyUpperBound {
  triangles: number;
  bytes: number;
}

export function contourTopologyUpperBound(
  sampleUpperBound: number,
  method: "marching-cubes" | "terrain-tetrahedra" = "marching-cubes",
): ContourTopologyUpperBound {
  if (!Number.isSafeInteger(sampleUpperBound) || sampleUpperBound < 1) {
    throw new Error("Contour topology requires a positive sample upper bound");
  }
  const trianglesPerCell = method === "terrain-tetrahedra" ? 12 : 5;
  const triangles = sampleUpperBound * trianglesPerCell;
  const bytes = triangles * 88;
  if (!Number.isSafeInteger(triangles) || !Number.isSafeInteger(bytes)) {
    throw new Error("Contour topology upper bound exceeds safe integer arithmetic");
  }
  return { triangles, bytes };
}

interface VtkArrayData {
  getData?: () => { byteLength: number };
}

interface VtkAttributesData {
  getArrays?: () => VtkArrayData[];
}

export interface VtkPolyDataOutput {
  getNumberOfPolys?: () => number;
  getPoints?: () => VtkArrayData;
  getPolys?: () => VtkArrayData;
  getPointData?: () => VtkAttributesData;
  getCellData?: () => VtkAttributesData;
}

export interface VtkContourSource {
  getOutputData: () => VtkPolyDataOutput;
}

export interface ContourOutputUsage {
  triangles: number;
  bytes: number;
}

/** Validate measured usage against a request-derived admission. */
export function contourUsageViolation(
  usage: ContourOutputUsage,
  label: string,
  maxBytes: number,
  maxTriangles: number,
): string | null {
  if (usage.bytes > maxBytes) {
    return `${label} is ${usage.bytes} bytes; limit is ${maxBytes}`;
  }
  if (usage.triangles > maxTriangles) {
    return `${label} has ${usage.triangles} triangles; limit is ${maxTriangles}`;
  }
  return null;
}

/** Measure arrays actually retained by one marching-cubes output. */
export function measureContourOutput(source: VtkContourSource): ContourOutputUsage {
  const output = source.getOutputData();
  const arrays = new Set<VtkArrayData>();
  const add = (array: VtkArrayData | undefined) => { if (array) arrays.add(array); };
  add(output.getPoints?.());
  add(output.getPolys?.());
  for (const array of output.getPointData?.().getArrays?.() ?? []) add(array);
  for (const array of output.getCellData?.().getArrays?.() ?? []) add(array);
  let bytes = 0;
  for (const array of arrays) {
    const byteLength = array.getData?.().byteLength ?? 0;
    if (Number.isFinite(byteLength) && byteLength > 0) bytes += byteLength;
  }
  return { triangles: output.getNumberOfPolys?.() ?? 0, bytes };
}

export function contourBudgetViolation(
  source: VtkContourSource,
  maxBytes: number,
  maxTriangles: number,
): string | null {
  const usage = measureContourOutput(source);
  if (usage.bytes > maxBytes) {
    return `Contour output is ${usage.bytes} bytes; limit is ${maxBytes}`;
  }
  if (usage.triangles > maxTriangles) {
    return `Contour output has ${usage.triangles} triangles; limit is ${maxTriangles}`;
  }
  return null;
}


/**
 * True when every source contoured to nothing.
 *
 * A budget check only reports the over side, so a threshold above the field's
 * maximum produces zero triangles, passes, and leaves visible actors with no
 * geometry — indistinguishable on screen from a broken layer. Callers use this
 * to say which it was.
 */
export function contoursAreEmpty(sources: readonly VtkContourSource[]): boolean {
  if (sources.length === 0) return false;
  return sources.every((source) => measureContourOutput(source).triangles === 0);
}

/** Enforce one retained-output ceiling across a multi-surface visualization. */
export function combinedContourBudgetViolation(
  sources: readonly VtkContourSource[],
  maxBytes: number,
  maxTriangles: number,
): string | null {
  const usage = sources.reduce<ContourOutputUsage>(
    (total, source) => {
      const current = measureContourOutput(source);
      total.bytes += current.bytes;
      total.triangles += current.triangles;
      return total;
    },
    { bytes: 0, triangles: 0 },
  );
  return contourUsageViolation(usage, "Combined contour output", maxBytes, maxTriangles);
}
