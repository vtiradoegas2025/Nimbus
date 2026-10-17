export interface StreamlineArrowGeometry {
  points: Float32Array;
  polys: Uint32Array;
  /** One scalar per generated cone point, when source magnitudes were supplied. */
  scalars: Float32Array | null;
  markerCount: number;
}

export function streamlineMagnitudeMaximum(paths: readonly (ArrayLike<number> | null | undefined)[]): number {
  let maximum = 0;
  for (const values of paths) {
    if (!values) continue;
    for (let index = 0; index < values.length; index++) {
      const value = values[index]!;
      if (Number.isFinite(value) && value > maximum) maximum = value;
    }
  }
  return maximum;
}

export function sampleStreamlineMagnitudes(
  points: ArrayLike<number>,
  vectorAt: (position: number[], vector: Float32Array) => boolean,
): Float32Array {
  if (points.length % 3 !== 0) throw new Error("Streamline points must contain complete coordinates");
  const magnitudes = new Float32Array(points.length / 3);
  const position = [0, 0, 0];
  const vector = new Float32Array(3);
  for (let point = 0; point < magnitudes.length; point++) {
    for (let axis = 0; axis < 3; axis++) position[axis] = points[point * 3 + axis];
    if (!vectorAt(position, vector)) throw new Error("Streamline magnitude lies outside the prepared vector field");
    magnitudes[point] = Math.hypot(vector[0], vector[1], vector[2]);
    if (!Number.isFinite(magnitudes[point])) throw new Error("Streamline magnitude is not finite");
  }
  return magnitudes;
}

export function vectorTupleMagnitudes(
  vectors: ArrayLike<number> | null | undefined,
  pointCount: number,
): Float32Array | null {
  if (!vectors || vectors.length !== pointCount * 3) return null;
  const out = new Float32Array(pointCount);
  for (let point = 0; point < pointCount; point += 1) {
    const base = point * 3;
    const magnitude = Math.hypot(vectors[base], vectors[base + 1], vectors[base + 2]);
    out[point] = Number.isFinite(magnitude) ? magnitude : 0;
  }
  return out;
}

interface Candidate {
  point: number;
  previous: number;
  next: number;
}

function candidatesFromLines(lines: ArrayLike<number>): Candidate[] {
  const candidates: Candidate[] = [];
  let cursor = 0;
  while (cursor < lines.length) {
    const count = Math.floor(lines[cursor++]);
    if (!(count >= 3) || cursor + count > lines.length) break;
    // One marker per branch is enough to orient a static streamline without
    // turning a dense seed set into a row of cones.
    const fractions = [0.58];
    for (const fraction of fractions) {
      const local = Math.max(1, Math.min(count - 2, Math.round((count - 1) * fraction)));
      candidates.push({
        point: Math.floor(lines[cursor + local]),
        previous: Math.floor(lines[cursor + local - 1]),
        next: Math.floor(lines[cursor + local + 1]),
      });
    }
    cursor += count;
  }
  return candidates;
}

/**
 * Build four-sided cone markers along streamline polylines. Backward-integrated
 * paths traverse against the vector field, so their tangent is reversed before
 * constructing the cone; every marker therefore points in the physical vector
 * direction even though both integration branches remain visible.
 */
export function buildStreamlineArrowGeometry(
  points: ArrayLike<number>,
  lines: ArrayLike<number>,
  direction: "forward" | "backward",
  markerLength: number,
  pointScalars: ArrayLike<number> | null = null,
  maxMarkers = 512,
): StreamlineArrowGeometry | null {
  if (points.length % 3 !== 0 || !(markerLength > 0) || lines.length === 0) return null;
  const pointCount = points.length / 3;
  const all = candidatesFromLines(lines).filter(({ point, previous, next }) =>
    point >= 0 && point < pointCount
    && previous >= 0 && previous < pointCount
    && next >= 0 && next < pointCount,
  );
  if (all.length === 0) return null;
  const step = Math.max(1, Math.ceil(all.length / Math.max(1, Math.floor(maxMarkers))));
  const selected = all.filter((_, index) => index % step === 0).slice(0, maxMarkers);
  const outPoints: number[] = [];
  const outPolys: number[] = [];
  const outScalars: number[] | null = pointScalars?.length === pointCount ? [] : null;
  const radius = markerLength * 0.22;

  for (const candidate of selected) {
    const p = candidate.point * 3;
    const a = candidate.previous * 3;
    const b = candidate.next * 3;
    let tx = points[b] - points[a];
    let ty = points[b + 1] - points[a + 1];
    let tz = points[b + 2] - points[a + 2];
    if (direction === "backward") {
      tx = -tx;
      ty = -ty;
      tz = -tz;
    }
    const length = Math.hypot(tx, ty, tz);
    if (!(length > 0) || !Number.isFinite(length)) continue;
    tx /= length;
    ty /= length;
    tz /= length;

    // Pick the least-parallel cardinal helper, then build an orthonormal base.
    const helper = Math.abs(tz) < 0.8 ? [0, 0, 1] : [0, 1, 0];
    let e1x = ty * helper[2] - tz * helper[1];
    let e1y = tz * helper[0] - tx * helper[2];
    let e1z = tx * helper[1] - ty * helper[0];
    const e1Length = Math.hypot(e1x, e1y, e1z);
    if (!(e1Length > 0)) continue;
    e1x /= e1Length;
    e1y /= e1Length;
    e1z /= e1Length;
    const e2x = ty * e1z - tz * e1y;
    const e2y = tz * e1x - tx * e1z;
    const e2z = tx * e1y - ty * e1x;

    const tip = [points[p], points[p + 1], points[p + 2]];
    const base = [
      tip[0] - tx * markerLength,
      tip[1] - ty * markerLength,
      tip[2] - tz * markerLength,
    ];
    const first = outPoints.length / 3;
    outPoints.push(...tip);
    for (const [u, v] of [[1, 0], [0, 1], [-1, 0], [0, -1]] as const) {
      outPoints.push(
        base[0] + radius * (u * e1x + v * e2x),
        base[1] + radius * (u * e1y + v * e2y),
        base[2] + radius * (u * e1z + v * e2z),
      );
    }
    outPolys.push(
      3, first, first + 1, first + 2,
      3, first, first + 2, first + 3,
      3, first, first + 3, first + 4,
      3, first, first + 4, first + 1,
      4, first + 1, first + 4, first + 3, first + 2,
    );
    if (outScalars) {
      const value = Number.isFinite(pointScalars![candidate.point])
        ? pointScalars![candidate.point]
        : 0;
      outScalars.push(value, value, value, value, value);
    }
  }

  const markerCount = outPoints.length / 15;
  if (markerCount === 0) return null;
  return {
    points: Float32Array.from(outPoints),
    polys: Uint32Array.from(outPolys),
    scalars: outScalars ? Float32Array.from(outScalars) : null,
    markerCount,
  };
}
