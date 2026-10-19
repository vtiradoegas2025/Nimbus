export function hasMissingValues(values: Float32Array): boolean {
  for (let index = 0; index < values.length; index++) if (!Number.isFinite(values[index])) return true;
  return false;
}

export function finiteVectorSupport(
  values: Float32Array,
  dimensions: readonly number[],
  origin: readonly number[],
  spacing: readonly number[],
  components: 1 | 3 = 3,
  order: "vtk" | "c" = "vtk",
) {
  const [nx, ny, nz] = dimensions;
  if (dimensions.length !== 3 || dimensions.some(n => !Number.isSafeInteger(n) || n < 2)
      || values.length !== nx * ny * nz * components || spacing.length !== 3 || origin.length !== 3
      || spacing.some(v => !Number.isFinite(v) || v <= 0)
      || origin.some(v => !Number.isFinite(v))) throw new Error("Invalid streamline support lattice");
  const sample = (point: ArrayLike<number>, vector: Float32Array | number[]): boolean => {
    const x = (point[0] - origin[0]) / spacing[0], y = (point[1] - origin[1]) / spacing[1], z = (point[2] - origin[2]) / spacing[2];
    if (!Number.isFinite(x + y + z) || x < 0 || y < 0 || z < 0 || x > nx - 1 || y > ny - 1 || z > nz - 1) return false;
    const ix = Math.min(nx - 2, Math.floor(x)), iy = Math.min(ny - 2, Math.floor(y)), iz = Math.min(nz - 2, Math.floor(z));
    const tx = Math.fround(x - ix), ty = Math.fround(y - iy), tz = Math.fround(z - iz);
    let u = 0, v = 0, w = 0;
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const weight = Math.fround((dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz));
      if (weight === 0) continue;
      const at = components * (order === "vtk" ? ix + dx + nx * (iy + dy + ny * (iz + dz)) : ((ix + dx) * ny + iy + dy) * nz + iz + dz);
      if (!Number.isFinite(values[at])) return false;
      u = Math.fround(u + weight * values[at]);
      if (components === 3) {
        if (!Number.isFinite(values[at + 1]) || !Number.isFinite(values[at + 2])) return false;
        v = Math.fround(v + weight * values[at + 1]); w = Math.fround(w + weight * values[at + 2]);
      }
    }
    if (!Number.isFinite(u + v + w)) return false;
    vector[0] = u; vector[1] = v; vector[2] = w;
    return true;
  };
  const start = new Float64Array(3), delta = new Float64Array(3), next = new Float64Array(3), step = new Float64Array(3);
  const point = [0, 0, 0], scratch = new Float32Array(3);
  const connects = (a: ArrayLike<number>, b: ArrayLike<number>): boolean => {
    if (!sample(a, scratch) || !sample(b, scratch)) return false;
    for (let axis = 0; axis < 3; axis++) {
      start[axis] = (a[axis] - origin[axis]) / spacing[axis];
      delta[axis] = (b[axis] - a[axis]) / spacing[axis];
      const boundary = delta[axis] > 0 ? Math.floor(start[axis]) + 1 : Math.ceil(start[axis]) - 1;
      next[axis] = delta[axis] === 0 ? Infinity : (boundary - start[axis]) / delta[axis];
      step[axis] = delta[axis] === 0 ? Infinity : 1 / Math.abs(delta[axis]);
    }
    let current = 0;
    // Every interpolation-cell interval needs support, even when a long RK2
    // step has valid endpoints and midpoint on opposite sides of a thin gap.
    for (let cell = 0; cell <= nx + ny + nz; cell++) {
      const end = Math.min(1, next[0], next[1], next[2]);
      const middle = (current + end) * 0.5;
      for (let axis = 0; axis < 3; axis++) point[axis] = a[axis] + middle * (b[axis] - a[axis]);
      if (!sample(point, scratch)) return false;
      if (end >= 1) return true;
      for (let axis = 0; axis < 3; axis++) if (next[axis] <= end) next[axis] += step[axis];
      current = end;
    }
    return false;
  };
  return { sample, connects };
}
