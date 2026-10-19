import type { Npy3D } from "@/lib/platform/npy";

/*
 * Precipitation-shaft hydrometeors, for runs that export only the lumped
 * `total_condensate`.
 * Pure — no vtk.js, no React.
 */

/**
 * Species excluded from Nimbus's `cloud_condensate` diagnostic.
 *
 * `total_condensate` is qc+qr+qi+qs+qg+qh. Removing only rain leaves graupel and
 * hail, whose shafts sit below cloud base and would render as cloud body — in a
 * supercell that is the part of the storm with the most mass. Snow (`qs`) stays
 * by design because the engine defines suspended frozen cloud as `qi+qs`; this
 * fallback must reproduce that published diagnostic exactly.
 */
export const PRECIPITATING_SPECIES: readonly string[] = ["qr", "qg", "qh"];

/** The precipitating species this step actually published, in a stable order. */
export function availablePrecipitatingSpecies(fields: readonly string[]): string[] {
  return PRECIPITATING_SPECIES.filter((name) => fields.includes(name));
}

function sameShape(a: Npy3D, b: Npy3D): boolean {
  return a.shape.length === b.shape.length
    && a.shape.every((extent, axis) => extent === b.shape[axis]);
}

interface SumCacheNode {
  children: WeakMap<Npy3D, SumCacheNode>;
  result?: Npy3D;
}

const sumCache: SumCacheNode = { children: new WeakMap() };

function cachedSum(fields: readonly Npy3D[]): Npy3D | undefined {
  let node = sumCache;
  for (const field of fields) {
    const child = node.children.get(field);
    if (!child) return undefined;
    node = child;
  }
  return node.result;
}

function rememberSum(fields: readonly Npy3D[], result: Npy3D): void {
  let node = sumCache;
  for (const field of fields) {
    let child = node.children.get(field);
    if (!child) {
      child = { children: new WeakMap() };
      node.children.set(field, child);
    }
    node = child;
  }
  node.result = result;
}

/**
 * Elementwise sum of same-shaped fields, or null when any is missing or the
 * shapes disagree. Refusing beats summing mismatched lattices, which would
 * subtract one cell's precip from another cell's cloud.
 */
export function sumFields(fields: ReadonlyArray<Npy3D | null>): Npy3D | null {
  const present = fields.filter((field): field is Npy3D => field != null);
  if (present.length === 0) return null;
  const first = present[0]!;
  if (!present.every((field) => sameShape(field, first) && field.data.length === first.data.length)) {
    return null;
  }
  if (present.length === 1) return first;
  const retained = cachedSum(present);
  if (retained) return retained;
  const data = new Float32Array(first.data.length);
  for (const field of present) {
    for (let i = 0; i < data.length; i++) data[i] += field.data[i]!;
  }
  const result: Npy3D = {
    shape: [...first.shape] as Npy3D["shape"],
    data,
    // Co-registered bounded fields share this geometry. Dropping it moves a
    // derived window back to the domain origin even though its samples came
    // from a storm-centred ROI.
    windowGeometry: first.windowGeometry,
  };
  // Weak-keyed by every immutable source field. This removes duplicate sums
  // across shared 3D panes without creating an independent strong-reference
  // cache of full-frame arrays.
  rememberSum(present, result);
  return result;
}
