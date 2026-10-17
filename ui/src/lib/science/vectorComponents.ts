/* Default in-plane component selection for the quiver plot. Column axis → x arrow, row axis → y arrow. */

import { sliceDisplayAxes, type SliceAxis } from "@/lib/platform/npy";

export interface VectorFamily {
  id: "velocity" | "vorticity";
  label: string;
  /** Component field on physical Cartesian axes x, y, z. */
  components: readonly [string, string, string];
}

export const VECTOR_FAMILIES: readonly VectorFamily[] = [
  { id: "velocity", label: "Velocity", components: ["u", "v", "w"] },
  {
    id: "vorticity",
    label: "Vorticity",
    components: ["vorticity_r", "vorticity_theta", "vorticity_z"],
  },
];

/** Display axes for a slice. Vertical sections always put z on screen y. */
export function inPlaneAxes(sliceAxis: SliceAxis): { rowAxis: SliceAxis; colAxis: SliceAxis } {
  return sliceDisplayAxes(sliceAxis);
}

/** Default (xField, yField) for a slice. xField = column-axis (horiz arrow), yField = row-axis (vert). */
export function defaultComponents(
  sliceAxis: SliceAxis,
  fields: string[],
): { xField: string | null; yField: string | null } {
  return availableVectorPairs(sliceAxis, fields)[0] ?? { xField: null, yField: null };
}

/** Physically valid in-plane component pairs. Arbitrary scalar fields are not
 * vectors, and components from different vector families must not be mixed. */
export function availableVectorPairs(
  sliceAxis: SliceAxis,
  fields: string[],
): Array<{ family: VectorFamily; xField: string; yField: string }> {
  const { rowAxis, colAxis } = inPlaneAxes(sliceAxis);
  // Resolve to the run's OWN spelling: matching case-insensitively and then handing back the
  // family's spelling would ask the loader for a field name the run does not publish.
  const bySpelling = new Map(fields.map((field) => [field.toLowerCase(), field]));
  const out: Array<{ family: VectorFamily; xField: string; yField: string }> = [];
  for (const family of VECTOR_FAMILIES) {
    const xField = bySpelling.get(family.components[colAxis].toLowerCase());
    const yField = bySpelling.get(family.components[rowAxis].toLowerCase());
    if (xField && yField) out.push({ family, xField, yField });
  }
  return out;
}
