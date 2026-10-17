/* The colormap and range mode a scalar field should be drawn with by default. Pure - no React. */

import { isDbzField, type ExtendedColorscale } from "./plotColormaps";
import type { FieldUnits } from "@/lib/science/fieldUnits";

/** Range modes this rule can produce. A superset consumer may offer more (e.g. "fixed"). */
export type DefaultRangeMode = "auto" | "symmetric" | "dBZ";

export interface ScalarDisplayDefault {
  colormap: ExtendedColorscale;
  rangeMode: DefaultRangeMode;
}

/**
 * Non-negative by construction, whatever they are derived from. Checked first because the
 * quantity they are built from is itself signed: a vorticity magnitude matches "vorticity",
 * and centring a diverging ramp on zero for it discards half the colour range.
 */
function isMagnitudeField(field: string): boolean {
  return field.includes("magnitude") || field.includes("speed");
}

/** Components retain directional sign even when one particular frame happens
 * to contain only one direction. A sequential ramp would make +v and -v look
 * like different strengths of the same quantity and can change semantics as a
 * run evolves through zero. */
function isSignedVectorComponent(field: string): boolean {
  return field === "u"
    || field === "v"
    || field === "u10"
    || field === "v10"
    || field === "surface_stress_u"
    || field === "surface_stress_v";
}

/** Surface exchange follows a signed upward/downward convention. Keep the
 * zero crossing visible even when one particular frame happens to be one-sided. */
function isSignedSurfaceFlux(field: string): boolean {
  return field === "surface_sensible_heat_flux"
    || field === "surface_latent_heat_flux"
    || field === "surface_moisture_flux";
}

/** Signed quantities: a diverging ramp is only honest when its neutral colour sits on zero. */
function isSignedField(field: string): boolean {
  const f = field.toLowerCase();
  if (isMagnitudeField(f)) return false;
  return f === "w"
    || isSignedVectorComponent(f)
    || isSignedSurfaceFlux(f)
    || f.includes("vorticity")
    || f.includes("prime")
    || f.includes("pressure")
    || f.includes("divergence")
    || f.includes("_term")
    // Helicity's sign is its rotation sense — cyclonic against anticyclonic — which is
    // the whole reason it is plotted, and a sequential ramp hides where it changes.
    || f.includes("helicity")
    || f.startsWith("srh")
    // Buoyancy separates buoyant from negatively buoyant air at zero.
    || f === "buoyancy";
}

/**
 * How a field should be coloured when nothing else has been chosen for it.
 *
 * Colour choice is a property of the quantity, not of the viewer: the NWS dBZ ramp anchors its
 * colours AT reflectivity values, so it means nothing over a field measured in kg/m^2, and a
 * diverging ramp centred on zero misreads a strictly positive field. Callers re-derive this when
 * the field changes so a palette chosen for one quantity cannot outlive it.
 */
export function defaultScalarDisplay(
  field: string | null,
  fieldUnits: FieldUnits,
): ScalarDisplayDefault {
  if (!field) return { colormap: "Viridis", rangeMode: "auto" };
  if (isDbzField(field, fieldUnits)) return { colormap: "dBZ", rangeMode: "dBZ" };
  if (isSignedField(field)) return { colormap: "RdBu", rangeMode: "symmetric" };
  return { colormap: "Viridis", rangeMode: "auto" };
}
