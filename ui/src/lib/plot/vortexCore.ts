/*
 * Which side of a signed field is the vortex core, and how to label the pair.
 * Pure — no vtk.js, no React.
 */

/** Fields the core layer can be thresholded on, in preference order. */
export const VORTEX_CORE_FIELDS: readonly string[] = ["vorticity_z", "p_prime"];

/** Conservative lower bound commonly used to identify mesocyclonic rotation. */
export const MESOCYCLONE_VORTICITY_FLOOR_PER_S = 0.01;

/**
 * Sign of the values that form the core itself.
 *
 * Vertical vorticity is positive for cyclonic rotation, so the core is the
 * positive side. Perturbation pressure is the opposite: a vortex core is a
 * dynamic LOW, so its values are negative. Emphasising one shared sign draws a
 * pressure high where the user asked for the well.
 */
export function vortexCoreSign(fieldName: string | null | undefined): 1 | -1 {
  return fieldName === "p_prime" ? -1 : 1;
}

/**
 * Signed feature peak used by the relative core-threshold control.
 *
 * A tornado can occupy far less than two percent of a large 3-D region, so a
 * whole-volume p98 can sit in environmental noise and make the slider unable
 * to reach the actual core. The selected sign's resolved extremum keeps the
 * full slider range meaningful for compact vortices.
 */
export function vortexCoreReferenceMagnitude(
  fieldName: string | null | undefined,
  stats: { min: number; max: number; p2: number; p98: number },
): number {
  const signedPeak = vortexCoreSign(fieldName) > 0 ? stats.max : -stats.min;
  if (Number.isFinite(signedPeak) && signedPeak > 0) return signedPeak;
  return Math.max(Math.abs(stats.p2), Math.abs(stats.p98));
}

/**
 * Choose the dimensional reference used by the relative vortex slider.
 *
 * Recomputing the reference from every animation frame changes the contour
 * value even when the slider is untouched. The resulting surface expands and
 * contracts with the frame maximum and reads as renderer flicker. Playback
 * therefore retains the last settled reference; paused inspection follows the
 * selected frame. A transient preparation gap also retains the last reference
 * instead of collapsing the contour to null for one React commit.
 */
export function stableVortexReferenceMagnitude(
  current: number | null | undefined,
  retained: number | null | undefined,
  freeze: boolean,
): number | null {
  const currentValid = current != null && Number.isFinite(current) && current > 0;
  const retainedValid = retained != null && Number.isFinite(retained) && retained > 0;
  if (freeze && retainedValid) return retained;
  if (currentValid) return current;
  return retainedValid ? retained : null;
}

export interface VortexCoreContours {
  /** Contour for the emphasised surface — the feature the layer is named for. */
  core: number;
  /** Contour for the de-emphasised surface of the opposite sign. */
  opposite: number;
}

/** Contour values for the pair, or null when the threshold cannot define a surface. */
export function vortexCoreContours(
  fieldName: string | null | undefined,
  threshold: number,
): VortexCoreContours | null {
  if (!Number.isFinite(threshold) || threshold <= 0) return null;
  const sign = vortexCoreSign(fieldName);
  return { core: sign * threshold, opposite: -sign * threshold };
}

/**
 * A lower, still dimensional vorticity contour for the circulation surrounding
 * the compact core. It is never inferred from pressure and never allowed to
 * collapse onto the core contour, because then it would disclose no hierarchy.
 */
export function mesocycloneEnvelopeThreshold(
  fieldName: string | null | undefined,
  coreThreshold: number,
): number | null {
  if (fieldName !== "vorticity_z" || !Number.isFinite(coreThreshold)) return null;
  const threshold = Math.max(MESOCYCLONE_VORTICITY_FLOOR_PER_S, coreThreshold * 0.25);
  return threshold < coreThreshold * 0.8 ? threshold : null;
}

export type VortexSurfaceBounds = [number, number, number, number, number, number];

export interface MesocycloneEnvelopeAssessment {
  present: boolean;
  depthM: number;
  diameterM: number;
  reason: "present" | "empty" | "detached" | "too-shallow" | "too-narrow" | "boundary-clipped";
}

export interface MesocycloneContextAssessment {
  qualified: boolean;
  spanM: [number, number, number];
  reason: "qualified" | "too-narrow" | "too-shallow" | "invalid";
}

/** A mesocyclone label requires a lattice physically capable of containing
 * the minimum qualifying circulation. At metre scale the tornado ROI may be
 * intentionally much smaller; skipping this contour avoids expensive work
 * that could never support the claimed meteorological scale. */
export function assessMesocycloneContext(
  shape: readonly number[] | null | undefined,
  spacing: readonly number[] | null | undefined,
): MesocycloneContextAssessment {
  if (!shape || !spacing || shape.length < 3 || spacing.length < 3
      || shape.slice(0, 3).some((count) => !Number.isSafeInteger(count) || count < 1)
      || spacing.slice(0, 3).some((step) => !Number.isFinite(step) || step <= 0)) {
    return { qualified: false, spanM: [0, 0, 0], reason: "invalid" };
  }
  const spanM = shape.slice(0, 3).map((count, axis) => count * spacing[axis]) as
    [number, number, number];
  if (Math.min(spanM[0], spanM[1]) < 1_500) {
    return { qualified: false, spanM, reason: "too-narrow" };
  }
  if (spanM[2] < 2_000) {
    return { qualified: false, spanM, reason: "too-shallow" };
  }
  return { qualified: true, spanM, reason: "qualified" };
}

export interface TornadoVortexAssessment {
  present: boolean;
  depthM: number;
  diameterM: number;
  reason: "present" | "empty" | "under-resolved" | "aloft" | "too-shallow" | "too-broad";
}

function finiteBounds(bounds: VortexSurfaceBounds | null): bounds is VortexSurfaceBounds {
  return bounds != null && bounds.every(Number.isFinite)
    && bounds[1] >= bounds[0] && bounds[3] >= bounds[2] && bounds[5] >= bounds[4];
}

/**
 * Decide whether the lower-threshold connected surface is resolved strongly
 * enough to call mesocyclone-scale. Bounds come from the actual marching-cubes
 * output. Grid-relative minima prevent a four-cell feature from acquiring a
 * meteorological label merely because the display lattice is coarse.
 */
export function assessMesocycloneEnvelope(
  envelope: VortexSurfaceBounds | null,
  core: VortexSurfaceBounds | null,
  spacing: readonly [number, number, number],
  contextBounds?: VortexSurfaceBounds | null,
): MesocycloneEnvelopeAssessment {
  if (!finiteBounds(envelope) || !finiteBounds(core)) {
    return { present: false, depthM: 0, diameterM: 0, reason: "empty" };
  }
  const depthM = envelope[5] - envelope[4];
  const diameterM = Math.max(envelope[1] - envelope[0], envelope[3] - envelope[2]);
  const coreCenterX = (core[0] + core[1]) * 0.5;
  const coreCenterY = (core[2] + core[3]) * 0.5;
  const coreCenterZ = (core[4] + core[5]) * 0.5;
  const enclosesCore = coreCenterX >= envelope[0] && coreCenterX <= envelope[1]
    && coreCenterY >= envelope[2] && coreCenterY <= envelope[3]
    && coreCenterZ >= envelope[4] && coreCenterZ <= envelope[5];
  if (!enclosesCore) return { present: false, depthM, diameterM, reason: "detached" };
  const finiteContextBounds = finiteBounds(contextBounds ?? null) ? contextBounds! : null;
  if (finiteContextBounds) {
    const padX = Math.max(Number.EPSILON, spacing[0] * 1.01);
    const padY = Math.max(Number.EPSILON, spacing[1] * 1.01);
    const padZ = Math.max(Number.EPSILON, spacing[2] * 1.01);
    // Ground contact is physically allowed. Contact with a horizontal wall or
    // the top of the retained brick means the circulation was clipped by the
    // renderer ROI and cannot be assigned a complete mesoscale envelope.
    const clipped = envelope[0] <= finiteContextBounds[0] + padX
      || envelope[1] >= finiteContextBounds[1] - padX
      || envelope[2] <= finiteContextBounds[2] + padY
      || envelope[3] >= finiteContextBounds[3] - padY
      || envelope[5] >= finiteContextBounds[5] - padZ;
    if (clipped) {
      return { present: false, depthM, diameterM, reason: "boundary-clipped" };
    }
  }
  if (depthM < Math.max(2_000, spacing[2] * 4)) {
    return { present: false, depthM, diameterM, reason: "too-shallow" };
  }
  if (diameterM < Math.max(1_500, Math.max(spacing[0], spacing[1]) * 4)) {
    return { present: false, depthM, diameterM, reason: "too-narrow" };
  }
  return { present: true, depthM, diameterM, reason: "present" };
}

/**
 * Resolution-aware classification for the compact component drawn as the
 * lowest member of the vortex hierarchy. This never manufactures a funnel:
 * it only decides whether solved vertical-vorticity geometry supports the
 * more specific "tornado-scale vortex" label.
 */
export function assessTornadoScaleVortex(
  core: VortexSurfaceBounds | null,
  spacing: readonly [number, number, number],
  lowestModelHeightM: number,
): TornadoVortexAssessment {
  if (!finiteBounds(core)) {
    return { present: false, depthM: 0, diameterM: 0, reason: "empty" };
  }
  const horizontalSpacing = Math.max(spacing[0], spacing[1]);
  const depthM = core[5] - core[4];
  const diameterM = Math.max(core[1] - core[0], core[3] - core[2]);
  if (horizontalSpacing > 250 || spacing[2] > 250) {
    return { present: false, depthM, diameterM, reason: "under-resolved" };
  }
  if (core[4] > lowestModelHeightM + Math.max(500, spacing[2] * 4)) {
    return { present: false, depthM, diameterM, reason: "aloft" };
  }
  if (depthM < Math.max(500, spacing[2] * 4)) {
    return { present: false, depthM, diameterM, reason: "too-shallow" };
  }
  if (diameterM > 3_000) {
    return { present: false, depthM, diameterM, reason: "too-broad" };
  }
  return { present: true, depthM, diameterM, reason: "present" };
}

/** Compact signed-threshold readout: zeta is ~1e-2, p' is ~1e2. */
export function formatCoreThreshold(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 0.01 || magnitude >= 1e5)) {
    return value.toExponential(1);
  }
  return Number(value.toPrecision(3)).toString();
}

/** Legend text for one contour, as the inequality it actually draws. */
export function vortexCoreLabel(contour: number): string {
  return contour >= 0
    ? `>= +${formatCoreThreshold(contour)}`
    : `<= -${formatCoreThreshold(Math.abs(contour))}`;
}

/**
 * What each surface means, for the legend. Rotation reads as a direction;
 * pressure reads as a departure from the base state, and calling a pressure
 * excess "anticyclonic" would name a rotation the field cannot show.
 */
export function vortexCoreDescription(
  fieldName: string | null | undefined,
  which: "core" | "opposite",
): string {
  if (fieldName === "p_prime") {
    return which === "core" ? "pressure deficit" : "pressure excess";
  }
  return which === "core" ? "cyclonic" : "anticyclonic";
}
