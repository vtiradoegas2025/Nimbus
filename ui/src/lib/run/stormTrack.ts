/*
 * Parse and validate storm-track JSON emitted by the engine.
 * Extracts both Earth-world and grid-relative position variants.
 * Pure module: no I/O, no vtk.js, no React.
 */

/** Rotation class string as written by the engine. */
export type RotationClass = string;

/** Grid-relative 2D position (metres). */
export interface GridPoint2D {
  xM: number;
  yM: number;
}

/** Typed summary of one storm-track snapshot (UI-relevant fields only). Immutable once constructed. */
export interface StormTrackSnapshot {
  /** Engine schema identifier ("nimbus.storm_feature"). */
  schema: string;
  /** Schema version number from the file. */
  schemaVersion: number;
  /** Simulation step that produced this snapshot. */
  completedStep: number;
  /** Simulation time (s). */
  simulationTimeS: number;

  /** False when no storm feature was detected; centroid and UH fields are null. */
  featureAvailable: boolean;
  /** Earth-relative core centroid (m), in the same world frame as current plot grids. */
  earthCoreCentroidM: GridPoint2D | null;
  /** Earth-relative updraft centroid (m), in the same world frame as current plot grids. */
  earthCentroidM: GridPoint2D | null;
  /** Grid-relative core centroid (m). From grid_core_centroid_m (no earth-origin offset). */
  gridCoreCentroidM: GridPoint2D | null;
  /** Grid-relative updraft centroid (m). From grid_centroid_m. */
  gridCentroidM: GridPoint2D | null;
  /** Maximum absolute updraft helicity (m² s⁻²). Null when featureAvailable is false. */
  maximumAbsoluteUpdraftHelicityM2S2: number | null;
  /** Rotation class (e.g. "rotating", "strongly_rotating"). Null when featureAvailable is false. */
  rotationClass: RotationClass | null;
}

/** Returned when parsing succeeds. */
export interface StormTrackParseSuccess {
  ok: true;
  snapshot: StormTrackSnapshot;
}

/** Returned on recoverable parse failure. Never thrown — partial JSON is expected during a live run. */
export interface StormTrackParseFailure {
  ok: false;
  reason:
    | "absent"       // caller passed null/undefined
    | "json_error"   // JSON.parse failed
    | "schema_mismatch" // not a nimbus.storm_feature document
    | "missing_keys";   // required top-level keys absent
  detail: string;
}

export type StormTrackParseResult =
  | StormTrackParseSuccess
  | StormTrackParseFailure;

/** Parse a storm-track JSON string or object into a typed snapshot. Never throws; returns a typed failure on bad input. */
export function parseStormTrack(
  input: string | object | null | undefined,
): StormTrackParseResult {
  if (input == null) {
    return { ok: false, reason: "absent", detail: "No input provided (tracking disabled or file absent)" };
  }

  let raw: unknown;
  if (typeof input === "string") {
    try {
      raw = JSON.parse(input);
    } catch (e) {
      return {
        ok: false,
        reason: "json_error",
        detail: `JSON.parse failed: ${(e as Error).message}`,
      };
    }
  } else {
    raw = input;
  }

  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return {
      ok: false,
      reason: "json_error",
      detail: "Parsed value is not a JSON object",
    };
  }

  const obj = raw as Record<string, unknown>;

  // Verify the schema identifier before reading any other field.
  if (obj["schema"] !== "nimbus.storm_feature") {
    return {
      ok: false,
      reason: "schema_mismatch",
      detail: `Expected schema "nimbus.storm_feature", got ${JSON.stringify(obj["schema"])}`,
    };
  }

  // Required structural keys.
  const requiredKeys = [
    "schema_version",
    "completed_step",
    "simulation_time_s",
    "selected_feature_available",
  ] as const;
  for (const key of requiredKeys) {
    if (!(key in obj)) {
      return {
        ok: false,
        reason: "missing_keys",
        detail: `Required key "${key}" is absent`,
      };
    }
  }

  const schemaVersion = Number(obj["schema_version"]);
  const completedStep = Number(obj["completed_step"]);
  const simulationTimeS = Number(obj["simulation_time_s"]);
  const featureAvailable = Boolean(obj["selected_feature_available"]);

  if (!Number.isFinite(schemaVersion) || !Number.isFinite(completedStep) || !Number.isFinite(simulationTimeS)) {
    return {
      ok: false,
      reason: "missing_keys",
      detail: "Numeric header fields (schema_version, completed_step, simulation_time_s) are not finite numbers",
    };
  }

  // Optional fields — all gated on featureAvailable so we do not misreport
  // stale values from a step where no storm was tracked.
  let earthCoreCentroidM: GridPoint2D | null = null;
  let earthCentroidM: GridPoint2D | null = null;
  let gridCoreCentroidM: GridPoint2D | null = null;
  let gridCentroidM: GridPoint2D | null = null;
  let maximumAbsoluteUpdraftHelicityM2S2: number | null = null;
  let rotationClass: RotationClass | null = null;

  if (featureAvailable) {
    const ecc = obj["core_centroid_m"];
    if (isXYObject(ecc)) {
      earthCoreCentroidM = { xM: Number(ecc.x), yM: Number(ecc.y) };
    }
    const ec = obj["centroid_m"];
    if (isXYObject(ec)) {
      earthCentroidM = { xM: Number(ec.x), yM: Number(ec.y) };
    }

    // grid_core_centroid_m: grid-relative, no earth-origin offset.
    const gcc = obj["grid_core_centroid_m"];
    if (isXYObject(gcc)) {
      gridCoreCentroidM = { xM: Number(gcc.x), yM: Number(gcc.y) };
    }

    // grid_centroid_m: grid-relative updraft centroid.
    const gc = obj["grid_centroid_m"];
    if (isXYObject(gc)) {
      gridCentroidM = { xM: Number(gc.x), yM: Number(gc.y) };
    }

    // maximum_absolute_updraft_helicity_m2s2 lives at the top level.
    const uh = obj["maximum_absolute_updraft_helicity_m2s2"];
    if (typeof uh === "number" && Number.isFinite(uh)) {
      maximumAbsoluteUpdraftHelicityM2S2 = uh;
    }

    // The primary rotation candidate is the first branch_candidates entry.
    // Fall back to the top-level if branch_candidates is absent.
    const candidates = obj["branch_candidates"];
    if (Array.isArray(candidates) && candidates.length > 0) {
      const primary = candidates[0];
      if (typeof primary === "object" && primary !== null) {
        const rc = (primary as Record<string, unknown>)["rotation_class"];
        if (typeof rc === "string") {
          rotationClass = rc;
        }
      }
    }
  }

  return {
    ok: true,
    snapshot: {
      schema: String(obj["schema"]),
      schemaVersion,
      completedStep,
      simulationTimeS,
      featureAvailable,
      earthCoreCentroidM,
      earthCentroidM,
      gridCoreCentroidM,
      gridCentroidM,
      maximumAbsoluteUpdraftHelicityM2S2,
      rotationClass,
    },
  };
}

/** Earth-world storm centre for VTK camera bearing/range calculations. */
export function stormTrackCentre(
  snapshot: StormTrackSnapshot | null,
): [number, number, number] | undefined {
  if (!snapshot?.featureAvailable) return undefined;
  // Current schemas publish Earth variants. Grid variants remain a stationary
  // legacy fallback; moving-domain files always carry the Earth fields.
  const centre = snapshot.earthCoreCentroidM
    ?? snapshot.earthCentroidM
    ?? snapshot.gridCoreCentroidM
    ?? snapshot.gridCentroidM;
  if (!centre) return undefined;
  return [centre.xM, centre.yM, 0];
}

/** True only when a run-level latest tracker snapshot belongs to the rendered frame. */
export function stormTrackMatchesFrame(
  snapshot: StormTrackSnapshot | null,
  simulationTimeS: number | null | undefined,
): boolean {
  if (!snapshot || simulationTimeS == null || !Number.isFinite(simulationTimeS)) return false;
  const scale = Math.max(1, Math.abs(snapshot.simulationTimeS), Math.abs(simulationTimeS));
  return Math.abs(snapshot.simulationTimeS - simulationTimeS) <= Math.max(1e-6, scale * 1e-9);
}

/** Type guard: object with numeric x and y properties. */
function isXYObject(value: unknown): value is { x: number; y: number } {
  return (
    typeof value === "object" &&
    value !== null &&
    "x" in value &&
    "y" in value &&
    typeof (value as Record<string, unknown>).x === "number" &&
    typeof (value as Record<string, unknown>).y === "number"
  );
}
