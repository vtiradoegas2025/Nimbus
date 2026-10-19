export type Cm1HistoryStateSemantic = "total" | "perturbation" | "diagnostic";

export interface Cm1HistoryFieldMapping {
  nativeName: string;
  canonicalName: string;
  canonicalUnit: string;
  acceptedUnits: readonly string[];
  stateSemantic: Cm1HistoryStateSemantic;
  rank: 2 | 3;
  gridLocation: "scalar_volume" | "surface" | "column";
  transform: "identity" | "scale";
  valueScale: number;
}

/**
 * Qualified identity mappings from CM1 history output to Nimbus plot fields.
 *
 * Every entry is written by CM1 on scalar points and has the same physical
 * quantity as the canonical Nimbus field. Non-identity units require an
 * explicit numeric scale in this table. Variables that require destaggering,
 * undeclared conversion, base-state reconstruction, or ambiguous
 * scheme-dependent interpretation intentionally do not appear here.
 */
export const CM1_HISTORY_FIELD_MAPPINGS: readonly Cm1HistoryFieldMapping[] = [
  { nativeName: "uinterp", canonicalName: "u", canonicalUnit: "m/s", acceptedUnits: ["m/s", "m s-1", "m s^-1", "m s**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "vinterp", canonicalName: "v", canonicalUnit: "m/s", acceptedUnits: ["m/s", "m s-1", "m s^-1", "m s**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "winterp", canonicalName: "w", canonicalUnit: "m/s", acceptedUnits: ["m/s", "m s-1", "m s^-1", "m s**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "th", canonicalName: "theta", canonicalUnit: "K", acceptedUnits: ["K", "kelvin"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "thpert", canonicalName: "theta_prime", canonicalUnit: "K", acceptedUnits: ["K", "kelvin"], stateSemantic: "perturbation", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "prs", canonicalName: "p", canonicalUnit: "Pa", acceptedUnits: ["Pa", "pascal", "pascals"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "prspert", canonicalName: "p_prime", canonicalUnit: "Pa", acceptedUnits: ["Pa", "pascal", "pascals"], stateSemantic: "perturbation", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "rho", canonicalName: "rho", canonicalUnit: "kg/m^3", acceptedUnits: ["kg/m^3", "kg m-3", "kg m^-3", "kg m**-3"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "qv", canonicalName: "qv", canonicalUnit: "kg/kg", acceptedUnits: ["kg/kg", "kg kg-1", "kg kg^-1", "kg kg**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "qc", canonicalName: "qc", canonicalUnit: "kg/kg", acceptedUnits: ["kg/kg", "kg kg-1", "kg kg^-1", "kg kg**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "qr", canonicalName: "qr", canonicalUnit: "kg/kg", acceptedUnits: ["kg/kg", "kg kg-1", "kg kg^-1", "kg kg**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "qi", canonicalName: "qi", canonicalUnit: "kg/kg", acceptedUnits: ["kg/kg", "kg kg-1", "kg kg^-1", "kg kg**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "qs", canonicalName: "qs", canonicalUnit: "kg/kg", acceptedUnits: ["kg/kg", "kg kg-1", "kg kg^-1", "kg kg**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "qg", canonicalName: "qg", canonicalUnit: "kg/kg", acceptedUnits: ["kg/kg", "kg kg-1", "kg kg^-1", "kg kg**-1"], stateSemantic: "total", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "dbz", canonicalName: "reflectivity_dbz", canonicalUnit: "dBZ", acceptedUnits: ["dBZ"], stateSemantic: "diagnostic", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "zvort", canonicalName: "vorticity_z", canonicalUnit: "1/s", acceptedUnits: ["1/s", "s-1", "s^-1", "s**-1"], stateSemantic: "diagnostic", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "buoyancy", canonicalName: "buoyancy", canonicalUnit: "m/s^2", acceptedUnits: ["m/s^2", "m s-2", "m s^-2", "m s**-2"], stateSemantic: "diagnostic", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "pv", canonicalName: "potential_vorticity", canonicalUnit: "K m^2/(kg s)", acceptedUnits: ["K m2/kg/s", "K m^2/kg/s", "K m2 kg-1 s-1", "K m^2 kg^-1 s^-1"], stateSemantic: "diagnostic", rank: 3, gridLocation: "scalar_volume", transform: "identity", valueScale: 1 },
  { nativeName: "cref", canonicalName: "composite_reflectivity", canonicalUnit: "dBZ", acceptedUnits: ["dBZ"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "cape", canonicalName: "cape", canonicalUnit: "J/kg", acceptedUnits: ["J/kg", "J kg-1", "J kg^-1", "J kg**-1"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "cin", canonicalName: "cin", canonicalUnit: "J/kg", acceptedUnits: ["J/kg", "J kg-1", "J kg^-1", "J kg**-1"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "lcl", canonicalName: "lcl", canonicalUnit: "m", acceptedUnits: ["m", "meter", "meters", "metre", "metres"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "lfc", canonicalName: "lfc", canonicalUnit: "m", acceptedUnits: ["m", "meter", "meters", "metre", "metres"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "uh", canonicalName: "updraft_helicity", canonicalUnit: "m^2/s^2", acceptedUnits: ["m2/s2", "m^2/s^2", "m2 s-2", "m^2 s^-2", "m**2 s**-2"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "pwat", canonicalName: "precipitable_water", canonicalUnit: "mm", acceptedUnits: ["m", "meter", "meters", "metre", "metres"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "scale", valueScale: 1_000 },
  { nativeName: "lwp", canonicalName: "liquid_water_path", canonicalUnit: "kg/m^2", acceptedUnits: ["kg/m^2", "kg/m2", "kg m-2", "kg m^-2", "kg m**-2"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "cwp", canonicalName: "cloud_water_path", canonicalUnit: "kg/m^2", acceptedUnits: ["kg/m^2", "kg/m2", "kg m-2", "kg m^-2", "kg m**-2"], stateSemantic: "diagnostic", rank: 2, gridLocation: "column", transform: "identity", valueScale: 1 },
  { nativeName: "rain", canonicalName: "accumulated_rainfall", canonicalUnit: "mm", acceptedUnits: ["cm", "centimeter", "centimeters", "centimetre", "centimetres"], stateSemantic: "diagnostic", rank: 2, gridLocation: "surface", transform: "scale", valueScale: 10 },
  { nativeName: "prate", canonicalName: "precip_rate", canonicalUnit: "mm/h", acceptedUnits: ["kg/m2/s", "kg/m^2/s", "kg m-2 s-1", "kg m^-2 s^-1", "kg m**-2 s**-1"], stateSemantic: "diagnostic", rank: 2, gridLocation: "surface", transform: "scale", valueScale: 3_600 },
] as const;

export const CM1_HISTORY_FIELD_MAP: Readonly<Record<string, string>> = Object.fromEntries(
  CM1_HISTORY_FIELD_MAPPINGS.map(({ nativeName, canonicalName }) => [nativeName, canonicalName]),
);

function normalizedUnit(unit: string): string {
  return unit.trim().toLowerCase().replace(/\u2212/g, "-").replace(/\s+/g, " ");
}

export function cm1HistoryFieldMapping(
  nativeName: string,
  units: string | null,
): Cm1HistoryFieldMapping | null {
  const mapping = CM1_HISTORY_FIELD_MAPPINGS.find(
    (candidate) => candidate.nativeName === nativeName,
  );
  if (!mapping || units == null) return null;
  const actual = normalizedUnit(units);
  return mapping.acceptedUnits.some((unit) => normalizedUnit(unit) === actual) ? mapping : null;
}

/** Name-only lookup for identities and UI copy; dataset qualification must use
 * cm1HistoryFieldMapping so a familiar name with missing/wrong units fails. */
export function canonicalCm1Field(nativeName: string): string | null {
  return CM1_HISTORY_FIELD_MAPPINGS.find(
    (mapping) => mapping.nativeName === nativeName,
  )?.canonicalName ?? null;
}

export function nativeCm1Field(canonicalName: string): string | null {
  return CM1_HISTORY_FIELD_MAPPINGS.find(
    (mapping) => mapping.canonicalName === canonicalName,
  )?.nativeName ?? null;
}

export function cm1HistoryMappingForCanonical(
  canonicalName: string,
): Cm1HistoryFieldMapping | null {
  return CM1_HISTORY_FIELD_MAPPINGS.find(
    (mapping) => mapping.canonicalName === canonicalName,
  ) ?? null;
}

/** Applies the catalog's declared value transform after native payload
 * validation. Identity fields retain their original read-only array. */
export function transformCm1HistoryValues(
  canonicalName: string,
  values: Float32Array,
): Float32Array {
  const mapping = cm1HistoryMappingForCanonical(canonicalName);
  if (!mapping || mapping.valueScale === 1) return values;
  return values.map((value) => value * mapping.valueScale);
}

export function cm1AxisIndex(name: string): 0 | 1 | 2 | null {
  const normalized = name.toLowerCase();
  if (normalized.startsWith("x") || normalized === "ni" || normalized === "nip1") return 0;
  if (normalized.startsWith("y") || normalized === "nj" || normalized === "njp1") return 1;
  if (normalized.startsWith("z") || normalized === "nk" || normalized === "nkp1") return 2;
  return null;
}

/** Returns the Studio x/y/z shape only when a CM1 variable is a numeric-style
 * 3-D field after an optional leading time dimension. Ambiguous, duplicate,
 * or unknown axes fail closed instead of becoming selectable plot fields. */
export function canonicalCm1Shape(
  dimensions: readonly string[],
  shape: readonly number[],
): [number, number, number] | null {
  if (dimensions.length !== shape.length) return null;
  const skip = dimensions[0]?.toLowerCase() === "time" ? 1 : 0;
  const spatialDimensions = dimensions.slice(skip);
  const spatialShape = shape.slice(skip);
  if (spatialDimensions.length !== 3) return null;
  const result: [number, number, number] = [0, 0, 0];
  for (let index = 0; index < 3; index += 1) {
    const axis = cm1AxisIndex(spatialDimensions[index]);
    const length = spatialShape[index];
    if (axis == null || result[axis] !== 0 || !Number.isSafeInteger(length) || length <= 0) return null;
    result[axis] = length;
  }
  return result.every((length) => length > 0) ? result : null;
}

const CM1_SCALAR_DIMENSIONS = [
  new Set(["xh", "x", "ni"]),
  new Set(["yh", "y", "nj"]),
  new Set(["zh", "z", "nk"]),
] as const;

/** Canonical shape for a CM1 scalar-point volume. Face dimensions (xf/yf/zf
 * and nip1/njp1/nkp1) are rejected even when their lengths happen to match. */
export function canonicalCm1ScalarShape(
  dimensions: readonly string[],
  shape: readonly number[],
): [number, number, number] | null {
  const canonical = canonicalCm1Shape(dimensions, shape);
  if (!canonical) return null;
  const skip = dimensions[0]?.toLowerCase() === "time" ? 1 : 0;
  const spatialDimensions = dimensions.slice(skip);
  for (let index = 0; index < spatialDimensions.length; index += 1) {
    const name = spatialDimensions[index].toLowerCase();
    const axis = cm1AxisIndex(name);
    if (axis == null || !CM1_SCALAR_DIMENSIONS[axis].has(name)) return null;
  }
  return canonical;
}

/** Canonical x/y shape for CM1's two-dimensional scalar grid. */
export function canonicalCm1HorizontalShape(
  dimensions: readonly string[],
  shape: readonly number[],
): [number, number] | null {
  if (dimensions.length !== shape.length) return null;
  const skip = dimensions[0]?.toLowerCase() === "time" ? 1 : 0;
  const spatialDimensions = dimensions.slice(skip);
  const spatialShape = shape.slice(skip);
  if (spatialDimensions.length !== 2) return null;
  const result: [number, number] = [0, 0];
  for (let index = 0; index < 2; index += 1) {
    const name = spatialDimensions[index].toLowerCase();
    const axis = cm1AxisIndex(name);
    const length = spatialShape[index];
    if ((axis !== 0 && axis !== 1) || !CM1_SCALAR_DIMENSIONS[axis].has(name)
        || result[axis] !== 0 || !Number.isSafeInteger(length) || length <= 0) return null;
    result[axis] = length;
  }
  return result.every((length) => length > 0) ? result : null;
}

export interface Cm1HistoryVariableAssessment {
  status: "ready" | "supporting" | "inspect_only";
  canonicalName: string | null;
  reason: string;
}

/** Explains the same fail-closed decision used by the frame indexer. This is
 * intentionally user-facing: an inspectable variable must not simply vanish
 * from Plot Studio when its units, rank, staggering, or semantics are unsafe. */
export function assessCm1HistoryVariable(variable: {
  name: string;
  units: string | null;
  dimensions: readonly string[];
  shape: readonly number[];
  cfDecode?: {
    status: "identity" | "applied" | "unsupported";
    issue: string | null;
    scaleFactor: number | null;
    addOffset: number | null;
  };
}): Cm1HistoryVariableAssessment {
  if (variable.cfDecode?.status === "unsupported") {
    return {
      status: "inspect_only",
      canonicalName: null,
      reason: `CF packed/missing-value metadata is unsupported: ${variable.cfDecode.issue ?? "invalid decode policy"}`,
    };
  }
  const normalizedName = variable.name.toLowerCase();
  const supportingCoordinate = normalizedName === "time"
    || variable.dimensions.length === 1 && (
      CM1_SCALAR_DIMENSIONS.some((names) => names.has(normalizedName))
      || variable.dimensions[0]?.toLowerCase() === normalizedName
    );
  if (supportingCoordinate) {
    return {
      status: "supporting",
      canonicalName: null,
      reason: normalizedName === "time" ? "Timeline coordinate" : "Physical grid coordinate",
    };
  }

  const declared = CM1_HISTORY_FIELD_MAPPINGS.find(({ nativeName }) => nativeName === variable.name);
  if (!declared) {
    if (["u", "v", "w"].includes(variable.name)) {
      return { status: "inspect_only", canonicalName: null, reason: "Native face-staggered wind requires a qualified destaggering transform" };
    }
    if (["xvort", "yvort"].includes(variable.name)) {
      return { status: "inspect_only", canonicalName: null, reason: "Cartesian horizontal vorticity is blocked until Nimbus migrates its legacy radial/azimuthal component IDs" };
    }
    if (variable.name === "div") {
      return { status: "inspect_only", canonicalName: null, reason: "No exact CM1 history definition is qualified against Nimbus flow divergence" };
    }
    if (["svs", "shs"].includes(variable.name)) {
      return { status: "inspect_only", canonicalName: null, reason: "CM1 running-extremum diagnostic is not interchangeable with an instantaneous Nimbus field" };
    }
    if (variable.name === "qh") {
      return { status: "inspect_only", canonicalName: null, reason: "CM1 qh species meaning depends on the selected microphysics scheme; no universal hail identity is qualified" };
    }
    return { status: "inspect_only", canonicalName: null, reason: "No qualified CM1-to-Nimbus field rule" };
  }

  const mapping = cm1HistoryFieldMapping(variable.name, variable.units);
  if (!mapping) {
    return {
      status: "inspect_only",
      canonicalName: declared.canonicalName,
      reason: variable.units == null
        ? `Units are missing; expected ${declared.acceptedUnits.join(" or ")}`
        : `Units '${variable.units}' do not match the qualified ${declared.canonicalUnit} contract`,
    };
  }
  const shape = mapping.rank === 2
    ? canonicalCm1HorizontalShape(variable.dimensions, variable.shape)
    : canonicalCm1ScalarShape(variable.dimensions, variable.shape);
  if (!shape) {
    return {
      status: "inspect_only",
      canonicalName: mapping.canonicalName,
      reason: mapping.rank === 2
        ? "Expected a two-dimensional scalar x/y field after optional time"
        : "Expected a three-dimensional scalar x/y/z field after optional time",
    };
  }
  return {
    status: "ready",
    canonicalName: mapping.canonicalName,
    reason: [
      mapping.valueScale === 1
        ? `Plot-ready as ${mapping.canonicalName}`
        : `Plot-ready as ${mapping.canonicalName} with declared ×${mapping.valueScale} unit scaling`,
      variable.cfDecode?.status === "applied"
        ? `CF stored-value decoding applied${variable.cfDecode.scaleFactor != null || variable.cfDecode.addOffset != null ? " before plotting" : " for missing values"}`
        : null,
    ].filter((value): value is string => value != null).join(" · "),
  };
}

export function cm1LengthScaleToMeters(units: string | null): number | null {
  const normalized = units?.trim().toLowerCase();
  if (!normalized) return null;
  if (["m", "meter", "meters", "metre", "metres"].includes(normalized)) return 1;
  if (["km", "kilometer", "kilometers", "kilometre", "kilometres"].includes(normalized)) return 1_000;
  return null;
}

export function cm1TimeScaleToSeconds(units: string | null): number | null {
  const normalized = units?.trim().toLowerCase();
  if (!normalized) return null;
  if (/^(?:s|sec|secs|second|seconds)(?:\s+since\b|$)/.test(normalized)) return 1;
  if (/^(?:min|mins|minute|minutes)(?:\s+since\b|$)/.test(normalized)) return 60;
  if (/^(?:h|hr|hrs|hour|hours)(?:\s+since\b|$)/.test(normalized)) return 3_600;
  return null;
}

export interface UniformCoordinate {
  origin: number;
  spacing: number;
  lowerBound: number;
  upperBound: number;
}

export interface Cm1ScalarCoordinate {
  valuesM: number[];
  lowerBound: number;
  upperBound: number;
  uniform: UniformCoordinate | null;
}

/** Validate and convert one declared CM1 scalar-center axis. Physical axes
 * must be finite, strictly increasing, have known length units, and contain at
 * least two centers. Stretched spacing is retained exactly. */
export function cm1ScalarCoordinate(
  values: readonly number[],
  scale: number | null,
): Cm1ScalarCoordinate | null {
  if (scale == null || !(scale > 0) || values.length < 2 || values.some((value) => !Number.isFinite(value))) return null;
  const valuesM = values.map((value) => value * scale);
  if (valuesM.some((value) => !Number.isFinite(value))) return null;
  const firstSpacing = valuesM[1] - valuesM[0];
  if (!(firstSpacing > 0)) return null;
  let isUniform = true;
  const tolerance = Math.max(1e-7, Math.abs(firstSpacing) * 1e-5);
  for (let index = 2; index < valuesM.length; index += 1) {
    const spacing = valuesM[index] - valuesM[index - 1];
    if (!(spacing > 0)) return null;
    if (Math.abs(spacing - firstSpacing) > tolerance) isUniform = false;
  }
  const lastSpacing = valuesM.at(-1)! - valuesM.at(-2)!;
  return {
    valuesM,
    lowerBound: valuesM[0] - firstSpacing / 2,
    upperBound: valuesM.at(-1)! + lastSpacing / 2,
    uniform: isUniform ? {
      origin: valuesM[0],
      spacing: firstSpacing,
      lowerBound: valuesM[0] - firstSpacing / 2,
      upperBound: valuesM.at(-1)! + firstSpacing / 2,
    } : null,
  };
}

/** Legacy uniform-spacing fast path. The broader cm1ScalarCoordinate contract
 * retains valid stretched axes; this helper returns null for them. */
export function uniformCm1Coordinate(
  values: readonly number[],
  scale: number | null,
): UniformCoordinate | null {
  return cm1ScalarCoordinate(values, scale)?.uniform ?? null;
}

export function isNetcdfDatasetPath(path: string): boolean {
  return /\.(?:nc|nc4|cdf|netcdf)$/i.test(path);
}
