/*
 * Domain-motion frame semantics for the plotting layer (see docs/domain.md).
 * A moving domain means cell (i,j) at t=0 and t=3600s are different Earth locations.
 * Time-accumulating plots must check canAccumulateByCell; policy is fail-closed.
 */

/** Motion modes declared by the strict `domain_motion.mode` configuration. */
export type DomainMotionMode =
  | "stationary"
  | "constant_velocity"
  | "tracked_window"
  | "moving_nest";

const KNOWN_MODES: ReadonlySet<string> = new Set([
  "stationary",
  "constant_velocity",
  "tracked_window",
  "moving_nest",
]);

/**
 * Frame provenance published with a step. Every field is optional because runs
 * predating the domain-motion contract carry none of it; absence is treated as
 * stationary, which is what those runs were.
 */
export interface FrameMotion {
  mode?: string | null;
  horizontalFrameContract?: string | null;
  /** Constant frame velocity in m/s, Earth-relative. */
  frameVelocityMs?: readonly [number, number] | null;
  /** Earth-coordinate origin of the grid at this frame, in metres. */
  earthOriginM?: readonly [number, number] | null;
  /** Cumulative whole-cell shift applied to the window. */
  cumulativeShiftCells?: readonly [number, number] | null;
  /** Whether exported winds are Earth-relative. Grid-relative winds must not be
   *  drawn as hodographs or storm-relative diagnostics without correction. */
  windsEarthRelative?: boolean | null;
}

/** A step carrying optional motion provenance. */
export interface MotionBearingStep {
  motion?: FrameMotion | null;
  grid?: {
    scalarOriginM?: readonly [number, number, number] | null;
    horizontalScalarOriginM?: readonly [number, number] | null;
  } | null;
}

/**
 * Resolved mode for a step, or null when the declared mode is not one this
 * build understands. Null is a refusal, not a default.
 */
export function motionMode(step: MotionBearingStep | null | undefined): DomainMotionMode | null {
  const declared = step?.motion?.mode;
  // No motion block at all: the run predates the contract and was stationary.
  if (declared == null || declared === "") return "stationary";
  return KNOWN_MODES.has(declared) ? (declared as DomainMotionMode) : null;
}

/** Whether the grid stays fixed to the Earth for the whole run. */
export function isStationary(step: MotionBearingStep | null | undefined): boolean {
  return motionMode(step) === "stationary";
}

export interface AccumulationVerdict {
  /** True when accumulating across time in grid indices is physically valid. */
  allowed: boolean;
  /** One line naming the reason, for on-canvas disclosure. Null when allowed. */
  blocker: string | null;
}

/** Whether a time-accumulating plot may combine these frames by grid index. All frames must be stationary. */
export function canAccumulateByCell(
  steps: readonly MotionBearingStep[],
): AccumulationVerdict {
  for (const step of steps) {
    const mode = motionMode(step);
    if (mode == null) {
      return {
        allowed: false,
        blocker:
          "This run declares a domain-motion mode this build does not recognise, "
          + "so a grid cell cannot be resolved to a fixed location.",
      };
    }
    if (mode !== "stationary") {
      return {
        allowed: false,
        blocker: describeMotionBlocker(mode),
      };
    }
  }
  return { allowed: true, blocker: null };
}

function describeMotionBlocker(mode: DomainMotionMode): string {
  const cause = mode === "constant_velocity"
    ? "the grid translates across the Earth"
    : mode === "tracked_window"
      ? "the grid window follows the storm"
      : "a moving nest follows the storm";
  return `Accumulating over time by grid cell needs a stationary domain, but ${cause}`
    + ` (domain_motion.mode: ${mode}). The same cell index is a different place in`
    + " each frame, so the result would smear the track rather than trace it.";
}

/** Whether two datasets may be compared cell by cell. Both must be stationary and share an Earth origin. */
export function canCompareByCell(
  left: MotionBearingStep | null | undefined,
  right: MotionBearingStep | null | undefined,
): AccumulationVerdict {
  const single = canAccumulateByCell([left, right].filter(Boolean) as MotionBearingStep[]);
  if (!single.allowed) return single;
  const a = left?.motion?.earthOriginM;
  const b = right?.motion?.earthOriginM;
  if (a && b && (a[0] !== b[0] || a[1] !== b[1])) {
    return {
      allowed: false,
      blocker:
        "These datasets declare different Earth origins, so the same cell index"
        + " is a different location in each. Differencing them by cell would"
        + " subtract one place from another.",
    };
  }
  return { allowed: true, blocker: null };
}

/** Whether exported winds are Earth-relative. Grid-relative winds must not be used for hodographs or storm-relative diagnostics without correction. */
export function windsAreEarthRelative(step: MotionBearingStep | null | undefined): boolean {
  const declared = step?.motion?.windsEarthRelative;
  // Absent declaration on a stationary run is unambiguous: the two frames
  // coincide, so the distinction does not exist.
  if (declared == null) return isStationary(step);
  return declared === true;
}

/** How to read the physical coordinates a plot's axes report. */
export type AxisFrame =
  /** Axis numbers name a fixed Earth location for the whole run. */
  | "earth-fixed"
  /** Axis numbers name a position in a grid that moves over the Earth. */
  | "grid-relative"
  /** The declared motion mode is unknown, so the frame cannot be named. */
  | "unverified";

export interface AxisFrameVerdict {
  frame: AxisFrame;
  /** Short headline for a run-level notice. Null when earth-fixed. */
  headline: string | null;
  /** One paragraph naming what the axis numbers do and do not mean. */
  detail: string | null;
}

/**
 * What a plot's axis coordinates mean for this run.
 * Current native manifests register scalar origins to the Earth each frame.
 * Older moving-grid metadata does not establish that coordinate contract.
 */
export function axisFrame(
  step: MotionBearingStep | null | undefined,
): AxisFrameVerdict {
  const mode = motionMode(step);
  if (mode == null) {
    return {
      frame: "unverified",
      headline: "Axis coordinates unverified",
      detail:
        "This run declares a domain-motion mode this build does not recognise,"
        + " so plot axes cannot be resolved to fixed Earth locations. Treat every"
        + " coordinate as grid-relative until the run is opened in a build that"
        + " understands its motion mode.",
    };
  }
  if (mode === "stationary") {
    return { frame: "earth-fixed", headline: null, detail: null };
  }
  const contract = step?.motion?.horizontalFrameContract;
  if (contract != null) {
    const origin = step?.grid?.scalarOriginM ?? step?.grid?.horizontalScalarOriginM;
    if (contract === "constant_translation_earth_wind_grid_transport_v1"
        && origin && origin.length >= 2 && origin.every(Number.isFinite)) {
      return { frame: "earth-fixed", headline: null, detail: null };
    }
    return {
      frame: "unverified",
      headline: "Axis coordinates unverified",
      detail: "This moving grid lacks a recognised Earth-coordinate contract with a finite scalar origin."
        + " Its axis coordinates cannot be identified as fixed Earth locations.",
    };
  }
  const cause = mode === "constant_velocity"
    ? "the grid translates across the Earth at a constant velocity"
    : mode === "tracked_window"
      ? "the grid window follows the storm"
      : "a moving nest follows the storm";
  return {
    frame: "grid-relative",
    headline: "Axis coordinates are grid-relative",
    detail:
      `Plot axes report position within the computational grid, and ${cause}`
      + ` (domain_motion.mode: ${mode}). The same axis coordinate is a different`
      + " Earth location in each frame, so a feature's apparent movement across"
      + " frames includes the grid's own motion.",
  };
}
