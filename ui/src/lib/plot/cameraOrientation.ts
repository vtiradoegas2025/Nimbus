/*
 * Camera orientation math for 3D plots. All positions in TRUE METRES (callers must undo VTK SceneScale).
 * Pure — no vtk.js, no React.
 */

/** Compass octant strings in meteorological order (N=0, clockwise). */
export type CompassOctant =
  | "N"
  | "NNE"
  | "NE"
  | "ENE"
  | "E"
  | "ESE"
  | "SE"
  | "SSE"
  | "S"
  | "SSW"
  | "SW"
  | "WSW"
  | "W"
  | "WNW"
  | "NW"
  | "NNW";

/** Orientation of the camera relative to north and optionally a storm centre. */
export interface CameraOrientation {
  /**
   * Compass bearing of the viewing direction projected onto the horizontal plane. Range [0, 360).
   * Convention: 0°=N, 90°=E, 180°=S, 270°=W. Computed as atan2(dx, dy) (east first, then north).
   */
  azimuthDeg: number;

  /** Camera elevation above horizontal, signed (pos = looking up). Range [−90, +90]. */
  elevationDeg: number;

  /** Camera roll in degrees from +z-up (0° when view-up is (0,0,1)). */
  rollDeg: number;

  /** True 3-D distance from camera position to focal point (m). */
  rangeM: number;

  /** Compass bearing from camera to storm centre (same convention as azimuthDeg). Optional. */
  stormBearingDeg?: number;

  /** Horizontal distance from camera to storm centre (m). Optional. */
  stormRangeM?: number;
}

/** Three-element position/vector in true metres [x (east), y (north), z (up)]. */
export type Vec3M = readonly [number, number, number];

/**
 * Compute camera orientation from intrinsics in true metres.
 * Returns null if the view vector is degenerate or any component is non-finite.
 */
export function computeCameraOrientation(
  positionM: Vec3M,
  focalPointM: Vec3M,
  viewUp: Vec3M,
  stormCentreM?: Vec3M,
): CameraOrientation | null {
  // Validate all inputs are finite.
  for (const v of [positionM, focalPointM, viewUp]) {
    for (const c of v) {
      if (!Number.isFinite(c)) return null;
    }
  }
  if (stormCentreM) {
    for (const c of stormCentreM) {
      if (!Number.isFinite(c)) return null;
    }
  }

  const dx = focalPointM[0] - positionM[0];
  const dy = focalPointM[1] - positionM[1];
  const dz = focalPointM[2] - positionM[2];

  const horizontalLen = Math.hypot(dx, dy);
  const totalLen = Math.hypot(horizontalLen, dz);

  // Degenerate: position and focal point coincide.
  if (totalLen < 1e-9) return null;

  // Azimuth: atan2(east, north) gives compass bearing (0°=N, CW).
  const azimuthRaw = Math.atan2(dx, dy) * (180 / Math.PI);
  const azimuthDeg = ((azimuthRaw % 360) + 360) % 360;

  // Elevation: signed angle above/below horizontal.
  const elevationDeg = Math.atan2(dz, horizontalLen) * (180 / Math.PI);

  // Roll: angle of view-up projected to the image plane.
  const rollDeg = computeRollDeg(
    [dx / totalLen, dy / totalLen, dz / totalLen],
    viewUp,
  );

  const result: CameraOrientation = { azimuthDeg, elevationDeg, rollDeg, rangeM: totalLen };

  if (stormCentreM) {
    const sdx = stormCentreM[0] - positionM[0];
    const sdy = stormCentreM[1] - positionM[1];
    const stormHorizontalLen = Math.hypot(sdx, sdy);
    const stormBearingRaw = Math.atan2(sdx, sdy) * (180 / Math.PI);
    result.stormBearingDeg = ((stormBearingRaw % 360) + 360) % 360;
    result.stormRangeM = stormHorizontalLen;
  }

  return result;
}

/** Camera roll (deg): project view-up onto the plane perp to view direction, measure from +z. */
function computeRollDeg(unitViewDir: Vec3M, viewUp: Vec3M): number {
  const upLen = Math.hypot(viewUp[0], viewUp[1], viewUp[2]);
  if (upLen < 1e-9) return 0;
  const ux = viewUp[0] / upLen;
  const uy = viewUp[1] / upLen;
  const uz = viewUp[2] / upLen;

  // Remove the view-direction component from view-up (Gram–Schmidt step).
  const dot = ux * unitViewDir[0] + uy * unitViewDir[1] + uz * unitViewDir[2];
  const px = ux - dot * unitViewDir[0];
  const py = uy - dot * unitViewDir[1];
  const pz = uz - dot * unitViewDir[2];
  const projLen = Math.hypot(px, py, pz);
  if (projLen < 1e-9) return 0;

  // right = viewDir × worldUp where worldUp = (0, 0, 1).
  // cross product: (vy*1 - vz*0, vz*0 - vx*1, vx*0 - vy*0) = (vy, -vx, 0).
  const rightX = unitViewDir[1];
  const rightY = -unitViewDir[0];

  // dot(projectedUp, worldUp) gives cos(roll); worldUp = (0,0,1) so dot = pz.
  const cosRoll = pz / projLen;
  // dot(projectedUp, right) gives sin(roll); right has no z component.
  const sinRoll = (px * rightX + py * rightY) / projLen;

  return Math.atan2(sinRoll, cosRoll) * (180 / Math.PI);
}

/** Named 16-point compass octant (22.5° sectors, wraps at 0°/360°). */
export function compassOctant(azimuthDeg: number): CompassOctant {
  if (!Number.isFinite(azimuthDeg)) return "N";
  // Normalise to [0, 360).
  const bearing = ((azimuthDeg % 360) + 360) % 360;
  // Each of 16 sectors is 22.5° wide. Add half a sector so boundaries fall on
  // the midpoints between named directions.
  const index = Math.floor((bearing + 11.25) / 22.5) % 16;
  const OCTANTS: CompassOctant[] = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
  ];
  return OCTANTS[index];
}
