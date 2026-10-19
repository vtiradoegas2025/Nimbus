import type { DomainBoundsM } from "@/lib/plot/domainAxes";
import type { SceneScale } from "@/lib/plot/sceneScale";

/*
 * Vertical camera placement: look into a chosen height above ground at a chosen
 * angle. Pure — no vtk.js, no React. Heights are TRUE metres above the domain
 * floor; the returned placement is DISPLAY metres, matching the scene scale.
 */

/** Camera state a renderer can apply directly, in display metres. */
export interface CameraPlacement {
  position: [number, number, number];
  focalPoint: [number, number, number];
  viewUp: [number, number, number];
}

export type CameraProjection = "perspective" | "orthographic";

/** A named look angle, in degrees above the horizontal. */
export interface ElevationPreset {
  id: string;
  label: string;
  degrees: number;
  /** Why a forecaster would pick it; surfaced as the control's tooltip. */
  hint: string;
}

/*
 * Level puts the eye inside the storm's inflow layer, which is where a hook
 * echo or a wall cloud reads as a shape rather than a plan-view blob. Top down
 * is the composite-reflectivity view.
 */
export const ELEVATION_PRESETS: readonly ElevationPreset[] = [
  { id: "level", label: "Level", degrees: 0, hint: "Eye level with the chosen height" },
  { id: "low", label: "Low", degrees: 15, hint: "Just above the inflow layer" },
  { id: "oblique", label: "Oblique", degrees: 30, hint: "Storm-chaser view" },
  { id: "top", label: "Top down", degrees: 90, hint: "Plan view from directly overhead" },
];

export const CAMERA_BEARING_PRESETS = [
  { id: "north", label: "N", degrees: 0 },
  { id: "east", label: "E", degrees: 90 },
  { id: "south", label: "S", degrees: 180 },
  { id: "west", label: "W", degrees: 270 },
] as const;

/**
 * Look angle a preset applies before the user picks one. The scenes' own reset
 * opens nearer 18 degrees, so until a pill is clicked this is a proposal rather
 * than a description of where the camera is.
 */
export const DEFAULT_ELEVATION_DEGREES = 30;

/** Steepest look angle before the view direction is parallel to the up vector. */
const MAX_ELEVATION_DEGREES = 90;
const MIN_ELEVATION_DEGREES = -89;

/** Beyond this the +z up vector is degenerate, so the camera rolls onto +y instead. */
const POLAR_ELEVATION_DEGREES = 89;

/** Vertical field of view the 3D scenes render with; shared with `applyVtkCameraPlacement`. */
export const CAMERA_VIEW_ANGLE_DEGREES = 30;

/** Breathing room around the domain, matching the reset presets' framing margin. */
const FIT_MARGIN = 1.08;

const DEGREES_TO_RADIANS = Math.PI / 180;

/** Height slider resolution, in metres. Storm-scale features resolve well below 1 km. */
export const CAMERA_HEIGHT_STEP_M = 50;

/** Domain height in true metres, or null when the bounds cannot supply one. */
export function domainHeightM(bounds: DomainBoundsM | null): number | null {
  if (!bounds) return null;
  const height = bounds.upper[2] - bounds.lower[2];
  return Number.isFinite(height) && height > 0 ? height : null;
}

/**
 * Opening camera height: a third of the domain, which sits in the mid-levels
 * where the updraft is widest, rather than at the ground where the first
 * useful surface is still half a cell away.
 */
export function defaultCameraHeightM(bounds: DomainBoundsM | null): number {
  const height = domainHeightM(bounds);
  return height == null ? 0 : height / 3;
}

/** Southwest of the storm, matching `resetVtkCameraToStormView`'s opening bearing. */
export const DEFAULT_AZIMUTH_RADIANS = Math.atan2(-1.35, -1.1);

/** Compass direction the camera currently sits in, relative to its focal point. */
export function cameraAzimuthRadians(
  position: readonly number[],
  focalPoint: readonly number[],
): number {
  const dx = position[0]! - focalPoint[0]!;
  const dy = position[1]! - focalPoint[1]!;
  // Directly overhead has no azimuth; keep the scenes' opening bearing instead
  // of letting atan2(0, 0) collapse the orbit to due east.
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) return DEFAULT_AZIMUTH_RADIANS;
  if (dx === 0 && dy === 0) return DEFAULT_AZIMUTH_RADIANS;
  return Math.atan2(dy, dx);
}

/** Clamp a look angle into the range the placement can represent. */
export function clampElevationDegrees(degrees: number): number {
  if (!Number.isFinite(degrees)) return DEFAULT_ELEVATION_DEGREES;
  return Math.min(MAX_ELEVATION_DEGREES, Math.max(MIN_ELEVATION_DEGREES, degrees));
}

/** Clamp a height to the domain it is measured in. */
export function clampHeightM(heightM: number, bounds: DomainBoundsM | null): number {
  if (!Number.isFinite(heightM)) return 0;
  const height = domainHeightM(bounds);
  if (height == null) return Math.max(0, heightM);
  return Math.min(height, Math.max(0, heightM));
}

/**
 * Eye distance that fits the whole domain, in display metres.
 *
 * The binding constraint is the narrower of the two fields of view, so a tall
 * viewport is fitted horizontally and a wide one vertically. Sizing on the
 * bounding sphere rather than the horizontal span is what keeps an exaggerated
 * domain — z can reach six times true scale — inside the frame.
 */
export function fitDistanceM(
  bounds: DomainBoundsM | null,
  scale: SceneScale,
  aspect: number,
): number | null {
  if (!bounds) return null;
  if (!Number.isFinite(scale.xy) || !Number.isFinite(scale.z)) return null;
  if (!Number.isFinite(aspect) || aspect <= 0) return null;
  const spanX = Math.abs(bounds.upper[0] - bounds.lower[0]) * scale.xy;
  const spanY = Math.abs(bounds.upper[1] - bounds.lower[1]) * scale.xy;
  const spanZ = Math.abs(bounds.upper[2] - bounds.lower[2]) * scale.z;
  const radius = 0.5 * Math.hypot(spanX, spanY, spanZ);
  if (!Number.isFinite(radius) || radius <= 0) return null;
  const vertical = CAMERA_VIEW_ANGLE_DEGREES * DEGREES_TO_RADIANS;
  const horizontal = 2 * Math.atan(Math.tan(0.5 * vertical) * aspect);
  const binding = Math.min(vertical, horizontal);
  return (radius * FIT_MARGIN) / Math.sin(0.5 * binding);
}

/**
 * Place the camera to look at `heightM` above the domain floor from
 * `elevationDeg` above the horizontal, keeping `azimuthRad` so a preset changes
 * the pitch without spinning the user's orbit. Returns null when the bounds or
 * scale cannot yield a finite placement.
 *
 * `view.distanceM` is how far back the eye already stood. Reusing it is what
 * makes the height slider an elevator: a user dollied in on a hook echo stays
 * dollied in when they drop to the inflow layer. Only when there is no usable
 * distance to keep does this fall back to fitting the whole domain.
 */
export function cameraPlacementAtHeight(
  bounds: DomainBoundsM | null,
  scale: SceneScale,
  heightM: number,
  elevationDeg: number,
  azimuthRad: number,
  view: {
    distanceM?: number | null;
    aspect?: number;
    minimumFitDistance?: boolean;
    /** Optional meteorological target; defaults to the horizontal domain centre. */
    targetXYM?: readonly [number, number] | null;
  } = {},
): CameraPlacement | null {
  if (!bounds) return null;
  if (!Number.isFinite(scale.xy) || !Number.isFinite(scale.z)) return null;
  if (scale.xy <= 0) return null;
  if (!Number.isFinite(azimuthRad)) return null;
  const finiteCorner = (corner: readonly number[]) =>
    corner.length === 3 && corner.every((value) => Number.isFinite(value));
  if (!finiteCorner(bounds.lower) || !finiteCorner(bounds.upper)) return null;

  const elevation = clampElevationDegrees(elevationDeg) * DEGREES_TO_RADIANS;
  const height = clampHeightM(heightM, bounds);

  const requestedTarget = view.targetXYM;
  const targetX = requestedTarget && Number.isFinite(requestedTarget[0])
    ? Math.min(bounds.upper[0], Math.max(bounds.lower[0], requestedTarget[0]))
    : 0.5 * (bounds.lower[0] + bounds.upper[0]);
  const targetY = requestedTarget && Number.isFinite(requestedTarget[1])
    ? Math.min(bounds.upper[1], Math.max(bounds.lower[1], requestedTarget[1]))
    : 0.5 * (bounds.lower[1] + bounds.upper[1]);
  const focalPoint: [number, number, number] = [
    targetX * scale.xy,
    targetY * scale.xy,
    (bounds.lower[2] + height) * scale.z,
  ];

  const kept = view.distanceM;
  const fitted = fitDistanceM(bounds, scale, view.aspect ?? 1);
  // A degenerate domain still has to yield a camera in front of its focal
  // point, or the scene renders from inside itself.
  const reusable = Number.isFinite(kept) && (kept ?? 0) > 0 ? kept! : null;
  const distance = reusable == null
    ? fitted ?? 1
    : view.minimumFitDistance && fitted != null
      ? Math.max(reusable, fitted)
      : reusable;

  const horizontal = Math.cos(elevation) * distance;
  const position: [number, number, number] = [
    focalPoint[0] + horizontal * Math.cos(azimuthRad),
    focalPoint[1] + horizontal * Math.sin(azimuthRad),
    focalPoint[2] + Math.sin(elevation) * distance,
  ];

  // Looking straight down makes +z parallel to the view direction, which leaves
  // the up vector undefined; roll onto +y so north stays at the top of frame.
  const polar = Math.abs(clampElevationDegrees(elevationDeg)) >= POLAR_ELEVATION_DEGREES;
  const viewUp: [number, number, number] = polar ? [0, 1, 0] : [0, 0, 1];

  return { position, focalPoint, viewUp };
}

/** Camera-position azimuth required for a camera looking toward a compass bearing. */
export function cameraPositionAzimuthForBearing(viewBearingDeg: number): number {
  const bearing = Number.isFinite(viewBearingDeg) ? viewBearingDeg : 0;
  const radians = bearing * DEGREES_TO_RADIANS;
  return Math.atan2(-Math.cos(radians), -Math.sin(radians));
}
