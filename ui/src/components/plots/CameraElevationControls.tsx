import { useCallback, useEffect, useRef, useState } from "react";
import type { StepInfo } from "@/lib/run/runOutputs";
import {
  CAMERA_BEARING_PRESETS,
  CAMERA_HEIGHT_STEP_M,
  DEFAULT_ELEVATION_DEGREES,
  ELEVATION_PRESETS,
  cameraPlacementAtHeight,
  cameraPositionAzimuthForBearing,
  defaultCameraHeightM,
  domainHeightM,
  type CameraProjection,
} from "@/lib/plot/cameraElevation";
import { domainBoundsM } from "@/lib/plot/domainAxes";
import type { SceneScale } from "@/lib/plot/sceneScale";
import {
  applyVtkCameraPlacement,
  ControlGroup,
  currentVtkCameraDistance,
  gridKeyOf,
  PillGroup,
  vtkRendererAspect,
  type VtkRendererLike,
} from "./common";

/* Vertical camera placement: what height to look at, and from what angle. */

export interface CameraElevationSelection {
  /** True metres above the domain floor that the camera looks at. */
  heightM: number;
  /** Degrees above the horizontal the camera looks from. */
  elevationDegrees: number;
  /** Compass bearing the camera is looking toward. */
  bearingDegrees: number;
  projection: CameraProjection;
  /**
   * Bumped only when the user moves a control. Scenes apply the placement on a
   * change, so a re-render never yanks a camera the user has orbited by hand.
   */
  requestGeneration: number;
  requestKind: "height" | "elevation" | "bearing" | "projection" | null;
  /** Domain lid in true metres, or null when the run publishes no usable height. */
  maxHeightM: number | null;
  setHeightM: (next: number) => void;
  setElevationDegrees: (next: number) => void;
  setBearingDegrees: (next: number) => void;
  setProjection: (next: CameraProjection) => void;
  /** Reflect a camera move without treating it as a new user command. */
  syncLiveState: (next: {
    heightM: number;
    elevationDegrees: number;
    bearingDegrees: number;
    projection: CameraProjection;
  }) => void;
}

export function useCameraElevation(
  grid: StepInfo["grid"] | null | undefined,
): CameraElevationSelection {
  const bounds = domainBoundsM(grid);
  const maxHeightM = domainHeightM(bounds);
  const gridKey = gridKeyOf(grid);

  const [heightM, setHeightState] = useState(() => defaultCameraHeightM(bounds));
  const [elevationDegrees, setElevationState] = useState(DEFAULT_ELEVATION_DEGREES);
  const [bearingDegrees, setBearingState] = useState(45);
  const [projection, setProjectionState] = useState<CameraProjection>("perspective");
  const [requestGeneration, setRequestGeneration] = useState(0);
  const [requestKind, setRequestKind] = useState<CameraElevationSelection["requestKind"]>(null);

  // A new run can have a very different lid; a height carried over from the
  // previous domain would sit outside it.
  useEffect(() => {
    setHeightState(defaultCameraHeightM(domainBoundsM(grid)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridKey]);

  const setHeightM = useCallback((next: number) => {
    setHeightState(next);
    setRequestKind("height");
    setRequestGeneration((generation) => generation + 1);
  }, []);

  const setElevationDegrees = useCallback((next: number) => {
    setElevationState(next);
    setRequestKind("elevation");
    setRequestGeneration((generation) => generation + 1);
  }, []);

  const setBearingDegrees = useCallback((next: number) => {
    setBearingState(next);
    setRequestKind("bearing");
    setRequestGeneration((generation) => generation + 1);
  }, []);

  const setProjection = useCallback((next: CameraProjection) => {
    setProjectionState(next);
    setRequestKind("projection");
    setRequestGeneration((generation) => generation + 1);
  }, []);

  const syncLiveState = useCallback((next: {
    heightM: number;
    elevationDegrees: number;
    bearingDegrees: number;
    projection: CameraProjection;
  }) => {
    setHeightState(next.heightM);
    setElevationState(next.elevationDegrees);
    setBearingState(next.bearingDegrees);
    setProjectionState(next.projection);
  }, []);

  return {
    heightM,
    elevationDegrees,
    bearingDegrees,
    projection,
    requestGeneration,
    requestKind,
    maxHeightM,
    setHeightM,
    setElevationDegrees,
    setBearingDegrees,
    setProjection,
    syncLiveState,
  };
}

/** Scene surface the placement is applied to; matches the radar floor's thunk. */
export interface CameraElevationRenderWindow {
  getRenderer: () => VtkRendererLike;
  getRenderWindow: () => { render: () => void };
}

/**
 * Apply the selected height and look angle to a scene's camera. Runs only when
 * the user moves a control, so orbiting by hand is never overridden, and keeps
 * the current bearing so a pitch change does not spin the view.
 */
export function useCameraElevationPlacement(
  getRenderWindow: () => CameraElevationRenderWindow | null,
  input: {
    grid: StepInfo["grid"] | null | undefined;
    scale: SceneScale;
    selection: CameraElevationSelection;
    targetXYM?: readonly [number, number] | null;
    lifecycleGeneration?: number;
    onCameraApplied?: () => void;
  },
): void {
  const { selection, lifecycleGeneration = 0 } = input;
  const getRwRef = useRef(getRenderWindow);
  getRwRef.current = getRenderWindow;
  const latest = useRef(input);
  latest.current = input;

  useEffect(() => {
    let disposed = false;
    let bindFrame = 0;
    let updateFrame = 0;
    let subscription: { unsubscribe: () => void } | undefined;

    const update = () => {
      updateFrame = 0;
      if (disposed) return;
      const current = latest.current;
      const bounds = domainBoundsM(current.grid);
      const camera = getRwRef.current()?.getRenderer().getActiveCamera() as unknown as {
        getPosition?: () => ArrayLike<number>;
        getFocalPoint?: () => ArrayLike<number>;
        getParallelProjection?: () => boolean;
      } | undefined;
      const position = camera?.getPosition?.();
      const focal = camera?.getFocalPoint?.();
      if (!bounds || !position || !focal) return;
      const safeXY = current.scale.xy || 1;
      const safeZ = current.scale.z || 1;
      const viewX = (focal[0]! - position[0]!) / safeXY;
      const viewY = (focal[1]! - position[1]!) / safeXY;
      const eyeAboveTarget = (position[2]! - focal[2]!) / safeZ;
      const horizontal = Math.hypot(viewX, viewY);
      const elevation = Math.atan2(eyeAboveTarget, horizontal) * 180 / Math.PI;
      const bearing = ((Math.atan2(viewX, viewY) * 180 / Math.PI) % 360 + 360) % 360;
      const height = focal[2]! / safeZ - bounds.lower[2];
      if (![elevation, bearing, height].every(Number.isFinite)) return;
      current.selection.syncLiveState({
        heightM: Math.min(bounds.upper[2] - bounds.lower[2], Math.max(0, height)),
        elevationDegrees: elevation,
        bearingDegrees: bearing,
        projection: camera?.getParallelProjection?.() ? "orthographic" : "perspective",
      });
    };
    const schedule = () => {
      if (!disposed && !updateFrame) updateFrame = requestAnimationFrame(update);
    };
    let attempts = 0;
    const bind = () => {
      bindFrame = 0;
      if (disposed) return;
      const camera = getRwRef.current()?.getRenderer().getActiveCamera() as unknown as {
        onModified?: (callback: () => void) => { unsubscribe: () => void };
      } | undefined;
      if (!camera?.onModified) {
        if (++attempts < 300) bindFrame = requestAnimationFrame(bind);
        return;
      }
      subscription = camera.onModified(schedule);
      update();
    };
    bindFrame = requestAnimationFrame(bind);
    return () => {
      disposed = true;
      if (bindFrame) cancelAnimationFrame(bindFrame);
      if (updateFrame) cancelAnimationFrame(updateFrame);
      subscription?.unsubscribe();
    };
  }, [lifecycleGeneration]);

  useEffect(() => {
    // Generation 0 is the opening state, which the scene's own reset already
    // framed; applying here would fight it on every mount.
    if (selection.requestGeneration === 0) return;
    let disposed = false;
    let frame = 0;
    let attempts = 0;
    const apply = () => {
      frame = 0;
      if (disposed) return;
      const rw = getRwRef.current();
      if (!rw) {
        if (++attempts < 120) frame = requestAnimationFrame(apply);
        return;
      }
      const current = latest.current;
      const renderer = rw.getRenderer();
      const placement = cameraPlacementAtHeight(
        domainBoundsM(current.grid),
        current.scale,
        current.selection.heightM,
        current.selection.elevationDegrees,
        cameraPositionAzimuthForBearing(current.selection.bearingDegrees),
        {
          distanceM: currentVtkCameraDistance(renderer),
          aspect: vtkRendererAspect(renderer) ?? 1,
          minimumFitDistance: current.selection.requestKind === "elevation"
            || current.selection.requestKind === "bearing",
          targetXYM: current.targetXYM,
        },
      );
      if (!placement) return;
      applyVtkCameraPlacement(renderer, placement, current.selection.projection);
      current.onCameraApplied?.();
      rw.getRenderWindow().render();
    };
    apply();
    return () => {
      disposed = true;
      if (frame) cancelAnimationFrame(frame);
    };
    // Only a user-driven change moves the camera; everything else is read fresh.
  }, [selection.requestGeneration, lifecycleGeneration]);
}

/** Metres for the low levels, kilometres above that; the slider reads as a sounding does. */
function formatHeight(heightM: number): string {
  if (heightM < 1000) return `${Math.round(heightM)} m`;
  return `${(heightM / 1000).toFixed(1)} km`;
}

export function CameraElevationControls({
  selection,
}: {
  selection: CameraElevationSelection;
}) {
  const { heightM, elevationDegrees, bearingDegrees, projection, maxHeightM } = selection;
  const unavailable = maxHeightM == null;
  const activePreset = ELEVATION_PRESETS.find(
    (preset) => Math.abs(preset.degrees - elevationDegrees) < 0.5,
  );

  return (
    <>
      <ControlGroup label="Target height">
        <input
          type="range"
          min={0}
          max={maxHeightM ?? CAMERA_HEIGHT_STEP_M}
          step={CAMERA_HEIGHT_STEP_M}
          value={Math.min(maxHeightM ?? 0, Math.max(0, heightM))}
          disabled={unavailable}
          aria-label="Camera target height above ground"
          onChange={(event) => selection.setHeightM(Number.parseFloat(event.target.value))}
          className="volume-render__slider"
          title={unavailable
            ? "This run publishes no domain height to place the camera in"
            : "Height above ground of the camera focal point"}
        />
        <span className="volume-render__slider-value">
          {unavailable ? "n/a" : formatHeight(heightM)}
        </span>
      </ControlGroup>
      <ControlGroup label="View toward">
        <PillGroup
          options={CAMERA_BEARING_PRESETS.map((preset) => ({
            value: preset.id,
            label: preset.label,
            hint: `Look toward ${preset.id}`,
          }))}
          disabled={unavailable}
          disabledReason="This run publishes no physical domain for camera placement"
          value={CAMERA_BEARING_PRESETS.find(
            (preset) => Math.abs((((bearingDegrees - preset.degrees) + 540) % 360) - 180) < 0.5,
          )?.id ?? ""}
          onChange={(next) => {
            const preset = CAMERA_BEARING_PRESETS.find((candidate) => candidate.id === next);
            if (preset) selection.setBearingDegrees(preset.degrees);
          }}
        />
      </ControlGroup>
      <ControlGroup label="Projection">
        <PillGroup
          options={[
            { value: "perspective" as const, label: "Perspective", hint: "Depth-scaled storm view" },
            { value: "orthographic" as const, label: "Orthographic", hint: "Scale-preserving plan and section view" },
          ]}
          value={projection}
          disabled={unavailable}
          disabledReason="This run publishes no physical domain for camera placement"
          onChange={selection.setProjection}
        />
      </ControlGroup>
      <ControlGroup label="Look angle">
        <PillGroup
          options={ELEVATION_PRESETS.map((preset) => ({
            value: preset.id,
            label: preset.label,
            hint: preset.hint,
          }))}
          disabled={unavailable}
          disabledReason="This run publishes no domain height to place the camera in"
          value={activePreset?.id ?? ""}
          onChange={(next) => {
            const preset = ELEVATION_PRESETS.find((candidate) => candidate.id === next);
            if (preset) selection.setElevationDegrees(preset.degrees);
          }}
        />
      </ControlGroup>
    </>
  );
}
