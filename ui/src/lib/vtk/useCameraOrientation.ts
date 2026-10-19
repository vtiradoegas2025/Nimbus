import { useEffect, useRef, useState } from "react";
import {
  computeCameraOrientation,
  type CameraOrientation,
  type Vec3M,
} from "@/lib/plot/cameraOrientation";
import type { SceneScale } from "@/components/plots/common";

/* Subscribes to a vtk.js camera and returns orientation in true metres.
 * Undoes display-space scale before computing azimuth/elevation. */

/** Minimal vtk.js camera interface required for orientation tracking. */
interface OrientationCamera {
  onModified: (cb: () => void) => { unsubscribe: () => void };
  getPosition: () => ArrayLike<number>;
  getFocalPoint: () => ArrayLike<number>;
  getViewUp: () => ArrayLike<number>;
}

/** Minimal vtk.js render-window wrapper required by this hook. */
export interface OrientationRenderWindow {
  getRenderer: () => { getActiveCamera: () => OrientationCamera };
}

const CAMERA_BIND_INITIAL_DELAY_MS = 16;
const CAMERA_BIND_MAX_DELAY_MS = 1_000;

/**
 * Maintains current camera orientation for a 3D scene.
 *
 * @param getRenderWindow  Stable getter returning the render window (or null
 *                         before the scene mounts). Use `useRef` at the call
 *                         site rather than capturing the value directly — the
 *                         hook re-reads on every camera event so a stale
 *                         closure never silently holds the wrong window.
 * @param scale            Scene scale in effect. The hook watches xy and z so
 *                         it re-subscribes when either changes.
 * @param stormCentreM     Optional storm centre in true metres. When provided,
 *                         stormBearingDeg and stormRangeM are populated.
 * @param lifecycleGeneration  Increment this when the scene is torn down and
 *                             rebuilt (e.g. on a new grid) to force a fresh
 *                             camera subscription.
 *
 * @returns The most-recent orientation, or null before the camera is ready or
 *          when the view vector is degenerate.
 */
export function useCameraOrientation(
  getRenderWindow: () => OrientationRenderWindow | null,
  scale: SceneScale,
  stormCentreM?: Vec3M,
  lifecycleGeneration = 0,
): CameraOrientation | null {
  const [orientation, setOrientation] = useState<CameraOrientation | null>(null);
  const getRwRef = useRef(getRenderWindow);
  getRwRef.current = getRenderWindow;

  // Stable ref for storm centre so the effect doesn't have to treat a new
  // object-identity as a dependency.
  const stormRef = useRef(stormCentreM);
  stormRef.current = stormCentreM;
  const stormX = stormCentreM?.[0];
  const stormY = stormCentreM?.[1];
  const stormZ = stormCentreM?.[2];

  useEffect(() => {
    let frame = 0;
    let disposed = false;
    let changeKey = "";

    const update = () => {
      frame = 0;
      if (disposed) return;
      const rw = getRwRef.current();
      if (!rw) return;
      const camera = rw.getRenderer().getActiveCamera();
      const rawPos = camera.getPosition();
      const rawFocal = camera.getFocalPoint();
      const rawUp = camera.getViewUp();

      const safeXY = scale.xy || 1;
      const safeZ = scale.z || 1;

      const posM: Vec3M = [
        rawPos[0] / safeXY,
        rawPos[1] / safeXY,
        rawPos[2] / safeZ,
      ];
      const focalM: Vec3M = [
        rawFocal[0] / safeXY,
        rawFocal[1] / safeXY,
        rawFocal[2] / safeZ,
      ];
      // view-up is a direction in scene space; for roll the absolute magnitude
      // doesn't matter, but the xy scale could skew the direction in the
      // horizontal plane, so undo it here too.
      const upM: Vec3M = [
        rawUp[0] / safeXY,
        rawUp[1] / safeXY,
        rawUp[2] / safeZ,
      ];

      const result = computeCameraOrientation(posM, focalM, upM, stormRef.current);

      // Build a change key from rounded values — 0.5° resolution for azimuth
      // and elevation (coarser than the display precision) so an orbit that
      // moves the camera by less than half a degree does not trigger a render.
      const nextKey = result
        ? [
            result.azimuthDeg.toFixed(0),
            result.elevationDeg.toFixed(0),
            result.rollDeg.toFixed(0),
            result.rangeM.toFixed(0),
            result.stormBearingDeg?.toFixed(0) ?? "",
            result.stormRangeM?.toFixed(0) ?? "",
          ].join("|")
        : "null";

      if (nextKey === changeKey) return;
      changeKey = nextKey;
      setOrientation(result);
    };

    const schedule = () => {
      if (disposed || frame) return;
      frame = requestAnimationFrame(update);
    };

    // The render window mounts asynchronously. Retry quickly at first, then
    // back off to one cheap probe per second instead of occupying rAF forever
    // when a hidden, failed, or suspended scene never creates a camera.
    let subscription: { unsubscribe: () => void } | undefined;
    let bindTimer: number | null = null;
    let bindDelayMs = CAMERA_BIND_INITIAL_DELAY_MS;
    const bind = () => {
      bindTimer = null;
      if (disposed || subscription) return;
      const camera = getRwRef.current()?.getRenderer().getActiveCamera();
      if (!camera) {
        bindTimer = window.setTimeout(bind, bindDelayMs);
        bindDelayMs = Math.min(CAMERA_BIND_MAX_DELAY_MS, bindDelayMs * 2);
        return;
      }
      subscription = camera.onModified(schedule);
      update();
    };
    bind();

    return () => {
      disposed = true;
      if (bindTimer != null) window.clearTimeout(bindTimer);
      if (frame) cancelAnimationFrame(frame);
      subscription?.unsubscribe();
    };
  }, [scale.xy, scale.z, stormX, stormY, stormZ, lifecycleGeneration]);

  return orientation;
}
