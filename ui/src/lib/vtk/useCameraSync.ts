import { useEffect, useRef } from "react";
import type { CameraState, CameraSyncBus } from "@/lib/plot/cameraSyncBus";
import type { SceneScale } from "@/lib/plot/sceneScale";

/** Bind one pane's vtk camera to the shared sync bus in true-metre space. */

interface SyncCamera {
  getPosition: () => number[];
  getFocalPoint: () => number[];
  getViewUp: () => number[];
  getParallelProjection?: () => boolean;
  getParallelScale?: () => number;
  getViewAngle?: () => number;
  setPosition: (x: number, y: number, z: number) => void;
  setFocalPoint: (x: number, y: number, z: number) => void;
  setViewUp: (x: number, y: number, z: number) => void;
  setParallelProjection?: (value: boolean) => void;
  setParallelScale?: (value: number) => void;
  setViewAngle?: (value: number) => void;
  orthogonalizeViewUp?: () => void;
  onModified: (cb: () => void) => { unsubscribe: () => void };
}

interface InteractionStyle {
  onStartInteractionEvent?: (callback: () => void) => { unsubscribe: () => void };
  onEndInteractionEvent?: (callback: () => void) => { unsubscribe: () => void };
}

export interface SyncRenderWindow {
  getRenderer: () => {
    getActiveCamera: () => SyncCamera;
    resetCameraClippingRange?: () => void;
  };
  getRenderWindow: () => { render: () => void };
  getInteractor?: () => { getInteractorStyle: () => InteractionStyle | null };
}

export interface CameraSyncOptions {
  paneId: string;
  enabled: boolean;
  scale: SceneScale;
  /** Exactly one pane seeds the opening/re-locked view. */
  authority: boolean;
  /** Prevent an empty vtk camera from becoming the shared opening view. */
  isReady: () => boolean;
  lifecycleGeneration?: number;
  onCameraApplied?: () => void;
  /** Remote camera streams use preview quality and settle at idle. */
  onRemoteInteractionChange?: (active: boolean) => void;
}

function normalize(values: readonly [number, number, number]): [number, number, number] {
  const length = Math.hypot(values[0], values[1], values[2]);
  return length > 0 && Number.isFinite(length)
    ? [values[0] / length, values[1] / length, values[2] / length]
    : [0, 0, 1];
}

function toTruePoint(values: ArrayLike<number>, scale: SceneScale): [number, number, number] {
  return [
    (values[0] ?? 0) / (scale.xy || 1),
    (values[1] ?? 0) / (scale.xy || 1),
    (values[2] ?? 0) / (scale.z || 1),
  ];
}

function toDisplayPoint(values: readonly [number, number, number], scale: SceneScale) {
  return [values[0] * scale.xy, values[1] * scale.xy, values[2] * scale.z] as const;
}

function toTrueDirection(values: ArrayLike<number>, scale: SceneScale) {
  return normalize(toTruePoint(values, scale));
}

function toDisplayDirection(values: readonly [number, number, number], scale: SceneScale) {
  return normalize([
    values[0] * scale.xy,
    values[1] * scale.xy,
    values[2] * scale.z,
  ]);
}

function directionDisplayScale(directionM: readonly [number, number, number], scale: SceneScale) {
  return Math.hypot(
    directionM[0] * scale.xy,
    directionM[1] * scale.xy,
    directionM[2] * scale.z,
  ) || 1;
}

export function useCameraSync(
  getRenderWindow: () => SyncRenderWindow | null,
  bus: CameraSyncBus | null,
  options: CameraSyncOptions,
): void {
  const getRwRef = useRef(getRenderWindow);
  getRwRef.current = getRenderWindow;
  const latest = useRef(options);
  latest.current = options;

  useEffect(() => {
    if (!bus || !options.enabled) return;

    let disposed = false;
    let publishFrame = 0;
    let bindFrame = 0;
    let seedFrame = 0;
    let remoteIdle: ReturnType<typeof setTimeout> | null = null;
    let pendingReadyTimer: ReturnType<typeof setTimeout> | null = null;
    let cameraSubscription: { unsubscribe: () => void } | undefined;
    let startSubscription: { unsubscribe: () => void } | undefined;
    let endSubscription: { unsubscribe: () => void } | undefined;
    let unsubscribeBus: (() => void) | undefined;
    let locallyInteracting = false;
    let seedAttempts = 0;
    let bindAttempts = 0;
    let pendingRemoteState: CameraState | null = null;

    const cameraState = (camera: SyncCamera): CameraState => {
      const current = latest.current;
      const viewUpM = toTrueDirection(camera.getViewUp(), current.scale);
      const parallelScale = camera.getParallelScale?.();
      return {
        position: toTruePoint(camera.getPosition(), current.scale),
        focalPoint: toTruePoint(camera.getFocalPoint(), current.scale),
        viewUp: viewUpM,
        parallelProjection: camera.getParallelProjection?.() ?? false,
        viewAngle: camera.getViewAngle?.() ?? 30,
        ...(parallelScale !== undefined
          ? { parallelScaleM: parallelScale / directionDisplayScale(viewUpM, current.scale) }
          : {}),
      };
    };

    const publish = (forceUser = false) => {
      publishFrame = 0;
      if (disposed || bus.isApplying()) return;
      const current = latest.current;
      if (!current.isReady()) return;
      // Internal resets/refits in follower panes must not seize the shared
      // camera. Direct interaction is still allowed from every pane.
      if (!current.authority && !locallyInteracting && !forceUser) return;
      const camera = getRwRef.current()?.getRenderer().getActiveCamera();
      if (camera) bus.publish(current.paneId, cameraState(camera));
    };

    const schedulePublish = () => {
      if (publishFrame || disposed) return;
      publishFrame = requestAnimationFrame(() => publish(false));
    };

    const seedWhenReady = () => {
      seedFrame = 0;
      if (disposed || !latest.current.authority) return;
      const rw = getRwRef.current();
      if (!rw || !latest.current.isReady()) {
        if (++seedAttempts < 600) seedFrame = requestAnimationFrame(seedWhenReady);
        return;
      }
      bus.seed(latest.current.paneId, cameraState(rw.getRenderer().getActiveCamera()));
    };

    const noteRemoteCamera = () => {
      latest.current.onRemoteInteractionChange?.(true);
      if (remoteIdle !== null) clearTimeout(remoteIdle);
      remoteIdle = setTimeout(() => {
        remoteIdle = null;
        latest.current.onRemoteInteractionChange?.(false);
      }, 220);
    };

    const applyRemoteState = () => {
      pendingReadyTimer = null;
      if (disposed || !pendingRemoteState) return;
      const current = latest.current;
      // Applying before the pane has actors lets its own opening fit overwrite
      // the shared camera a moment later. Retain only the newest bus state and
      // wait cheaply for the renderer to finish its first fit.
      if (!current.isReady()) {
        pendingReadyTimer = setTimeout(applyRemoteState, 50);
        return;
      }
      const rw = getRwRef.current();
      if (!rw) {
        pendingReadyTimer = setTimeout(applyRemoteState, 50);
        return;
      }
      const state = pendingRemoteState;
      pendingRemoteState = null;
      const camera = rw.getRenderer().getActiveCamera();
      const position = toDisplayPoint(state.position, current.scale);
      const focalPoint = toDisplayPoint(state.focalPoint, current.scale);
      const viewUp = toDisplayDirection(state.viewUp, current.scale);
      camera.setPosition(...position);
      camera.setFocalPoint(...focalPoint);
      camera.setViewUp(...viewUp);
      camera.orthogonalizeViewUp?.();
      camera.setParallelProjection?.(state.parallelProjection);
      camera.setViewAngle?.(state.viewAngle);
      if (state.parallelScaleM !== undefined && camera.setParallelScale) {
        camera.setParallelScale(
          state.parallelScaleM * directionDisplayScale(state.viewUp, current.scale),
        );
      }
      current.onCameraApplied?.();
      noteRemoteCamera();
      rw.getRenderer().resetCameraClippingRange?.();
      rw.getRenderWindow().render();
    };

    const bind = () => {
      bindFrame = 0;
      if (disposed) return;
      const rw = getRwRef.current();
      if (!rw) {
        if (++bindAttempts < 300) bindFrame = requestAnimationFrame(bind);
        return;
      }
      const camera = rw.getRenderer().getActiveCamera();
      cameraSubscription = camera.onModified(schedulePublish);

      const style = rw.getInteractor?.().getInteractorStyle();
      startSubscription = style?.onStartInteractionEvent?.(() => {
        locallyInteracting = true;
        publish(true);
      });
      endSubscription = style?.onEndInteractionEvent?.(() => {
        // Publish the post-stability camera on the following frame. Other end
        // listeners can correct view-up and clipping in this same event turn.
        requestAnimationFrame(() => {
          publish(true);
          locallyInteracting = false;
        });
      });

      unsubscribeBus = bus.subscribe(options.paneId, (state) => {
        if (disposed) return;
        pendingRemoteState = state;
        if (pendingReadyTimer === null) applyRemoteState();
      }, { replayLatest: !options.authority });

      if (options.authority) seedFrame = requestAnimationFrame(seedWhenReady);
    };

    bindFrame = requestAnimationFrame(bind);

    return () => {
      disposed = true;
      if (publishFrame) cancelAnimationFrame(publishFrame);
      if (bindFrame) cancelAnimationFrame(bindFrame);
      if (seedFrame) cancelAnimationFrame(seedFrame);
      if (remoteIdle !== null) clearTimeout(remoteIdle);
      if (pendingReadyTimer !== null) clearTimeout(pendingReadyTimer);
      latest.current.onRemoteInteractionChange?.(false);
      cameraSubscription?.unsubscribe();
      startSubscription?.unsubscribe();
      endSubscription?.unsubscribe();
      unsubscribeBus?.();
    };
  }, [
    bus,
    options.paneId,
    options.enabled,
    options.authority,
    options.scale.xy,
    options.scale.z,
    options.lifecycleGeneration,
  ]);
}
