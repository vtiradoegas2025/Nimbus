/* Camera synchronisation bus across panes. Pure — no React, no vtk.js. */

export interface CameraState {
  /** Camera coordinates in true metres, before a pane's display exaggeration. */
  position: readonly [number, number, number];
  focalPoint: readonly [number, number, number];
  viewUp: readonly [number, number, number];
  parallelProjection: boolean;
  viewAngle: number;
  /** Parallel half-height in true metres along view-up. */
  parallelScaleM?: number;
}

export type CameraSyncListener = (state: CameraState, fromPaneId: string) => void;

export interface CameraSyncBus {
  /** @returns an unsubscribe function. */
  subscribe(
    paneId: string,
    listener: CameraSyncListener,
    options?: { replayLatest?: boolean },
  ): () => void;
  /** Broadcast this pane's camera to every other subscriber. */
  publish(paneId: string, state: CameraState): void;
  /** Establish the authoritative opening view and always deliver it. */
  seed(paneId: string, state: CameraState): void;
  /** True while a broadcast is being applied, so appliers can suppress echo. */
  isApplying(): boolean;
}

function sameTriple(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

export function cameraStatesEqual(a: CameraState | null, b: CameraState | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    sameTriple(a.position, b.position) &&
    sameTriple(a.focalPoint, b.focalPoint) &&
    sameTriple(a.viewUp, b.viewUp) &&
    a.parallelProjection === b.parallelProjection &&
    a.viewAngle === b.viewAngle &&
    a.parallelScaleM === b.parallelScaleM
  );
}

export function createCameraSyncBus(): CameraSyncBus {
  const listeners = new Map<string, CameraSyncListener>();
  let applying = false;
  let last: CameraState | null = null;

  return {
    subscribe(paneId, listener, options) {
      listeners.set(paneId, listener);
      if (options?.replayLatest && last) {
        const wasApplying = applying;
        applying = true;
        try {
          try {
            listener(last, "camera-authority");
          } catch {
            // A remounting pane can disappear between subscribe and replay.
          }
        } finally {
          applying = wasApplying;
        }
      }
      return () => {
        // Guard against remount race: only drop if still our listener.
        if (listeners.get(paneId) === listener) listeners.delete(paneId);
      };
    },

    publish(paneId, state) {
      // `applying` breaks immediate echo; equality check breaks slower re-broadcast loop.
      if (applying) return;
      if (cameraStatesEqual(last, state)) return;
      last = state;

      applying = true;
      try {
        for (const [id, listener] of listeners) {
          if (id === paneId) continue;
          // Isolate each delivery; one broken pane must not desynchronise others.
          try {
            listener(state, paneId);
          } catch {
            // Not actionable; next publish will resynchronise if scene recovers.
          }
        }
      } finally {
        // `finally` ensures applying is always restored.
        applying = false;
      }
    },

    seed(paneId, state) {
      last = null;
      this.publish(paneId, state);
    },

    isApplying() {
      return applying;
    },
  };
}
