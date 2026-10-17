/** Small structural contracts keep lifecycle policy independent of vtk.js. */
export interface VtkDisposable {
  delete?: () => void;
  getPointData?: () => {
    getArrays: () => Array<{ delete?: () => void }>;
    removeAllArrays: () => void;
  };
}

export interface VtkRenderWindowDisposable extends VtkDisposable {
  /** The interactor, which `delete()` does NOT release — see `disposeVtkScene`. */
  getInteractor?: () => VtkDisposable | null | undefined;
  getApiSpecificRenderWindow?: () => {
    setContainer?: (container: null) => void;
    getCanvas?: () => {
      dataset?: Record<string, string>;
      getContext?: (kind: "webgl2" | "webgl") => {
        getExtension?: (name: "WEBGL_lose_context") => {
          loseContext: () => void;
        } | null;
      } | null;
    } | null;
  };
}

export interface DisposeVtkSceneOptions {
  /** Must clear the owning ref before any VTK callback can re-enter cleanup. */
  clearScene: () => void;
  /** Listener/observer/registry and renderer-detach operations, in order. */
  cleanupSteps?: readonly (() => void)[];
  /** Pipeline objects, ordered from actors/mappers back toward source data. */
  resources?: readonly VtkDisposable[];
  renderWindow: VtkRenderWindowDisposable;
}

function attempt(step: () => void, errors: unknown[]): void {
  try {
    step();
  } catch (error) {
    errors.push(error);
  }
}

export function runVtkCleanupSteps(steps: readonly (() => void)[]): unknown[] {
  const errors: unknown[] = [];
  for (const step of steps) attempt(step, errors);
  return errors;
}

/** Dispose one scene; never throws. Returns captured errors for diagnostics. */
export function disposeVtkScene(options: DisposeVtkSceneOptions): unknown[] {
  const errors: unknown[] = [];
  attempt(options.clearScene, errors);
  // Capture now, delete last: render window must be deleted before the interactor.
  let interactor: VtkDisposable | null | undefined;
  attempt(() => { interactor = options.renderWindow.getInteractor?.(); }, errors);
  for (const error of runVtkCleanupSteps(options.cleanupSteps ?? [])) {
    errors.push(error);
  }
  for (const resource of options.resources ?? []) {
    const pointData = resource.getPointData?.();
    if (pointData) {
      const arrays = [...new Set(pointData.getArrays())];
      attempt(() => pointData.removeAllArrays(), errors);
      for (const array of arrays) {
        if (typeof array.delete === "function") attempt(() => array.delete!(), errors);
      }
    }
    if (typeof resource.delete === "function") {
      attempt(() => resource.delete!(), errors);
    }
  }
  // Chromium does not release WebGL contexts promptly on delete; explicitly
  // retire via WEBGL_lose_context to stay under the browser's context ceiling.
  attempt(() => {
    const canvas = options.renderWindow.getApiSpecificRenderWindow?.().getCanvas?.();
    if (!canvas?.getContext) return;
    if (canvas.dataset) canvas.dataset.nimbusVtkRelease = "planned";
    const context = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    context?.getExtension?.("WEBGL_lose_context")?.loseContext();
  }, errors);
  attempt(
    () => options.renderWindow.getApiSpecificRenderWindow?.().setContainer?.(null),
    errors,
  );
  // vtkGenericRenderWindow.delete() never releases model.interactor; must delete
  // the interactor explicitly (after the render window) or the document
  // visibilitychange listener leaks and holds the entire renderer graph alive.
  if (typeof options.renderWindow.delete === "function") {
    attempt(() => options.renderWindow.delete!(), errors);
  }
  if (interactor && typeof interactor.delete === "function") {
    attempt(() => interactor!.delete!(), errors);
  }
  return errors;
}

export interface VtkContextState {
  lost: boolean;
  /** False between loss and restoration. Owners dispose the stale vtk pipeline
   * immediately and do not create another one against the lost drawing surface. */
  available: boolean;
  /** Incremented at both loss and restoration so scene effects first dispose,
   * then rebuild from a clean canvas. */
  generation: number;
}

export type VtkContextEvent =
  | { type: "lost" }
  | { type: "restored" }
  | { type: "frame_rendered"; generation: number };

/** Pure state machine used by the React hook and deterministic regressions. */
export function reduceVtkContextState(
  state: VtkContextState,
  event: VtkContextEvent,
): VtkContextState {
  switch (event.type) {
    case "lost":
      return state.lost
        ? state
        : { lost: true, available: false, generation: state.generation + 1 };
    case "restored":
      return state.lost && !state.available
        ? { lost: true, available: true, generation: state.generation + 1 }
        : state;
    case "frame_rendered":
      return state.lost && state.available && event.generation === state.generation
        ? { ...state, lost: false }
        : state;
  }
}
