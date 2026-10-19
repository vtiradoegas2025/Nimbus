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
  getRenderer?: () => VtkDisposable | null | undefined;
  getRenderWindow?: () => VtkDisposable | null | undefined;
  getApiSpecificRenderWindow?: () => {
    releaseGraphicsResources?: () => void;
    setContainer?: (container: null) => void;
    getCanvas?: () => {
      dataset?: Record<string, string>;
      width?: number;
      height?: number;
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
  let renderer: VtkDisposable | null | undefined;
  let coreWindow: VtkDisposable | null | undefined;
  attempt(() => { renderer = options.renderWindow.getRenderer?.(); }, errors);
  attempt(() => { coreWindow = options.renderWindow.getRenderWindow?.(); }, errors);
  // VTK window deletion drops its JS model without releasing cached graphics resources.
  attempt(() => options.renderWindow.getApiSpecificRenderWindow?.().releaseGraphicsResources?.(), errors);
  for (const error of runVtkCleanupSteps(options.cleanupSteps ?? [])) {
    errors.push(error);
  }
  for (const resource of options.resources ?? []) {
    let pointData: ReturnType<NonNullable<VtkDisposable["getPointData"]>> | undefined;
    attempt(() => { pointData = resource.getPointData?.(); }, errors);
    if (pointData) {
      const data = pointData;
      let arrays: ReturnType<typeof data.getArrays> = [];
      attempt(() => { arrays = [...new Set(data.getArrays())]; }, errors);
      attempt(() => data.removeAllArrays(), errors);
      for (const array of arrays) {
        if (typeof array.delete === "function") attempt(() => array.delete!(), errors);
      }
    }
    if (typeof resource.delete === "function") {
      attempt(() => resource.delete!(), errors);
    }
  }
  let retireContext: (() => void) | undefined;
  let retireDrawingBuffer: (() => void) | undefined;
  attempt(() => {
    const canvas = options.renderWindow.getApiSpecificRenderWindow?.().getCanvas?.();
    if (!canvas) return;
    retireDrawingBuffer = () => {
      if (typeof canvas.width === "number") canvas.width = 0;
      if (typeof canvas.height === "number") canvas.height = 0;
    };
    if (canvas.dataset) canvas.dataset.nimbusVtkRelease = "planned";
    const context = canvas.getContext?.("webgl2") ?? canvas.getContext?.("webgl");
    const extension = context?.getExtension?.("WEBGL_lose_context");
    if (extension) retireContext = () => extension.loseContext();
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
  // Mapper/framebuffer deletion needs a live GL context. Retire the captured
  // context afterward; waiting for browser GC can exhaust the context ceiling.
  if (retireContext) attempt(retireContext, errors);
  // Detached canvases can retain native drawing surfaces until collection.
  if (retireDrawingBuffer) attempt(retireDrawingBuffer, errors);
  if (interactor && typeof interactor.delete === "function") {
    attempt(() => interactor!.delete!(), errors);
  }
  // GenericRenderWindow creates these core owners but only deletes its API view
  // and wrapper. Keep them alive until interactor callbacks have stopped.
  const coreOwners = new Set([renderer, coreWindow]);
  coreOwners.delete(options.renderWindow);
  coreOwners.delete(interactor);
  for (const resource of options.resources ?? []) coreOwners.delete(resource);
  for (const owner of coreOwners) attempt(() => owner?.delete?.(), errors);
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
