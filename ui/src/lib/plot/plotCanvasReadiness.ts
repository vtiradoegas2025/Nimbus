interface CanvasState {
  revision: number;
  pending: number;
  errors: Partial<Record<"render" | "resize", Error>>;
  latest: Record<"render" | "resize", number>;
  retired: boolean;
  listeners: Set<() => void>;
}

export interface PlotCanvasReceipt {
  readonly element: HTMLElement;
  readonly revision: number;
  readonly owner: object;
}

const canvases = new WeakMap<HTMLElement, CanvasState>();

export function beginPlotCanvasOperation(element: HTMLElement, kind: "render" | "resize" = "render"): (error?: unknown) => void {
  let state = canvases.get(element);
  if (!state || state.retired) {
    state = { revision: 0, pending: 0, errors: {}, latest: { render: 0, resize: 0 }, retired: false, listeners: new Set() };
    canvases.set(element, state);
  }
  const owner = state;
  const revision = ++owner.revision;
  owner.pending += 1;
  owner.latest[kind] = revision;
  delete owner.errors[kind];
  let finished = false;
  return (error?: unknown) => {
    if (finished) return;
    finished = true;
    owner.pending -= 1;
    if (error !== undefined && revision === owner.latest[kind]) {
      owner.errors[kind] = error instanceof Error ? error : new Error(String(error));
    }
    for (const listener of owner.listeners) listener();
  };
}

export function retirePlotCanvas(element: HTMLElement): void {
  const state = canvases.get(element);
  if (!state) return;
  state.retired = true;
  for (const listener of state.listeners) listener();
  canvases.delete(element);
}

export function isPlotCanvasReceiptCurrent(receipt: PlotCanvasReceipt): boolean {
  const state = canvases.get(receipt.element);
  return Boolean(receipt.element.isConnected && state === receipt.owner && state
    && !state.retired && state.pending === 0 && !state.errors.render && !state.errors.resize && state.revision === receipt.revision);
}

export function waitForPlotCanvasReady(element: HTMLElement, timeoutMs = 15_000): Promise<PlotCanvasReceipt> {
  const state = canvases.get(element);
  if (!state || state.retired || !element.isConnected) {
    return Promise.reject(new Error("The plot is no longer available for capture."));
  }
  return new Promise((resolve, reject) => {
    const finish = (error?: Error) => {
      clearTimeout(timer);
      state.listeners.delete(check);
      if (error) reject(error);
      else resolve({ element, revision: state.revision, owner: state });
    };
    const check = () => {
      if (state.retired || !element.isConnected || canvases.get(element) !== state) {
        finish(new Error("The plot changed while preparing capture."));
      } else if (state.pending === 0) {
        finish(state.errors.render ?? state.errors.resize);
      }
    };
    const timer = setTimeout(() => finish(new Error("The plot did not finish rendering before capture.")), timeoutMs);
    state.listeners.add(check);
    check();
  });
}
