export type GpuRenderTimingStatus = "pending" | "available" | "unavailable" | "disjoint" | "failed";
export interface GpuRenderTimer {
  begin(): boolean;
  end(onComplete: (milliseconds: number) => void): void;
  cancel(): void;
  dispose(): void;
  status(): GpuRenderTimingStatus;
}

export function createVtkGpuRenderTimer(getContext: () => WebGL2RenderingContext | null): GpuRenderTimer {
  let status: GpuRenderTimingStatus = "pending";
  let query: WebGLQuery | null = null;
  let context: WebGL2RenderingContext | null = null;
  let extension: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null = null;
  let active = false, frame = 0, disposed = false, startedAt = 0;
  const cancel = () => {
    cancelAnimationFrame(frame); frame = 0;
    try { if (active && context && extension && !context.isContextLost()) context.endQuery(extension.TIME_ELAPSED_EXT); } catch { /* A lost driver must not prevent render cleanup. */ }
    active = false;
    try { if (query && context) context.deleteQuery(query); } catch { /* A retired context owns no usable timing result. */ }
    query = null;
  };
  return {
    begin() {
      if (disposed || query) return false;
      try {
        context = getContext();
        if (!context || typeof context.createQuery !== "function" || context.isContextLost()) { status = "unavailable"; return false; }
        extension = context.getExtension("EXT_disjoint_timer_query_webgl2");
        if (!extension) { status = "unavailable"; return false; }
        if (context.getParameter(extension.GPU_DISJOINT_EXT)) { status = "disjoint"; return false; }
        if (context.getQuery(extension.TIME_ELAPSED_EXT, context.CURRENT_QUERY)) return false;
        query = context.createQuery();
        if (!query) { status = "failed"; return false; }
        context.beginQuery(extension.TIME_ELAPSED_EXT, query);
        active = true; startedAt = performance.now(); status = "pending";
        return true;
      } catch { status = "failed"; cancel(); return false; }
    },
    end(onComplete) {
      if (!active || !context || !extension || !query) return;
      try { context.endQuery(extension.TIME_ELAPSED_EXT); active = false; }
      catch { status = "failed"; cancel(); return; }
      const poll = () => {
        frame = 0;
        if (disposed || !context || !extension || !query) return;
        try {
          if (context.isContextLost() || context.getParameter(extension.GPU_DISJOINT_EXT)) { status = "disjoint"; cancel(); return; }
          if (context.getQueryParameter(query, context.QUERY_RESULT_AVAILABLE)) {
            const milliseconds = Number(context.getQueryParameter(query, context.QUERY_RESULT)) / 1_000_000;
            cancel();
            if (!(milliseconds > 0) || !Number.isFinite(milliseconds)) { status = "failed"; return; }
            status = "available"; onComplete(milliseconds); return;
          }
          if (performance.now() - startedAt >= 2000) { status = "failed"; cancel(); return; }
          frame = requestAnimationFrame(poll);
        } catch { status = "failed"; cancel(); }
      };
      frame = requestAnimationFrame(poll);
    },
    cancel,
    dispose() { disposed = true; cancel(); },
    status: () => status,
  };
}
