interface PaintScheduler {
  requestAnimationFrame: (callback: FrameRequestCallback) => number;
  cancelAnimationFrame: (id: number) => void;
}

export function waitForCapturePaint(signal?: AbortSignal, timeoutMs = 8000, scheduler: PaintScheduler = globalThis): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException("Capture cancelled", "AbortError"));
  return new Promise((resolve, reject) => {
    let frame = 0;
    let remaining = 2;
    const cleanup = () => { scheduler.cancelAnimationFrame(frame); clearTimeout(timer); signal?.removeEventListener("abort", abort); };
    const abort = () => { cleanup(); reject(new DOMException("Capture cancelled", "AbortError")); };
    const timer = setTimeout(() => { cleanup(); reject(new Error("Viewpoint presentation paused. Bring Studio to the foreground and try again.")); }, timeoutMs);
    const paint = () => {
      if (--remaining === 0) { cleanup(); resolve(); }
      else frame = scheduler.requestAnimationFrame(paint);
    };
    signal?.addEventListener("abort", abort, { once: true });
    try { frame = scheduler.requestAnimationFrame(paint); }
    catch (error) { cleanup(); reject(error); }
  });
}
