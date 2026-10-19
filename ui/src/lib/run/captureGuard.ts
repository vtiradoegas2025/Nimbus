export interface CaptureGuard {
  signal?: AbortSignal;
  assertCurrent?: () => void;
}

export function assertCaptureCurrent(guard: CaptureGuard = {}): void {
  if (guard.signal?.aborted) throw new DOMException("Capture cancelled", "AbortError");
  guard.assertCurrent?.();
}

export async function guardedCapture<T>(capture: () => Promise<T>, guard: CaptureGuard = {}): Promise<T> {
  assertCaptureCurrent(guard);
  const result = await capture();
  assertCaptureCurrent(guard);
  return result;
}

export function gifFrameDelayMs(frameIndex: number, fps: number): number {
  if (!Number.isSafeInteger(frameIndex) || frameIndex < 0 || !Number.isFinite(fps) || fps < 1 || fps > 60) throw new Error("Invalid animation cadence");
  return 10 * (Math.round((frameIndex + 1) * 100 / fps) - Math.round(frameIndex * 100 / fps));
}
