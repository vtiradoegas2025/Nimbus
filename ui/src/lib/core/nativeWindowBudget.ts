// Startup fallback and per-request defaults; the shared cap follows native preferences.
export const NATIVE_WINDOW_IO_BYTES = 512 * 1024 * 1024;
export const DEFAULT_NATIVE_WINDOW_SOURCE_BYTES = 64 * 1024 * 1024;
export const DEFAULT_NATIVE_WINDOW_OUTPUT_BYTES = 32 * 1024 * 1024;

let currentIoLimit = NATIVE_WINDOW_IO_BYTES;
const limitListeners = new Set<() => void>();
export function publishNativeIoBudgetLimit(bytes: number): void {
  if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error("Invalid native I/O limit");
  if (bytes === currentIoLimit) return;
  currentIoLimit = bytes;
  for (const listener of limitListeners) listener();
}
export function subscribeNativeIoBudgetLimit(listener: () => void): () => void {
  limitListeners.add(listener);
  return () => limitListeners.delete(listener);
}

export function nativeWindowReservationBytes(window: { maxSourceBytes?: number; maxOutputBytes?: number }): number {
  const source = window.maxSourceBytes ?? DEFAULT_NATIVE_WINDOW_SOURCE_BYTES;
  const output = window.maxOutputBytes ?? DEFAULT_NATIVE_WINDOW_OUTPUT_BYTES;
  const bytes = source + 2 * output;
  return [source, output, bytes].every(value => Number.isSafeInteger(value) && value > 0)
    ? bytes : Number.MAX_SAFE_INTEGER;
}
