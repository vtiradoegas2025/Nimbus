/*
 * Frame-readiness signaling for the movie recorder.
 * Keyed to the exact step/publication identity; other plot types fail closed.
 */

let inFlight = 0;
let recording = false;

/** Set while recording so 3D widgets bypass their scrub debounce — a debounced load could miss the min-frame floor and capture a stale frame. */
export function setRecording(active: boolean): void {
  recording = active;
}

/** True while a recording is capturing frames. */
export function isRecording(): boolean {
  return recording;
}

/** Called by a field loader when it starts fetching. */
export function beginFieldLoad(): void {
  inFlight += 1;
}

/** Called when a field load settles (success OR failure). Must be paired with
 *  beginFieldLoad, including when the originating effect was cancelled. */
export function endFieldLoad(): void {
  inFlight = Math.max(0, inFlight - 1);
}

/** How many field loads are currently fetching. */
export function fieldLoadsInFlight(): number {
  return inFlight;
}

/** Absolute source-frame positions selected for an export. Striding changes
 * which solved frames are included; output FPS independently controls how long
 * those frames are displayed in the encoded animation. */
export function captureFrameIndices(
  totalSteps: number,
  startStep: number,
  endStep: number,
  stride: number,
): number[] {
  const total = Math.max(0, Math.floor(totalSteps));
  if (total === 0) return [];
  const start = Math.max(0, Math.min(total - 1, Math.floor(startStep)));
  const end = Math.max(0, Math.min(total - 1, Math.floor(endStep)));
  if (start > end) return [];
  const safeStride = Math.max(1, Math.floor(stride));
  const indices: number[] = [];
  for (let index = start; index <= end; index += safeStride) indices.push(index);
  return indices;
}

function nextFrame(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException("Capture cancelled", "AbortError")); return; }
    const abort = () => { cancelAnimationFrame(frame); reject(new DOMException("Capture cancelled", "AbortError")); };
    const frame = requestAnimationFrame(() => { signal?.removeEventListener("abort", abort); resolve(); });
    signal?.addEventListener("abort", abort, { once: true });
  });
}

/**
 * Resolve once the next exact rendered frame is ready to capture and at least
 * `minMs` has elapsed, bounded by `maxMs`. Ends on a paint frame so the
 * post-render canvas is on screen before the snapshot.
 *
 * Returns `true` only for the requested absolute step and publication
 * generation. The recorder turns a timeout into a visible failure instead of
 * silently capturing the previous frame or a reused step from another run.
 */
export async function waitForFrameReady(
  minMs: number,
  maxMs: number,
  exactFrameReady: () => boolean,
  signal?: AbortSignal,
): Promise<boolean> {
  const start = performance.now();
  // Let React commit the new step index and let the load hooks begin fetching.
  await nextFrame(signal);
  await nextFrame(signal);
  for (;;) {
    const elapsed = performance.now() - start;
    // Readiness wins over the cap: a load that drains right at the deadline is
    // still a real frame, so only the "cap elapsed while still loading" case fails.
    if (exactFrameReady() && elapsed >= minMs) {
      await nextFrame(signal);
      if (exactFrameReady()) return true;
    }
    if (elapsed >= maxMs) {
      await nextFrame(signal);
      return false;
    }
    await nextFrame(signal);
  }
}
