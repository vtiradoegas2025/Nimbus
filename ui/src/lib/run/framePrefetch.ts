/*
 * Pre-warm frames playback is about to need.
 * Prefetches carry no latestScope so speculative reads cannot cancel the interactive request.
 * Failures are swallowed; depth is bounded to avoid occupying all scheduler slots.
 */

import { fieldLoadSchedulerStats } from "@/lib/core/loadScheduler";

/** Frames ahead of the rendering one to warm. Depth 1 hides one read behind a render; deeper depth starves the scheduler at multi-field scenes. */
export const PREFETCH_DEPTH = 1;

/** Default speculative ceiling when the loader has three slots. */
export const MAX_INFLIGHT_PREFETCH = 2;

let inflight = 0;
const controllers = new Set<AbortController>();

/** Reserve a speculative slot, or refuse when the ceiling is reached. */
export function tryBeginPrefetch(): boolean {
  const scheduler = fieldLoadSchedulerStats();
  const limit = Math.max(1, Math.min(4, scheduler.maxConcurrent - 1));
  if (inflight >= limit || scheduler.queued > 0 || scheduler.activeBytes >= scheduler.maxBytes) return false;
  inflight += 1;
  return true;
}

export function endPrefetch(): void {
  inflight = Math.max(0, inflight - 1);
}

/** Register one speculative read so interaction can cooperatively preempt it. */
export function createPrefetchController(): AbortController {
  const controller = new AbortController();
  controllers.add(controller);
  return controller;
}

export function releasePrefetchController(controller: AbortController): void {
  controllers.delete(controller);
}

/** Interactive field changes own the machine; obsolete speculation is expendable. */
export function cancelInflightPrefetches(): void {
  for (const controller of controllers) controller.abort();
  controllers.clear();
}

/** Speculative reads currently in flight. Exposed for tests. */
export function inflightPrefetchCount(): number {
  return inflight;
}

export interface PlaybackPrefetchHint {
  runDir: string;
  publicationGeneration: string | null;
  /** Absolute step indices, nearest first. */
  upcoming: readonly number[];
}

let hint: PlaybackPrefetchHint | null = null;
const runAliases = new Map<string, { ownerRunDir: string; users: number }>();
const MAX_METADATA_RUNS = 64;
const MAX_METADATA_FRAMES = 50_000;
const framesByRun = new Map<string, {
  frames: Map<number, string | null>;
  timelineGeneration: string | null;
}>();
let retainedFrames = 0;

export interface PrefetchFrameIdentity {
  stepIndex: number;
  publicationGeneration: string | null;
}

export function registerPlaybackFrameMetadata(
  runDir: string,
  steps: readonly { index: number; publicationGeneration: string | null }[],
  fullSnapshot = false,
): void {
  const previous = framesByRun.get(runDir);
  if (fullSnapshot && previous) retainedFrames -= previous.frames.size;
  const entry = fullSnapshot || !previous
    ? { frames: new Map<number, string | null>(), timelineGeneration: null as string | null }
    : previous;
  framesByRun.delete(runDir);
  framesByRun.set(runDir, entry);
  for (const step of steps) {
    if (!Number.isSafeInteger(step.index) || step.index < 0) continue;
    if (!entry.frames.has(step.index)) retainedFrames++;
    entry.frames.delete(step.index);
    entry.frames.set(step.index, step.publicationGeneration);
  }
  let earliest = Infinity;
  entry.timelineGeneration = null;
  for (const [index, generation] of entry.frames) {
    if (generation != null && index < earliest) {
      earliest = index;
      entry.timelineGeneration = generation;
    }
  }
  while (framesByRun.size > MAX_METADATA_RUNS || retainedFrames > MAX_METADATA_FRAMES) {
    const oldestRunDir = framesByRun.keys().next().value!;
    const oldest = framesByRun.get(oldestRunDir)!;
    if (framesByRun.size > MAX_METADATA_RUNS || oldest.frames.size === 0) {
      retainedFrames -= oldest.frames.size;
      framesByRun.delete(oldestRunDir);
    } else {
      oldest.frames.delete(oldest.frames.keys().next().value!);
      retainedFrames--;
      if (oldest.frames.size === 0) framesByRun.delete(oldestRunDir);
    }
  }
}

export function clearPlaybackFrameMetadata(): void {
  framesByRun.clear();
  retainedFrames = 0;
  setPlaybackPrefetchHint(null);
}

export function playbackFrameMetadataCount(): number {
  return retainedFrames;
}

/** Let a nested history consume the root timeline's upcoming frame hint while
 * retaining its own read/cache identity. Reference counting keeps multiple 3-D
 * panes from removing an alias still used by another pane. */
export function registerPlaybackPrefetchAlias(
  nestedRunDir: string,
  ownerRunDir: string,
): () => void {
  if (nestedRunDir === ownerRunDir) return () => undefined;
  const current = runAliases.get(nestedRunDir);
  if (current && current.ownerRunDir !== ownerRunDir) {
    throw new Error(`Prefetch alias '${nestedRunDir}' already belongs to another root run`);
  }
  runAliases.set(nestedRunDir, {
    ownerRunDir,
    users: (current?.users ?? 0) + 1,
  });
  return () => {
    const active = runAliases.get(nestedRunDir);
    if (!active || active.ownerRunDir !== ownerRunDir) return;
    if (active.users <= 1) runAliases.delete(nestedRunDir);
    else runAliases.set(nestedRunDir, { ...active, users: active.users - 1 });
  };
}

/**
 * Publish what playback will need next, or `null` when not playing.
 * Owned by the studio shell, which is the only place that knows the frame order.
 */
export function setPlaybackPrefetchHint(next: PlaybackPrefetchHint | null): void {
  if (next == null || (hint != null && (hint.runDir !== next.runDir
      || hint.publicationGeneration !== next.publicationGeneration))) {
    cancelInflightPrefetches();
  }
  hint = next;
}

export function playbackPrefetchHint(): PlaybackPrefetchHint | null {
  return hint;
}

/**
 * Steps to warm for a field in `runDir`, or an empty list when the hint does not
 * apply. Scoped by run AND publication generation so a republished run cannot
 * inherit the previous generation's queue.
 */
export function framesToPrefetch(
  runDir: string,
  publicationGeneration: string | null,
  currentStepIndex?: number,
): readonly number[] {
  if (currentStepIndex != null) {
    return frameIdentitiesToPrefetch(runDir, currentStepIndex, publicationGeneration)
      .map(frame => frame.stepIndex);
  }
  const timelineRunDir = runAliases.get(runDir)?.ownerRunDir ?? runDir;
  if (!hint || hint.runDir !== timelineRunDir) return [];
  if (hint.publicationGeneration !== publicationGeneration) return [];
  return hint.upcoming.slice(0, PREFETCH_DEPTH);
}

export function frameIdentitiesToPrefetch(
  runDir: string,
  currentStepIndex: number,
  currentPublicationGeneration: string | null,
): readonly PrefetchFrameIdentity[] {
  const timelineRunDir = runAliases.get(runDir)?.ownerRunDir ?? runDir;
  if (!hint || hint.runDir !== timelineRunDir) return [];
  const owner = framesByRun.get(timelineRunDir);
  const source = framesByRun.get(runDir);
  if (!owner || !source || owner.timelineGeneration !== hint.publicationGeneration
      || !source.frames.has(currentStepIndex)
      || source.frames.get(currentStepIndex) !== currentPublicationGeneration) return [];
  const result: PrefetchFrameIdentity[] = [];
  for (const stepIndex of hint.upcoming.slice(0, PREFETCH_DEPTH)) {
    if (stepIndex === currentStepIndex || !source.frames.has(stepIndex)) continue;
    result.push({ stepIndex, publicationGeneration: source.frames.get(stepIndex)! });
  }
  return result;
}

/**
 * The next `PREFETCH_DEPTH` absolute step indices after `stepIdx`, wrapping at
 * the end because playback loops. Returns fewer than requested when the run has
 * fewer frames, and never repeats a step (a two-frame run yields one entry).
 */
export function upcomingSteps(
  steps: readonly { index: number }[],
  stepIdx: number,
  stride = 1,
): number[] {
  const out: number[] = [];
  if (steps.length <= 1) return out;
  const safeStride = Number.isSafeInteger(stride) && stride > 0 ? stride : 1;
  const effectiveStride = Math.min(safeStride, steps.length - 1);
  for (let offset = 1; offset <= PREFETCH_DEPTH && offset < steps.length; offset++) {
    const frame = steps[(stepIdx + offset * effectiveStride) % steps.length];
    if (frame) out.push(frame.index);
  }
  return out;
}
