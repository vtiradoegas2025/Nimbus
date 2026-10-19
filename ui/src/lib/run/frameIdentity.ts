/**
 * Requested, loaded, and rendered frame identity are intentionally distinct.
 * Keeping this policy pure makes the scientific-label contract deterministic
 * without requiring a browser/React test harness.
 */

export interface FrameIdentityState {
  requestedStepIndex: number | null;
  requestedPublicationGeneration: string | null;
  loadedStepIndex: number | null;
  loadedPublicationGeneration: string | null;
}

export type FrameTransitionKind = "empty" | "loading" | "stale" | "ready";

export interface FrameTransition {
  kind: FrameTransitionKind;
  label: string;
}

export interface CaptureFrameIdentity {
  stepIndex: number;
  publicationGeneration: string | null;
}

export interface RequestedFrameIdentity {
  stepIndex: number | null;
  publicationGeneration: string | null;
}

interface LoadedFrameIdentity {
  loadedStepIndex: number | null;
  loadedPublicationGeneration: string | null;
}

interface PublishedFrameIdentity {
  index: number;
  publicationGeneration: string | null;
}

export interface DisplayedFrameIdentity {
  renderedStepIndex: number | null;
  renderedPublicationGeneration: string | null;
  loading: boolean;
}

/** Exact capture gate: ordinal equality is insufficient across run reuse. */
export function captureFrameIsReady(
  expected: CaptureFrameIdentity,
  displayed: DisplayedFrameIdentity | null,
): boolean {
  return displayed != null &&
    displayed.loading === false &&
    displayed.renderedStepIndex === expected.stepIndex &&
    displayed.renderedPublicationGeneration === expected.publicationGeneration;
}

/** React subtree identity for a deliberate run-directory open. The open epoch
 * distinguishes repeated opens of a markerless legacy directory, while the
 * publication generation distinguishes current runs that reuse the same path.
 */
export function runViewIdentityKey(
  runDir: string | null,
  publicationGeneration: string | null,
  openEpoch: number,
): string {
  return JSON.stringify([runDir, publicationGeneration, openEpoch]);
}

/** Find metadata for the exact loaded publication, not merely the same numeric
 * step. A live run may republish a step at a new generation. */
export function findPublishedFrame<T extends PublishedFrameIdentity>(
  frames: readonly T[],
  loadedStepIndex: number | null,
  loadedPublicationGeneration: string | null,
): T | null {
  if (loadedStepIndex == null) return null;
  return frames.find(
    (frame) => frame.index === loadedStepIndex &&
      frame.publicationGeneration === loadedPublicationGeneration,
  ) ?? null;
}

/** Multi-field plots may render only when every participating array belongs to
 * the same exact publication. Shape equality alone cannot prevent a torn frame
 * while independent field reads finish at different times. */
export function loadedFramesAreCoherent(
  ...frames: readonly LoadedFrameIdentity[]
): boolean {
  if (frames.length === 0 || frames[0].loadedStepIndex == null) return false;
  const first = frames[0];
  return frames.every(
    (frame) => frame.loadedStepIndex === first.loadedStepIndex &&
      frame.loadedPublicationGeneration === first.loadedPublicationGeneration,
  );
}

/** Identity state immediately after a field request begins. */
export function beginFrameRequest(
  previousLoadedStepIndex: number | null,
  previousLoadedPublicationGeneration: string | null,
  requestedStepIndex: number,
  requestedPublicationGeneration: string | null,
  retainPrevious: boolean,
): FrameIdentityState {
  return {
    requestedStepIndex,
    requestedPublicationGeneration,
    loadedStepIndex: retainPrevious ? previousLoadedStepIndex : null,
    loadedPublicationGeneration: retainPrevious
      ? previousLoadedPublicationGeneration
      : null,
  };
}

/** Identity state after the matching request resolves successfully. */
export function completeFrameRequest(
  requestedStepIndex: number,
  requestedPublicationGeneration: string | null,
): FrameIdentityState {
  return {
    requestedStepIndex,
    requestedPublicationGeneration,
    loadedStepIndex: requestedStepIndex,
    loadedPublicationGeneration: requestedPublicationGeneration,
  };
}

/** Honest user-facing description of selected versus rendered pixels. */
export function describeFrameTransition(
  requestedStepIndex: number | null,
  renderedStepIndex: number | null,
  requestedPublicationGeneration: string | null,
  renderedPublicationGeneration: string | null,
): FrameTransition {
  if (requestedStepIndex == null) {
    return { kind: "empty", label: "No frame selected" };
  }
  if (
    renderedStepIndex === requestedStepIndex &&
    renderedPublicationGeneration === requestedPublicationGeneration
  ) {
    return { kind: "ready", label: `Showing step ${renderedStepIndex}` };
  }
  if (renderedStepIndex != null) {
    if (renderedStepIndex === requestedStepIndex) {
      return {
        kind: "stale",
        label: `Showing the previous run's step ${renderedStepIndex} while the reopened run loads`,
      };
    }
    return {
      kind: "stale",
      label: `Showing step ${renderedStepIndex} while step ${requestedStepIndex} loads`,
    };
  }
  return { kind: "loading", label: `Loading step ${requestedStepIndex}` };
}
