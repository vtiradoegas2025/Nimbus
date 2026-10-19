/*
 * Step-completeness policy. Pure, no Tauri dependency, so it is unit-testable.
 * publicationState is set by the Rust bridge; the UI never guesses "legacy" from missing manifests.
 */

/** Minimal shape needed to judge completeness (StepInfo satisfies it). */
interface HasPublicationState {
  publicationState: "complete" | "incomplete" | "invalid_manifest" | "legacy";
}

/** Current complete frames and deliberately classified legacy frames are visible. */
export function stepIsComplete<T extends HasPublicationState>(step: T): boolean {
  return step.publicationState === "complete" || step.publicationState === "legacy";
}

/** Drop incomplete/torn steps. A current run with zero complete frames returns [] (waiting state); legacy runs stay readable. */
export function filterCompleteSteps<T extends HasPublicationState>(steps: T[]): T[] {
  return steps.filter(stepIsComplete);
}
