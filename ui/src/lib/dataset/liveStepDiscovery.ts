import { stepIsComplete } from "@/lib/run/stepCompleteness";

/** Minimal frame contract needed by the live-run discovery reconciler. */
export interface DiscoverableStep {
  index: number;
  publicationState: "complete" | "incomplete" | "invalid_manifest" | "legacy";
  publicationGeneration: string | null;
}

export interface LiveStepDiscovery<T extends DiscoverableStep> {
  steps: T[];
  selectedIndex: number;
  publicationGeneration: string | null;
  generationChanged: boolean;
  incompleteCount: number;
}

function stepIdentity(step: DiscoverableStep): string {
  return `${step.publicationGeneration ?? "legacy"}:${step.index}`;
}

/**
 * Reconcile one directory scan with the selectable frames already in Studio.
 *
 * Within one publication generation, completed frames are immutable and scans
 * are append-only. Retaining the existing objects avoids reloading every
 * time-series consumer on an unchanged poll. A generation change is different:
 * it replaces the inventory so no frame/cache identity crosses between runs.
 */
export function reconcileLiveSteps<T extends DiscoverableStep>(
  currentSteps: T[],
  discoveredSteps: readonly T[],
  currentSelectedIndex: number,
  currentPublicationGeneration: string | null,
  followTail: boolean,
): LiveStepDiscovery<T> {
  const visible = discoveredSteps.filter(stepIsComplete);
  const discoveredGeneration = discoveredSteps.find(
    (step) => step.publicationGeneration != null,
  )?.publicationGeneration;
  // An empty scan carries no run-marker metadata. Preserve the known
  // generation and inventory instead of mistaking a transient empty listing
  // for replacement by a legacy run.
  const publicationGeneration =
    discoveredGeneration ??
    (discoveredSteps.length === 0
      ? currentPublicationGeneration
      : null);
  const generationChanged =
    publicationGeneration !== currentPublicationGeneration;

  let steps: T[];
  if (generationChanged) {
    steps = [...visible].sort((a, b) => a.index - b.index);
  } else {
    const known = new Set(currentSteps.map(stepIdentity));
    const additions = visible.filter((step) => !known.has(stepIdentity(step)));
    if (additions.length === 0) {
      steps = currentSteps;
    } else {
      steps = [...currentSteps, ...additions].sort((a, b) => a.index - b.index);
    }
  }

  const oldSelected = currentSteps[currentSelectedIndex] ?? null;
  const wasAtTail =
    currentSteps.length === 0 || currentSelectedIndex >= currentSteps.length - 1;
  let selectedIndex = 0;
  if (steps.length > 0) {
    if (!generationChanged && followTail && wasAtTail) {
      selectedIndex = steps.length - 1;
    } else if (!generationChanged && oldSelected) {
      const preserved = steps.findIndex(
        (step) => stepIdentity(step) === stepIdentity(oldSelected),
      );
      selectedIndex = preserved >= 0
        ? preserved
        : Math.min(currentSelectedIndex, steps.length - 1);
    } else {
      selectedIndex = followTail ? steps.length - 1 : 0;
    }
  }

  return {
    steps,
    selectedIndex,
    publicationGeneration,
    generationChanged,
    incompleteCount: discoveredSteps.length - visible.length,
  };
}
