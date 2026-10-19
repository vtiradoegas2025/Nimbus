import { useEffect, useMemo, useState } from "react";
import { augmentStepWithDerivedFields } from "@/lib/derived/derivedFieldRegistry";
import {
  listNestedHistoryLevels,
  type NestedHistoryLevel,
  type StepInfo,
} from "./runOutputs";
import { resolveOutputHierarchyFrame, type OutputHierarchyVerdict } from "./outputHierarchy";
import { registerPlaybackPrefetchAlias } from "./framePrefetch";

interface HierarchyState {
  runDir: string;
  publicationGeneration: string | null;
  levels: NestedHistoryLevel[];
  checkedStepIndices: number[];
  refreshingStepIndex: number | null;
  error: string | null;
  loading: boolean;
}

const NO_NESTED_LEVELS: NestedHistoryLevel[] = [];

function augmentLevels(levels: NestedHistoryLevel[]): NestedHistoryLevel[] {
  return levels.map((level) => ({
    ...level,
    steps: level.steps.map(augmentStepWithDerivedFields),
  }));
}

function mergeLevelFrames(
  existing: NestedHistoryLevel[],
  incoming: NestedHistoryLevel[],
): NestedHistoryLevel[] {
  const merged = new Map(existing.map((level) => [level.levelId, level]));
  for (const level of augmentLevels(incoming)) {
    const previous = merged.get(level.levelId);
    if (!previous) {
      merged.set(level.levelId, level);
      continue;
    }
    const steps = new Map(previous.steps.map((step) => [step.index, step]));
    for (const step of level.steps) steps.set(step.index, step);
    merged.set(level.levelId, {
      ...level,
      steps: [...steps.values()].sort((left, right) => left.index - right.index),
    });
  }
  return [...merged.values()].sort((left, right) =>
    left.depth - right.depth || left.levelId.localeCompare(right.levelId)
  );
}

/** Generation-safe nested-history discovery shared by hierarchy-aware plots. */
export function useOutputHierarchy(
  runDir: string,
  rootStep: StepInfo | null,
  enabled = true,
): {
  verdict: OutputHierarchyVerdict | null;
  levels: NestedHistoryLevel[];
  loading: boolean;
  error: string | null;
} {
  const [state, setState] = useState<HierarchyState>({
    runDir: "",
    publicationGeneration: null,
    levels: [],
    checkedStepIndices: [],
    refreshingStepIndex: null,
    error: null,
    loading: false,
  });
  useEffect(() => {
    let cancelled = false;
    const publicationGeneration = rootStep?.publicationGeneration ?? null;
    if (!enabled) {
      setState({
        runDir,
        publicationGeneration,
        levels: [],
        checkedStepIndices: [],
        refreshingStepIndex: null,
        error: null,
        loading: false,
      });
      return () => { cancelled = true; };
    }
    setState({
      runDir,
      publicationGeneration,
      levels: [],
      checkedStepIndices: [],
      refreshingStepIndex: null,
      error: null,
      loading: true,
    });
    void listNestedHistoryLevels(runDir, publicationGeneration).then(
      (levels) => {
        if (cancelled) return;
        const augmented = augmentLevels(levels);
        const checkedStepIndices = [...new Set(
          augmented.flatMap((level) => level.steps.map((step) => step.index)),
        )];
        setState({
          runDir,
          publicationGeneration,
          levels: augmented,
          checkedStepIndices,
          refreshingStepIndex: null,
          error: null,
          loading: false,
        });
      },
      (error: unknown) => {
        if (!cancelled) setState({
          runDir,
          publicationGeneration,
          levels: [],
          checkedStepIndices: [],
          refreshingStepIndex: null,
          error: error instanceof Error ? error.message : String(error),
          loading: false,
        });
      },
    );
    return () => { cancelled = true; };
  }, [enabled, rootStep?.publicationGeneration, runDir]);

  const currentIdentity = state.runDir === runDir
    && state.publicationGeneration === rootStep?.publicationGeneration;
  const needsStepRefresh = enabled
    && currentIdentity
    && !state.loading
    && state.levels.length > 0
    && rootStep != null
    && !state.checkedStepIndices.includes(rootStep.index);

  useEffect(() => {
    if (!needsStepRefresh || !rootStep) return;
    let cancelled = false;
    let retryTimer = 0;
    let releaseDelay: (() => void) | null = null;
    const stepIndex = rootStep.index;
    const publicationGeneration = rootStep.publicationGeneration;
    setState(current => ({ ...current, refreshingStepIndex: stepIndex }));
    void (async () => {
      try {
        for (let attempt = 0; attempt < 3; attempt++) {
          const levels = await listNestedHistoryLevels(runDir, publicationGeneration, stepIndex);
          if (cancelled) return;
          const arrived = levels.some(level => level.steps.some(step => step.index === stepIndex));
          if (arrived || attempt === 2) {
            setState(current => current.runDir === runDir && current.publicationGeneration === publicationGeneration
              ? { ...current, levels: mergeLevelFrames(current.levels, levels),
                checkedStepIndices: [...new Set([...current.checkedStepIndices, stepIndex])],
                refreshingStepIndex: null, error: null } : current);
            return;
          }
          await new Promise<void>(resolve => {
            releaseDelay = resolve;
            retryTimer = window.setTimeout(resolve, 150 + attempt * 150);
          });
          releaseDelay = null;
          if (cancelled) return;
        }
      } catch (error) {
        if (cancelled) return;
        setState(current => current.runDir === runDir && current.publicationGeneration === publicationGeneration
          ? { ...current, checkedStepIndices: [...new Set([...current.checkedStepIndices, stepIndex])],
            refreshingStepIndex: null, error: error instanceof Error ? error.message : String(error) } : current);
      }
    })();
    return () => {
      cancelled = true;
      if (retryTimer) window.clearTimeout(retryTimer);
      releaseDelay?.();
    };
  }, [needsStepRefresh, rootStep?.index, rootStep?.publicationGeneration, runDir]);

  useEffect(() => {
    if (!enabled || state.runDir !== runDir
        || state.publicationGeneration !== rootStep?.publicationGeneration) return;
    const unregister = state.levels.map((level) =>
      registerPlaybackPrefetchAlias(level.runDir, runDir)
    );
    return () => unregister.forEach((release) => release());
  }, [enabled, rootStep?.publicationGeneration, runDir, state.levels, state.publicationGeneration, state.runDir]);

  const activeLevels = enabled && currentIdentity ? state.levels : NO_NESTED_LEVELS;
  const loading = enabled && (!currentIdentity || state.loading || needsStepRefresh);
  const verdict = useMemo(() => rootStep && !loading
    ? resolveOutputHierarchyFrame(runDir, rootStep, activeLevels)
    : null, [activeLevels, loading, rootStep, runDir]);
  return {
    verdict,
    loading,
    error: enabled && currentIdentity ? state.error : null,
    levels: activeLevels,
  };
}
