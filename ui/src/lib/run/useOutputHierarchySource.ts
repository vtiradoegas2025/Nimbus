import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ThreeDControlScope } from "@/lib/plot/threeDControlScope";
import {
  publishShared3DSetting,
  readShared3DSetting,
  subscribeShared3DSetting,
} from "@/lib/plot/threeDControlScope";
import type { StepInfo } from "./runOutputs";
import {
  selectOutputHierarchyLevel,
  type OutputHierarchySelection,
  type ResolvedOutputLevel,
} from "./outputHierarchy";
import { useOutputHierarchy } from "./useOutputHierarchy";

const HIERARCHY_LEVEL_SETTING = "output-hierarchy-level";

export interface OutputHierarchySource {
  rootRunDir: string;
  rootCurrentStep: StepInfo | null;
  currentStep: StepInfo | null;
  requestedStep: StepInfo | null;
  sourceRunDir: string;
  sourceSteps: StepInfo[];
  /** Position of `currentStep` inside `sourceSteps`, or -1 when the selected
   *  level has no independently addressable history frame. Never reuse the
   *  root array position: a child may publish on a different cadence. */
  sourceStepIdx: number;
  activeLevel: ResolvedOutputLevel | null;
  selection: OutputHierarchySelection | null;
  requestedLevelId: "auto" | string;
  levelAtFocus: (point: readonly [number, number]) => string | null;
  setRequestedLevelId: (levelId: "auto" | string) => void;
  availableLevels: Array<{ levelId: string; depth: number; spacingM?: number }>;
  loading: boolean;
  error: string | null;
  valid: boolean;
}

interface LevelSetting {
  runDir: string;
  levelId: string;
}

function isLevelSetting(value: unknown): value is LevelSetting {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const candidate = value as Partial<LevelSetting>;
  return typeof candidate.runDir === "string"
    && candidate.runDir.length > 0
    && typeof candidate.levelId === "string"
    && candidate.levelId.length > 0
    && candidate.levelId.length <= 256;
}

/**
 * One hierarchy routing authority for field-based plots. Three-dimensional
 * views pass their existing shared/independent control scope; spatial 1-D/2-D
 * and Python views pass `shared` so one dataset-level Domain detail choice
 * follows the user between those analysis surfaces. Switching runs resets to
 * Auto so a branch identity can never leak across datasets.
 */
export function useOutputHierarchySource(options: {
  runDir: string;
  steps: StepInfo[];
  stepIdx: number;
  scope: ThreeDControlScope;
  requiredFields?: readonly string[];
  focusPointM?: readonly [number, number] | null;
}): OutputHierarchySource {
  const {
    runDir,
    steps,
    stepIdx,
    scope,
    requiredFields = [],
    focusPointM = null,
  } = options;
  const rootCurrentStep = steps[stepIdx] ?? null;
  const hierarchyEligible = rootCurrentStep?.publicationState === "complete"
    && rootCurrentStep.publicationGeneration != null;
  const hierarchy = useOutputHierarchy(runDir, rootCurrentStep, hierarchyEligible);
  const [levelSetting, setLevelSetting] = useState<LevelSetting>(() => {
    const shared = scope === "shared"
      ? readShared3DSetting(HIERARCHY_LEVEL_SETTING, isLevelSetting)
      : null;
    return shared?.runDir === runDir ? shared : { runDir, levelId: "auto" };
  });
  const requestedLevelId = levelSetting.runDir === runDir ? levelSetting.levelId : "auto";

  useEffect(() => {
    const shared = scope === "shared"
      ? readShared3DSetting(HIERARCHY_LEVEL_SETTING, isLevelSetting)
      : null;
    setLevelSetting(shared?.runDir === runDir ? shared : { runDir, levelId: "auto" });
  }, [runDir, scope]);
  useEffect(() => {
    if (scope !== "shared") return;
    return subscribeShared3DSetting(
      HIERARCHY_LEVEL_SETTING,
      isLevelSetting,
      (value) => {
        if (value.runDir === runDir) setLevelSetting(value);
      },
    );
  }, [runDir, scope]);

  const setRequestedLevelId = useCallback((levelId: "auto" | string) => {
    const next = { runDir, levelId };
    setLevelSetting(next);
    if (scope === "shared") publishShared3DSetting(HIERARCHY_LEVEL_SETTING, next);
  }, [runDir, scope]);

  const focusX = focusPointM?.[0] ?? null;
  const focusY = focusPointM?.[1] ?? null;
  const selection = useMemo(() => hierarchy.verdict
    ? selectOutputHierarchyLevel(hierarchy.verdict, {
        requestedLevelId,
        focusPointM: focusX != null && focusY != null ? [focusX, focusY] : null,
        requiredFields,
      })
    : null, [focusX, focusY, hierarchy.verdict, requestedLevelId, requiredFields]);
  const levelAtFocus = useCallback((point: readonly [number, number]) => hierarchy.verdict
    ? selectOutputHierarchyLevel(hierarchy.verdict, { requestedLevelId, focusPointM: point, requiredFields }).level.levelId
    : null, [hierarchy.verdict, requestedLevelId, requiredFields]);
  const activeLevel = selection?.level ?? null;
  const sourceRunDir = activeLevel?.runDir ?? runDir;
  const sourceSteps = activeLevel?.depth
    ? hierarchy.levels.find((level) => level.levelId === activeLevel.levelId)?.steps ?? []
    : steps;
  const currentStep = activeLevel?.step ?? rootCurrentStep;
  const sourceStepIdx = currentStep == null
    ? -1
    : sourceSteps.findIndex((step) =>
        step.index === currentStep.index
          && step.publicationGeneration === currentStep.publicationGeneration
      );
  const availableLevels = useMemo(() => [
    { levelId: "root", depth: 0, spacingM: rootCurrentStep?.grid?.drM },
    ...hierarchy.levels.map((level) => ({
      levelId: level.levelId,
      depth: level.depth,
      spacingM: level.steps.find((step) => step.grid?.drM != null)?.grid?.drM,
    })),
  ], [hierarchy.levels, rootCurrentStep?.grid?.drM]);
  const routeIdentity = JSON.stringify([runDir, requestedLevelId, requiredFields]);
  const loading = hierarchyEligible && hierarchy.loading;
  const retainedRoute = useRef<{
    identity: string;
    route: Pick<OutputHierarchySource, "currentStep" | "sourceRunDir" | "sourceSteps" | "sourceStepIdx" | "activeLevel" | "selection" | "availableLevels">;
  } | null>(null);
  const resolvedRoute = { currentStep, sourceRunDir, sourceSteps, sourceStepIdx, activeLevel, selection, availableLevels };
  if (!loading) retainedRoute.current = { identity: routeIdentity, route: resolvedRoute };
  // A pending hierarchy lookup must not temporarily replace a displayed child
  // with the root domain and its different controls, bounds and field lattice.
  const route = loading && retainedRoute.current?.identity === routeIdentity
    ? retainedRoute.current.route : resolvedRoute;
  return {
    rootRunDir: runDir,
    rootCurrentStep,
    ...route,
    requestedStep: loading ? route.sourceRunDir === runDir ? rootCurrentStep : null : currentStep,
    requestedLevelId,
    levelAtFocus,
    setRequestedLevelId,
    loading,
    error: hierarchy.error ?? (hierarchy.verdict && !hierarchy.verdict.ok ? hierarchy.verdict.reason : null),
    valid: hierarchy.verdict?.ok ?? false,
  };
}
