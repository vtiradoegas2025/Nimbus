import { useEffect, useMemo, useState } from "react";
import { readFieldBrickHierarchy, type StepInfo } from "@/lib/run/runOutputs";
import { planPublishedVolumeOverview, PublishedVolumeOverviewUnavailableError, type PublishedVolumeOverviewPlan } from "@/lib/plot/publishedVolumeOverview";
import { interactiveFullDomainSourceSafe, planBoundedVolumeWindows, windowsForFieldAggregation } from "@/lib/plot/fieldLoadPlanning";
import { beginFieldLoad, endFieldLoad } from "@/lib/run/frameReady";
import { useField, type UseFieldResult } from "./common";

const plans = new Map<string, Promise<PublishedVolumeOverviewPlan>>();

export function usePublishedVolumeOverview(input: {
  enabled: boolean;
  runDir: string;
  step: StepInfo | null | undefined;
  fieldName: string | null;
  maxCells: number;
}): UseFieldResult & { sourceLevel: number | null; planning: boolean; disclosure: string | null } {
  const { enabled, runDir, step, fieldName, maxCells } = input;
  const declaration = fieldName ? step?.fieldBrickHierarchies?.[fieldName] : null;
  const identity = useMemo(() => JSON.stringify([enabled, runDir, step?.index, step?.publicationGeneration,
    declaration, fieldName, maxCells, step?.grid?.terrainFollowing,
    step?.grid?.nr, step?.grid?.nth, step?.grid?.nz, step?.grid?.validIndexRange]),
  [enabled, runDir, step?.index, step?.publicationGeneration, declaration, fieldName, maxCells, step?.grid?.terrainFollowing,
    step?.grid?.nr, step?.grid?.nth, step?.grid?.nz, step?.grid?.validIndexRange]);
  const [state, setState] = useState<{ identity: string; plan: PublishedVolumeOverviewPlan | null; error: string | null }>({
    identity: "", plan: null, error: null,
  });
  useEffect(() => {
    if (!enabled || !step?.grid || !fieldName || !declaration) return;
    let cancelled = false;
    beginFieldLoad();
    let work = plans.get(identity);
    if (!work) {
      const grid = step.grid;
      work = readFieldBrickHierarchy(runDir, step, fieldName)
        .then(hierarchy => {
          try { return planPublishedVolumeOverview(hierarchy, grid, maxCells); }
          catch (error) {
            if (!(error instanceof PublishedVolumeOverviewUnavailableError) || !interactiveFullDomainSourceSafe(grid)) throw error;
            const raw = planBoundedVolumeWindows(grid, maxCells);
            if (!raw.ok) throw new Error(raw.reason);
            return { level: 0, sampledCells: raw.plan.sampledCells,
              windows: windowsForFieldAggregation(raw.plan.windows, fieldName, {
                sourceKind: step.fieldOrigins?.[fieldName], terrainFollowing: grid.terrainFollowing,
              }),
              disclosure: `bounded raw cell-mean overview ${raw.plan.stride.join("×")} · display ${raw.plan.sampledShape.join("×")}`,
            };
          }
        });
      plans.set(identity, work);
      while (plans.size > 16) plans.delete(plans.keys().next().value!);
      void work.catch(() => { if (plans.get(identity) === work) plans.delete(identity); });
    }
    void work.then(plan => {
      if (!cancelled) setState({ identity, plan, error: null });
    }, error => {
      if (!cancelled) setState({ identity, plan: null, error: error instanceof Error ? error.message : String(error) });
    }).finally(endFieldLoad);
    return () => { cancelled = true; };
  }, [identity]);
  const active = enabled && state.identity === identity ? state.plan : null;
  const field = useField(runDir, active ? step?.index ?? null : null, active ? fieldName : null,
    step?.publicationGeneration ?? null, active?.windows,
    { prefetch: false, brickHierarchyLevel: active && active.level > 0 ? active.level : undefined });
  const planning = enabled && Boolean(step?.grid && fieldName && declaration) && state.identity !== identity;
  return useMemo(() => ({
    ...field,
    loading: planning || field.loading,
    error: enabled && !declaration ? "The field has no published overview"
      : enabled && state.identity === identity && state.error ? state.error : field.error,
    requestedStepIndex: enabled ? step?.index ?? null : null,
    requestedPublicationGeneration: enabled ? step?.publicationGeneration ?? null : null,
    sourceLevel: active?.level ?? null, planning, disclosure: active?.disclosure ?? null,
  }), [field, planning, enabled, state, identity, active, step?.index, step?.publicationGeneration]);
}
