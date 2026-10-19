import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { usePublishedVolumeOverview } from "@nimbus/components/plots/usePublishedVolumeOverview";
import { clearFieldCache } from "@nimbus/lib/core/fieldCache";
import { readFieldWindows, registerListedStepMetadata, type StepInfo } from "@nimbus/lib/run/runOutputs";
import { planPublishedVolumeOverview, type PublishedVolumeOverviewPlan } from "@nimbus/lib/plot/publishedVolumeOverview";
import { planBoundedVolumeWindows, windowsForFieldAggregation } from "@nimbus/lib/plot/fieldLoadPlanning";
import type { FieldWindowGeometry } from "@nimbus/lib/platform/npy";
import { derivedBrickSteps, makeDerivedBrickHierarchy } from "../support/fixtures";
import { OVERVIEW_RUN, overviewCalls, setOverviewHierarchyMode, setOverviewMetadataDelay, type OverviewHierarchyMode } from "../support/publishedOverviewFixture";

type Result = ReturnType<typeof usePublishedVolumeOverview>;
const MAX_CELLS = 50_000;

function geometry(value: FieldWindowGeometry | undefined) {
  if (!value) return null;
  return { ...value, centroidsM: value.centroidsM && {
    x: Array.from(value.centroidsM.x), y: Array.from(value.centroidsM.y), z: Array.from(value.centroidsM.z),
  }, volumesM3: value.volumesM3 && Array.from(value.volumesM3) };
}

declare global {
  interface Window {
    __nimbusPublishedOverviewHarness?: {
      setFrame: (index: number) => void;
      setMode: (mode: OverviewHierarchyMode) => void;
      setMetadataDelay: (milliseconds: number) => void;
      rerender: () => void;
      snapshot: () => {
        frame: number; publication: string | null; mode: OverviewHierarchyMode; revision: number;
        calls: typeof overviewCalls;
        consumers: {
          loading: boolean; planning: boolean; error: string | null;
          requestedStepIndex: number | null; requestedPublicationGeneration: string | null;
          loadedStepIndex: number | null; loadedPublicationGeneration: string | null;
          level: number | null; disclosure: string | null; shape: number[] | null; cells: number;
          geometry: ReturnType<typeof geometry>;
        }[];
      };
      compareOrdinaryRead: () => Promise<{
        level: number; cells: number; shape: number[]; windows: PublishedVolumeOverviewPlan["windows"];
        geometry: ReturnType<typeof geometry>; comparedValues: number; mismatches: number[];
        sameConsumerBacking: boolean; independentReference: boolean; geometryMatches: boolean[];
      }>;
    };
  }
}

function Consumer({ step, index, results }: { step: StepInfo; index: number; results: { current: (Result | null)[] } }) {
  const result = usePublishedVolumeOverview({ enabled: true, runDir: OVERVIEW_RUN, step, fieldName: "w", maxCells: MAX_CELLS });
  useLayoutEffect(() => { results.current[index] = result; }, [result, index, results]);
  return <output data-consumer={index}>{result.loading ? "loading" : result.error ?? `${result.sourceLevel}: ${result.field?.shape.join("×")}`}</output>;
}

export function PublishedVolumeOverviewHarness() {
  const [frame, setFrame] = useState(0);
  const [mode, setMode] = useState<OverviewHierarchyMode>("durable");
  const [revision, setRevision] = useState(0);
  const results = useRef<(Result | null)[]>([null, null]);
  const step = useMemo(() => {
    const original = derivedBrickSteps[frame];
    const declared = original.fieldBrickHierarchies!.w;
    const next: StepInfo = { ...original, fieldBrickHierarchies: { ...original.fieldBrickHierarchies,
      w: { ...declared, contentSha256: ["durable", "native-only", "changed-publication", "invalid-shape"].indexOf(mode).toString(16).repeat(64) } } };
    registerListedStepMetadata(OVERVIEW_RUN, [next]);
    return next;
  }, [frame, mode]);
  useLayoutEffect(() => {
    window.__nimbusPublishedOverviewHarness = {
      setFrame, setMetadataDelay: setOverviewMetadataDelay,
      setMode: next => { setOverviewHierarchyMode(next); setMode(next); },
      rerender: () => setRevision(value => value + 1),
      snapshot: () => ({ frame, publication: step.publicationGeneration ?? null, mode, revision,
        calls: [...overviewCalls], consumers: results.current.flatMap(result => result ? [{
          loading: result.loading, planning: result.planning, error: result.error,
          requestedStepIndex: result.requestedStepIndex, requestedPublicationGeneration: result.requestedPublicationGeneration,
          loadedStepIndex: result.loadedStepIndex, loadedPublicationGeneration: result.loadedPublicationGeneration,
          level: result.sourceLevel, disclosure: result.disclosure, shape: result.field?.shape ?? null,
          cells: result.field?.data.length ?? 0, geometry: geometry(result.field?.windowGeometry),
        }] : []) }),
      compareOrdinaryRead: async () => {
        const fields = results.current.map(result => result?.field);
        if (fields.some(field => !field)) throw new Error("Both mounted overview consumers must have complete fields");
        let plan: PublishedVolumeOverviewPlan;
        if (mode === "durable") plan = planPublishedVolumeOverview(makeDerivedBrickHierarchy(frame, "w"), step.grid!, MAX_CELLS);
        else if (mode === "native-only") {
          const raw = planBoundedVolumeWindows(step.grid!, MAX_CELLS);
          if (!raw.ok) throw new Error(raw.reason);
          plan = { level: 0, sampledCells: raw.plan.sampledCells, disclosure: "reference",
            windows: windowsForFieldAggregation(raw.plan.windows, "w", { sourceKind: step.fieldOrigins?.w, terrainFollowing: step.grid!.terrainFollowing }) };
        } else throw new Error("Invalid hierarchy cannot have a reference overview");
        clearFieldCache();
        const reference = await readFieldWindows(OVERVIEW_RUN, frame, "w", step.publicationGeneration ?? null, plan.windows,
          { brickHierarchyLevel: plan.level > 0 ? plan.level : undefined });
        const mismatches = fields.map(field => {
          if (field!.data.length !== reference.data.length || field!.shape.join() !== reference.shape.join()) return -1;
          let count = 0;
          for (let index = 0; index < reference.data.length; index++) if (!Object.is(field!.data[index], reference.data[index])) count++;
          return count;
        });
        return { level: plan.level, cells: reference.data.length, shape: reference.shape, windows: plan.windows,
          geometry: geometry(reference.windowGeometry), comparedValues: reference.data.length * fields.length, mismatches,
          sameConsumerBacking: fields[0]!.data === fields[1]!.data,
          independentReference: fields.every(field => field!.data !== reference.data),
          geometryMatches: fields.map(field => JSON.stringify(geometry(field!.windowGeometry)) === JSON.stringify(geometry(reference.windowGeometry))) };
      },
    };
    return () => { delete window.__nimbusPublishedOverviewHarness; };
  }, [frame, mode, revision, step]);
  return <main data-revision={revision}><Consumer step={step} index={0} results={results} /><Consumer step={step} index={1} results={results} /></main>;
}
