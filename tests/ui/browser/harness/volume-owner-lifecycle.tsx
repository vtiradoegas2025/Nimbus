import { useEffect, useMemo, useState } from "react";
import { useBrickedVolumePreparation } from "@nimbus/lib/vtk/useBrickedVolumePreparation";
import { useNestedBrickedVolumePreparation } from "@nimbus/lib/vtk/useNestedBrickedVolumePreparation";
import { useVolumeBackground } from "@nimbus/lib/vtk/useVolumeBackground";
import { registerListedStepMetadata } from "@nimbus/lib/run/runOutputs";
import { visualizationResourceGovernor } from "@nimbus/lib/core/visualizationResourceGovernor";
import type { NestedVolumeOwnershipLayer } from "@nimbus/lib/run/outputHierarchy";
import { derivedBrickSteps, nestedChildBrickSteps } from "../support/fixtures";
import { snapshotBrowserResources } from "../support/instrumentation";
import { publishNativeIoBudgetLimit } from "@nimbus/lib/core/nativeWindowBudget";

const RUN = "/__nimbus_browser_fixture__/bricked-derived";
const CHILD = "/__nimbus_browser_fixture__/nested-child";
const TRANSFER = { colormap: "Viridis", opacityScale: 1, scalarTransform: "none" as const };
const LIMIT = 2_000_000;
registerListedStepMetadata(RUN, derivedBrickSteps);
registerListedStepMetadata(CHILD, nestedChildBrickSteps);

const params = new URLSearchParams(window.location.search);
let rejectResident = params.get("rejectResident") === "1";
let rejectedResidency = 0;
let rejectReads = params.get("rejectReads") === "1";
let holdPreparation = params.get("holdPreparation") === "1";
let heldPreparations = 0;
export async function volumeOwnerInvoke(command: string): Promise<{ value: unknown } | null> {
  if (rejectReads && command === "read_field_window") throw new Error("Analytical fixture I/O budget exceeded");
  return null;
}
if (params.get("mode") === "volume-owner") {
  const post = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function(this: Worker, ...args: Parameters<Worker["postMessage"]>) {
    if (holdPreparation && (args[0]?.kind === "prepare" || args[0]?.kind === "prepare-anomaly-batch")) {
      heldPreparations++;
      return;
    }
    return post.apply(this, args);
  } as Worker["postMessage"];
  const transition = visualizationResourceGovernor.transition.bind(visualizationResourceGovernor);
  visualizationResourceGovernor.transition = (id, claims) => {
    const owner = visualizationResourceGovernor.getReservation(id)?.owner;
    if (rejectResident && (owner?.startsWith("vtk:nested-volume:") || owner?.startsWith("vtk:bricked-volume:"))
        && claims.length > 0 && claims.every(claim => (claim.accounting ?? "resident") === "resident")) {
      rejectedResidency++;
      return { ok: false, reason: "Analytical fixture resident capacity rejection", requiredBytes: 1, availableBytes: 0, stage: null };
    }
    return transition(id, claims);
  };
}

type State = { current: boolean; loading: boolean; error: string | null; bricks: readonly unknown[] };
declare global {
  interface Window {
    __nimbusVolumeOwner?: {
      enable: (enabled: boolean) => void;
      limit: (cells: number) => void;
      permitResidency: () => void;
      increaseIoLimit: () => void;
      permitPreparation: () => void;
      snapshot: () => { current: boolean; loading: boolean; error: string | null; bricks: number;
        preparedBytes: number; reservedBytes: number; workers: number; rejectedResidency: number; heldPreparations: number };
    };
  }
}

function Controls({ state, enable, limit }: { state: State; enable: (value: boolean) => void; limit: (value: number) => void }) {
  useEffect(() => {
    window.__nimbusVolumeOwner = { enable, limit, permitResidency: () => { rejectResident = false; },
      increaseIoLimit: () => { rejectReads = false; publishNativeIoBudgetLimit(1024 ** 3); },
      permitPreparation: () => { holdPreparation = false; },
      snapshot: () => ({ current: state.current, loading: state.loading, error: state.error, bricks: state.bricks.length,
        preparedBytes: visualizationResourceGovernor.stats().byStage["prepared-main"].residentBytes,
        reservedBytes: visualizationResourceGovernor.stats().reservedBytes,
        workers: snapshotBrowserResources().workers, rejectedResidency, heldPreparations }),
    };
    return () => { delete window.__nimbusVolumeOwner; };
  }, [state, enable, limit]);
  return <output>{state.error ?? (state.loading ? "Loading" : `${state.bricks.length} bricks`)}</output>;
}

function SingleOwner() {
  const [enabled, enable] = useState(true);
  const [maximumPreparedCells, limit] = useState(LIMIT);
  const state = useBrickedVolumePreparation({ enabled, runDir: RUN, step: derivedBrickSteps[0],
    fieldName: "dense_signed", transfer: TRANSFER, maximumPreparedCells, maximumTextureDimension: 2048 });
  return <Controls state={state} enable={enable} limit={limit} />;
}

function NestedOwner() {
  const [enabled, enable] = useState(true);
  const [maximumPreparedCells, limit] = useState(LIMIT);
  const layers = useMemo<NestedVolumeOwnershipLayer[]>(() => [
    { level: { levelId: "root", parentLevelId: null, runDir: RUN, step: derivedBrickSteps[0], depth: 0,
      horizontalCellAreaM2: derivedBrickSteps[0].grid!.drM! * derivedBrickSteps[0].grid!.dyM! },
      excludeChildBoundsM: nestedChildBrickSteps[0].grid!.physicalBoundsM! },
    { level: { levelId: "child", parentLevelId: "root", runDir: CHILD, step: nestedChildBrickSteps[0], depth: 1,
      horizontalCellAreaM2: nestedChildBrickSteps[0].grid!.drM! * nestedChildBrickSteps[0].grid!.dyM! },
      excludeChildBoundsM: null },
  ], []);
  const state = useNestedBrickedVolumePreparation({ enabled, layers, fieldName: "dense_signed", transfer: TRANSFER,
    maximumPreparedCells, maximumTextureDimension: 2048 });
  return <Controls state={state} enable={enable} limit={limit} />;
}

function BackgroundOwner() {
  const [enabled, enable] = useState(true);
  const state = useVolumeBackground(enabled, RUN, derivedBrickSteps[0], "u");
  return <Controls state={{ ...state, current: state.profile != null, bricks: state.profile ? [state.profile] : [] }}
    enable={enable} limit={() => undefined} />;
}

export function VolumeOwnerLifecycleHarness() {
  return params.get("owner") === "background" ? <BackgroundOwner />
    : params.get("owner") === "nested" ? <NestedOwner /> : <SingleOwner />;
}
