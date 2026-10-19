import { useCallback, useEffect, useRef, useState } from "react";
import type { PlotComponentProps } from "@nimbus/components/plots/common";
import { TimeSeriesPlot } from "@nimbus/components/plots/TimeSeriesPlot";
import { TimeHeightPlot } from "@nimbus/components/plots/TimeHeightPlot";
import { SwathPlot } from "@nimbus/components/plots/SwathPlot";
import { CompositePlot } from "@nimbus/components/plots/CompositePlot";
import { SlicePlot } from "@nimbus/components/plots/SlicePlot";
import { VectorField } from "@nimbus/components/plots/VectorField";
import { Hodograph } from "@nimbus/components/plots/Hodograph";
import { VerticalProfile } from "@nimbus/components/plots/VerticalProfile";
import { RadialProfile } from "@nimbus/components/plots/RadialProfile";
import { registerListedStepMetadata } from "@nimbus/lib/run/runOutputs";
import { fieldLoadScheduler } from "@nimbus/lib/core/loadScheduler";
import { fieldCacheStats } from "@nimbus/lib/core/fieldCache";
import { inflightPrefetchCount, setPlaybackPrefetchHint } from "@nimbus/lib/run/framePrefetch";
import { REGULAR_RUN, regularCalls, regularFixture, regularSource, regularSteps } from "../support/regularDataflowFixture";

type Plot = "series" | "height" | "swath" | "composite" | "slice" | "vector" | "hodograph" | "vertical" | "radial";
declare global {
  interface Window {
    __nimbusRegularHarness?: {
      setPlot: (plot: Plot) => void;
      setPreset: (preset: PlotComponentProps["preset"]) => void;
      setStep: (index: number) => void;
      setFrames: (indices: number[]) => void;
      releaseReads: () => void;
      revise: (index: number, offset: number) => void;
      setMissing: (index: number, missing: boolean) => void;
      setUnit: (index: number, unit: string) => void;
      setDelay: (field: string, milliseconds: number) => void;
      setSample: (index: number, field: string, cell: number, value: number) => void;
      setComponentUnit: (index: number, field: string, unit: string) => void;
      sourceFor: (field: string) => number[];
      snapshot: () => { calls: typeof regularCalls; source: number[]; frameReports: Array<Parameters<NonNullable<PlotComponentProps["onFrameDisplayChange"]>>[0]>;
        ownership: ReturnType<typeof regularFixture.readOwnership>; activeLoads: number; queuedLoads: number; prefetches: number; deduplicatedRequests: number };
    };
  }
}

const components = { series: TimeSeriesPlot, height: TimeHeightPlot, swath: SwathPlot,
  composite: CompositePlot, slice: SlicePlot, vector: VectorField, hodograph: Hodograph, vertical: VerticalProfile, radial: RadialProfile };
const initial = regularSteps([0, 1, 2]);
registerListedStepMetadata(REGULAR_RUN, initial);
const parameters = new URLSearchParams(window.location.search);
if (parameters.has("heldPrefetch") || parameters.has("failedRegularField")) {
  setPlaybackPrefetchHint({ runDir: REGULAR_RUN, publicationGeneration: initial[0].publicationGeneration, upcoming: [1] });
}

export function RegularDataflowHarness() {
  const [plot, setPlot] = useState<Plot>((new URLSearchParams(window.location.search).get("regularPlot") as Plot) || "series");
  const [frames, setFrames] = useState(initial);
  const [stepIdx, setStep] = useState(0);
  const [preset, setPreset] = useState<PlotComponentProps["preset"]>(undefined);
  const frameReports = useRef<Array<Parameters<NonNullable<PlotComponentProps["onFrameDisplayChange"]>>[0]>>([]);
  const onFrameDisplayChange = useCallback<NonNullable<PlotComponentProps["onFrameDisplayChange"]>>(
    report => { frameReports.current.push(report); }, []);
  useEffect(() => {
    const refresh = (indices: number[]) => { const next = regularSteps(indices); registerListedStepMetadata(REGULAR_RUN, next); setFrames(next); };
    const indices = () => frames.map(frame => frame.index);
    window.__nimbusRegularHarness = { setPlot, setPreset, setStep, setFrames: refresh, releaseReads: regularFixture.releaseReads,
      revise: (index, offset) => { regularFixture.revise(index, offset); refresh(indices()); },
      setMissing: (index, missing) => { regularFixture.setMissing(index, missing); refresh(indices()); },
      setUnit: (index, unit) => { regularFixture.setUnit(index, unit); refresh(indices()); },
      setDelay: regularFixture.setDelay,
      setSample: (index, field, cell, value) => { regularFixture.setSample(index, field, cell, value); refresh(indices()); },
      setComponentUnit: (index, field, unit) => { regularFixture.setComponentUnit(index, field, unit); refresh(indices()); },
      sourceFor: field => [...regularSource(0, field).data],
      snapshot: () => ({ calls: [...regularCalls], source: [...regularSource(0, "w").data], frameReports: [...frameReports.current],
        ownership: regularFixture.readOwnership(), activeLoads: fieldLoadScheduler.stats().active,
        queuedLoads: fieldLoadScheduler.stats().queued, prefetches: inflightPrefetchCount(), deduplicatedRequests: fieldCacheStats().deduplicatedRequests }),
    };
    return () => { delete window.__nimbusRegularHarness; };
  }, [frames]);
  const Component = components[plot];
  return <main style={{ width: 1000, height: 700, display: "flex", flexDirection: "column" }}>
    <Component key={plot} runDir={REGULAR_RUN} steps={frames} stepIdx={stepIdx}
      preset={preset} onPresetCleared={() => setPreset(undefined)}
      defaultSliceAxis={new URLSearchParams(window.location.search).get("section") === "1" ? 1 : undefined}
      onFrameDisplayChange={onFrameDisplayChange} />
  </main>;
}
