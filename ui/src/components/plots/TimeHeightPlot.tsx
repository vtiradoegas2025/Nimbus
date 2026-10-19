import { timeHeightGeometry, type TimeHeightColumn } from "@/lib/plot/timeHeightGeometry";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import {
  REDUCE_CONCURRENCY,
  reduceFieldAxis,
  type FieldReducer,
} from "@/lib/run/runOutputs";
import { unitFor, type FieldUnits } from "@/lib/science/fieldUnits";
import { mapPool } from "@/lib/core/concurrency";
import { fieldsAcrossStepsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { axisOptions } from "@/lib/science/gridCoords";
import {
  plotlyColorRange,
  resolveColorscale,
  type ExtendedColorscale,
} from "@/lib/plot/plotColormaps";
import { defaultScalarDisplay } from "@/lib/plot/scalarDisplayDefaults";
import type { SliceAxis } from "@/lib/platform/npy";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  ControlGroup,
  PillGroup,
  type PlotChrome,
  type PlotComponentProps,
  uniformFieldAnnotation,
  usePlotChrome,
} from "./common";
import { PlotCompressionControls, RootHistoryScopeNotice } from "./PlotSourceDisclosure";
import { historyFrameKey, historyUnitMismatch, newHistoryReductionCache, reconcileHistoryReductionCache } from "@/lib/plot/historyReductionCache";

/* Time-height (Hovmöller) heatmap: per-level field reducer vs time, server-side (reduceFieldAxis). */

const REDUCERS: ReadonlyArray<{ value: FieldReducer; label: string }> = [
  { value: "max", label: "max" },
  { value: "min", label: "min" },
  { value: "mean", label: "mean" },
  { value: "rms", label: "rms" },
];

const COLORMAPS: ReadonlyArray<{ value: ExtendedColorscale; label: string }> = [
  { value: "Viridis", label: "Viridis" },
  { value: "Plasma", label: "Plasma" },
  { value: "Inferno", label: "Inferno" },
  { value: "RdBu", label: "RdBu" },
  { value: "Greys", label: "Greys" },
  { value: "dBZ", label: "NWS dBZ" },
  { value: "ColdPool", label: "Cold pool (BuRd)" },
];
const NATIVE_FIELD_ORIGINS = new Set(["native"] as const);

type SeriesColumn = TimeHeightColumn & { loaded: boolean };

export function TimeHeightPlot({ runDir, steps, stepIdx }: PlotComponentProps) {
  const chrome = usePlotChrome();
  const allFields = useMemo(
    () => fieldsAcrossStepsForPlotRanks(runDir, steps, [3], NATIVE_FIELD_ORIGINS),
    [runDir, steps],
  );

  const [fieldName, setFieldName] = useState<string | null>(allFields[0] ?? null);
  const [reducer, setReducer] = useState<FieldReducer>("max");
  const [axis, setAxis] = useState<SliceAxis>(2); // z → time-height
  const [colormap, setColormap] = useState<ExtendedColorscale>(() =>
    defaultScalarDisplay(
      allFields[0] ?? null,
      steps.find((step) => step.fields.includes(allFields[0] ?? ""))?.fieldUnits,
    ).colormap
  );
  const [series, setSeries] = useState<SeriesColumn[]>([]);
  const [loadingPct, setLoadingPct] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reductionCacheRef = useRef(newHistoryReductionCache<SeriesColumn>());
  const referenceStep = useMemo(
    () => steps.find((step) => fieldName != null && step.fields.includes(fieldName)) ?? null,
    [steps, fieldName],
  );
  const grid = referenceStep?.grid ?? null;
  const fieldUnits = referenceStep?.fieldUnits;

  // Reselect the field if the current one disappears (e.g. opening another run).
  useEffect(() => {
    if (fieldName && allFields.includes(fieldName)) return;
    const next = allFields[0] ?? null;
    setFieldName(next);
    setColormap(defaultScalarDisplay(
      next,
      steps.find((step) => next != null && step.fields.includes(next))?.fieldUnits,
    ).colormap);
  }, [allFields, fieldName, steps]);

  // Load every step's per-axis reduction. Re-runs when field / axis / reducer
  // changes (each is computed Rust-side per request).
  useEffect(() => {
    if (!fieldName || steps.length === 0) {
      reductionCacheRef.current = newHistoryReductionCache<SeriesColumn>();
      setSeries([]);
      setLoadingPct(null);
      return;
    }
    let cancelled = false;
    const unitError = historyUnitMismatch(steps, fieldName);
    if (unitError) { setSeries([]); setLoadingPct(null); setError(unitError); return; }
    const seriesKey = JSON.stringify([
      runDir,
      fieldName,
      axis,
      reducer,
    ]);
    if (reductionCacheRef.current.seriesKey !== seriesKey) {
      reductionCacheRef.current = newHistoryReductionCache<SeriesColumn>(seriesKey);
    }
    const cache = reductionCacheRef.current;
    const requests = steps.map((step, index) => ({
      index,
      step,
      key: historyFrameKey(step, fieldName),
    }));
    reconcileHistoryReductionCache(cache, requests.map(request => request.key));
    for (const request of requests) {
      if (!request.step.fields.includes(fieldName) && !cache.values.has(request.key)) {
        cache.values.set(request.key, null);
      }
    }
    const results = requests.map(({ key, step }) => cache.values.get(key)
      ? { ...cache.values.get(key)!, timeS: step.simulationTimeS, grid: step.grid ?? null }
      : { step: step.index, timeS: step.simulationTimeS, loaded: false, values: [], grid: step.grid ?? null });
    let settled = requests.reduce((count, { key }) => count + (cache.values.has(key) ? 1 : 0), 0);
    const outstanding = requests.filter(({ key }) => !cache.values.has(key));
    let publishFrame: number | null = null;
    setSeries([...results]);
    setError(null);
    setLoadingPct(settled < steps.length ? (settled / steps.length) * 100 : null);

    const publish = () => {
      publishFrame = null;
      if (cancelled) return;
      setLoadingPct((settled / steps.length) * 100);
      setSeries([...results]);
    };
    const schedulePublish = () => {
      if (publishFrame == null) publishFrame = requestAnimationFrame(publish);
    };
    const cancelPublish = () => {
      if (publishFrame != null) cancelAnimationFrame(publishFrame);
      publishFrame = null;
    };

    // Overlap the per-step axis reductions (see TimeSeriesPlot) up to
    // REDUCE_CONCURRENCY, drawing the heatmap progressively as columns land.
    void mapPool(
      outstanding.length,
      REDUCE_CONCURRENCY,
      async (i): Promise<SeriesColumn | null> => {
        const request = outstanding[i];
        let pending = cache.pending.get(request.key);
        if (!pending) {
          pending = reduceFieldAxis(
            runDir,
            request.step.index,
            fieldName,
            axis,
            reducer,
            request.step.publicationGeneration,
          ).then((values) => {
            const column = {
              loaded: true,
              step: request.step.index,
              timeS: request.step.simulationTimeS,
              values,
              grid: request.step.grid ?? null,
            } satisfies SeriesColumn;
            if (cache.activeKeys.has(request.key)) cache.values.set(request.key, column);
            cache.pending.delete(request.key);
            return column;
          }, (reason) => {
            cache.pending.delete(request.key);
            throw reason;
          });
          cache.pending.set(request.key, pending);
        }
        return pending;
      },
      (i, res) => {
        if (cancelled) return false;
        if (!res.ok) {
          cancelPublish();
          setError(res.error instanceof Error ? res.error.message : String(res.error));
          setLoadingPct(null);
          return false;
        }
        if (res.value) results[outstanding[i].index] = { ...res.value, timeS: outstanding[i].step.simulationTimeS, grid: outstanding[i].step.grid ?? null };
        settled += 1;
        schedulePublish();
        return true;
      },
    ).then(() => {
      if (!cancelled) {
        cancelPublish();
        setSeries([...results]);
        setLoadingPct(null);
      }
    });

    return () => {
      cancelled = true;
      cancelPublish();
    };
  }, [runDir, steps, fieldName, axis, reducer]);

  const geometry = useMemo(() => timeHeightGeometry(series, axis), [series, axis]);
  const plot = useMemo(
    () => buildHeatmap(geometry, axis, fieldName, reducer, colormap, fieldUnits, chrome),
    [geometry, axis, fieldName, reducer, colormap, fieldUnits, chrome],
  );

  return (
    <>
      <RootHistoryScopeNotice runDir={runDir} step={steps[stepIdx] ?? null} />
      <div className="plot-controls">
        <ControlGroup label="Field">
          <select
            aria-label="Field"
            value={fieldName ?? ""}
            onChange={(e) => {
              const field = e.target.value || null;
              setFieldName(field);
              setColormap(defaultScalarDisplay(
                field,
                steps.find((step) => field != null && step.fields.includes(field))?.fieldUnits,
              ).colormap);
            }}
          >
            {allFields.length === 0 && <option value="">(no fields)</option>}
            {allFields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </ControlGroup>
        <PlotCompressionControls steps={steps} fields={[fieldName]} />

        <ControlGroup label="Reducer">
          <PillGroup options={REDUCERS} value={reducer} onChange={setReducer} />
        </ControlGroup>

        <ControlGroup label="Profile axis">
          <PillGroup options={axisOptions(grid)} value={axis} onChange={setAxis} />
        </ControlGroup>

        <ControlGroup label="Colormap">
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value as ExtendedColorscale)}
          >
            {COLORMAPS.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </ControlGroup>

        <ControlGroup label="Frames" stretch>
          <span className="plot-controls__status">
            {series.filter((column) => column.loaded).length} / {steps.length} loaded
          </span>
        </ControlGroup>
      </div>

      <div className="plot-body">
        {loadingPct != null && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>Loading {loadingPct.toFixed(0)}%</span>
          </div>
        )}
        {error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{error}</span>
          </div>
        )}
        {plot ? (
          <PlotCanvas data={plot.data} layout={plot.layout} />
        ) : (
          !error && (
            <div className="plot-overlay">
              <span>No data to plot.</span>
            </div>
          )
        )}
      </div>
    </>
  );
}

interface BuiltPlot {
  data: Data[];
  layout: Partial<Layout>;
}

function buildHeatmap(
  geometry: ReturnType<typeof timeHeightGeometry>,
  axis: SliceAxis,
  fieldName: string | null,
  reducer: FieldReducer,
  colormap: ExtendedColorscale,
  fieldUnits: FieldUnits,
  chrome: PlotChrome,
): BuiltPlot | null {
  if (!geometry || !fieldName) return null;
  if ("error" in geometry) return {
    data: [], layout: { annotations: [{ text: geometry.error, showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5 }] },
  };
  const { coordinates: yc, range: valueRange, xTitle: xUnit, xHover } = geometry;
  const unit = unitFor(fieldName, fieldUnits);
  const label = `${reducer}(${fieldName})${unit ? ` (${unit})` : ""}`;

  const data: Data[] = geometry.groups.map((group, index) => ({
      type: "heatmap",
      // Time and height samples are discrete output/model levels. Preserve
      // missing intervals and do not reconstruct unreported intermediate data.
      zsmooth: false,
      hoverongaps: false,
      z: group.z,
      x: group.x,
      y: group.y,
      customdata: group.y.map(() => group.times),
      showscale: index === 0,
      colorscale: resolveColorscale(colormap) as never,
      zmin: valueRange?.min, zmax: valueRange?.max,
      ...plotlyColorRange(colormap, valueRange),
      colorbar: { title: { text: label, side: "right" } },
      hovertemplate:
        `${xHover.replace("%{x", "%{customdata")}<br>${yc.title}=%{y${yc.hoverFmt}}` +
        `<br>${label}=%{z:.4g}<extra></extra>`,
    }));

  const layout: Partial<Layout> = {
    xaxis: { title: { text: xUnit }, range: geometry.xRange, uirevision: xUnit },
    yaxis: { title: { text: yc.title }, uirevision: `${axis}:${yc.unit}` },
    title: {
      text: `${label} — time vs ${yc.title}`,
      font: { size: 13 },
    },
    annotations: (() => {
      const uniform = uniformFieldAnnotation(valueRange, label, chrome);
      return uniform ? [uniform] : [];
    })(),
  };

  return { data, layout };
}
