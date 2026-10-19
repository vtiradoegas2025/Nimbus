import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { REDUCE_CONCURRENCY, reduceField } from "@/lib/run/runOutputs";
import { unitFor, type FieldUnits } from "@/lib/science/fieldUnits";
import { mapPool } from "@/lib/core/concurrency";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  COMMON_LINE_LAYOUT,
  ControlGroup,
  PillGroup,
  type PlotComponentProps,
} from "./common";
import { PlotCompressionControls, RootHistoryScopeNotice } from "./PlotSourceDisclosure";
import { fieldsAcrossStepsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { historyFrameKey, historyUnitMismatch, newHistoryReductionCache, reconcileHistoryReductionCache } from "@/lib/plot/historyReductionCache";

/* Field reducer (max/min/mean/rms) vs time. All reducers pre-computed server-side per request. */

type TsReducer =
  | "max"
  | "min"
  | "mean"
  | "rms"
  | "countAbove"
  | "countBelow"
  | "countAbsAbove";

interface SeriesPoint {
  loaded: boolean;
  timeS: number | null;
  step: number;
  max: number;
  min: number;
  mean: number;
  rms: number;
  countAbove: number;
  countBelow: number;
  countAbsAbove: number;
  finite: number;
  total: number;
}

const REDUCERS: ReadonlyArray<{ value: TsReducer; label: string }> = [
  { value: "max", label: "max" },
  { value: "min", label: "min" },
  { value: "mean", label: "mean" },
  { value: "rms", label: "rms" },
  { value: "countAbove", label: "# >" },
  { value: "countBelow", label: "# <" },
  { value: "countAbsAbove", label: "# |·|>" },
];

/** Reducers that count cells against a threshold. A count is a cell tally, not
 * physical area; it drives the threshold input + re-fetch. */
const COUNT_REDUCERS: ReadonlySet<TsReducer> = new Set<TsReducer>([
  "countAbove",
  "countBelow",
  "countAbsAbove",
]);
const NATIVE_FIELD_ORIGINS = new Set(["native"] as const);

export function TimeSeriesPlot({ runDir, steps, stepIdx }: PlotComponentProps) {
  const allFields = useMemo(
    () => fieldsAcrossStepsForPlotRanks(runDir, steps, [2, 3], NATIVE_FIELD_ORIGINS),
    [runDir, steps],
  );
  const [fieldName, setFieldName] = useState<string | null>(allFields[0] ?? null);
  const [reducer, setReducer] = useState<TsReducer>("max");
  const [threshold, setThreshold] = useState(0.01);
  const isCount = COUNT_REDUCERS.has(reducer);
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [loadingPct, setLoadingPct] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reductionCacheRef = useRef(newHistoryReductionCache<SeriesPoint>());
  const fieldUnits = useMemo(
    () => steps.find((step) => fieldName != null && step.fields.includes(fieldName))?.fieldUnits,
    [steps, fieldName],
  );
  const thresholdUnit = fieldName ? unitFor(fieldName, fieldUnits) : null;

  // Reselect field if the current one disappears (e.g. opening a different run).
  useEffect(() => {
    if (fieldName && allFields.includes(fieldName)) return;
    setFieldName(allFields[0] ?? null);
  }, [allFields, fieldName]);

  // Load all steps when (runDir, fieldName, steps reference) changes.
  useEffect(() => {
    if (!fieldName || steps.length === 0) {
      reductionCacheRef.current = newHistoryReductionCache<SeriesPoint>();
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
      threshold,
    ]);
    if (reductionCacheRef.current.seriesKey !== seriesKey) {
      reductionCacheRef.current = newHistoryReductionCache<SeriesPoint>(seriesKey);
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
    // Indexed by step position so out-of-order completions still assemble into
    // a step-ordered series. Completed reductions survive append-only live
    // discovery; only newly published frames cross IPC on the next scan.
    const results = requests.map(({ key, step }) => cache.values.get(key) ? { ...cache.values.get(key)!, timeS: step.simulationTimeS } : {
      step: step.index, timeS: step.simulationTimeS, loaded: false,
      max: NaN, min: NaN, mean: NaN, rms: NaN,
      countAbove: NaN, countBelow: NaN, countAbsAbove: NaN,
      finite: 0, total: 0,
    });
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

    // Overlap the per-step reductions (each an off-main-thread Rust read) up to
    // REDUCE_CONCURRENCY at a time, instead of a strictly serial 90-frame wait,
    // while still drawing the line progressively as results land.
    void mapPool(
      outstanding.length,
      REDUCE_CONCURRENCY,
      async (i): Promise<SeriesPoint | null> => {
        const request = outstanding[i];
        let pending = cache.pending.get(request.key);
        if (!pending) {
          pending = reduceField(
            runDir,
            request.step.index,
            fieldName,
            request.step.publicationGeneration,
            threshold,
          ).then((r) => {
            const point = {
              loaded: true,
              timeS: request.step.simulationTimeS,
              step: request.step.index,
              max: r.max,
              min: r.min,
              mean: r.mean,
              rms: r.rms,
              countAbove: r.finite > 0 ? r.countAbove : NaN,
              countBelow: r.finite > 0 ? r.countBelow : NaN,
              countAbsAbove: r.finite > 0 ? r.countAbsAbove : NaN,
              finite: r.finite, total: r.total,
            } satisfies SeriesPoint;
            if (cache.activeKeys.has(request.key)) cache.values.set(request.key, point);
            cache.pending.delete(request.key);
            return point;
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
        if (res.value) results[outstanding[i].index] = { ...res.value, timeS: outstanding[i].step.simulationTimeS };
        settled += 1;
        // A fast native reducer can complete many frames in one display frame.
        // Coalesce those completions so React/Plotly never redraws thousands of
        // intermediate arrays that the screen could not display.
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
  }, [runDir, steps, fieldName, threshold]);

  const plot = useMemo(
    () => buildPlot(series, reducer, fieldName, threshold, fieldUnits),
    [series, reducer, fieldName, threshold, fieldUnits],
  );

  return (
    <>
      <RootHistoryScopeNotice runDir={runDir} step={steps[stepIdx] ?? null} />
      <div className="plot-controls">
        <ControlGroup label="Field">
          <select
            aria-label="Field"
            value={fieldName ?? ""}
            onChange={(e) => setFieldName(e.target.value || null)}
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

        {isCount && (
          <ControlGroup label="Threshold">
            <input
              type="number"
              step="any"
              value={threshold}
              onChange={(e) => {
                const t = Number(e.target.value);
                if (Number.isFinite(t)) setThreshold(t);
              }}
              title="Count cells passing this threshold each step (changing it re-loads the series)"
            />
            {thresholdUnit && (
              <span className="plot-controls__status">{thresholdUnit}</span>
            )}
          </ControlGroup>
        )}

        <ControlGroup label="Frames" stretch>
          <span className="plot-controls__status">
            {series.filter((point) => point.loaded).length} / {steps.length} loaded
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

/** y-axis / legend label for a reducer. Count reducers read as
 *  "count(field > t)" so the threshold is explicit on the plot. The field's
 *  physical unit is appended only for value-preserving reducers (max/min/
 *  mean/rms) — a count is a dimensionless cell tally, not a value in the
 *  field's units. */
function reducerLabel(
  reducer: TsReducer,
  fieldName: string,
  threshold: number,
  fieldUnits: FieldUnits,
): string {
  const thresholdUnit = unitFor(fieldName, fieldUnits);
  const thresholdLabel = `${threshold}${thresholdUnit ? ` ${thresholdUnit}` : ""}`;
  switch (reducer) {
    case "countAbove":
      return `count(${fieldName} > ${thresholdLabel})`;
    case "countBelow":
      return `count(${fieldName} < ${thresholdLabel})`;
    case "countAbsAbove":
      return `count(|${fieldName}| > ${thresholdLabel})`;
    default: {
      const unit = unitFor(fieldName, fieldUnits);
      return `${reducer}(${fieldName})${unit ? ` (${unit})` : ""}`;
    }
  }
}

function buildPlot(
  series: SeriesPoint[],
  reducer: TsReducer,
  fieldName: string | null,
  threshold: number,
  fieldUnits: FieldUnits,
): BuiltPlot | null {
  if (series.length === 0 || !fieldName) return null;
  const label = reducerLabel(reducer, fieldName, threshold, fieldUnits);

  // Use simulation time only if every loaded frame has it; otherwise the x-axis
  // would silently interleave seconds with step indices (and could go
  // non-monotone). Fall back to the step index for the whole series.
  const haveAllTimes = series.every(
    (s) => s.timeS != null && Number.isFinite(s.timeS),
  );
  const xs = series.map((s) => (haveAllTimes ? (s.timeS as number) : s.step));
  const ys = series.map((s) => Number.isFinite(s[reducer]) ? s[reducer] : null);
  const xUnit = haveAllTimes ? "t (s)" : "step";
  const xHover = haveAllTimes ? "t=%{x:.1f}s" : "step %{x}";

  const data: Data[] = [
    {
      type: "scatter",
      connectgaps: false,
      mode: "lines+markers",
      x: xs,
      y: ys,
      customdata: series.map(point => [point.finite, point.total]),
      name: label,
      line: { color: "#4a8cff", width: 1.8 },
      marker: { size: 4, color: "#4a8cff" },
      hovertemplate: `${xHover}<br>${label}=%{y:.4g}<br>finite cells %{customdata[0]} / %{customdata[1]}<extra></extra>`,
    },
  ];

  const layout: Partial<Layout> = {
    ...COMMON_LINE_LAYOUT,
    // One series is already identified by both the title and y-axis. Reserve
    // legend space for plots where it distinguishes multiple quantities.
    showlegend: false,
    xaxis: { title: { text: xUnit } },
    yaxis: { title: { text: label }, uirevision: JSON.stringify([fieldName, reducer, threshold, label]) },
    title: {
      text: `${label} vs ${haveAllTimes ? "t" : "step"}`,
      font: { size: 13 },
    },
  };

  return { data, layout };
}
