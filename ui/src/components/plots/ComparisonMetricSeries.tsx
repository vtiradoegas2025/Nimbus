import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import {
  emptyPairMetricMoments,
  finishPairMetricMoments,
  formatComparisonMetric,
  mergePairMetricMoments,
  matchedPhysicalVolumes,
  scanPairMetricMoments,
  type PairMetricMoments,
  type PairMetrics,
} from "@/lib/comparison/comparisonMetrics";
import {
  buildIncrementalComparisonSeries,
  comparisonSeriesMatchKey,
  OrderedComparisonSeriesCache,
} from "@/lib/comparison/comparisonSeriesCache";
import type { DatasetTimeMatch } from "@/lib/dataset/datasetAlignment";
import { fieldDefinitionIdentity, readFieldAnyWindows, registeredFieldRank, type StepInfo } from "@/lib/run/runOutputs";
import {
  fieldSupportsBoundedWindows,
  planExactValidDomainWindows,
} from "@/lib/plot/fieldLoadPlanning";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";

interface MetricFrame extends PairMetrics {
  simulationTimeS: number;
}

interface CachedMetricFrame extends MetricFrame {
  moments: PairMetricMoments;
}

type SeriesState = { identity: string } & (
  | { status: "loading"; rows: MetricFrame[]; completed: number; total: number; error: null }
  | { status: "ready"; rows: MetricFrame[]; completed: number; total: number; error: null }
  | { status: "error"; rows: MetricFrame[]; completed: number; total: number; error: string });

export function ComparisonMetricSeries({
  leftRunDir,
  leftDatasetIdentity,
  leftSteps,
  rightRunDir,
  rightDatasetIdentity,
  rightSteps,
  matches,
  fieldName,
  unit,
}: {
  leftRunDir: string;
  leftDatasetIdentity: string;
  leftSteps: StepInfo[];
  rightRunDir: string;
  rightDatasetIdentity: string;
  rightSteps: StepInfo[];
  matches: readonly DatasetTimeMatch[];
  fieldName: string;
  unit: string | null;
}) {
  const cache = useRef(new OrderedComparisonSeriesCache<CachedMetricFrame>());
  const [state, setState] = useState<SeriesState>({
    identity: "",
    status: "loading",
    rows: [],
    completed: 0,
    total: matches.length,
    error: null,
  });

  const plannedRequests = useMemo(() => {
    try {
    const requests = matches.map((match) => {
      const leftStep = leftSteps[match.leftFrameIndex];
      const rightStep = rightSteps[match.rightFrameIndex];
      if (!leftStep || !rightStep) throw new Error(`Matched time ${match.simulationTimeS} s no longer resolves to both datasets.`);
      if (!fieldSupportsBoundedWindows(leftStep.fieldOrigins?.[fieldName])
          || !fieldSupportsBoundedWindows(rightStep.fieldOrigins?.[fieldName])) {
        throw new Error("A matched comparison field cannot serve bounded windows.");
      }
      const leftRank = registeredFieldRank(
        leftRunDir, leftStep.index, fieldName, leftStep.publicationGeneration,
      );
      const rightRank = registeredFieldRank(
        rightRunDir, rightStep.index, fieldName, rightStep.publicationGeneration,
      );
      if ((leftRank !== 2 && leftRank !== 3) || rightRank !== leftRank) {
        throw new Error(`Matched time ${match.simulationTimeS} s changed rank after alignment.`);
      }
      const leftPlan = planExactValidDomainWindows(leftStep.grid, leftRank);
      const rightPlan = planExactValidDomainWindows(rightStep.grid, rightRank);
      if (!leftPlan.ok || !rightPlan.ok
          || leftPlan.plan.shape.join("x") !== rightPlan.plan.shape.join("x")
          || leftPlan.plan.windows.length !== rightPlan.plan.windows.length) {
        throw new Error(`Matched time ${match.simulationTimeS} s changed shape after alignment.`);
      }
      return {
        key: comparisonSeriesMatchKey({
          left: {
            datasetIdentity: leftDatasetIdentity,
            runDir: leftRunDir,
            publicationGeneration: leftStep.publicationGeneration,
            stepIndex: leftStep.index,
            fieldDefinitions: [fieldName].map(field => fieldDefinitionIdentity(leftRunDir, leftStep.index, field, leftStep.publicationGeneration)),
            fieldUnits: [fieldName].map(field => leftStep.fieldUnits?.[field] ?? null),
          },
          right: {
            datasetIdentity: rightDatasetIdentity,
            runDir: rightRunDir,
            publicationGeneration: rightStep.publicationGeneration,
            stepIndex: rightStep.index,
            fieldDefinitions: [fieldName].map(field => fieldDefinitionIdentity(rightRunDir, rightStep.index, field, rightStep.publicationGeneration)),
            fieldUnits: [fieldName].map(field => rightStep.fieldUnits?.[field] ?? null),
          },
          simulationTimeS: match.simulationTimeS,
          analysis: {
            kind: "scalar",
            field: fieldName,
            rank: leftRank,
            leftGeometry: leftPlan.plan,
            rightGeometry: rightPlan.plan,
          },
        }),
        request: { match, leftStep, rightStep, leftPlan: leftPlan.plan, rightPlan: rightPlan.plan },
      };
    });
      return { requests, error: null as string | null };
    } catch (reason) {
      return { requests: [], error: reason instanceof Error ? reason.message : String(reason) };
    }
  }, [fieldName, leftDatasetIdentity, leftRunDir, leftSteps, matches,
    rightDatasetIdentity, rightRunDir, rightSteps]);
  const requestIdentity = useMemo(() => JSON.stringify([fieldName, unit,
    plannedRequests.requests.map(({ key }) => key), plannedRequests.error]),
  [fieldName, unit, plannedRequests]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    void (async () => {
      if (plannedRequests.error) throw new Error(plannedRequests.error);
      const requests = plannedRequests.requests;
      const cachedRows = await buildIncrementalComparisonSeries({
        cache: cache.current,
        requests,
        cancelled: () => cancelled,
        onProgress: (rows) => setState({
          identity: requestIdentity,
          status: rows.length === matches.length ? "ready" : "loading",
          rows: [...rows],
          completed: rows.length,
          total: matches.length,
          error: null,
        }),
        load: async ({ match, leftStep, rightStep, leftPlan, rightPlan }) => {
          const moments = emptyPairMetricMoments();
          for (let tile = 0; tile < leftPlan.windows.length; tile += 1) {
            if (cancelled) throw new DOMException("Comparison series cancelled", "AbortError");
            const [left, right] = await Promise.all([
              readFieldAnyWindows(
                leftRunDir, leftStep.index, fieldName, leftStep.publicationGeneration,
                [leftPlan.windows[tile]], { signal: controller.signal },
              ),
              readFieldAnyWindows(
                rightRunDir, rightStep.index, fieldName, rightStep.publicationGeneration,
                [rightPlan.windows[tile]], { signal: controller.signal },
              ),
            ]);
            if (left.shape.join("x") !== right.shape.join("x")) {
              throw new Error(`Matched time ${match.simulationTimeS} s changed tile geometry.`);
            }
            mergePairMetricMoments(
              moments,
              await scanPairMetricMoments(left.data, right.data, {
                cancelled: () => cancelled,
                weights: matchedPhysicalVolumes(left, right) ?? undefined,
              }),
            );
          }
          const metrics = finishPairMetricMoments(moments);
          return { ...metrics, moments, simulationTimeS: match.simulationTimeS };
        },
      });
      if (!cancelled) setState({
        identity: requestIdentity,
        status: "ready",
        rows: cachedRows,
        completed: cachedRows.length,
        total: matches.length,
        error: null,
      });
    })().catch((error: unknown) => {
      if (cancelled || (error instanceof DOMException && error.name === "AbortError")) return;
      setState((current) => ({
        identity: requestIdentity,
        status: "error",
        rows: current.identity === requestIdentity ? current.rows : [],
        completed: current.identity === requestIdentity ? current.completed : 0,
        total: matches.length,
        error: error instanceof Error ? error.message : String(error),
      }));
    });
    return () => { cancelled = true; controller.abort(); };
  }, [plannedRequests, requestIdentity, matches.length, leftRunDir, rightRunDir, fieldName]);

  const retainedPrefix = state.identity === requestIdentity ? state.rows
    : cache.current.matchingPrefix(plannedRequests.requests.map(({ key }) => key));
  const displayState: SeriesState = state.identity === requestIdentity ? state : {
    identity: requestIdentity, status: plannedRequests.error ? "error" : "loading", rows: retainedPrefix,
    completed: retainedPrefix.length, total: matches.length, error: plannedRequests.error,
  } as SeriesState;

  const plot = useMemo((): { data: Data[]; layout: Partial<Layout> } | null => {
    if (displayState.rows.length === 0) return null;
    const x = displayState.rows.map((row) => row.simulationTimeS);
    const trace = (name: string, values: number[], color: string, dash: "solid" | "dot" = "solid"): Data => ({
      type: "scatter",
      mode: "lines+markers",
      name,
      x,
      y: values,
      line: { color, width: name === "RMSE" ? 2.2 : 1.4, dash },
      marker: { size: 5 },
      hovertemplate: `t=%{x:.4g} s<br>${name}=%{y:.5g}${unit ? ` ${unit}` : ""}<extra></extra>`,
    } as Data);
    return {
      data: [
        trace("RMSE", displayState.rows.map((row) => row.rmse), "#d9784b"),
        trace("MAE", displayState.rows.map((row) => row.mae), "#c9a84b"),
        trace("Mean A−B", displayState.rows.map((row) => row.mean), "#6f9fbd"),
        trace("Max |A−B|", displayState.rows.map((row) => row.maxAbs), "#879099", "dot"),
      ],
      layout: {
        xaxis: { title: { text: "Physical time (s)" } },
        yaxis: { title: { text: unit ? `Difference (${unit})` : "Difference" }, zeroline: true },
        legend: { orientation: "h", x: 0, y: 1.18 },
        margin: { l: 64, r: 28, t: 48, b: 46 },
        hovermode: "x unified",
        uirevision: `comparison-series:${fieldName}`,
      },
    };
  }, [fieldName, displayState.rows, unit]);

  const summary = useMemo(() => {
    if (displayState.rows.length === 0) return null;
    const meanRmse = displayState.rows.reduce((sum, row) => sum + row.rmse, 0) / displayState.rows.length;
    const peak = displayState.rows.reduce((best, row) => row.rmse > best.rmse ? row : best);
    const meanBias = displayState.rows.reduce((sum, row) => sum + row.mean, 0) / displayState.rows.length;
    return { meanRmse, peak, meanBias };
  }, [displayState.rows]);

  return (
    <section className="comparison-series" aria-label="Matched-time comparison metrics">
      <header className="comparison-series__header">
        <div>
          <strong>Validation over matched time</strong>
          <span>Full valid grid · {fieldName}{unit ? ` · ${unit}` : ""}</span>
        </div>
        {displayState.status !== "error" && (
          <span className="comparison-series__progress" role="status">
            {displayState.status === "loading" && <Icon name="dot" size={10} />} {displayState.completed}/{displayState.total} frames
          </span>
        )}
        {summary && (
          <div className="comparison-series__summary">
            <span>Time-mean RMSE <b>{formatComparisonMetric(summary.meanRmse)}</b></span>
            <span>Mean bias <b>{formatComparisonMetric(summary.meanBias)}</b></span>
            <span>Peak RMSE <b>{formatComparisonMetric(summary.peak.rmse)}</b> at {summary.peak.simulationTimeS.toFixed(3)} s</span>
          </div>
        )}
      </header>
      {displayState.status === "error" && (
        <div className="comparison-series__error" role="alert"><Icon name="warning" size={11} />{displayState.error}</div>
      )}
      {plot && <PlotCanvas data={plot.data} layout={plot.layout} className="comparison-series__canvas" />}
    </section>
  );
}
