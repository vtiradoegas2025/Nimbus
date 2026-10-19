import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import {
  emptyVectorMetricMoments,
  finishVectorMetricMoments,
  formatComparisonMetric,
  mergeVectorMetricMoments,
  matchedPhysicalVolumes,
  scanVectorMetricMoments,
  type IndexRegion3D,
  type VectorMetricMoments,
  type VectorPairMetrics,
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

interface VectorMetricFrame extends VectorPairMetrics {
  simulationTimeS: number;
}

interface CachedVectorMetricFrame extends VectorMetricFrame {
  moments: VectorMetricMoments;
}

export function ComparisonVectorMetricSeries({
  leftRunDir,
  leftDatasetIdentity,
  leftSteps,
  rightRunDir,
  rightDatasetIdentity,
  rightSteps,
  matches,
  xField,
  yField,
  region,
  unit,
}: {
  leftRunDir: string;
  leftDatasetIdentity: string;
  leftSteps: StepInfo[];
  rightRunDir: string;
  rightDatasetIdentity: string;
  rightSteps: StepInfo[];
  matches: readonly DatasetTimeMatch[];
  xField: string;
  yField: string;
  region: IndexRegion3D;
  unit: string | null;
}) {
  const cache = useRef(new OrderedComparisonSeriesCache<CachedVectorMetricFrame>());
  const [state, setState] = useState({ identity: "", rows: [] as VectorMetricFrame[], progress: 0, error: null as string | null });

  const plannedRequests = useMemo(() => {
    try {
    const requests = matches.map((match) => {
      const leftStep = leftSteps[match.leftFrameIndex];
      const rightStep = rightSteps[match.rightFrameIndex];
      if (!leftStep || !rightStep) throw new Error(`Matched vector frame at ${match.simulationTimeS} s is unavailable.`);
      const fields = [xField, yField] as const;
      if (fields.some((field) => !fieldSupportsBoundedWindows(leftStep.fieldOrigins?.[field])
        || !fieldSupportsBoundedWindows(rightStep.fieldOrigins?.[field]))) {
        throw new Error("A matched vector component cannot serve bounded windows.");
      }
      if (fields.some((field) =>
        registeredFieldRank(leftRunDir, leftStep.index, field, leftStep.publicationGeneration) !== 3
        || registeredFieldRank(rightRunDir, rightStep.index, field, rightStep.publicationGeneration) !== 3
      )) throw new Error(`Vector component rank changed at ${match.simulationTimeS} s.`);
      const leftPlan = planExactValidDomainWindows(leftStep.grid, 3, region);
      const rightPlan = planExactValidDomainWindows(rightStep.grid, 3, region);
      if (!leftPlan.ok || !rightPlan.ok
          || leftPlan.plan.shape.join("x") !== rightPlan.plan.shape.join("x")
          || leftPlan.plan.windows.length !== rightPlan.plan.windows.length) {
        throw new Error(`Vector region geometry changed at ${match.simulationTimeS} s.`);
      }
      return {
        key: comparisonSeriesMatchKey({
          left: {
            datasetIdentity: leftDatasetIdentity,
            runDir: leftRunDir,
            publicationGeneration: leftStep.publicationGeneration,
            stepIndex: leftStep.index,
            fieldDefinitions: [xField, yField].map(field => fieldDefinitionIdentity(leftRunDir, leftStep.index, field, leftStep.publicationGeneration)),
            fieldUnits: [xField, yField].map(field => leftStep.fieldUnits?.[field] ?? null),
          },
          right: {
            datasetIdentity: rightDatasetIdentity,
            runDir: rightRunDir,
            publicationGeneration: rightStep.publicationGeneration,
            stepIndex: rightStep.index,
            fieldDefinitions: [xField, yField].map(field => fieldDefinitionIdentity(rightRunDir, rightStep.index, field, rightStep.publicationGeneration)),
            fieldUnits: [xField, yField].map(field => rightStep.fieldUnits?.[field] ?? null),
          },
          simulationTimeS: match.simulationTimeS,
          analysis: {
            kind: "vector",
            components: [xField, yField],
            region,
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
  }, [leftDatasetIdentity, leftRunDir, leftSteps, matches, region,
    rightDatasetIdentity, rightRunDir, rightSteps, xField, yField]);
  const requestIdentity = useMemo(() => JSON.stringify([xField, yField, unit,
    plannedRequests.requests.map(({ key }) => key), plannedRequests.error]),
  [xField, yField, unit, plannedRequests]);

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
        onProgress: (next) => {
          setState({ identity: requestIdentity, rows: [...next],
            progress: matches.length === 0 ? 1 : next.length / matches.length, error: null });
        },
        load: async ({ match, leftStep, rightStep, leftPlan, rightPlan }) => {
          const moments = emptyVectorMetricMoments();
          for (let tile = 0; tile < leftPlan.windows.length; tile += 1) {
            if (cancelled) throw new DOMException("Vector comparison series cancelled", "AbortError");
            const [lx, ly, rx, ry] = await Promise.all([
              readFieldAnyWindows(leftRunDir, leftStep.index, xField, leftStep.publicationGeneration,
                [leftPlan.windows[tile]], { signal: controller.signal }),
              readFieldAnyWindows(leftRunDir, leftStep.index, yField, leftStep.publicationGeneration,
                [leftPlan.windows[tile]], { signal: controller.signal }),
              readFieldAnyWindows(rightRunDir, rightStep.index, xField, rightStep.publicationGeneration,
                [rightPlan.windows[tile]], { signal: controller.signal }),
              readFieldAnyWindows(rightRunDir, rightStep.index, yField, rightStep.publicationGeneration,
                [rightPlan.windows[tile]], { signal: controller.signal }),
            ]);
            const shape = lx.shape.join("x");
            if ([ly, rx, ry].some((field) => field.shape.join("x") !== shape)) {
              throw new Error(`Vector tile geometry changed at ${match.simulationTimeS} s.`);
            }
            mergeVectorMetricMoments(
              moments,
              await scanVectorMetricMoments(lx.data, ly.data, rx.data, ry.data, {
                cancelled: () => cancelled,
                weights: matchedPhysicalVolumes(lx, rx) ?? undefined,
              }),
            );
          }
          const metrics = finishVectorMetricMoments(moments);
          return { ...metrics, moments, simulationTimeS: match.simulationTimeS };
        },
      });
      if (!cancelled) {
        setState({ identity: requestIdentity, rows: cachedRows, progress: 1, error: null });
      }
    })().catch((reason: unknown) => {
      if (!cancelled && !(reason instanceof DOMException && reason.name === "AbortError")) {
        setState(current => ({ identity: requestIdentity,
          rows: current.identity === requestIdentity ? current.rows : [],
          progress: current.identity === requestIdentity ? current.progress : 0,
          error: reason instanceof Error ? reason.message : String(reason) }));
      }
    });
    return () => { cancelled = true; controller.abort(); };
  }, [plannedRequests, requestIdentity, matches.length, leftRunDir, rightRunDir, xField, yField]);

  const displayState = state.identity === requestIdentity ? state : {
    identity: requestIdentity,
    rows: cache.current.matchingPrefix(plannedRequests.requests.map(({ key }) => key)),
    progress: 0, error: plannedRequests.error,
  };
  const { rows, progress, error } = displayState;

  const plot = useMemo((): { data: Data[]; layout: Partial<Layout> } | null => {
    if (rows.length === 0) return null;
    const x = rows.map((row) => row.simulationTimeS);
    const trace = (name: string, values: number[], color: string, dash: "solid" | "dot" = "solid"): Data => ({
      type: "scatter",
      mode: "lines+markers",
      name,
      x,
      y: values,
      line: { color, width: name === "Vector RMSE" ? 2.2 : 1.35, dash },
      marker: { size: 4 },
      hovertemplate: `t=%{x:.4g} s<br>${name}=%{y:.5g}${unit ? ` ${unit}` : ""}<extra></extra>`,
    } as Data);
    return {
      data: [
        trace("Vector RMSE", rows.map((row) => row.rmse), "#d9784b"),
        trace("Mean |ΔV|", rows.map((row) => row.meanMagnitude), "#c9a84b"),
        trace("Mean Δ→", rows.map((row) => row.meanX), "#6f9fbd"),
        trace("Mean Δ↑", rows.map((row) => row.meanY), "#80a47a"),
        trace("Max |ΔV|", rows.map((row) => row.maxMagnitude), "#879099", "dot"),
      ],
      layout: {
        xaxis: { title: { text: "Physical time (s)" } },
        yaxis: { title: { text: unit ? `Vector difference (${unit})` : "Vector difference" }, zeroline: true },
        legend: { orientation: "h", x: 0, y: 1.2 },
        margin: { l: 64, r: 26, t: 50, b: 44 },
        hovermode: "x unified",
        uirevision: `vector-metric-series:${xField}:${yField}`,
      },
    };
  }, [rows, unit, xField, yField]);

  const summary = useMemo(() => {
    if (rows.length === 0) return null;
    const meanRmse = rows.reduce((sum, row) => sum + row.rmse, 0) / rows.length;
    const peak = rows.reduce((best, row) => row.rmse > best.rmse ? row : best);
    return { meanRmse, peak };
  }, [rows]);

  return (
    <section className="comparison-vector-series" aria-label="Matched-time vector metrics">
      <header>
        <strong>Vector validation over matched time</strong>
        <span>{xField} → · {yField} ↑ · selected volume</span>
        {!error && <em role="status">{rows.length}/{matches.length} frames · {Math.round(progress * 100)}%</em>}
        {summary && <p>Time-mean vector RMSE <b>{formatComparisonMetric(summary.meanRmse)}</b> · peak <b>{formatComparisonMetric(summary.peak.rmse)}</b> at {summary.peak.simulationTimeS.toFixed(3)} s</p>}
        {error && <p className="comparison-vector-series__error" role="alert"><Icon name="warning" size={10} />{error}</p>}
      </header>
      {plot && <PlotCanvas data={plot.data} layout={plot.layout} className="comparison-vector-series__canvas" />}
    </section>
  );
}
