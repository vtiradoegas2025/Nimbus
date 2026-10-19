import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import {
  emptyPairMetricMoments,
  finishPairMetricMoments,
  mergePairMetricMoments,
  matchedPhysicalVolumes,
  scanPairMetricMoments,
  volumeMeanPairSample,
  type IndexRegion3D,
  type PairMetrics,
  type VolumeDisplaySample,
} from "@/lib/comparison/comparisonMetrics";
import { axisCoords } from "@/lib/science/gridCoords";
import { labelWithUnit, unitFor } from "@/lib/science/fieldUnits";
import type { NpyN } from "@/lib/platform/npy";
import { resolveColorscale } from "@/lib/plot/plotColormaps";
import { cameraSignature } from "@/lib/plot/viewSync";
import type { StepInfo } from "@/lib/run/runOutputs";
import { readFieldAnyWindows } from "@/lib/run/runOutputs";
import {
  fieldSupportsBoundedWindows,
  planBoundedVolumeWindows,
  planExactValidDomainWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { gridForWindowedField } from "@/lib/science/windowedGrid";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import { useFieldAny } from "./common";

// Three Plotly volume traces share this budget. 100k conservative block means
// per panel retains materially more storm structure than the former 20k target
// while exact comparison metrics continue to use every selected source cell.
const TARGET_VOLUME_SAMPLES = 100_000;

function sampleVolumes(
  left: NpyN,
  right: NpyN,
  region: IndexRegion3D,
  grid: StepInfo["grid"],
): VolumeDisplaySample | null {
  if (left.shape.length !== 3 || right.shape.join("x") !== left.shape.join("x")) return null;
  const terrainCentroids = left.windowGeometry?.centroidsM;
  if (terrainCentroids) {
    matchedPhysicalVolumes(left, right);
    const sample: VolumeDisplaySample = {
      x: [], y: [], z: [], left: [], right: [], difference: [], stride: 1,
      selectedCells: left.data.length,
    };
    for (let index = 0; index < left.data.length; index++) {
      const a = left.data[index];
      const b = right.data[index];
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      sample.x.push(terrainCentroids.x[index] / 1000);
      sample.y.push(terrainCentroids.y[index] / 1000);
      sample.z.push(terrainCentroids.z[index] / 1000);
      sample.left.push(a);
      sample.right.push(b);
      sample.difference.push(a - b);
    }
    return sample;
  }
  const shape = left.shape as [number, number, number];
  const counts = region.upper.map((upper, axis) => upper - region.lower[axis] + 1) as [number, number, number];
  const selectedCells = counts[0] * counts[1] * counts[2];
  const stride = Math.max(1, Math.ceil(Math.cbrt(selectedCells / TARGET_VOLUME_SAMPLES)));
  const coordinates = ([0, 1, 2] as const).map((axis) =>
    axisCoords(axis, shape[axis], grid ?? null).values ?? Array.from({ length: shape[axis] }, (_, index) => index)
  );
  return volumeMeanPairSample(
    left.data, right.data, shape, region, stride,
    coordinates as [number[], number[], number[]],
  );
}

function range(values: readonly number[], symmetric = false): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (!Number.isFinite(min)) return [0, 1];
  if (symmetric) {
    const magnitude = Math.max(Math.abs(min), Math.abs(max), 1e-12);
    return [-magnitude, magnitude];
  }
  if (min === max) return [min - 0.5, max + 0.5];
  return [min, max];
}

function metric(value: number): string {
  if (value === 0) return "0";
  return Math.abs(value) < 0.001 || Math.abs(value) >= 10_000 ? value.toExponential(3) : value.toPrecision(4);
}

export function ComparisonVolumePlot({
  leftRunDir,
  leftLabel,
  leftStep,
  rightRunDir,
  rightLabel,
  rightStep,
  fieldName,
}: {
  leftRunDir: string;
  leftLabel: string;
  leftStep: StepInfo | null;
  rightRunDir: string;
  rightLabel: string;
  rightStep: StepInfo | null;
  fieldName: string;
}) {
  const terrainBlocked = false;
  const sourcePlan = useMemo(() => {
    const left = planExactValidDomainWindows(leftStep?.grid, 3);
    const right = planExactValidDomainWindows(rightStep?.grid, 3);
    if (!left.ok || !right.ok || left.plan.shape.join("x") !== right.plan.shape.join("x")) {
      return null;
    }
    return { left: left.plan, right: right.plan };
  }, [leftStep?.grid, rightStep?.grid]);
  const shape = sourcePlan ? sourcePlan.left.shape as [number, number, number] : null;
  const shapeKey = shape?.join("x") ?? "";
  const [region, setRegion] = useState<IndexRegion3D>({ lower: [0, 0, 0], upper: [0, 0, 0] });
  const [metrics, setMetrics] = useState<PairMetrics | null>(null);
  const [metricError, setMetricError] = useState<string | null>(null);
  const [camera, setCamera] = useState<unknown>(null);
  const [cameraRevision, setCameraRevision] = useState(0);
  /** Signature of the camera last broadcast, so the echo can be recognised. */
  const cameraSignatureRef = useRef("");

  useEffect(() => {
    if (!shape) return;
    setRegion({ lower: [0, 0, 0], upper: [shape[0] - 1, shape[1] - 1, shape[2] - 1] });
  }, [shapeKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const regionReady = Boolean(shape && region.lower.every((lower, axis) =>
    lower >= 0 && region.upper[axis] >= lower && region.upper[axis] < shape[axis]
  ));
  const boundedSupported = fieldSupportsBoundedWindows(leftStep?.fieldOrigins?.[fieldName])
    && fieldSupportsBoundedWindows(rightStep?.fieldOrigins?.[fieldName]);
  const displayPlans = useMemo(() => {
    if (!regionReady || terrainBlocked || !boundedSupported || !leftStep?.grid || !rightStep?.grid) {
      return null;
    }
    const withRegion = (step: StepInfo): StepInfo["grid"] => {
      const base = step.grid?.validIndexRange?.lower ?? [0, 0, 0];
      return step.grid ? {
        ...step.grid,
        validIndexRange: {
          lower: base.map((value, axis) => value + region.lower[axis]) as [number, number, number],
          upper: base.map((value, axis) => value + region.upper[axis]) as [number, number, number],
        },
      } : null;
    };
    const left = planBoundedVolumeWindows(withRegion(leftStep), TARGET_VOLUME_SAMPLES);
    const right = planBoundedVolumeWindows(withRegion(rightStep), TARGET_VOLUME_SAMPLES);
    if (!left.ok || !right.ok || left.plan.sampledShape.join("x") !== right.plan.sampledShape.join("x")) {
      return null;
    }
    return {
      left: windowsForFieldAggregation(left.plan.windows, fieldName),
      right: windowsForFieldAggregation(right.plan.windows, fieldName),
    };
  }, [boundedSupported, fieldName, leftStep, region, regionReady, rightStep, terrainBlocked]);
  const loadEnabled = displayPlans != null && !terrainBlocked;
  const leftLoad = useFieldAny(
    leftRunDir, loadEnabled ? leftStep?.index ?? null : null, fieldName,
    leftStep?.publicationGeneration ?? null, displayPlans?.left,
  );
  const rightLoad = useFieldAny(
    rightRunDir, loadEnabled ? rightStep?.index ?? null : null, fieldName,
    rightStep?.publicationGeneration ?? null, displayPlans?.right,
  );
  const left = leftLoad.field;
  const right = rightLoad.field;
  const leftGrid = useMemo(
    () => left ? gridForWindowedField(left, leftStep?.grid) : null,
    [left, leftStep?.grid],
  );

  useEffect(() => {
    if (!regionReady || terrainBlocked || !boundedSupported || !leftStep || !rightStep) {
      setMetrics(null);
      setMetricError(null);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    setMetrics(null);
    setMetricError(null);
    void (async () => {
      const leftPlan = planExactValidDomainWindows(leftStep.grid, 3, region);
      const rightPlan = planExactValidDomainWindows(rightStep.grid, 3, region);
      if (!leftPlan.ok || !rightPlan.ok || leftPlan.plan.windows.length !== rightPlan.plan.windows.length) {
        throw new Error("The selected exact comparison region could not be tiled consistently.");
      }
      const moments = emptyPairMetricMoments();
      for (let tile = 0; tile < leftPlan.plan.windows.length; tile += 1) {
        const [leftTile, rightTile] = await Promise.all([
          readFieldAnyWindows(leftRunDir, leftStep.index, fieldName, leftStep.publicationGeneration, [leftPlan.plan.windows[tile]], { signal: controller.signal }),
          readFieldAnyWindows(rightRunDir, rightStep.index, fieldName, rightStep.publicationGeneration, [rightPlan.plan.windows[tile]], { signal: controller.signal }),
        ]);
        mergePairMetricMoments(moments, await scanPairMetricMoments(
          leftTile.data, rightTile.data, {
            cancelled: () => cancelled,
            weights: matchedPhysicalVolumes(leftTile, rightTile) ?? undefined,
          },
        ));
      }
      return finishPairMetricMoments(moments);
    })().then(
      (value) => { if (!cancelled) setMetrics(value); },
      (reason: unknown) => {
        if (!cancelled && !(reason instanceof DOMException && reason.name === "AbortError")) setMetricError(reason instanceof Error ? reason.message : String(reason));
      },
    );
    return () => { cancelled = true; controller.abort(); };
  }, [boundedSupported, fieldName, leftRunDir, leftStep, region, regionReady, rightRunDir, rightStep, terrainBlocked]);

  const sample = useMemo(() => {
    if (!left || !right || left.shape.length !== 3 || terrainBlocked) return null;
    const sampledShape = left.shape as [number, number, number];
    return sampleVolumes(left, right, {
      lower: [0, 0, 0],
      upper: [sampledShape[0] - 1, sampledShape[1] - 1, sampledShape[2] - 1],
    }, leftGrid);
  }, [left, leftGrid, right, terrainBlocked]);
  const selectedCells = regionReady
    ? region.upper.reduce((cells, upper, axis) => cells * (upper - region.lower[axis] + 1), 1)
    : 0;
  const fieldUnit = unitFor(fieldName, leftStep?.fieldUnits);
  const plot = useMemo((): { data: Data[]; layout: Partial<Layout> } | null => {
    if (!sample) return null;
    const shared = range([...sample.left, ...sample.right]);
    const difference = range(sample.difference, true);
    const trace = (values: number[], scene: "scene" | "scene2" | "scene3", name: string, bounds: [number, number], colorscale: "Viridis" | "RdBu", showscale: boolean): Data => ({
      type: "volume",
      x: sample.x,
      y: sample.y,
      z: sample.z,
      value: values,
      scene,
      name,
      isomin: bounds[0],
      isomax: bounds[1],
      opacity: 0.13,
      surface: { count: 13 },
      colorscale: resolveColorscale(colorscale) as never,
      showscale,
      colorbar: showscale
        ? {
          // A difference carries the same unit as the quantity differenced.
          title: {
            text: scene === "scene3"
              ? (fieldUnit ? `A−B (${fieldUnit})` : "A−B")
              : labelWithUnit(fieldName, leftStep?.fieldUnits),
            side: "right",
          },
          len: 0.72,
        }
        : undefined,
      hovertemplate: `${name}<br>x=%{x:.3g}<br>y=%{y:.3g}<br>z=%{z:.3g}<br>value=%{value:.4g}<extra></extra>`,
    } as unknown as Data);
    const scene = (domain: [number, number]) => ({
      domain: { x: domain, y: [0, 1] },
      aspectmode: "data",
      ...(camera ? { camera } : {}),
      xaxis: { title: { text: "x (km)" } },
      yaxis: { title: { text: "y (km)" } },
      zaxis: { title: { text: "z (km)" } },
    });
    return {
      data: [
        trace(sample.left, "scene", `A · ${leftLabel}`, shared, "Viridis", false),
        trace(sample.right, "scene2", `B · ${rightLabel}`, shared, "Viridis", true),
        trace(sample.difference, "scene3", "A−B volume", difference, "RdBu", true),
      ],
      layout: ({
        scene: scene([0, 0.3]),
        scene2: scene([0.35, 0.65]),
        scene3: scene([0.7, 1]),
        annotations: [
          { text: `A · ${leftLabel}`, x: 0.15, y: 1.05, xref: "paper", yref: "paper", showarrow: false },
          { text: `B · ${rightLabel}`, x: 0.5, y: 1.05, xref: "paper", yref: "paper", showarrow: false },
          { text: "A−B volume", x: 0.85, y: 1.05, xref: "paper", yref: "paper", showarrow: false },
        ],
        margin: { l: 8, r: 68, t: 42, b: 8 },
        // Bump revision on each camera broadcast — constant uirevision lets orbited scenes refuse the broadcast.
        uirevision: `comparison-volume:${fieldName}:${cameraRevision}`,
      } as unknown as Partial<Layout>),
    };
  }, [camera, cameraRevision, fieldName, fieldUnit, leftLabel, leftStep?.fieldUnits, rightLabel, sample]);

  const setBound = (bound: "lower" | "upper", axis: number, value: number) => {
    if (!shape) return;
    setRegion((current) => {
      const lower = [...current.lower] as [number, number, number];
      const upper = [...current.upper] as [number, number, number];
      const clamped = Math.max(0, Math.min(shape[axis] - 1, Math.floor(value)));
      if (bound === "lower") lower[axis] = Math.min(clamped, upper[axis]);
      else upper[axis] = Math.max(clamped, lower[axis]);
      return { lower, upper };
    });
  };
  const loading = leftLoad.loading || rightLoad.loading;
  const error = leftLoad.error ?? rightLoad.error ?? metricError;
  return (
    <section className="comparison-volume" aria-label="Synchronized three-dimensional comparison">
      <header>
        <div>
          <strong>Volumetric A/B/A−B · {fieldName}</strong>
          <span>Exact region metrics · bounded display sampling · synchronized cameras</span>
        </div>
        {sample && <span>{sample.left.length.toLocaleString()} volume-mean blocks from {selectedCells.toLocaleString()} selected cells</span>}
        {metrics && <div><span>Mean A−B <b>{metric(metrics.mean)}</b></span><span>RMSE <b>{metric(metrics.rmse)}</b></span><span>Max |A−B| <b>{metric(metrics.maxAbs)}</b></span><span>Pairs <b>{metrics.count.toLocaleString()}</b></span></div>}
      </header>
      {shape && (
        <div className="comparison-volume__region">
          {(["x", "y", "z"] as const).map((axisName, axis) => (
            <label key={axisName}><span>{axisName}</span><input aria-label={`3D ${axisName} lower index`} type="number" min={0} max={shape[axis] - 1} value={region.lower[axis]} onChange={(event) => setBound("lower", axis, Number(event.target.value))} /><i>to</i><input aria-label={`3D ${axisName} upper index`} type="number" min={0} max={shape[axis] - 1} value={region.upper[axis]} onChange={(event) => setBound("upper", axis, Number(event.target.value))} /></label>
          ))}
        </div>
      )}
      <div className="comparison-volume__body">
        {left && left.shape.length !== 3 && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{fieldName} is not a qualified 3-D field.</span></div>}
        {!terrainBlocked && !boundedSupported && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>The comparison field cannot serve bounded windows.</span></div>}
        {loading && <div className="plot-overlay"><Icon name="dot" size={12} /><span>Loading matched volumes…</span></div>}
        {error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{error}</span></div>}
        {plot && !error && <PlotCanvas data={plot.data} layout={plot.layout} className="comparison-volume__canvas" onRelayout={(event) => {
          // Take the camera from whichever scene emitted one (search all three; ?? would drop batched updates).
          const sceneKey = (["scene.camera", "scene2.camera", "scene3.camera"] as const)
            .find((key) => key in event);
          const nextCamera = sceneKey ? event[sceneKey] : undefined;
          if (!nextCamera) return;
          // Guard echo: without this, re-applying the broadcast triggers a relayout on each scene
          // -> re-entry -> unbounded re-render storm across three WebGL volume traces.
          const key = cameraSignature(nextCamera);
          if (!key || key === cameraSignatureRef.current) return;
          cameraSignatureRef.current = key;
          setCamera(nextCamera);
          setCameraRevision((revision) => revision + 1);
        }} />}
      </div>
    </section>
  );
}
