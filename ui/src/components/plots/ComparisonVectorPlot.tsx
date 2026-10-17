import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import {
  emptyVectorMetricMoments,
  finishVectorMetricMoments,
  matchedPhysicalVolumes,
  mergeVectorMetricMoments,
  scanVectorMetricMoments,
  type IndexRegion3D,
  type VectorPairMetrics,
} from "@/lib/comparison/comparisonMetrics";
import type { DatasetTimeMatch } from "@/lib/dataset/datasetAlignment";
import {
  axisCoords,
  LOCKED_ASPECT_AXIS,
  lockedAspectPartnerAxis,
} from "@/lib/science/gridCoords";
import { slice2D, type Npy3D, type NpyN, type SliceAxis } from "@/lib/platform/npy";
import { resolveColorscale } from "@/lib/plot/plotColormaps";
import { buildQuiver } from "@/lib/plot/quiver";
import { describeVectorSampling, vectorReferenceKey } from "@/lib/plot/vectorPresentation";
import { readFieldAnyWindows, type StepInfo } from "@/lib/run/runOutputs";
import {
  fieldSupportsBoundedWindows,
  planBoundedSliceWindows,
  planExactValidDomainWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { gridForWindowedField } from "@/lib/science/windowedGrid";
import { remapTerrainVerticalSlice } from "@/lib/science/terrainSliceRemap";
import { availableVectorPairs } from "@/lib/science/vectorComponents";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import { unitFor } from "@/lib/science/fieldUnits";
import { useFieldAny, useRetainedPlot } from "./common";
import { ComparisonVectorMetricSeries } from "./ComparisonVectorMetricSeries";

interface Plane {
  data: number[][];
  rows: number;
  cols: number;
  rowAxis: SliceAxis;
  colAxis: SliceAxis;
}

type VectorMetrics = Omit<VectorPairMetrics, "meanMagnitude">;

type VolumeState =
  | { status: "idle"; metrics: null; progress: 0; error: null }
  | { status: "loading"; metrics: null; progress: number; error: null }
  | { status: "ready"; metrics: VectorPairMetrics; progress: 1; error: null }
  | { status: "error"; metrics: null; progress: 0; error: string };

const IDLE_VOLUME: VolumeState = { status: "idle", metrics: null, progress: 0, error: null };

function plane(field: NpyN | null, axis: SliceAxis, index: number): Plane | null {
  if (!field || field.shape.length !== 3) return null;
  const volume = field as Npy3D;
  const sliced = slice2D(volume, axis, Math.min(index, volume.shape[axis] - 1));
  return {
    data: sliced.data,
    rows: sliced.rows,
    cols: sliced.cols,
    rowAxis: sliced.rowAxis as SliceAxis,
    colAxis: sliced.colAxis as SliceAxis,
  };
}

function vectorMagnitude(x: Plane, y: Plane): number[][] | null {
  if (x.rows !== y.rows || x.cols !== y.cols || x.rowAxis !== y.rowAxis || x.colAxis !== y.colAxis) return null;
  return x.data.map((row, rowIndex) => row.map((value, colIndex) => {
    const other = y.data[rowIndex][colIndex];
    return Number.isFinite(value) && Number.isFinite(other) ? Math.hypot(value, other) : Number.NaN;
  }));
}

function vectorDifference(
  leftX: Plane,
  leftY: Plane,
  rightX: Plane,
  rightY: Plane,
): { x: number[][]; y: number[][]; magnitude: number[][]; metrics: VectorMetrics } | null {
  if ([leftY, rightX, rightY].some((item) => item.rows !== leftX.rows || item.cols !== leftX.cols)) return null;
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  let squareSum = 0;
  let maxMagnitude = 0;
  const x: number[][] = [];
  const y: number[][] = [];
  const magnitude: number[][] = [];
  for (let rowIndex = 0; rowIndex < leftX.rows; rowIndex += 1) {
    const xRow: number[] = [];
    const yRow: number[] = [];
    const magnitudeRow: number[] = [];
    for (let colIndex = 0; colIndex < leftX.cols; colIndex += 1) {
      const values = [
        leftX.data[rowIndex][colIndex],
        leftY.data[rowIndex][colIndex],
        rightX.data[rowIndex][colIndex],
        rightY.data[rowIndex][colIndex],
      ];
      if (values.some((value) => !Number.isFinite(value))) {
        xRow.push(Number.NaN);
        yRow.push(Number.NaN);
        magnitudeRow.push(Number.NaN);
        continue;
      }
      const dx = values[0] - values[2];
      const dy = values[1] - values[3];
      const length = Math.hypot(dx, dy);
      xRow.push(dx);
      yRow.push(dy);
      magnitudeRow.push(length);
      count += 1;
      sumX += dx;
      sumY += dy;
      squareSum += length * length;
      maxMagnitude = Math.max(maxMagnitude, length);
    }
    x.push(xRow);
    y.push(yRow);
    magnitude.push(magnitudeRow);
  }
  if (count === 0) return null;
  return { x, y, magnitude, metrics: { count, meanX: sumX / count, meanY: sumY / count, rmse: Math.sqrt(squareSum / count), maxMagnitude } };
}

function maximum(matrix: number[][]): number {
  let value = 0;
  for (const row of matrix) for (const item of row) if (Number.isFinite(item)) value = Math.max(value, item);
  return value;
}

function valueLabel(value: number): string {
  if (value === 0) return "0";
  return Math.abs(value) < 0.001 || Math.abs(value) >= 10_000 ? value.toExponential(3) : value.toPrecision(4);
}

export function ComparisonVectorPlot({
  leftRunDir,
  leftDatasetIdentity,
  leftLabel,
  leftStep,
  rightRunDir,
  rightDatasetIdentity,
  rightLabel,
  rightStep,
  leftSteps,
  rightSteps,
  timeMatches,
  eligibleFields,
  sliceAxis,
  sliceIndex,
}: {
  leftRunDir: string;
  leftDatasetIdentity: string;
  leftLabel: string;
  leftStep: StepInfo | null;
  rightRunDir: string;
  rightDatasetIdentity: string;
  rightLabel: string;
  rightStep: StepInfo | null;
  leftSteps: StepInfo[];
  rightSteps: StepInfo[];
  timeMatches: readonly DatasetTimeMatch[];
  eligibleFields: readonly string[];
  sliceAxis: SliceAxis;
  sliceIndex: number;
}) {
  const pair = useMemo(
    () => availableVectorPairs(sliceAxis, [...eligibleFields])[0] ?? null,
    [eligibleFields, sliceAxis],
  );
  // Both components of a vector share a unit; `unitFor` also suppresses a dimensionless "1".
  const unit = pair ? unitFor(pair.xField, leftStep?.fieldUnits) : null;
  const terrainFollowing = leftStep?.grid?.terrainFollowing === true
    || rightStep?.grid?.terrainFollowing === true;
  const terrainSliceBlocked = false;
  const boundedSupported = Boolean(pair
    && fieldSupportsBoundedWindows(leftStep?.fieldOrigins?.[pair.xField])
    && fieldSupportsBoundedWindows(leftStep?.fieldOrigins?.[pair.yField])
    && fieldSupportsBoundedWindows(rightStep?.fieldOrigins?.[pair.xField])
    && fieldSupportsBoundedWindows(rightStep?.fieldOrigins?.[pair.yField]));
  const slicePlans = useMemo(() => {
    if (!pair || terrainSliceBlocked || !boundedSupported) return null;
    const left = planBoundedSliceWindows(leftStep?.grid, sliceAxis, sliceIndex);
    const right = planBoundedSliceWindows(rightStep?.grid, sliceAxis, sliceIndex);
    if (!left.ok || !right.ok || left.plan.sampledShape.join("x") !== right.plan.sampledShape.join("x")) {
      return null;
    }
    return {
      leftX: windowsForFieldAggregation(left.plan.windows, pair.xField),
      leftY: windowsForFieldAggregation(left.plan.windows, pair.yField),
      rightX: windowsForFieldAggregation(right.plan.windows, pair.xField),
      rightY: windowsForFieldAggregation(right.plan.windows, pair.yField),
    };
  }, [boundedSupported, leftStep?.grid, pair, rightStep?.grid, sliceAxis, sliceIndex, terrainSliceBlocked]);
  const loadEnabled = slicePlans != null && !terrainSliceBlocked;
  const leftX = useFieldAny(leftRunDir, loadEnabled ? leftStep?.index ?? null : null, pair?.xField ?? null, leftStep?.publicationGeneration ?? null, slicePlans?.leftX);
  const leftY = useFieldAny(leftRunDir, loadEnabled ? leftStep?.index ?? null : null, pair?.yField ?? null, leftStep?.publicationGeneration ?? null, slicePlans?.leftY);
  const rightX = useFieldAny(rightRunDir, loadEnabled ? rightStep?.index ?? null : null, pair?.xField ?? null, rightStep?.publicationGeneration ?? null, slicePlans?.rightX);
  const rightY = useFieldAny(rightRunDir, loadEnabled ? rightStep?.index ?? null : null, pair?.yField ?? null, rightStep?.publicationGeneration ?? null, slicePlans?.rightY);

  const fields = useMemo(() => {
    const raw = {
      leftX: leftX.field,
      leftY: leftY.field,
      rightX: rightX.field,
      rightY: rightY.field,
    };
    const leftGrid = raw.leftX ? gridForWindowedField(raw.leftX, leftStep?.grid) : null;
    const rightGrid = raw.rightX ? gridForWindowedField(raw.rightX, rightStep?.grid) : null;
    if (!raw.leftX || !raw.leftY || !raw.rightX || !raw.rightY) {
      return { ...raw, grid: leftGrid, remapped: false, error: null as string | null };
    }
    try {
      if (terrainFollowing) {
        matchedPhysicalVolumes(raw.leftX, raw.rightX);
        matchedPhysicalVolumes(raw.leftY, raw.rightY);
      }
      const lx = remapTerrainVerticalSlice(raw.leftX, leftGrid, sliceAxis);
      const ly = remapTerrainVerticalSlice(raw.leftY, leftGrid, sliceAxis);
      const rx = remapTerrainVerticalSlice(raw.rightX, rightGrid, sliceAxis);
      const ry = remapTerrainVerticalSlice(raw.rightY, rightGrid, sliceAxis);
      return {
        leftX: lx.field,
        leftY: ly.field,
        rightX: rx.field,
        rightY: ry.field,
        grid: lx.grid,
        remapped: lx.remapped || rx.remapped,
        error: null as string | null,
      };
    } catch (remapError) {
      return {
        leftX: null,
        leftY: null,
        rightX: null,
        rightY: null,
        grid: leftGrid,
        remapped: false,
        error: remapError instanceof Error ? remapError.message : String(remapError),
      };
    }
  }, [leftStep?.grid, leftX.field, leftY.field, rightStep?.grid, rightX.field, rightY.field, sliceAxis, terrainFollowing]);
  const sourcePlans = useMemo(() => {
    const left = planExactValidDomainWindows(leftStep?.grid, 3);
    const right = planExactValidDomainWindows(rightStep?.grid, 3);
    return left.ok && right.ok && left.plan.shape.join("x") === right.plan.shape.join("x")
      ? { left: left.plan, right: right.plan }
      : null;
  }, [leftStep?.grid, rightStep?.grid]);
  const volumeShape = sourcePlans
    ? sourcePlans.left.shape as [number, number, number]
    : null;
  const volumeShapeKey = volumeShape?.join("x") ?? "";
  const [metricScope, setMetricScope] = useState<"slice" | "volume">("slice");
  const [region, setRegion] = useState<IndexRegion3D>({ lower: [0, 0, 0], upper: [0, 0, 0] });
  const [volumeState, setVolumeState] = useState<VolumeState>(IDLE_VOLUME);
  const [showTimeSeries, setShowTimeSeries] = useState(false);

  useEffect(() => {
    if (!volumeShape) return;
    setRegion({ lower: [0, 0, 0], upper: [volumeShape[0] - 1, volumeShape[1] - 1, volumeShape[2] - 1] });
  }, [volumeShapeKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if ((metricScope === "slice" && terrainSliceBlocked)
        || !pair || !boundedSupported || !leftStep || !rightStep || !volumeShape) {
      setVolumeState(IDLE_VOLUME);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    setVolumeState({ status: "loading", metrics: null, progress: 0, error: null });
    const metricLower = metricScope === "volume"
      ? [...region.lower] as [number, number, number]
      : [0, 0, 0] as [number, number, number];
    const metricUpper = metricScope === "volume"
      ? [...region.upper] as [number, number, number]
      : [volumeShape[0] - 1, volumeShape[1] - 1, volumeShape[2] - 1] as [number, number, number];
    if (metricScope === "slice") {
      metricLower[sliceAxis] = Math.min(sliceIndex, volumeShape[sliceAxis] - 1);
      metricUpper[sliceAxis] = metricLower[sliceAxis];
    }
    const metricRegion: IndexRegion3D = { lower: metricLower, upper: metricUpper };
    void (async () => {
      const leftPlan = planExactValidDomainWindows(leftStep.grid, 3, metricRegion);
      const rightPlan = planExactValidDomainWindows(rightStep.grid, 3, metricRegion);
      if (!leftPlan.ok || !rightPlan.ok || leftPlan.plan.windows.length !== rightPlan.plan.windows.length) {
        throw new Error("The exact vector comparison region could not be tiled consistently.");
      }
      const moments = emptyVectorMetricMoments();
      for (let tile = 0; tile < leftPlan.plan.windows.length; tile += 1) {
        const [lx, ly, rx, ry] = await Promise.all([
          readFieldAnyWindows(leftRunDir, leftStep.index, pair.xField, leftStep.publicationGeneration, [leftPlan.plan.windows[tile]], { signal: controller.signal }),
          readFieldAnyWindows(leftRunDir, leftStep.index, pair.yField, leftStep.publicationGeneration, [leftPlan.plan.windows[tile]], { signal: controller.signal }),
          readFieldAnyWindows(rightRunDir, rightStep.index, pair.xField, rightStep.publicationGeneration, [rightPlan.plan.windows[tile]], { signal: controller.signal }),
          readFieldAnyWindows(rightRunDir, rightStep.index, pair.yField, rightStep.publicationGeneration, [rightPlan.plan.windows[tile]], { signal: controller.signal }),
        ]);
        mergeVectorMetricMoments(
          moments,
          await scanVectorMetricMoments(lx.data, ly.data, rx.data, ry.data, {
            cancelled: () => cancelled,
            weights: matchedPhysicalVolumes(lx, rx) ?? undefined,
          }),
        );
        if (!cancelled) setVolumeState({
          status: "loading", metrics: null,
          progress: (tile + 1) / leftPlan.plan.windows.length, error: null,
        });
      }
      return finishVectorMetricMoments(moments);
    })().then(
      (metrics) => { if (!cancelled) setVolumeState({ status: "ready", metrics, progress: 1, error: null }); },
      (error: unknown) => {
        if (!cancelled && !(error instanceof DOMException && error.name === "AbortError")) {
          setVolumeState({ status: "error", metrics: null, progress: 0, error: error instanceof Error ? error.message : String(error) });
        }
      },
    );
    return () => { cancelled = true; controller.abort(); };
  }, [boundedSupported, leftRunDir, leftStep, metricScope, pair, region, rightRunDir, rightStep, sliceAxis, sliceIndex, terrainFollowing, terrainSliceBlocked, volumeShape, volumeShapeKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const result = useMemo((): {
    data: Data[];
    layout: Partial<Layout>;
    metrics: VectorMetrics;
    samplingDisclosure: string | null;
  } | null => {
    if (!pair || terrainSliceBlocked) return null;
    const lx = plane(fields.leftX, sliceAxis, sliceIndex);
    const ly = plane(fields.leftY, sliceAxis, sliceIndex);
    const rx = plane(fields.rightX, sliceAxis, sliceIndex);
    const ry = plane(fields.rightY, sliceAxis, sliceIndex);
    if (!lx || !ly || !rx || !ry) return null;
    const leftMagnitude = vectorMagnitude(lx, ly);
    const rightMagnitude = vectorMagnitude(rx, ry);
    const difference = vectorDifference(lx, ly, rx, ry);
    if (!leftMagnitude || !rightMagnitude || !difference) return null;
    const xCoords = axisCoords(lx.colAxis, lx.cols, fields.grid);
    const yCoords = axisCoords(lx.rowAxis, lx.rows, fields.grid);
    const stride = Math.max(1, Math.ceil(Math.max(lx.rows, lx.cols) / 18));
    const sharedMaximum = Math.max(maximum(leftMagnitude), maximum(rightMagnitude), 1e-12);
    const differenceMaximum = Math.max(difference.metrics.maxMagnitude, 1e-12);
    const spacing = Math.min(xCoords.spacing, yCoords.spacing);
    const quiver = (x: number[][], y: number[][], referenceMagnitude: number) => buildQuiver({
      u: x,
      v: y,
      rows: lx.rows,
      cols: lx.cols,
      xCoords: xCoords.values,
      yCoords: yCoords.values,
      xSpacing: spacing,
      ySpacing: spacing,
      stride,
      scale: 0.72,
      referenceMagnitude,
    });
    const leftQuiver = quiver(lx.data, ly.data, sharedMaximum);
    const rightQuiver = quiver(rx.data, ry.data, sharedMaximum);
    const differenceQuiver = quiver(difference.x, difference.y, differenceMaximum);
    const samplingDisclosure = describeVectorSampling(fields.leftX, lx.rowAxis, lx.colAxis);
    const heatmap = (z: number[][], xaxis: "x" | "x2" | "x3", yaxis: "y" | "y2" | "y3", coloraxis: "coloraxis" | "coloraxis2", name: string): Data => ({
      type: "heatmap",
      // Magnitude cells and sampled arrows must refer to the same resolved
      // field; smoothing only the background would visually decouple them.
      zsmooth: false,
      z,
      x: xCoords.values,
      y: yCoords.values,
      xaxis,
      yaxis,
      coloraxis,
      name,
      hovertemplate: `${name}<br>x=%{x:.4g}<br>y=%{y:.4g}<br>|V|=%{z:.4g}<extra></extra>`,
    } as Data);
    const arrows = (built: ReturnType<typeof buildQuiver>, xaxis: "x" | "x2" | "x3", yaxis: "y" | "y2" | "y3", name: string, color: string): Data => ({
      type: "scatter",
      mode: "lines",
      x: built.x,
      y: built.y,
      xaxis,
      yaxis,
      name,
      line: { color, width: 1.15 },
      hoverinfo: "skip",
      showlegend: false,
    } as Data);
    const hoverTargets = (
      built: ReturnType<typeof buildQuiver>,
      xaxis: "x" | "x2" | "x3",
      yaxis: "y" | "y2" | "y3",
      name: string,
    ): Data => ({
      type: "scatter",
      mode: "markers",
      x: built.originX,
      y: built.originY,
      xaxis,
      yaxis,
      customdata: built.components,
      marker: { size: 9, color: "rgba(0,0,0,0)" },
      hovertemplate:
        `${name}<br>${pair.xField} (→): %{customdata[0]:.3g}${unit ? ` ${unit}` : ""}`
        + `<br>${pair.yField} (↑): %{customdata[1]:.3g}${unit ? ` ${unit}` : ""}`
        + `<br>|V|: %{customdata[2]:.3g}${unit ? ` ${unit}` : ""}<extra></extra>`,
      showlegend: false,
      name: `${name} values`,
    } as Data);
    const keyAnnotation = (
      built: ReturnType<typeof buildQuiver>,
      referenceMagnitude: number,
      xref: "x" | "x2" | "x3",
      yref: "y" | "y2" | "y3",
      color: string,
    ) => {
      const key = vectorReferenceKey(
        referenceMagnitude,
        spacing,
        built.effectiveStride,
        0.72,
        xCoords.values,
        yCoords.values,
      );
      return key ? {
        x: key.headX, y: key.y, ax: key.tailX, ay: key.y,
        xref, yref, axref: xref, ayref: yref,
        text: `${key.magnitude.toPrecision(3)}${unit ? ` ${unit}` : ""}`,
        showarrow: true, arrowhead: 3, arrowsize: 1, arrowwidth: 1.3, arrowcolor: color,
        font: { size: 9, color }, bgcolor: "rgba(0,0,0,0)", borderpad: 2,
      } : null;
    };
    const scaleKeys = [
      keyAnnotation(leftQuiver, sharedMaximum, "x", "y", "#eef1f3"),
      keyAnnotation(rightQuiver, sharedMaximum, "x2", "y2", "#eef1f3"),
      keyAnnotation(differenceQuiver, differenceMaximum, "x3", "y3", "#f2c078"),
    ].filter((annotation) => annotation != null);
    return {
      data: [
        heatmap(leftMagnitude, "x", "y", "coloraxis", `A |${pair.family.label}|`),
        arrows(leftQuiver, "x", "y", "A vectors", "#eef1f3"),
        hoverTargets(leftQuiver, "x", "y", "A vector"),
        heatmap(rightMagnitude, "x2", "y2", "coloraxis", `B |${pair.family.label}|`),
        arrows(rightQuiver, "x2", "y2", "B vectors", "#eef1f3"),
        hoverTargets(rightQuiver, "x2", "y2", "B vector"),
        heatmap(difference.magnitude, "x3", "y3", "coloraxis2", "|A−B vector|"),
        arrows(differenceQuiver, "x3", "y3", "A−B vectors", "#f2c078"),
        hoverTargets(differenceQuiver, "x3", "y3", "A−B vector"),
      ],
      layout: ({
        grid: { rows: 1, columns: 3, pattern: "independent", xgap: 0.09 },
        xaxis: { title: { text: xCoords.title }, ...LOCKED_ASPECT_AXIS },
        yaxis: { title: { text: yCoords.title }, ...lockedAspectPartnerAxis() },
        xaxis2: { title: { text: xCoords.title }, matches: "x", ...LOCKED_ASPECT_AXIS },
        yaxis2: { title: { text: yCoords.title }, matches: "y", ...lockedAspectPartnerAxis("x2") },
        xaxis3: { title: { text: xCoords.title }, matches: "x", ...LOCKED_ASPECT_AXIS },
        yaxis3: { title: { text: yCoords.title }, matches: "y", ...lockedAspectPartnerAxis("x3") },
        coloraxis: { colorscale: resolveColorscale("Viridis") as never, cmin: 0, cmax: sharedMaximum, colorbar: { title: { text: unit ? `|V| (${unit})` : "|V|", side: "right" }, x: 0.646, len: 0.8 } },
        coloraxis2: { colorscale: resolveColorscale("Inferno") as never, cmin: 0, cmax: differenceMaximum, colorbar: { title: { text: unit ? `|ΔV| (${unit})` : "|ΔV|", side: "right" }, x: 1.015, len: 0.8 } },
        annotations: [
          { text: `A · ${leftLabel}`, x: 0.145, y: 1.07, xref: "paper", yref: "paper", showarrow: false },
          { text: `B · ${rightLabel}`, x: 0.5, y: 1.07, xref: "paper", yref: "paper", showarrow: false },
          { text: "A−B vector", x: 0.855, y: 1.07, xref: "paper", yref: "paper", showarrow: false },
          ...scaleKeys,
        ],
        margin: { l: 55, r: 80, t: 45, b: 45 },
        dragmode: "zoom",
        uirevision: `comparison-vector:${pair.family.id}:${sliceAxis}`,
      } as unknown as Partial<Layout>),
      metrics: difference.metrics,
      samplingDisclosure,
    };
  }, [fields, leftLabel, pair, rightLabel, sliceAxis, sliceIndex, terrainSliceBlocked, unit]);

  // Four components load independently here, so `result` is null for the whole gap between the first
  // and last resolving — on every plane change and every frame.
  const { shown: shownResult } = useRetainedPlot(
    result,
    [leftRunDir, rightRunDir, pair?.xField ?? "", pair?.yField ?? "", sliceAxis].join(" "),
  );

  const loading = leftX.loading || leftY.loading || rightX.loading || rightY.loading;
  const error = leftX.error ?? leftY.error ?? rightX.error ?? rightY.error;
  const displayedMetrics = volumeState.metrics ?? null;
  const setRegionBound = (bound: "lower" | "upper", axis: number, value: number) => {
    if (!volumeShape) return;
    setRegion((current) => {
      const lower = [...current.lower] as [number, number, number];
      const upper = [...current.upper] as [number, number, number];
      const clamped = Math.max(0, Math.min(volumeShape[axis] - 1, Math.floor(value)));
      if (bound === "lower") lower[axis] = Math.min(clamped, upper[axis]);
      else upper[axis] = Math.max(clamped, lower[axis]);
      return { lower, upper };
    });
  };
  return (
    <section className="comparison-vector" aria-label="Synchronized vector comparison">
      <header className="comparison-vector__header">
        <div>
          <strong>{pair ? `${pair.family.label}: ${pair.xField} → · ${pair.yField} ↑` : "Vector comparison unavailable on this plane"}</strong>
          <span>
            Same physical slice and matched time · shared A/B arrow normalization · Δ arrows separately normalized
            {shownResult?.samplingDisclosure ? ` · ${shownResult.samplingDisclosure}` : ""}
            {unit ? ` · ${unit}` : ""}
          </span>
        </div>
        <div className="comparison-vector__scope" role="group" aria-label="Vector statistics scope">
          <button type="button" className={metricScope === "slice" ? "active" : ""} onClick={() => setMetricScope("slice")}>Slice</button>
          <button type="button" className={metricScope === "volume" ? "active" : ""} onClick={() => setMetricScope("volume")}>Volume region</button>
        </div>
        {pair && (
          <button type="button" className="comparison-vector__series-toggle" aria-pressed={showTimeSeries} onClick={() => setShowTimeSeries((visible) => !visible)}>
            {showTimeSeries ? "Hide vector series" : "Build vector series"}
          </button>
        )}
        {displayedMetrics && (
          <div className="comparison-vector__metrics">
            <span>Mean Δ→ <b>{valueLabel(displayedMetrics.meanX)}</b></span>
            <span>Mean Δ↑ <b>{valueLabel(displayedMetrics.meanY)}</b></span>
            {"meanMagnitude" in displayedMetrics && <span>Mean |ΔV| <b>{valueLabel((displayedMetrics as VectorPairMetrics).meanMagnitude)}</b></span>}
            <span>Vector RMSE <b>{valueLabel(displayedMetrics.rmse)}</b></span>
            <span>Max |ΔV| <b>{valueLabel(displayedMetrics.maxMagnitude)}</b></span>
            <span>Pairs <b>{displayedMetrics.count.toLocaleString()}</b></span>
          </div>
        )}
        {volumeState.status === "loading" && (
          <span className="comparison-vector__scan" role="status">Exact {metricScope} scan · {Math.round(volumeState.progress * 100)}%</span>
        )}
        {loading && <span className="comparison-vector__scan" role="status">Loading matched vector components…</span>}
        {volumeState.status === "error" && (
          <span className="comparison-vector__scan comparison-vector__scan--error" role="alert">{volumeState.error}</span>
        )}
      </header>
      <div className="comparison-vector__main">
        {volumeShape && (
          <details className="comparison-vector__region" open={metricScope === "volume"}>
            <summary>Volumetric selection · inclusive scalar-cell indices</summary>
            <div>
              {(["x", "y", "z"] as const).map((axisName, axis) => (
                <label key={axisName}>
                  <span>{axisName}</span>
                  <input aria-label={`${axisName} lower index`} type="number" min={0} max={volumeShape[axis] - 1} value={region.lower[axis]} onChange={(event) => setRegionBound("lower", axis, Number(event.target.value))} />
                  <i>to</i>
                  <input aria-label={`${axisName} upper index`} type="number" min={0} max={volumeShape[axis] - 1} value={region.upper[axis]} onChange={(event) => setRegionBound("upper", axis, Number(event.target.value))} />
                </label>
              ))}
            </div>
          </details>
        )}
        <div className="comparison-vector__body plot-notice-layout">
        {fields.error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{fields.error}</span></div>}
        {fields.remapped && <div className="plot-overlay"><Icon name="dot" size={12} /><span>physical-volume terrain→Cartesian vector remap</span></div>}
        {!pair && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>This slice needs both qualified in-plane components in A and B.</span></div>}
        {pair && !terrainSliceBlocked && !boundedSupported && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>A vector comparison component cannot serve bounded windows.</span></div>}
        {error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{error}</span></div>}
        {shownResult && !error && <PlotCanvas data={shownResult.data} layout={shownResult.layout} className="comparison-vector__canvas" />}
        </div>
        {showTimeSeries && pair && (
          <ComparisonVectorMetricSeries
            leftRunDir={leftRunDir}
            leftDatasetIdentity={leftDatasetIdentity}
            leftSteps={leftSteps}
            rightRunDir={rightRunDir}
            rightDatasetIdentity={rightDatasetIdentity}
            rightSteps={rightSteps}
            matches={timeMatches}
            xField={pair.xField}
            yField={pair.yField}
            region={region}
            unit={unit}
          />
        )}
      </div>
    </section>
  );
}
