import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import type { DatasetTimeMatch } from "@/lib/dataset/datasetAlignment";
import {
  COMPARISON_WORKSPACE_SCHEMA,
  COMPARISON_WORKSPACE_VERSION,
  loadComparisonWorkspace,
  saveComparisonWorkspace,
  type ComparisonWorkspaceRecipe,
} from "@/lib/comparison/comparisonWorkspace";
import { axesAreCoLocated, axisCoords, axisLabel } from "@/lib/science/gridCoords";
import { unitFor } from "@/lib/science/fieldUnits";
import { slice2D, type Npy3D, type NpyN, type SliceAxis } from "@/lib/platform/npy";
import { canCompareByCell } from "@/lib/run/frameMotion";
import { DBZ_RANGE, resolveColorscale, usesFixedAnchorRange, type ExtendedColorscale } from "@/lib/plot/plotColormaps";
import { readFieldAnyWindows, registeredFieldRank, type StepInfo } from "@/lib/run/runOutputs";
import {
  fieldSupportsBoundedWindows,
  describeBoundedSampling,
  PLOTLY_HEATMAP_CELL_CAP,
  planBoundedPlaneWindows,
  planBoundedSliceWindows,
  planExactValidDomainWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import {
  emptyPairMetricMoments,
  finishPairMetricMoments,
  mergePairMetricMoments,
  matchedPhysicalVolumes,
  scanPairMetricMoments,
  type IndexRegion3D,
} from "@/lib/comparison/comparisonMetrics";
import { gridForWindowedField } from "@/lib/science/windowedGrid";
import { fieldsWithSharedPlotRank } from "@/lib/plot/fieldPlotEligibility";
import { remapTerrainVerticalSlice } from "@/lib/science/terrainSliceRemap";
import { availableVectorPairs } from "@/lib/science/vectorComponents";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import { ComparisonMetricSeries } from "./ComparisonMetricSeries";
import { ComparisonVectorPlot } from "./ComparisonVectorPlot";
import { ComparisonVolumePlot } from "./ComparisonVolumePlot";
import { ComparisonCompositePlot } from "./ComparisonCompositePlot";
import {
  ControlGroup,
  PillGroup,
  useDebouncedValue,
  useFieldAny,
  type FrameDisplayReport,
} from "./common";
import "./ComparisonSlicePlot.css";

type SharedRangeMode = "auto" | "symmetric" | "dBZ";
type MetricScope = "slice" | "region" | "volume";

const RANGE_OPTIONS: ReadonlyArray<{ value: SharedRangeMode; label: string }> = [
  { value: "auto", label: "Shared auto" },
  { value: "symmetric", label: "Shared ±" },
  { value: "dBZ", label: "dBZ" },
];

const AXIS_OPTIONS: ReadonlyArray<{ value: SliceAxis; label: string }> = [
  { value: 0, label: "x" },
  { value: 1, label: "y" },
  { value: 2, label: "z" },
];

const METRIC_SCOPE_OPTIONS: ReadonlyArray<{ value: MetricScope; label: string }> = [
  { value: "slice", label: "Visible slice" },
  { value: "region", label: "Zoomed region" },
  { value: "volume", label: "Full grid" },
];

interface ViewportBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function defaultRange(field: string): SharedRangeMode {
  if (field.includes("reflectivity") || field === "radar") return "dBZ";
  if (field === "w" || field.includes("prime") || field.includes("vorticity") || field === "buoyancy") return "symmetric";
  return "auto";
}

function defaultColormap(field: string): ExtendedColorscale {
  if (field.includes("reflectivity") || field === "radar") return "dBZ";
  if (defaultRange(field) === "symmetric") return "RdBu";
  return "Viridis";
}

interface Heatmap2D {
  data: number[][];
  rowAxis: SliceAxis;
  colAxis: SliceAxis;
}

interface PairMetrics {
  count: number;
  mean: number;
  mae: number;
  rmse: number;
  maxAbs: number;
}

interface SliceDifference extends PairMetrics {
  heatmap: Heatmap2D;
}

type VolumeMetricState =
  | { status: "idle"; metrics: null; progress: 0; error: null }
  | { status: "loading"; metrics: null; progress: number; error: null }
  | { status: "ready"; metrics: PairMetrics; progress: 1; error: null }
  | { status: "error"; metrics: null; progress: 0; error: string };

const IDLE_VOLUME_METRICS: VolumeMetricState = {
  status: "idle",
  metrics: null,
  progress: 0,
  error: null,
};

function heatmap2D(field: NpyN, axis: SliceAxis, index: number): Heatmap2D | null {
  if (field.shape.length === 3) {
    const volume = field as Npy3D;
    const safe = Math.max(0, Math.min(volume.shape[axis] - 1, index));
    const sliced = slice2D(volume, axis, safe);
    return { data: sliced.data, rowAxis: sliced.rowAxis as SliceAxis, colAxis: sliced.colAxis as SliceAxis };
  }
  if (field.shape.length !== 2) return null;
  const [nx, ny] = field.shape;
  const data = Array.from({ length: nx }, (_, x) =>
    Array.from({ length: ny }, (_, y) => field.data[x * ny + y])
  );
  return { data, rowAxis: 0, colAxis: 1 };
}

function combinedRange(
  left: Heatmap2D,
  right: Heatmap2D,
  mode: SharedRangeMode,
  /** When true, colormap anchors are at physical values — use fixed DBZ_RANGE, not fitted. */
  fixedAnchors = false,
): [number, number] {
  if (mode === "dBZ" || fixedAnchors) return [DBZ_RANGE[0], DBZ_RANGE[1]];
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let finiteCount = 0;
  for (const heatmap of [left, right]) {
    for (const row of heatmap.data) {
      for (const value of row) {
        if (!Number.isFinite(value)) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
        finiteCount += 1;
      }
    }
  }
  if (finiteCount === 0) return [0, 1];
  if (mode === "symmetric") {
    const magnitude = Math.max(Math.abs(min), Math.abs(max), 1e-12);
    return [-magnitude, magnitude];
  }
  if (Math.abs(max - min) <= 1e-12) {
    const pad = Math.max(0.5, Math.abs(min) * 1e-3);
    min -= pad;
    max += pad;
  }
  return [min, max];
}

function differenceSlice(left: Heatmap2D, right: Heatmap2D): SliceDifference | null {
  if (
    left.rowAxis !== right.rowAxis
    || left.colAxis !== right.colAxis
    || left.data.length !== right.data.length
  ) return null;

  let count = 0;
  let sum = 0;
  let absoluteSum = 0;
  let squareSum = 0;
  let maxAbs = 0;
  const data: number[][] = [];
  for (let rowIndex = 0; rowIndex < left.data.length; rowIndex += 1) {
    const leftRow = left.data[rowIndex];
    const rightRow = right.data[rowIndex];
    if (leftRow.length !== rightRow.length) return null;
    const row = new Array<number>(leftRow.length);
    for (let colIndex = 0; colIndex < leftRow.length; colIndex += 1) {
      const leftValue = leftRow[colIndex];
      const rightValue = rightRow[colIndex];
      if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) {
        row[colIndex] = Number.NaN;
        continue;
      }
      const difference = leftValue - rightValue;
      const absolute = Math.abs(difference);
      row[colIndex] = difference;
      count += 1;
      sum += difference;
      absoluteSum += absolute;
      squareSum += difference * difference;
      maxAbs = Math.max(maxAbs, absolute);
    }
    data.push(row);
  }
  if (count === 0) return null;
  return {
    heatmap: { data, rowAxis: left.rowAxis, colAxis: left.colAxis },
    count,
    mean: sum / count,
    mae: absoluteSum / count,
    rmse: Math.sqrt(squareSum / count),
    maxAbs,
  };
}

function regionMetrics(
  difference: SliceDifference,
  x: readonly number[],
  y: readonly number[],
  bounds: ViewportBounds | null,
): PairMetrics | null {
  if (!bounds) return difference;
  let count = 0;
  let sum = 0;
  let absoluteSum = 0;
  let squareSum = 0;
  let maxAbs = 0;
  for (let rowIndex = 0; rowIndex < difference.heatmap.data.length; rowIndex += 1) {
    const yValue = y[rowIndex];
    if (yValue == null || yValue < bounds.yMin || yValue > bounds.yMax) continue;
    const row = difference.heatmap.data[rowIndex];
    for (let colIndex = 0; colIndex < row.length; colIndex += 1) {
      const xValue = x[colIndex];
      const value = row[colIndex];
      if (xValue == null || xValue < bounds.xMin || xValue > bounds.xMax || !Number.isFinite(value)) continue;
      const absolute = Math.abs(value);
      count += 1;
      sum += value;
      absoluteSum += absolute;
      squareSum += value * value;
      maxAbs = Math.max(maxAbs, absolute);
    }
  }
  if (count === 0) return null;
  return { count, mean: sum / count, mae: absoluteSum / count, rmse: Math.sqrt(squareSum / count), maxAbs };
}

function relayoutRange(event: Readonly<Record<string, unknown>>, axisNames: readonly string[]): [number, number] | null | undefined {
  for (const axis of axisNames) {
    if (event[`${axis}.autorange`] === true) return null;
    const combined = event[`${axis}.range`];
    if (Array.isArray(combined) && combined.length === 2 && combined.every((value) => Number.isFinite(Number(value)))) {
      const values = combined.map(Number).sort((left, right) => left - right);
      return [values[0], values[1]];
    }
    const first = Number(event[`${axis}.range[0]`]);
    const second = Number(event[`${axis}.range[1]`]);
    if (Number.isFinite(first) && Number.isFinite(second)) return first <= second ? [first, second] : [second, first];
  }
  return undefined;
}

function metric(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  return magnitude >= 10_000 || magnitude < 0.001
    ? value.toExponential(3)
    : value.toPrecision(4);
}

function useExactMetrics({
  leftRunDir,
  leftStep,
  rightRunDir,
  rightStep,
  fieldName,
  rank,
  region,
  enabled,
}: {
  leftRunDir: string;
  leftStep: StepInfo | null;
  rightRunDir: string;
  rightStep: StepInfo | null;
  fieldName: string | null;
  rank: 2 | 3 | null;
  region?: IndexRegion3D;
  enabled: boolean;
}): VolumeMetricState {
  const [state, setState] = useState<VolumeMetricState>(IDLE_VOLUME_METRICS);

  useEffect(() => {
    if (!enabled) {
      setState(IDLE_VOLUME_METRICS);
      return;
    }
    if (!leftStep || !rightStep || !fieldName || !rank) {
      setState(IDLE_VOLUME_METRICS);
      return;
    }
    if (!fieldSupportsBoundedWindows(leftStep.fieldOrigins?.[fieldName])
        || !fieldSupportsBoundedWindows(rightStep.fieldOrigins?.[fieldName])) {
      setState({ status: "error", metrics: null, progress: 0, error: "Exact metrics require a bounded field provider." });
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    setState({ status: "loading", metrics: null, progress: 0, error: null });
    void (async () => {
      const leftPlan = planExactValidDomainWindows(leftStep.grid, rank, region);
      const rightPlan = planExactValidDomainWindows(rightStep.grid, rank, region);
      if (!leftPlan.ok || !rightPlan.ok
          || leftPlan.plan.shape.join("x") !== rightPlan.plan.shape.join("x")
          || leftPlan.plan.windows.length !== rightPlan.plan.windows.length) {
        throw new Error("Matched exact-metric regions changed shape after alignment.");
      }
      const moments = emptyPairMetricMoments();
      for (let tile = 0; tile < leftPlan.plan.windows.length; tile += 1) {
        const [left, right] = await Promise.all([
          readFieldAnyWindows(leftRunDir, leftStep.index, fieldName, leftStep.publicationGeneration, [leftPlan.plan.windows[tile]], { signal: controller.signal }),
          readFieldAnyWindows(rightRunDir, rightStep.index, fieldName, rightStep.publicationGeneration, [rightPlan.plan.windows[tile]], { signal: controller.signal }),
        ]);
        mergePairMetricMoments(
          moments,
          await scanPairMetricMoments(left.data, right.data, {
            cancelled: () => cancelled,
            weights: matchedPhysicalVolumes(left, right) ?? undefined,
          }),
        );
        if (!cancelled) setState({
          status: "loading", metrics: null,
          progress: (tile + 1) / leftPlan.plan.windows.length, error: null,
        });
      }
      return finishPairMetricMoments(moments);
    })().then(
      (metrics) => { if (!cancelled) setState({ status: "ready", metrics, progress: 1, error: null }); },
      (error: unknown) => {
        if (!cancelled && !(error instanceof DOMException && error.name === "AbortError")) {
          setState({ status: "error", metrics: null, progress: 0, error: error instanceof Error ? error.message : String(error) });
        }
      },
    );
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [enabled, fieldName, leftRunDir, leftStep, rank, region, rightRunDir, rightStep]);

  return state;
}

export function ComparisonSlicePlot({
  leftRunDir,
  leftDatasetIdentity,
  leftLabel,
  leftSteps,
  leftStepIdx,
  rightRunDir,
  rightDatasetIdentity,
  rightLabel,
  rightSteps,
  rightStepIdx,
  eligibleFields,
  timeMatches,
  initialField,
  onFrameDisplayChange,
}: {
  leftRunDir: string;
  leftDatasetIdentity: string;
  leftLabel: string;
  leftSteps: StepInfo[];
  leftStepIdx: number;
  rightRunDir: string;
  rightDatasetIdentity: string;
  rightLabel: string;
  rightSteps: StepInfo[];
  rightStepIdx: number | null;
  eligibleFields: readonly string[];
  timeMatches: readonly DatasetTimeMatch[];
  initialField?: string | null;
  onFrameDisplayChange?: (report: FrameDisplayReport) => void;
}) {
  const leftStep = leftSteps[leftStepIdx] ?? null;
  const rightStep = rightStepIdx == null ? null : rightSteps[rightStepIdx] ?? null;
  // Alignment by name/unit is necessary but not sufficient for a spatial
  // comparison. Only advertise fields whose adapters report the same exact
  // plottable rank on both sides of the currently matched frame.
  const plotEligibleFields = useMemo(() => fieldsWithSharedPlotRank(
    eligibleFields,
    [{ runDir: leftRunDir, step: leftStep }, { runDir: rightRunDir, step: rightStep }],
    [2, 3],
  ), [eligibleFields, leftRunDir, leftStep, rightRunDir, rightStep]);
  const preferred = initialField && plotEligibleFields.includes(initialField)
    ? initialField
    : plotEligibleFields.includes("reflectivity_dbz")
      ? "reflectivity_dbz"
      : plotEligibleFields[0] ?? null;
  const [fieldName, setFieldName] = useState<string | null>(preferred);
  const [sliceAxis, setSliceAxis] = useState<SliceAxis>(2);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [rangeMode, setRangeMode] = useState<SharedRangeMode>(preferred ? defaultRange(preferred) : "auto");
  const [colormap, setColormap] = useState<ExtendedColorscale>(preferred ? defaultColormap(preferred) : "Viridis");
  const [metricScope, setMetricScope] = useState<MetricScope>("slice");
  const [viewportBounds, setViewportBounds] = useState<ViewportBounds | null>(null);
  const sourceViewport = useDebouncedValue(viewportBounds, 180);
  const [showMetricSeries, setShowMetricSeries] = useState(false);
  const [showVectorComparison, setShowVectorComparison] = useState(false);
  const [showVolumeComparison, setShowVolumeComparison] = useState(false);
  const [showCompositeComparison, setShowCompositeComparison] = useState(false);
  const [workspaceStatus, setWorkspaceStatus] = useState<"pending" | "restored" | "saved" | "unavailable">("pending");
  const [renderedLeftIndex, setRenderedLeftIndex] = useState<number | null>(null);
  const workspaceReadyRef = useRef(false);
  const pendingWorkspaceSaveRef = useRef<ComparisonWorkspaceRecipe | null>(null);

  const flushPendingWorkspaceSave = useCallback((updateStatus: boolean) => {
    const pending = pendingWorkspaceSaveRef.current;
    if (!pending || typeof window === "undefined") return;
    pendingWorkspaceSaveRef.current = null;
    const saved = saveComparisonWorkspace(window.localStorage, pending);
    if (updateStatus) setWorkspaceStatus(saved ? "saved" : "unavailable");
  }, []);

  // Preserve the last debounced edit when the comparison pair changes or the
  // pane closes; routine slider changes still coalesce into one durable write.
  useEffect(() => () => flushPendingWorkspaceSave(false), [flushPendingWorkspaceSave, leftRunDir, rightRunDir]);
  const restoredFieldRef = useRef<string | null>(null);
  const eligibleFieldKey = plotEligibleFields.join("\u0000");
  const leftRank = fieldName && leftStep
    ? registeredFieldRank(leftRunDir, leftStep.index, fieldName, leftStep.publicationGeneration)
    : null;
  const rightRank = fieldName && rightStep
    ? registeredFieldRank(rightRunDir, rightStep.index, fieldName, rightStep.publicationGeneration)
    : null;
  const matchedRank = (leftRank === 2 || leftRank === 3) && rightRank === leftRank
    ? leftRank
    : null;
  const terrainFollowing = matchedRank === 3 && (leftStep?.grid?.terrainFollowing === true
    || rightStep?.grid?.terrainFollowing === true);
  const terrainBlocked = false;
  const boundedSupported = Boolean(fieldName
    && fieldSupportsBoundedWindows(leftStep?.fieldOrigins?.[fieldName])
    && fieldSupportsBoundedWindows(rightStep?.fieldOrigins?.[fieldName]));
  const displayPlans = useMemo(() => {
    if (!fieldName || !matchedRank || terrainBlocked || !boundedSupported) return null;
    const plan = (step: StepInfo | null) => matchedRank === 3
      ? planBoundedSliceWindows(
          step?.grid, sliceAxis, sliceIndex, PLOTLY_HEATMAP_CELL_CAP, sourceViewport,
        )
      : planBoundedPlaneWindows(step?.grid, PLOTLY_HEATMAP_CELL_CAP, sourceViewport);
    const left = plan(leftStep);
    const right = plan(rightStep);
    if (!left.ok || !right.ok || left.plan.sampledShape.join("x") !== right.plan.sampledShape.join("x")) {
      return null;
    }
    return {
      left: windowsForFieldAggregation(left.plan.windows, fieldName),
      right: windowsForFieldAggregation(right.plan.windows, fieldName),
      samplingDisclosure: sourceViewport
        ? `viewport ${describeBoundedSampling(left.plan, fieldName) ?? "native cells"}`
        : describeBoundedSampling(left.plan, fieldName),
    };
  }, [boundedSupported, fieldName, leftStep, matchedRank, rightStep, sliceAxis, sliceIndex, sourceViewport, terrainBlocked]);
  const loadEnabled = displayPlans != null;
  const leftResult = useFieldAny(
    leftRunDir, loadEnabled ? leftStep?.index ?? null : null, fieldName,
    leftStep?.publicationGeneration ?? null, displayPlans?.left,
    JSON.stringify([leftRunDir, leftStep?.publicationGeneration, fieldName, sliceAxis]),
  );
  const rightResult = useFieldAny(
    rightRunDir, loadEnabled ? rightStep?.index ?? null : null, fieldName,
    rightStep?.publicationGeneration ?? null, displayPlans?.right,
    JSON.stringify([rightRunDir, rightStep?.publicationGeneration, fieldName, sliceAxis]),
  );

  useEffect(() => {
    workspaceReadyRef.current = false;
    setWorkspaceStatus("pending");
    const recipe = loadComparisonWorkspace(window.localStorage, leftRunDir, rightRunDir);
    if (recipe && plotEligibleFields.includes(recipe.fieldName)) {
      restoredFieldRef.current = recipe.fieldName;
      setFieldName(recipe.fieldName);
      setSliceAxis(recipe.sliceAxis);
      setSliceIndex(recipe.sliceIndex);
      setRangeMode(recipe.rangeMode);
      setColormap(recipe.colormap);
      setMetricScope(recipe.metricScope);
      setShowMetricSeries(recipe.showMetricSeries);
      setShowVectorComparison(recipe.showVectorComparison);
      setShowVolumeComparison(recipe.showVolumeComparison);
      setShowCompositeComparison(recipe.showCompositeComparison);
      setViewportBounds(recipe.viewportBounds);
      setWorkspaceStatus("restored");
    }
    queueMicrotask(() => { workspaceReadyRef.current = true; });
  }, [eligibleFieldKey, leftRunDir, rightRunDir]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!workspaceReadyRef.current || !fieldName) return;
    setWorkspaceStatus("pending");
    pendingWorkspaceSaveRef.current = {
      schema: COMPARISON_WORKSPACE_SCHEMA,
      schemaVersion: COMPARISON_WORKSPACE_VERSION,
      leftIdentity: leftRunDir,
      rightIdentity: rightRunDir,
      fieldName,
      sliceAxis,
      sliceIndex,
      rangeMode,
      colormap,
      metricScope,
      showMetricSeries,
      showVectorComparison,
      showVolumeComparison,
      showCompositeComparison,
      viewportBounds,
      savedAt: new Date().toISOString(),
    };
    const timer = window.setTimeout(() => {
      flushPendingWorkspaceSave(true);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [colormap, fieldName, flushPendingWorkspaceSave, leftRunDir, metricScope, rangeMode, rightRunDir, showMetricSeries, showVectorComparison, showVolumeComparison, showCompositeComparison, sliceAxis, sliceIndex, viewportBounds]);

  useEffect(() => {
    if (fieldName && plotEligibleFields.includes(fieldName)) return;
    setFieldName(preferred);
  }, [fieldName, plotEligibleFields, preferred]);

  useEffect(() => {
    if (!fieldName) return;
    if (restoredFieldRef.current === fieldName) {
      restoredFieldRef.current = null;
      return;
    }
    setRangeMode(defaultRange(fieldName));
    setColormap(defaultColormap(fieldName));
  }, [fieldName]);

  useEffect(() => {
    setRenderedLeftIndex(null);
  }, [fieldName, leftStep?.publicationGeneration, rightStep?.publicationGeneration]);

  const leftValid = leftResult.field;
  const rightValid = rightResult.field;
  const loadedLeftGrid = useMemo(
    () => leftValid ? gridForWindowedField(leftValid, leftStep?.grid) : null,
    [leftStep?.grid, leftValid],
  );
  const loadedRightGrid = useMemo(
    () => rightValid ? gridForWindowedField(rightValid, rightStep?.grid) : null,
    [rightStep?.grid, rightValid],
  );
  const terrainSlices = useMemo(() => {
    try {
      return {
        left: leftValid ? remapTerrainVerticalSlice(leftValid, loadedLeftGrid, sliceAxis) : null,
        right: rightValid ? remapTerrainVerticalSlice(rightValid, loadedRightGrid, sliceAxis) : null,
        error: null as string | null,
      };
    } catch (remapError) {
      return {
        left: null,
        right: null,
        error: remapError instanceof Error ? remapError.message : String(remapError),
      };
    }
  }, [leftValid, loadedLeftGrid, loadedRightGrid, rightValid, sliceAxis]);
  const displayLeft = terrainSlices.left?.field ?? leftValid;
  const displayRight = terrainSlices.right?.field ?? rightValid;
  const leftGrid = terrainSlices.left?.grid ?? loadedLeftGrid;
  const rightGrid = terrainSlices.right?.grid ?? loadedRightGrid;
  const sourcePlans = useMemo(() => {
    if (!matchedRank) return null;
    const left = planExactValidDomainWindows(leftStep?.grid, matchedRank);
    const right = planExactValidDomainWindows(rightStep?.grid, matchedRank);
    return left.ok && right.ok && left.plan.shape.join("x") === right.plan.shape.join("x")
      ? { left: left.plan, right: right.plan }
      : null;
  }, [leftStep?.grid, matchedRank, rightStep?.grid]);
  const sourceShape = sourcePlans?.left.shape ?? null;
  const sourceGrid = useMemo(() => {
    if (!sourceShape || !matchedRank) return { left: null, right: null };
    const build = (step: StepInfo | null) => {
      if (!step?.grid) return null;
      const origin = (step.grid.validIndexRange?.lower ?? [0, 0, 0]).slice(0, matchedRank);
      return gridForWindowedField({
        shape: [...sourceShape],
        data: new Float32Array(0),
        windowGeometry: {
          aggregation: "cell_mean",
          weighting: "uniform_cell_volume",
          sampleLocation: "block_center",
          centroidsM: null,
          sourceShape: matchedRank === 3
            ? [step.grid.nr ?? 1, step.grid.nth ?? 1, step.grid.nz ?? 1]
            : [step.grid.nr ?? 1, step.grid.nth ?? 1],
          origin,
          extent: [...sourceShape],
          stride: Array(matchedRank).fill(1),
        },
      }, step.grid);
    };
    return { left: build(leftStep), right: build(rightStep) };
  }, [leftStep, matchedRank, rightStep, sourceShape]);
  const is3D = matchedRank === 3;
  const alignedFieldsReady = leftValid != null && rightValid != null;
  const axisLength = is3D && sourceShape ? sourceShape[sliceAxis] : 1;
  const sliceCoordinateKm = is3D
    ? axisCoords(sliceAxis, axisLength, sourceGrid.left).values?.[sliceIndex]
    : undefined;
  const sliceControlLabel = sliceCoordinateKm == null
    ? `Index (${axisLabel(sliceAxis, leftGrid)} = ${sliceIndex})`
    : `${axisLabel(sliceAxis, leftGrid)} = ${sliceCoordinateKm.toFixed(3)} km · index ${sliceIndex}`;
  // Via unitFor so a dimensionless "1" is suppressed rather than printed as a unit.
  const metricUnit = fieldName
    ? unitFor(fieldName, leftStep?.fieldUnits) ?? unitFor(fieldName, rightStep?.fieldUnits)
    : null;
  const vectorComparisonAvailable = ([0, 1, 2] as SliceAxis[]).some(
    (axis) => availableVectorPairs(axis, plotEligibleFields).length > 0,
  );
  const compositeComparisonAvailable = ["reflectivity_dbz", "w", "vorticity_z"].every((field) => plotEligibleFields.includes(field));
  useEffect(() => {
    if (!alignedFieldsReady) return;
    setSliceIndex((index) => Math.min(index, Math.max(0, axisLength - 1)));
  }, [alignedFieldsReady, axisLength]);

  const leftHeatmap = useMemo(
    () => displayLeft ? heatmap2D(displayLeft, sliceAxis, 0) : null,
    [displayLeft, sliceAxis],
  );
  const rightHeatmap = useMemo(
    () => displayRight ? heatmap2D(displayRight, sliceAxis, 0) : null,
    [displayRight, sliceAxis],
  );
  // The axis vectors both panels are drawn against. Hoisted out of the plot
  // builder so the registration check below and the traces agree by construction.
  const panelAxes = useMemo(() => {
    if (!leftHeatmap || !rightHeatmap) return null;
    return {
      leftX: axisCoords(leftHeatmap.colAxis, leftHeatmap.data[0]?.length ?? 0, leftGrid),
      leftY: axisCoords(leftHeatmap.rowAxis, leftHeatmap.data.length, leftGrid),
      rightX: axisCoords(rightHeatmap.colAxis, rightHeatmap.data[0]?.length ?? 0, rightGrid),
      rightY: axisCoords(rightHeatmap.rowAxis, rightHeatmap.data.length, rightGrid),
    };
  }, [leftGrid, leftHeatmap, rightGrid, rightHeatmap]);

  // Require co-located axes AND matching motion frame before differencing: same index ≠ same place
  // across domains that translate (frameMotion.ts). Co-located alone is necessary but not sufficient.
  const frameComparison = useMemo(
    () => canCompareByCell(leftStep, rightStep),
    [leftStep, rightStep],
  );
  const terrainGeometryRegistered = useMemo(() => {
    if (!terrainFollowing) return true;
    if (!leftValid || !rightValid) return false;
    try {
      matchedPhysicalVolumes(leftValid, rightValid);
      return true;
    } catch {
      return false;
    }
  }, [leftValid, rightValid, terrainFollowing]);
  const differenceRegistered = panelAxes != null
    && frameComparison.allowed
    && terrainGeometryRegistered
    && axesAreCoLocated(panelAxes.leftX, panelAxes.rightX)
    && axesAreCoLocated(panelAxes.leftY, panelAxes.rightY);

  const difference = useMemo(
    () => leftHeatmap && rightHeatmap && differenceRegistered
      ? differenceSlice(leftHeatmap, rightHeatmap)
      : null,
    [leftHeatmap, rightHeatmap, differenceRegistered],
  );
  const sliceCoordinates = useMemo(() => {
    if (!leftHeatmap) return null;
    return {
      x: axisCoords(leftHeatmap.colAxis, leftHeatmap.data[0]?.length ?? 0, leftGrid).values,
      y: axisCoords(leftHeatmap.rowAxis, leftHeatmap.data.length, leftGrid).values,
    };
  }, [leftHeatmap, leftGrid]);
  const viewportMetrics = useMemo(
    () => difference && sliceCoordinates?.x && sliceCoordinates.y
      ? regionMetrics(difference, sliceCoordinates.x, sliceCoordinates.y, viewportBounds)
      : difference,
    [difference, sliceCoordinates, viewportBounds],
  );
  // Volume scan needs co-location on all three axes; 2-D check is insufficient.
  const volumeRegistered = useMemo(() => {
    if (!sourceShape || !is3D) return false;
    if (terrainFollowing) return frameComparison.allowed;
    return ([0, 1, 2] as SliceAxis[]).every((axis) => axesAreCoLocated(
      axisCoords(axis, sourceShape[axis], sourceGrid.left),
      axisCoords(axis, sourceShape[axis], sourceGrid.right),
    ));
  }, [frameComparison.allowed, is3D, sourceGrid.left, sourceGrid.right, sourceShape, terrainFollowing]);
  // Secondary tools perform full-grid or multi-frame cellwise calculations.
  // A matching current plane alone is not enough to qualify those operations.
  const extendedComparisonRegistered = is3D ? volumeRegistered : differenceRegistered;
  const exactRegion = useMemo<IndexRegion3D | undefined>(() => {
    if (!is3D || !sourceShape) return undefined;
    if (metricScope === "volume") return undefined;
    const lower: [number, number, number] = [0, 0, 0];
    const upper: [number, number, number] = [sourceShape[0] - 1, sourceShape[1] - 1, sourceShape[2] - 1];
    lower[sliceAxis] = Math.min(sliceIndex, sourceShape[sliceAxis] - 1);
    upper[sliceAxis] = lower[sliceAxis];
    return { lower, upper };
  }, [is3D, metricScope, sliceAxis, sliceIndex, sourceShape]);
  const exactMetricsEnabled = metricScope !== "region"
    && (metricScope === "volume" ? is3D && volumeRegistered : differenceRegistered);
  const exactMetrics = useExactMetrics({
    leftRunDir,
    leftStep,
    rightRunDir,
    rightStep,
    fieldName,
    rank: matchedRank,
    region: exactRegion,
    enabled: exactMetricsEnabled,
  });
  const displayedMetrics = metricScope === "volume"
    ? exactMetrics.metrics
    : metricScope === "region"
      ? viewportMetrics
      : exactMetrics.metrics;

  const plot = useMemo((): { data: Data[]; layout: Partial<Layout> } | null => {
    // A and B stay drawable when the difference is not: each is a valid field in
    // its own right, and seeing them side by side is most of this view's value.
    // Only the third panel depends on the two being co-located.
    if (!leftHeatmap || !rightHeatmap || !panelAxes || !fieldName) return null;
    const [cmin, cmax] = combinedRange(
      leftHeatmap, rightHeatmap, rangeMode, usesFixedAnchorRange(colormap),
    );
    const differenceMagnitude = Math.max(difference?.maxAbs ?? 0, 1e-12);
    const { leftX, leftY, rightX, rightY } = panelAxes;
    const colorscale = resolveColorscale(colormap) as never;
    const trace = (
      z: number[][],
      x: number[],
      y: number[],
      xaxis: "x" | "x2" | "x3",
      yaxis: "y" | "y2" | "y3",
      name: string,
      coloraxis: "coloraxis" | "coloraxis2",
    ): Data => ({
      type: "heatmap",
      // Comparisons must show the co-located samples used by the metrics, not
      // a renderer-specific interpolation of them.
      zsmooth: false,
      z,
      x,
      y,
      xaxis,
      yaxis,
      name,
      coloraxis,
      hovertemplate: `${name}<br>x=%{x:.4g}<br>y=%{y:.4g}<br>${fieldName}=%{z:.4g}<extra></extra>`,
    } as Data);
    const layout = {
        grid: { rows: 1, columns: 3, pattern: "independent", xgap: 0.09 },
        xaxis: { title: { text: leftX.title }, ...(viewportBounds ? { range: [viewportBounds.xMin, viewportBounds.xMax] } : {}) },
        yaxis: { title: { text: leftY.title }, ...(viewportBounds ? { range: [viewportBounds.yMin, viewportBounds.yMax] } : {}) },
        xaxis2: { title: { text: rightX.title }, matches: "x" },
        yaxis2: { title: { text: rightY.title }, matches: "y" },
        xaxis3: { title: { text: rightX.title }, matches: "x" },
        yaxis3: { title: { text: rightY.title }, matches: "y" },
        coloraxis: {
          colorscale,
          cmin,
          cmax,
          colorbar: {
            title: {
              text: metricUnit && fieldName ? `${fieldName} (${metricUnit})` : fieldName,
              side: "right",
            },
            x: 0.646,
            len: 0.82,
          },
        },
        coloraxis2: {
          colorscale: resolveColorscale("RdBu") as never,
          cmin: -differenceMagnitude,
          cmax: differenceMagnitude,
          cmid: 0,
          // A difference carries the same unit as the quantity differenced.
          colorbar: {
            title: { text: metricUnit ? `A−B (${metricUnit})` : "A−B", side: "right" },
            x: 1.015,
            len: 0.82,
          },
        },
        annotations: [
          { text: `A · ${leftLabel}`, x: 0.145, y: 1.07, xref: "paper", yref: "paper", showarrow: false },
          { text: `B · ${rightLabel}`, x: 0.5, y: 1.07, xref: "paper", yref: "paper", showarrow: false },
          difference
            ? { text: "A−B difference", x: 0.855, y: 1.07, xref: "paper", yref: "paper", showarrow: false }
            : {
                text: "A−B unavailable · grids not co-located",
                x: 0.855,
                y: 1.07,
                xref: "paper",
                yref: "paper",
                showarrow: false,
                font: { color: "#d9a441" },
              },
        ],
        margin: { l: 55, r: 80, t: 48, b: 48 },
        dragmode: "zoom",
        uirevision: `${fieldName}:${sliceAxis}`,
    } as unknown as Partial<Layout>;
    return {
      data: [
        trace(leftHeatmap.data, leftX.values ?? [], leftY.values ?? [], "x", "y", `A · ${leftLabel}`, "coloraxis"),
        trace(rightHeatmap.data, rightX.values ?? [], rightY.values ?? [], "x2", "y2", `B · ${rightLabel}`, "coloraxis"),
        // Panel 3 uses A's coordinates (same index space as the subtraction).
        ...(difference
          ? [trace(difference.heatmap.data, leftX.values ?? [], leftY.values ?? [], "x3", "y3", "A−B difference", "coloraxis2")]
          : []),
      ],
      layout,
    };
  }, [colormap, difference, fieldName, leftHeatmap, leftLabel, metricUnit, panelAxes, rangeMode, rightHeatmap, rightLabel, sliceAxis, viewportBounds]);

  const loading = leftResult.loading || rightResult.loading;
  const error = leftResult.error ?? rightResult.error;
  useEffect(() => {
    onFrameDisplayChange?.({
      requestedStepIndex: leftStep?.index ?? null,
      requestedPublicationGeneration: leftStep?.publicationGeneration ?? null,
      renderedStepIndex: renderedLeftIndex,
      renderedPublicationGeneration: renderedLeftIndex == null ? null : leftResult.loadedPublicationGeneration,
      loading: loading || renderedLeftIndex !== leftStep?.index,
    });
  }, [leftResult.loadedPublicationGeneration, leftStep?.index, leftStep?.publicationGeneration, loading, onFrameDisplayChange, renderedLeftIndex]);

  return (
    <section className="comparison-slice" aria-label="Synchronized A and B slices">
      <div className="plot-controls comparison-slice__controls">
        <ControlGroup label="Shared field">
          <select value={fieldName ?? ""} onChange={(event) => setFieldName(event.target.value)}>
            {plotEligibleFields.map((field) => <option value={field} key={field}>{field}</option>)}
          </select>
        </ControlGroup>
        {is3D && (
          <>
            <ControlGroup label="Slice axis"><PillGroup options={AXIS_OPTIONS} value={sliceAxis} onChange={setSliceAxis} /></ControlGroup>
            <ControlGroup label={sliceControlLabel} stretch>
              <input type="range" min={0} max={Math.max(0, axisLength - 1)} value={sliceIndex} onChange={(event) => setSliceIndex(Number(event.target.value))} />
            </ControlGroup>
          </>
        )}
        <ControlGroup label="Matched time">
          <button
            type="button"
            className={`comparison-slice__series-toggle ${showMetricSeries ? "comparison-slice__series-toggle--active" : ""}`}
            aria-pressed={showMetricSeries}
            disabled={!extendedComparisonRegistered}
            title={extendedComparisonRegistered
              ? "Build exact metrics over every matched physical time"
              : "Matched-time metrics require a fully co-located grid and motion frame"}
            onClick={() => setShowMetricSeries((visible) => !visible)}
          >
            {showMetricSeries ? "Hide series" : "Build series"}
          </button>
        </ControlGroup>
        {vectorComparisonAvailable && (
          <ControlGroup label="Vector validation">
            <button
              type="button"
              className={`comparison-slice__series-toggle ${showVectorComparison ? "comparison-slice__series-toggle--active" : ""}`}
              aria-pressed={showVectorComparison}
              disabled={!extendedComparisonRegistered}
              title={extendedComparisonRegistered
                ? "Open co-located vector validation"
                : "Vector validation requires a fully co-located grid and motion frame"}
              onClick={() => setShowVectorComparison((visible) => !visible)}
            >
              {showVectorComparison ? "Hide vectors" : "Open vectors"}
            </button>
          </ControlGroup>
        )}
        <ControlGroup label="3-D validation">
          <button
            type="button"
            className={`comparison-slice__series-toggle ${showVolumeComparison ? "comparison-slice__series-toggle--active" : ""}`}
            aria-pressed={showVolumeComparison}
            disabled={!volumeRegistered}
            title={volumeRegistered
              ? "Open co-located 3-D validation"
              : "3-D validation requires a fully co-located grid and motion frame"}
            onClick={() => setShowVolumeComparison((visible) => !visible)}
          >
            {showVolumeComparison ? "Hide 3-D" : "Open 3-D"}
          </button>
        </ControlGroup>
        {compositeComparisonAvailable && (
          <ControlGroup label="Storm composite">
            <button
              type="button"
              className={`comparison-slice__series-toggle ${showCompositeComparison ? "comparison-slice__series-toggle--active" : ""}`}
              aria-pressed={showCompositeComparison}
              disabled={!differenceRegistered}
              title={differenceRegistered
                ? "Reflectivity / w / vorticity, A beside B beside A−B"
                : "The composite panel is an A−B difference, so it needs the same co-located axes and motion frame the difference panel needs."}
              onClick={() => setShowCompositeComparison((visible) => !visible)}
            >
              {showCompositeComparison ? "Hide composite" : "Open composite"}
            </button>
          </ControlGroup>
        )}
        <ControlGroup label="Shared range"><PillGroup options={RANGE_OPTIONS} value={rangeMode} onChange={setRangeMode} /></ControlGroup>
        <ControlGroup label="Colormap">
          <select value={colormap} onChange={(event) => setColormap(event.target.value as ExtendedColorscale)}>
            {(["Viridis", "Plasma", "Inferno", "RdBu", "Greys", "dBZ", "ColdPool"] as const).map((map) => <option key={map}>{map}</option>)}
          </select>
        </ControlGroup>
        {is3D && (
          <ControlGroup label="Statistics">
            <PillGroup options={METRIC_SCOPE_OPTIONS} value={metricScope} onChange={setMetricScope} />
          </ControlGroup>
        )}
      </div>
      <div className="comparison-slice__time">
        <span>A t = {leftStep?.simulationTimeS?.toFixed(3) ?? "missing"} s</span>
        <span>B t = {rightStep?.simulationTimeS?.toFixed(3) ?? "unmatched"} s</span>
        <span className="comparison-slice__workspace-status">
          {workspaceStatus === "restored"
            ? "saved view restored"
            : workspaceStatus === "saved"
              ? "view recipe retained"
              : workspaceStatus === "unavailable"
                ? "local save unavailable"
                : "preparing view recipe"}
        </span>
        <strong>linked axes · shared color scale</strong>
      </div>
      {!differenceRegistered && leftHeatmap && rightHeatmap && (
        <div className="comparison-slice__metric-error" role="alert">
          <small>Difference blocked</small>
          <strong>
            {frameComparison.blocker ?? (
              <>
                A and B do not sample the same physical positions on this plane,
                so a cell-by-cell difference would subtract values from different
                places. The two fields are still shown side by side; regrid the
                runs onto a common lattice to difference them.
              </>
            )}
          </strong>
        </div>
      )}
      {difference && (
        <div className="comparison-slice__metrics" aria-label="A minus B statistics">
          <span className="comparison-slice__metric-scope">
            <small>Statistics scope</small>
            <strong>{metricScope === "volume" ? "Full valid grid" : metricScope === "region" ? "Zoomed region" : "Visible slice"}</strong>
            <em>{fieldName}{metricUnit ? ` · ${metricUnit}` : ""}</em>
          </span>
          {metricScope === "volume" && is3D && !volumeRegistered ? (
            <span className="comparison-slice__metric-error" role="alert">
              <small>Full-grid blocker</small>
              <strong>A and B do not share a 3-D lattice — a whole-grid scan would pair cells from different positions.</strong>
            </span>
          ) : metricScope !== "region" && exactMetrics.status === "loading" ? (
            <span className="comparison-slice__metric-progress" role="status">
              <small>Exact scan</small>
              <strong>{Math.round(exactMetrics.progress * 100)}% complete</strong>
            </span>
          ) : metricScope !== "region" && exactMetrics.status === "error" ? (
            <span className="comparison-slice__metric-error" role="alert">
              <small>Exact-metric blocker</small>
              <strong>{exactMetrics.error}</strong>
            </span>
          ) : displayedMetrics && (
            <>
              <span><small>Mean A−B</small><strong>{metric(displayedMetrics.mean)}</strong></span>
              <span><small>MAE</small><strong>{metric(displayedMetrics.mae)}</strong></span>
              <span><small>RMSE</small><strong>{metric(displayedMetrics.rmse)}</strong></span>
              <span><small>Max |A−B|</small><strong>{metric(displayedMetrics.maxAbs)}</strong></span>
              <span><small>Finite pairs</small><strong>{displayedMetrics.count.toLocaleString()}</strong></span>
            </>
          )}
        </div>
      )}
      {showMetricSeries && fieldName && extendedComparisonRegistered && (
        <ComparisonMetricSeries
          leftRunDir={leftRunDir}
          leftDatasetIdentity={leftDatasetIdentity}
          leftSteps={leftSteps}
          rightRunDir={rightRunDir}
          rightDatasetIdentity={rightDatasetIdentity}
          rightSteps={rightSteps}
          matches={timeMatches}
          fieldName={fieldName}
          unit={metricUnit}
        />
      )}
      {showVectorComparison && extendedComparisonRegistered && (
        <ComparisonVectorPlot
          leftRunDir={leftRunDir}
          leftDatasetIdentity={leftDatasetIdentity}
          leftLabel={leftLabel}
          leftStep={leftStep}
          rightRunDir={rightRunDir}
          rightDatasetIdentity={rightDatasetIdentity}
          rightLabel={rightLabel}
          rightStep={rightStep}
          leftSteps={leftSteps}
          rightSteps={rightSteps}
          timeMatches={timeMatches}
          eligibleFields={plotEligibleFields}
          sliceAxis={sliceAxis}
          sliceIndex={sliceIndex}
        />
      )}
      {showVolumeComparison && fieldName && volumeRegistered && (
        <ComparisonVolumePlot
          leftRunDir={leftRunDir}
          leftLabel={leftLabel}
          leftStep={leftStep}
          rightRunDir={rightRunDir}
          rightLabel={rightLabel}
          rightStep={rightStep}
          fieldName={fieldName}
        />
      )}
      {/* Same gate as the difference panel: this composite subtracts A from B cell-by-cell, and a
          matching shape is not a matching place across domains that translate. */}
      {showCompositeComparison && compositeComparisonAvailable && differenceRegistered && (
        <ComparisonCompositePlot
          leftRunDir={leftRunDir}
          leftLabel={leftLabel}
          leftStep={leftStep}
          rightRunDir={rightRunDir}
          rightLabel={rightLabel}
          rightStep={rightStep}
          sliceAxis={sliceAxis}
          sliceIndex={sliceIndex}
        />
      )}
      <div className="plot-body comparison-slice__body">
        {!rightStep && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>No exact B time partner for this A frame.</span></div>}
        {terrainSlices.error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{terrainSlices.error}</span></div>}
        {(terrainSlices.left?.remapped || terrainSlices.right?.remapped) && <div className="plot-overlay"><Icon name="dot" size={12} /><span>physical-volume terrain→Cartesian vertical remap</span></div>}
        {!terrainBlocked && fieldName && !boundedSupported && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>This provider cannot serve bounded comparison windows.</span></div>}
        {displayPlans?.samplingDisclosure && <div className="plot-overlay"><Icon name="dot" size={12} /><span>{displayPlans.samplingDisclosure}</span></div>}
        {error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{error}</span></div>}
        {plot && !error ? (
          <PlotCanvas
            data={plot.data}
            layout={plot.layout}
            className="comparison-slice__canvas"
            onRelayout={(event) => {
              const xRange = relayoutRange(event, ["xaxis", "xaxis2", "xaxis3"]);
              const yRange = relayoutRange(event, ["yaxis", "yaxis2", "yaxis3"]);
              if (xRange === null || yRange === null) {
                setViewportBounds(null);
                return;
              }
              if (xRange && yRange) {
                setViewportBounds({ xMin: xRange[0], xMax: xRange[1], yMin: yRange[0], yMax: yRange[1] });
              }
            }}
            onRendered={() => {
              if (
                leftResult.loadedStepIndex === leftStep?.index
                && rightResult.loadedStepIndex === rightStep?.index
              ) setRenderedLeftIndex(leftStep.index);
            }}
          />
        ) : !loading && !error && <div className="plot-overlay"><span>No aligned slice is available.</span></div>}
      </div>
    </section>
  );
}
