import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { slice2D, type SliceAxis, type Npy3D, type NpyN } from "@/lib/platform/npy";
import { axisCoords, axisLabel, axisOptions } from "@/lib/science/gridCoords";
import { buildQuiver, sampledMaxMagnitude } from "@/lib/plot/quiver";
import { describeVectorSampling, vectorReferenceKey } from "@/lib/plot/vectorPresentation";
import { carriedMagnitude, latchMagnitude, type LatchedMagnitude } from "@/lib/plot/viewSync";
import { availableVectorPairs, defaultComponents } from "@/lib/science/vectorComponents";
import {
  describeFrameTransition,
  findPublishedFrame,
  loadedFramesAreCoherent,
} from "@/lib/run/frameIdentity";
import { unitFor, type FieldUnits } from "@/lib/science/fieldUnits";
import { fieldDefinitionIdentity, registeredFieldRank, type GridInfo } from "@/lib/run/runOutputs";
import { validAxisLength } from "@/lib/science/validDomain";
import { loadedSliceWindowIndex } from "@/lib/plot/sliceWindowSelection";
import {
  fieldSupportsBoundedWindows,
  PLOTLY_HEATMAP_CELL_CAP,
  planBoundedPlaneWindows,
  planBoundedSliceWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { remapTerrainVerticalSlice } from "@/lib/science/terrainSliceRemap";
import { gridForWindowedField } from "@/lib/science/windowedGrid";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  ControlGroup,
  fitAxisFragments,
  PillGroup,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useFieldAny,
  usePlaneViewport,
  usePlotChrome,
  useRetainedPlot,
  type PlotComponentProps,
} from "./common";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsSharePlotRank } from "@/lib/plot/fieldPlotEligibility";
import { SlicePlaneContext } from "./SlicePlaneContext";
import { SliceIndexInput } from "./SliceIndexInput";

/* 2D quiver plot: two component fields + slice axis. Default pairing stays in-plane (defaultComponents). */

type DragMode = "pan" | "zoom";
const DRAG_OPTIONS: ReadonlyArray<{ value: DragMode; label: string }> = [
  { value: "pan", label: "Pan" },
  { value: "zoom", label: "Box zoom" },
];

const STRIDE_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 1, label: "Every" },
  { value: 2, label: "1/2" },
  { value: 3, label: "1/3" },
  { value: 4, label: "1/4" },
];

interface Component2D {
  data: number[][];
  rows: number;
  cols: number;
  rowAxis: SliceAxis;
  colAxis: SliceAxis;
}

/** Slice a 3D field, or pass a 2D (nr, nθ) surface field through directly. */
function componentSlice(
  field: NpyN,
  axis: SliceAxis,
  index: number,
): Component2D | null {
  if (field.shape.length === 3) {
    const f = field as Npy3D;
    const dim = f.shape[axis];
    const s = slice2D(f, axis, Math.min(index, dim - 1));
    return {
      data: s.data,
      rows: s.rows,
      cols: s.cols,
      rowAxis: s.rowAxis as SliceAxis,
      colAxis: s.colAxis as SliceAxis,
    };
  }
  if (field.shape.length === 2) {
    const [nr, nth] = field.shape;
    const data: number[][] = new Array(nr);
    for (let i = 0; i < nr; i++) {
      const row = new Array<number>(nth);
      const base = i * nth;
      for (let j = 0; j < nth; j++) row[j] = field.data[base + j];
      data[i] = row;
    }
    return { data, rows: nr, cols: nth, rowAxis: 0, colAxis: 1 };
  }
  return null;
}

export function VectorField({
  runDir: rootRunDir,
  steps: rootSteps,
  stepIdx: rootStepIdx,
  onPresetCleared,
  defaultSliceAxis,
  onFrameDisplayChange,
}: PlotComponentProps) {
  const rootCurrentStep = rootSteps[rootStepIdx] ?? null;
  const rootFields = useMemo(() => rootCurrentStep?.fields ?? [], [rootCurrentStep?.fields]);

  // Default components follow the slice plane (see defaultComponents): the
  // Default θ-slice → (u rightward, w upward); plan view stays horizontal.
  const initialSliceAxis: SliceAxis = defaultSliceAxis ?? 1;
  const [xField, setXField] = useState<string | null>(
    () => defaultComponents(initialSliceAxis, rootFields).xField,
  );
  const [yField, setYField] = useState<string | null>(
    () => defaultComponents(initialSliceAxis, rootFields).yField,
  );
  const hierarchyRequiredFields = useMemo(
    () => [xField, yField].filter((field): field is string => field != null),
    [xField, yField],
  );
  const outputHierarchySource = useOutputHierarchySource({
    runDir: rootRunDir,
    steps: rootSteps,
    stepIdx: rootStepIdx,
    scope: "shared",
    requiredFields: hierarchyRequiredFields,
  });
  const runDir = outputHierarchySource.sourceRunDir;
  const steps = outputHierarchySource.sourceSteps;
  const currentStep = outputHierarchySource.currentStep;
  const fieldsForStep = useMemo(() => currentStep?.fields ?? [], [currentStep?.fields]);
  const grid = currentStep?.grid ?? null;
  const [sliceAxis, setSliceAxis] = useState<SliceAxis>(initialSliceAxis); // θ-slice → r-z plane
  const [sliceIndex, setSliceIndex] = useState(0);
  const [displayedSliceIndex, setDisplayedSliceIndex] = useState(0);
  const [stride, setStride] = useState(2);
  const [scale, setScale] = useState(1);
  const [dragMode, setDragMode] = useState<DragMode>("pan");
  // See PlotFitMode. Vectors are the most aspect-sensitive plot in Studio — a
  // stretched pane rotates every arrow — so true shape is firmly the default.
  const [resetTick, setResetTick] = useState(0);
  const viewIdentity = JSON.stringify([runDir, xField, yField, sliceAxis]);
  const { viewport, onRelayout, resetViewport } = usePlaneViewport(viewIdentity);
  const chrome = usePlotChrome();
  const vectorPairs = useMemo(
    () => availableVectorPairs(sliceAxis, fieldsForStep).filter((pair) =>
      fieldsSharePlotRank(runDir, currentStep, [pair.xField, pair.yField], [2, 3])
    ),
    [currentStep, fieldsForStep, runDir, sliceAxis],
  );

  const clearPreset = () => onPresetCleared?.();

  // Re-pick defaults when the plane changes or a component becomes invalid; manual choice persists.
  const prevSliceAxisRef = useRef<SliceAxis>(sliceAxis);
  useEffect(() => {
    if (fieldsForStep.length === 0) return;
    const planeChanged = prevSliceAxisRef.current !== sliceAxis;
    prevSliceAxisRef.current = sliceAxis;
    const d = vectorPairs[0] ?? { xField: null, yField: null };
    if (planeChanged || !xField || !fieldsForStep.includes(xField)) {
      setXField(d.xField);
    }
    if (planeChanged || !yField || !fieldsForStep.includes(yField)) {
      setYField(d.yField);
    }
  }, [sliceAxis, fieldsForStep, vectorPairs, xField, yField]);

  // Debounce the load step (see SlicePlot): a scrub/autoplay otherwise fires two
  // full field reads per frame swept through. Bypassed while recording.
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const xRank = xField && currentStep
    ? registeredFieldRank(runDir, currentStep.index, xField, currentStep.publicationGeneration)
    : null;
  const yRank = yField && currentStep
    ? registeredFieldRank(runDir, currentStep.index, yField, currentStep.publicationGeneration)
    : null;
  const vectorRank = (xRank === 2 || xRank === 3) && yRank === xRank ? xRank : null;
  const terrainVectorBlocked = false;
  const boundedSupported = Boolean(xField && yField
    && fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[xField])
    && fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[yField]));
  const windowPlan = useMemo(() => {
    if (!vectorRank || terrainVectorBlocked || !boundedSupported) return null;
    if (vectorRank === 2) return planBoundedPlaneWindows(
      grid, PLOTLY_HEATMAP_CELL_CAP, viewport,
    );
    return planBoundedSliceWindows(
      grid, sliceAxis, sliceIndex, PLOTLY_HEATMAP_CELL_CAP, viewport,
    );
  }, [boundedSupported, grid, sliceAxis, sliceIndex, terrainVectorBlocked, vectorRank, viewport]);
  const loadEnabled = windowPlan?.ok === true;
  const xLoad = useFieldAny(
    runDir, loadEnabled ? loadFrame.stepIndex : null, xField, loadFrame.publicationGeneration,
    windowPlan?.ok ? windowsForFieldAggregation(windowPlan.plan.windows, xField) : null,
    JSON.stringify([runDir, xField, sliceAxis]),
  );
  const yLoad = useFieldAny(
    runDir, loadEnabled ? loadFrame.stepIndex : null, yField, loadFrame.publicationGeneration,
    windowPlan?.ok ? windowsForFieldAggregation(windowPlan.plan.windows, yField) : null,
    JSON.stringify([runDir, yField, sliceAxis]),
  );
  const sourceGeometryKey = (field: NpyN | null) => JSON.stringify([
    field?.shape, field?.windowGeometry?.origin, field?.windowGeometry?.extent, field?.windowGeometry?.stride,
  ]);
  const coherentFrame = loadedFramesAreCoherent(xLoad, yLoad)
    && sourceGeometryKey(xLoad.field) === sourceGeometryKey(yLoad.field);
  const loadedStep = coherentFrame
    ? findPublishedFrame(
        steps,
        xLoad.loadedStepIndex,
        xLoad.loadedPublicationGeneration,
      )
    : null;
  const settledSources = coherentFrame && !xLoad.loading && !yLoad.loading;
  const xFieldValue = settledSources ? xLoad.field : null;
  const yFieldValue = settledSources ? yLoad.field : null;
  const loadedSliceIndex = loadedSliceWindowIndex(xLoad.field, sliceAxis, loadedStep?.grid?.validIndexRange?.lower,
    xLoad.loading || yLoad.loading ? displayedSliceIndex : sliceIndex);
  const displayGrid = useMemo(
    () => xFieldValue ? gridForWindowedField(xFieldValue, loadedStep?.grid) : null,
    [loadedStep?.grid, xFieldValue],
  );
  const terrainVectors = useMemo(() => {
    if (!xFieldValue || !yFieldValue) {
      return { x: xFieldValue, y: yFieldValue, grid: displayGrid, remapped: false, error: null as string | null };
    }
    try {
      const x = remapTerrainVerticalSlice(xFieldValue, displayGrid, sliceAxis);
      const y = remapTerrainVerticalSlice(yFieldValue, displayGrid, sliceAxis);
      return { x: x.field, y: y.field, grid: x.grid, remapped: x.remapped, error: null as string | null };
    } catch (remapError) {
      return {
        x: null,
        y: null,
        grid: displayGrid,
        remapped: false,
        error: remapError instanceof Error ? remapError.message : String(remapError),
      };
    }
  }, [displayGrid, sliceAxis, xFieldValue, yFieldValue]);
  const displayFieldUnits = loadedStep?.fieldUnits;
  const loading = xLoad.loading || yLoad.loading || vectorRank === 3 && displayedSliceIndex !== sliceIndex;
  const error = xLoad.error || yLoad.error;
  const [renderedStepIndex, setRenderedStepIndex] = useState<number | null>(null);
  const [renderedPublicationGeneration, setRenderedPublicationGeneration] =
    useState<string | null>(null);
  const requestedStepIndex = currentStep?.index ?? null;
  const requestedPublicationGeneration = currentStep?.publicationGeneration ?? null;
  const frameTransition = describeFrameTransition(
    requestedStepIndex,
    renderedStepIndex,
    requestedPublicationGeneration,
    renderedPublicationGeneration,
  );
  useEffect(() => {
    setRenderedStepIndex(null);
    setRenderedPublicationGeneration(null);
  }, [runDir, xField, yField, sliceAxis]);
  useEffect(() => {
    onFrameDisplayChange?.({
      requestedStepIndex,
      requestedPublicationGeneration,
      renderedStepIndex,
      renderedPublicationGeneration,
      loading: !terrainVectorBlocked && boundedSupported
        && !error && (loading || frameTransition.kind !== "ready"),
    });
  }, [
    boundedSupported,
    error,
    frameTransition.kind,
    loading,
    onFrameDisplayChange,
    renderedPublicationGeneration,
    renderedStepIndex,
    requestedPublicationGeneration,
    requestedStepIndex,
    terrainVectorBlocked,
  ]);

  const sliceDim = vectorRank === 3
    ? validAxisLength(sliceAxis, grid) ?? [grid?.nr, grid?.nth, grid?.nz][sliceAxis] ?? 0 : 0;
  useEffect(() => {
    if (sliceDim > 0) setSliceIndex((prev) => Math.min(prev, sliceDim - 1));
  }, [sliceDim, sliceAxis]);

  // Reset to the category's default orientation when the category changes
  // (Plan views → z / horizontal winds, Cross-sections → θ / r-z circulation).
  // Keyed on the prop only, so manual axis changes within a category persist.
  useEffect(() => {
    if (defaultSliceAxis == null) return;
    setSliceAxis(defaultSliceAxis);
  }, [defaultSliceAxis]);

  /*
   * The latch only ever RISES (latchMagnitude), so one transient frame shrinks every later frame's
   * arrows for the rest of the session with no way back. Folding an epoch into the scope makes
   * re-anchoring a scope change rather than a second code path.
   */
  const [latchEpoch, setLatchEpoch] = useState(0);
  // Arrow scale latched per run/pair/plane scope; scope resets at render time to avoid stale carry-over.
  const scaleScope = JSON.stringify([runDir, xField, yField, sliceAxis,
    fieldDefinitionIdentity(runDir, currentStep?.index ?? null, xField, currentStep?.publicationGeneration ?? null),
    fieldDefinitionIdentity(runDir, currentStep?.index ?? null, yField, currentStep?.publicationGeneration ?? null),
    currentStep?.fieldUnits?.[xField ?? ""], currentStep?.fieldUnits?.[yField ?? ""]]);
  // Ref not state: raising the latch must not trigger a re-render (would repaint every playback frame twice).
  const latchedMagnitudeRef = useRef<LatchedMagnitude>({ scope: scaleScope, reference: 0 });
  const latchScope = `${scaleScope} epoch=${latchEpoch}`;
  const magnitudeReference = carriedMagnitude(latchedMagnitudeRef.current, latchScope);
  const vectorPlane = useMemo(() => prepareVectorPlane(terrainVectors.x, terrainVectors.y,
    sliceAxis, 0, terrainVectors.grid), [terrainVectors, sliceAxis]);
  const effectiveReference = Math.max(magnitudeReference, vectorPlane?.frameMax ?? 0);
  const vectorGeometry = useMemo(() => prepareVectorGeometry(vectorPlane, stride, scale, effectiveReference),
    [vectorPlane, stride, scale, effectiveReference]);

  const plot = useMemo(() => {
    if (terrainVectorBlocked || !settledSources || !loadedStep || error || terrainVectors.error) return null;
    const built = buildVectorPlot({
        plane: vectorPlane,
        geometry: vectorGeometry,
        xName: xField,
        yName: yField,
        sliceAxis,
        stride,
        grid: terrainVectors.grid,
        fieldUnits: displayFieldUnits,
        dragMode,
        arrowColor: chrome.ink,
      });
    return built ? { ...built, stepIndex: loadedStep.index, publicationGeneration: loadedStep.publicationGeneration,
      sourceGrid: loadedStep.grid, sliceIndex: loadedSliceIndex, remapped: terrainVectors.remapped } : null;
  }, [terrainVectors.grid, terrainVectors.error, terrainVectors.remapped, vectorPlane, vectorGeometry, xField, yField,
    sliceAxis, stride, displayFieldUnits, dragMode, terrainVectorBlocked, chrome.ink, settledSources, loadedStep, loadedSliceIndex, error]);

  // Persist a new peak for the frames that follow. Writing a ref rather than
  // state is the point: this must not schedule a render (see above).
  useEffect(() => {
    if (!plot) return;
    latchedMagnitudeRef.current = latchMagnitude(
      latchedMagnitudeRef.current,
      latchScope,
      plot.maxMagnitude,
    );
  }, [plot, latchScope]);

  // The two components load independently, so a plane change (which repicks xField) or a playback
  // frame leaves the coherence gate false for the gap between them, nulling `plot`.
  const { shown: shownPlot } = useRetainedPlot(plot, scaleScope);
  const [committedSection, setCommittedSection] = useState<{ scope: string; grid: GridInfo | null; index: number; remapped: boolean } | null>(null);
  const displayedSection = committedSection?.scope === scaleScope ? committedSection : null;

  const is3D = vectorRank === 3;

  /*
   * "Index (z = 5)" read as a height of 5, not as cell k=5 — the one number a cross-section is
   * interrogated by. Show the physical position when the grid knows it, and keep the index visible
   * because it is what the slider actually moves.
   */
  const slicePositionLabel = `Index (${axisLabel(sliceAxis, grid)} = ${sliceIndex})`;

  return (
    <>
      <div className="plot-controls">
        <OutputHierarchyControls source={outputHierarchySource} />
        <ControlGroup label="Vector field">
          <select
            value={xField && yField ? `${xField}\u0000${yField}` : ""}
            onChange={(e) => {
              const pair = vectorPairs.find(
                (candidate) => `${candidate.xField}\u0000${candidate.yField}` === e.target.value,
              );
              setXField(pair?.xField ?? null);
              setYField(pair?.yField ?? null);
              clearPreset();
            }}
          >
            {vectorPairs.length === 0 && <option value="">(no in-plane vector family)</option>}
            {vectorPairs.map((pair) => (
              <option
                key={pair.family.id}
                value={`${pair.xField}\u0000${pair.yField}`}
              >
                {pair.family.label} ({pair.xField} →, {pair.yField} ↑)
              </option>
            ))}
          </select>
        </ControlGroup>

        <PlotCompressionControls step={loadedStep} fields={[xField, yField]} />

        {is3D && (
          <>
            <ControlGroup label="Slice axis">
              <PillGroup options={axisOptions(grid)} value={sliceAxis} onChange={setSliceAxis} />
            </ControlGroup>
            <ControlGroup label={slicePositionLabel} stretch>
              <input
                type="range"
                min={0}
                max={Math.max(0, sliceDim - 1)}
                value={sliceIndex}
                aria-label={`Selected ${axisLabel(sliceAxis, grid)} index`}
                onChange={(e) => setSliceIndex(Number(e.target.value))}
                disabled={sliceDim <= 0}
              />
              <SliceIndexInput value={sliceIndex} max={Math.max(0, sliceDim - 1)} disabled={sliceDim <= 0}
                label={`Go to ${axisLabel(sliceAxis, grid)} index`} onCommit={setSliceIndex} />
            </ControlGroup>
          </>
        )}

        <ControlGroup label="Density">
          <PillGroup options={STRIDE_OPTIONS} value={stride} onChange={setStride} />
        </ControlGroup>

        <ControlGroup label={`Scale (${scale.toFixed(1)}×)`}>
          <input
            type="range"
            min={0.2}
            max={4}
            step={0.1}
            value={scale}
            onChange={(e) => setScale(Number(e.target.value))}
          />
            <button
              type="button"
              className="plot-reanchor-btn"
              disabled={shownPlot == null || !(magnitudeReference > shownPlot.maxMagnitude)}
              onClick={() => setLatchEpoch((epoch) => epoch + 1)}
              title={shownPlot != null && magnitudeReference > shownPlot.maxMagnitude
                ? `Arrows are normalised to an earlier peak of ${magnitudeReference.toPrecision(3)}. Re-anchor to this frame's ${shownPlot.maxMagnitude.toPrecision(3)}.`
                : "Arrow scaling already matches this frame"}
            >
              Re-anchor
            </button>
        </ControlGroup>

        <ControlGroup label="Drag">
          <PillGroup options={DRAG_OPTIONS} value={dragMode} onChange={setDragMode} />
        </ControlGroup>

        <button
          type="button"
          className="plot-reset-btn"
          onClick={() => {
            resetViewport();
            setResetTick((t) => t + 1);
          }}
          title="Reset pan/zoom to fit"
        >
          Reset view
        </button>
      </div>

      <div className="plot-body">
        {(is3D || defaultSliceAxis === 1 && vectorRank == null) && <SlicePlaneContext grid={displayedSection?.grid} axis={sliceAxis}
          index={displayedSection?.index ?? null} physicalHeight={displayedSection?.remapped} />}
        {loading && shownPlot && (
          <div className="plot-overlay" role="status">
            <Icon name="dot" size={12} />
            <span>
              Displaying index {displayedSliceIndex} · loading selected {axisLabel(sliceAxis, grid)} index {sliceIndex}…
            </span>
          </div>
        )}
        {error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{error}</span>
          </div>
        )}
        {terrainVectors.error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{terrainVectors.error}</span>
          </div>
        )}
        {terrainVectors.remapped && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>physical-volume terrain→Cartesian vector remap</span>
          </div>
        )}
        {!terrainVectorBlocked && !boundedSupported && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>A vector component cannot serve bounded windows.</span>
          </div>
        )}
        {terrainVectorBlocked && !loading && !error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>Terrain vertical-vector planes require the non-separable physical height lattice; model-level arrows would have a false direction.</span>
          </div>
        )}
        {viewport && windowPlan?.ok && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>
              viewport {windowPlan.plan.stride.every((value) => value === 1)
                ? "native vectors"
                : `mean ${windowPlan.plan.stride.join("×")}`}
              {windowPlan.plan.fullSourceShape
                ? ` · full plane ${windowPlan.plan.fullSourceShape.join("×")}`
                : ""}
            </span>
          </div>
        )}
        {shownPlot ? (
          <PlotCanvas
            key={resetTick}
            viewIdentity={viewIdentity}
            data={shownPlot.data}
            layout={shownPlot.layout}
            onRelayout={onRelayout}
            onRendered={() => {
              if (shownPlot) {
                setRenderedStepIndex(shownPlot.stepIndex);
                setRenderedPublicationGeneration(shownPlot.publicationGeneration);
                setDisplayedSliceIndex(shownPlot.sliceIndex);
                setCommittedSection(previous => previous?.scope === scaleScope && previous.grid === shownPlot.sourceGrid
                  && previous.index === shownPlot.sliceIndex && previous.remapped === shownPlot.remapped ? previous
                  : { scope: scaleScope, grid: shownPlot.sourceGrid, index: shownPlot.sliceIndex, remapped: shownPlot.remapped });
              }
            }}
          />
        ) : (
          !loading &&
          !error && !terrainVectorBlocked && (
            <div className="plot-overlay">
              {/* Telling the user to "pick two fields" when the select has nothing to offer is the
                  one thing this state must not do. Name what the plane needs instead. */}
              <span>{vectorPairs.length === 0
                ? `This run exports no matching pair of in-plane components for the `
                  + `${axisLabel(sliceAxis, grid)}-slice. A vector plot needs both components of one `
                  + `family (velocity or vorticity) on this plane — try another slice axis, or `
                  + `re-export with the missing component.`
                : "Choose a vector field above to draw the arrows."}</span>
            </div>
          )
        )}
      </div>
    </>
  );
}

interface BuildVectorArgs {
  plane: ReturnType<typeof prepareVectorPlane>;
  geometry: ReturnType<typeof prepareVectorGeometry>;
  xName: string | null;
  yName: string | null;
  sliceAxis: SliceAxis;
  stride: number;
  grid: GridInfo | null;
  fieldUnits: FieldUnits;
  dragMode: DragMode;
  /** Theme-resolved neutral ink color (fixed color breaks in one theme). */
  arrowColor: string;
}

function prepareVectorPlane(xField: NpyN | null, yField: NpyN | null, sliceAxis: SliceAxis,
  sliceIndex: number, grid: GridInfo | null) {
  if (!xField || !yField) return null;
  const xc = componentSlice(xField, sliceAxis, sliceIndex);
  const yc = componentSlice(yField, sliceAxis, sliceIndex);
  if (!xc || !yc || xc.rows !== yc.rows || xc.cols !== yc.cols
      || xc.rowAxis !== yc.rowAxis || xc.colAxis !== yc.colAxis) return null;
  const xCoords = axisCoords(xc.colAxis, xc.cols, grid);
  const yCoords = axisCoords(xc.rowAxis, xc.rows, grid);
  return { xc, yc, xCoords, yCoords, arrowSpacing: Math.min(xCoords.spacing, yCoords.spacing),
    frameMax: sampledMaxMagnitude(xc.data, yc.data, xc.rows, xc.cols, 1),
    samplingDisclosure: describeVectorSampling(xField, xc.rowAxis, xc.colAxis) };
}

function prepareVectorGeometry(plane: ReturnType<typeof prepareVectorPlane>, stride: number, scale: number, magnitudeReference: number) {
  if (!plane) return null;
  const { xc, yc, xCoords, yCoords, arrowSpacing, frameMax } = plane;
  const reference = Math.max(magnitudeReference, frameMax);
  const quiver = buildQuiver({ u: xc.data, v: yc.data, rows: xc.rows, cols: xc.cols,
    xCoords: xCoords.values, yCoords: yCoords.values, xSpacing: arrowSpacing, ySpacing: arrowSpacing,
    stride, scale, referenceMagnitude: reference });
  const referenceKey = vectorReferenceKey(reference, arrowSpacing, quiver.effectiveStride, scale, xCoords.values, yCoords.values);
  return { quiver, reference, referenceKey };
}

function buildVectorPlot({
  plane,
  geometry,
  xName,
  yName,
  sliceAxis,
  stride,
  grid,
  fieldUnits,
  dragMode,
  arrowColor,
}: BuildVectorArgs): { data: Data[]; layout: Partial<Layout>; maxMagnitude: number } | null {
  if (!plane || !geometry) return null;
  const { xCoords, yCoords, frameMax, samplingDisclosure } = plane;
  const { quiver, reference, referenceKey } = geometry;

  const xUnitHover = xName ? unitFor(xName, fieldUnits) : null;
  const yUnitHover = yName ? unitFor(yName, fieldUnits) : null;
  const hoverUnit = (unit: string | null) => (unit ? ` ${unit}` : "");
  const data: Data[] = [
    {
      type: "scatter",
      mode: "lines",
      x: quiver.x,
      y: quiver.y,
      line: { color: arrowColor, width: 1.2 },
      hoverinfo: "skip",
      // The physical reference arrow below is the useful key. A one-item
      // legend labelled only "vectors" repeats the obvious and steals space.
      showlegend: false,
      name: "vectors",
    },
    /*
     * The arrows are one polyline with null separators, so Plotly cannot hover them per arrow — the
     * plot previously had no readout at all and a vector could only be eyeballed against the title's
     * peak. This invisible point per tail carries the components, making the field queryable.
     */
    {
      type: "scatter",
      mode: "markers",
      x: quiver.originX,
      y: quiver.originY,
      marker: { size: 9, color: "rgba(0,0,0,0)" },
      customdata: quiver.components,
      hovertemplate:
        `${xCoords.title}: %{x:.2f}<br>${yCoords.title}: %{y:.2f}`
        + `<br>${xName ?? "?"} (→): %{customdata[0]:.3g}${hoverUnit(xUnitHover)}`
        + `<br>${yName ?? "?"} (↑): %{customdata[1]:.3g}${hoverUnit(yUnitHover)}`
        + `<br>|V|: %{customdata[2]:.3g}${hoverUnit(xUnitHover === yUnitHover ? xUnitHover : null)}`
        + "<extra></extra>",
      showlegend: false,
      name: "vector values",
    },
  ];

  // Report full-plane peak, not decimated — "max |V_plane|" must not change with stride.
  const mag = frameMax;
  const xUnit = xUnitHover;
  const yUnit = yUnitHover;
  const vectorUnit = xUnit && xUnit === yUnit ? xUnit : null;
  const unitSuffix = vectorUnit ? ` ${vectorUnit}` : "";
  // Disclose extra thinning: control may say "1/2" while picture is coarser.
  const thinned = quiver.effectiveStride > Math.max(1, Math.floor(stride));
  const title =
    `${xName ?? "?"} (→) , ${yName ?? "?"} (↑) — ${axisLabel(sliceAxis, grid)}-slice` +
    (mag > 0
      ? `  ·  max |V_${samplingDisclosure ? "display" : "plane"}| = ${mag.toPrecision(3)}${unitSuffix}`
        + (reference > mag ? `  ·  arrows scaled to ${reference.toPrecision(3)}${unitSuffix}` : "")
      : "")
    + (samplingDisclosure ? `  ·  ${samplingDisclosure}` : "")
    + (thinned
      ? `  ·  thinned to every ${quiver.effectiveStride} cells (${quiver.arrowCount} arrows)`
      : "");

  // A quiver plane is always commensurable. Releasing this lock would change
  // apparent vector direction, so unlike scalar plots vectors cannot opt into Fill.
  const fit = fitAxisFragments(true, "true-shape");
  const layout: Partial<Layout> = {
    xaxis: { title: { text: xCoords.title }, ...fit.x },
    yaxis: { title: { text: yCoords.title }, ...fit.y },
    dragmode: dragMode,
    title: { text: title, font: { size: 13 } },
    annotations: referenceKey ? [{
      x: referenceKey.headX,
      y: referenceKey.y,
      ax: referenceKey.tailX,
      ay: referenceKey.y,
      xref: "x",
      yref: "y",
      axref: "x",
      ayref: "y",
      text: `${referenceKey.magnitude.toPrecision(3)}${unitSuffix}`,
      showarrow: true,
      arrowhead: 3,
      arrowsize: 1,
      arrowwidth: 1.4,
      arrowcolor: arrowColor,
      font: { size: 10, color: arrowColor },
      bgcolor: "rgba(0,0,0,0)",
      borderpad: 2,
    }] : [],
  };

  return { data, layout, maxMagnitude: frameMax };
}
